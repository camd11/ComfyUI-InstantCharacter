from typing import Optional

import torch.nn as nn
import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from einops import rearrange

from .norm_layer import RMSNorm


class FluxIPAttnProcessor(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(
        self,
        hidden_size=None,
        ip_hidden_states_dim=None,
    ):
        super().__init__()
        self.norm_ip_q = RMSNorm(128, eps=1e-6)
        self.to_k_ip = nn.Linear(ip_hidden_states_dim, hidden_size)
        self.norm_ip_k = RMSNorm(128, eps=1e-6)
        self.to_v_ip = nn.Linear(ip_hidden_states_dim, hidden_size)
        # Store references passed during init (though not strictly needed if accessed via attn.model)
        # self.main_transformer_model_ref = main_transformer_model_ref
        # self.image_projector_module_ref = image_projector_module_ref


    def __call__(
        self,
        attn, # The attention module this processor is attached to (e.g., FluxAttention)
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        # **joint_attention_kwargs might be passed here by diffusers/sampler
        **kwargs,
    ) -> torch.FloatTensor:
        
        # Retrieve conditioning data stored on the main model by InstantCharacterApply node
        # Assumes the main model is accessible via attn.model (common pattern)
        # or needs to be passed/accessed differently depending on FluxAttention structure
        main_model = attn.model if hasattr(attn, "model") else None # Need a way to access the main model
        ip_kwargs = getattr(main_model, "_ip_kwargs", None) if main_model else None

        ip_hidden_states = None
        scale = 0.0
        if ip_kwargs and ip_kwargs.get("image_embeds") is not None and ip_kwargs.get("projector") is not None:
            image_tokens = ip_kwargs["image_embeds"]
            projector = ip_kwargs["projector"]
            scale = ip_kwargs["scale"]
            
            # Get current timestep - crucial for projection
            # How timestep is passed depends on the transformer's forward signature.
            # Common names: 'timestep', 't', 'added_cond_kwargs' containing timestep. Check Flux transformer impl.
            # Placeholder: Assume timestep is available in kwargs
            timestep = kwargs.get("timestep", None)
            if timestep is None and "added_cond_kwargs" in kwargs and isinstance(kwargs["added_cond_kwargs"], dict):
                 timestep = kwargs["added_cond_kwargs"].get("timestep", None)

            if timestep is not None and scale > 0:
                 # Ensure projector is on correct device (might have been offloaded)
                 projector.to(hidden_states.device, dtype=hidden_states.dtype)
                 # Project image tokens using the current timestep
                 with torch.no_grad():
                     # Projector expects specific inputs: low_res_shallow, low_res_deep, high_res_deep
                     # Retrieve the dict prepared by InstantCharacterApply node
                     image_embeds_dict = ip_kwargs.get("image_embeds") # This should now be the dict
                     if image_embeds_dict is None or not all(k in image_embeds_dict for k in ['image_embeds_low_res_shallow', 'image_embeds_low_res_deep', 'image_embeds_high_res_deep']):
                          print("Warning: image_embeds_dict not found in ip_kwargs or missing required keys.")
                          ip_hidden_states = None
                     else:
                          # Ensure embeddings are on the correct device/dtype before passing to projector
                          low_res_shallow = image_embeds_dict['image_embeds_low_res_shallow'].to(hidden_states.device, dtype=hidden_states.dtype)
                          low_res_deep = image_embeds_dict['image_embeds_low_res_deep'].to(hidden_states.device, dtype=hidden_states.dtype)
                          high_res_deep = image_embeds_dict['image_embeds_high_res_deep'].to(hidden_states.device, dtype=hidden_states.dtype)

                          proj_output = projector(
                              low_res_shallow=low_res_shallow,
                              low_res_deep=low_res_deep,
                              high_res_deep=high_res_deep,
                              timesteps=timestep.to(dtype=hidden_states.dtype),
                              need_temb=False # We only need the projected embeddings here
                          )
                          if isinstance(proj_output, tuple):
                              ip_hidden_states = proj_output[0]
                          else:
                              ip_hidden_states = proj_output
            else:
                 if scale <= 0: print("IP-Adapter scale is 0, skipping projection.")
                 if timestep is None: print("Warning: Timestep not found for IP-Adapter projection.")

        else:
            # print("IP-Adapter conditioning data not found on model or incomplete.")
            pass # No IP-Adapter effect if data is missing


        # Original Attention Calculation starts here
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Calculate IP-Adapter attention contribution if ip_hidden_states are available
        ip_attn_contribution = None
        if ip_hidden_states is not None and scale > 0:
             # Use the IP-Adapter's K, V projections with the projected image embeddings
             # Note: query comes from the main hidden_states
             ip_key = self.to_k_ip(ip_hidden_states)
             ip_value = self.to_v_ip(ip_hidden_states)

             # Apply norms if they exist (RMSNorm in this class)
             # Query norm needs to happen *after* potential rotary embeddings if used
             # Key norm:
             ip_key = self.norm_ip_k(rearrange(ip_key, 'b l (h d) -> b h l d', h=attn.heads))
             ip_key = rearrange(ip_key, 'b h l d -> (b h) l d')

             # Reshape query, ip_key, ip_value for attention
             inner_dim = key.shape[-1]
             head_dim = inner_dim // attn.heads
             ip_query_for_ip = self.norm_ip_q(rearrange(query, 'b l (h d) -> b h l d', h=attn.heads))
             ip_query_for_ip = rearrange(ip_query_for_ip, 'b h l d -> (b h) l d')

             ip_key = attn.head_to_batch_dim(ip_key) # Shape: (bs*heads, num_ip_tokens, head_dim)
             ip_value = attn.head_to_batch_dim(ip_value) # Shape: (bs*heads, num_ip_tokens, head_dim)

             # Calculate scaled dot product attention between query and ip_key/ip_value
             ip_attn_contribution = self._scaled_dot_product_attention(
                 ip_query_for_ip.to(ip_value.dtype),
                 ip_key.to(ip_value.dtype),
                 ip_value,
                 heads=attn.heads
             ) # Shape: (bs*heads, seq_len, head_dim)
             ip_attn_contribution = ip_attn_contribution.to(query.dtype)
             ip_attn_contribution = attn.batch_to_head_dim(ip_attn_contribution) # Shape: (bs, seq_len, inner_dim)


        # Reshape main Q, K, V
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)


        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
                
            # Apply IP-Adapter contribution scaled
            if ip_attn_contribution is not None:
                 # Check if encoder_hidden_states were used (cross-attention)
                 if encoder_hidden_states is not None:
                      # Only add contribution to the non-encoder part of hidden_states
                      hidden_states[:, encoder_hidden_states.shape[1] :] = \
                          hidden_states[:, encoder_hidden_states.shape[1] :] + ip_attn_contribution * scale
                 else:
                      # Add contribution to all hidden_states (self-attention)
                      hidden_states = hidden_states + ip_attn_contribution * scale

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:

            # Apply IP-Adapter contribution scaled
            if ip_attn_contribution is not None:
                 hidden_states = hidden_states + ip_attn_contribution * scale

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            return hidden_states


    def _scaled_dot_product_attention(self, query, key, value, attention_mask=None, heads=None):
        query = rearrange(query, '(b h) l c -> b h l c', h=heads)
        key = rearrange(key, '(b h) l c -> b h l c', h=heads)
        value = rearrange(value, '(b h) l c -> b h l c', h=heads)
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=None)
        hidden_states = rearrange(hidden_states, 'b h l c -> (b h) l c', h=heads)
        hidden_states = hidden_states.to(query)
        return hidden_states


    # This helper function is replaced by the logic integrated into __call__
    # def _get_ip_hidden_states( ... )

