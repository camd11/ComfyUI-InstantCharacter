# Copyright 2025 Tencent InstantX Team. All rights reserved.
#

from PIL import Image
from einops import rearrange
import torch
import torch.nn as nn # Added
import numpy as np # Added for __call__
from typing import Union, List, Optional, Dict, Any, Callable # Added for __call__

# Removed: from diffusers.pipelines.flux.pipeline_flux import *
# Removed: from transformers import SiglipVisionModel, SiglipImageProcessor, AutoModel, AutoImageProcessor
# Import for FluxPipelineOutput if used, or define a similar structure
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput # Keep if __call__ returns this
from diffusers.utils import replace_example_docstring # Keep if EXAMPLE_DOC_STRING is used
from diffusers.utils.torch_utils import randn_tensor # For latent generation if not from model object

from .models.attn_processor import FluxIPAttnProcessor
from .models.resampler import CrossLayerCrossScaleProjector
from .models.utils import flux_load_lora # Keep for with_style_lora, may need adaptation


# TODO: This EXAMPLE_DOC_STRING might need to be adapted if FluxPipeline is not the base
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> # pipe = InstantCharacterFluxPipeline(...) # Initialization will change
        >>> # pipe.to("cuda")
        >>> # prompt = "A cat holding a sign that says hello world"
        >>> # image = pipe(prompt, subject_image=...).images[0] # Call will change
        >>> # image.save("flux_instant_character.png")
        ```
"""

# Helper functions that might be needed if not inheriting from FluxPipeline
# These are simplified placeholders and would need robust implementation
def calculate_shift(image_seq_len, base_image_seq_len, max_image_seq_len, base_shift, max_shift):
    # Placeholder for shift calculation logic from FLUX
    return base_shift + (max_shift - base_shift) * (image_seq_len - base_image_seq_len) / (
        max_image_seq_len - base_image_seq_len
    )

def retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=None, mu=0.0):
    # Placeholder for timestep retrieval
    if hasattr(scheduler, "set_timesteps_mu"):
        scheduler.set_timesteps_mu(num_inference_steps, sigmas=sigmas, mu=mu, device=device)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    return timesteps, len(timesteps)


class InstantCharacterFluxPipeline(nn.Module): # Changed base class
    def __init__(self,
                 flux_unet_model_object,      # ComfyUI MODEL object
                 vae_module,                  # VAE nn.Module
                 siglip_vision_model_object,  # ComfyUI CLIP_VISION object for SigLIP
                 dinov2_vision_model_object,  # ComfyUI CLIP_VISION object for DINOv2
                 ipadapter_model_data_dict,   # Dict of IPAdapter weights
                 dtype=torch.bfloat16):
        """
        Initializes the InstantCharacterFluxPipeline with pre-loaded model components.
        """
        super().__init__()

        self.dtype = dtype
        self.device = flux_unet_model_object.model.device # Assuming model object has a device attribute or a model attribute with device

        # Assign FLUX components from flux_unet_model_object
        # These attributes are based on typical ComfyUI MODEL object structure and FLUX needs
        self.transformer = flux_unet_model_object.model
        if not hasattr(self.transformer, 'attn_processors'):
            self.transformer.attn_processors = {}
        self.vae = vae_module
        
        # Text encoders, tokenizers, and scheduler are expected to be part of flux_unet_model_object
        # or need to be passed separately if not.
        # For FLUX, there are typically two text encoders.
        if hasattr(flux_unet_model_object, 'text_encoder_one') and hasattr(flux_unet_model_object, 'tokenizer_one'):
            self.text_encoder = flux_unet_model_object.text_encoder_one
            self.tokenizer = flux_unet_model_object.tokenizer_one
        else:
            # Fallback or error if not found, as these are crucial for prompt encoding
            # This part depends heavily on how the ComfyUI MODEL object for FLUX is structured
            print("Warning: Text Encoder One / Tokenizer One not found directly on flux_unet_model_object.")
            self.text_encoder = None # Placeholder, will cause issues if not properly set
            self.tokenizer = None  # Placeholder

        if hasattr(flux_unet_model_object, 'text_encoder_two') and hasattr(flux_unet_model_object, 'tokenizer_two'):
            self.text_encoder_2 = flux_unet_model_object.text_encoder_two
            self.tokenizer_2 = flux_unet_model_object.tokenizer_two
        else:
            print("Warning: Text Encoder Two / Tokenizer Two not found directly on flux_unet_model_object.")
            self.text_encoder_2 = None # Placeholder
            self.tokenizer_2 = None    # Placeholder

        if hasattr(flux_unet_model_object, 'scheduler'):
            self.scheduler = flux_unet_model_object.scheduler
        else:
            # Fallback: try to get config and initialize a scheduler
            # from diffusers.schedulers import FluxScheduler # Example
            # self.scheduler = FluxScheduler.from_config(flux_unet_model_object.model_config.scheduler)
            print("Warning: Scheduler not found directly on flux_unet_model_object. Needs manual setup or to be included in MODEL.")
            self.scheduler = None # Placeholder

        # Assign Image Encoders from CLIP_VISION objects
        self.siglip_image_encoder_model = siglip_vision_model_object.model
        self.siglip_image_processor = siglip_vision_model_object # ComfyUI CLIP_VISION object itself for preprocessing info

        self.dinov2_image_encoder_model = dinov2_vision_model_object.model
        self.dinov2_image_processor = dinov2_vision_model_object # ComfyUI CLIP_VISION object itself for preprocessing info

        self._initialize_ip_adapter_components(ipadapter_model_data_dict, self.dtype)

        # For progress bar in __call__ if not inheriting from DiffusionPipeline
        self.progress_bar = lambda x: x # Simple placeholder

        # Attributes that were previously inherited from FluxPipeline/DiffusionPipeline
        # These might need to be set if __call__ logic relies on them.
        # self.vae_scale_factor = getattr(self.vae.config, "scale_factor", 0.13025) # Example, verify FLUX VAE scale
        # self.default_sample_size = getattr(self.transformer.config, "sample_size", 128) # Example for FLUX unet sample size (latent)
        # self._execution_device = self.device # Already set as self.device

    def _initialize_ip_adapter_components(self, ipadapter_state_dict, dtype):
        """
        Initializes IPAdapter attention processors and image projection model.
        """
        device = self.device # Use device from __init__

        # Initialize and load FluxIPAttnProcessor
        attn_procs = {}
        # Dimensions need to be accurate for FLUX.1-dev or configurable.
        # Try to get from transformer/text_encoder config if available
        try:
            flux_transformer_hidden_size = self.transformer.config.hidden_size # Or equivalent for FLUX UNet
            # flux_transformer_hidden_size = self.transformer.config.attention_head_dim * self.transformer.config.num_attention_heads
        except AttributeError:
            print("Warning: Could not infer flux_transformer_hidden_size from transformer.config. Using default 4096.")
            flux_transformer_hidden_size = 4096 # Default from design doc

        try:
            # FLUX often uses text_encoder_2 for IP-Adapter conditioning
            flux_text_encoder_dim = self.text_encoder_2.config.hidden_size # Or d_model
        except AttributeError:
            print("Warning: Could not infer flux_text_encoder_dim from text_encoder_2.config. Using default 4096.")
            flux_text_encoder_dim = 4096    # Default from design doc

        for name in self.transformer.attn_processors.keys():
            attn_procs[name] = FluxIPAttnProcessor(
                hidden_size=flux_transformer_hidden_size,
                ip_hidden_states_dim=flux_text_encoder_dim,
            ).to(device, dtype=dtype)
        self.transformer.attn_processors = attn_procs
        
        tmp_ip_layers = torch.nn.ModuleList(list(self.transformer.attn_processors.values())) # Ensure it's a list for ModuleList
        if "ip_adapter" in ipadapter_state_dict:
            tmp_ip_layers.load_state_dict(ipadapter_state_dict["ip_adapter"], strict=False)
            print("=> IP Adapter Attention Processor weights loaded.")
        else:
            print("Warning: 'ip_adapter' key not found in ipadapter_state_dict. Attn processor weights not loaded.")


        # Initialize and load CrossLayerCrossScaleProjector (image_proj_model)
        self.subject_image_proj_model = CrossLayerCrossScaleProjector(
            inner_dim=1152 + 1536, num_attention_heads=42, attention_head_dim=64,
            cross_attention_dim=1152 + 1536, num_layers=4, dim=1280, depth=4,
            dim_head=64, heads=20, num_queries=1024, # nb_token
            embedding_dim=1152 + 1536, output_dim=flux_transformer_hidden_size, # output_dim should match transformer's input for IP
            ff_mult=4,
            timestep_in_dim=320, timestep_flip_sin_to_cos=True, timestep_freq_shift=0,
        ).to(device, dtype=dtype)
        self.subject_image_proj_model.eval()
        if "image_proj" in ipadapter_state_dict:
            self.subject_image_proj_model.load_state_dict(ipadapter_state_dict["image_proj"], strict=False)
            print("=> IP Adapter Image Projection weights loaded.")
        else:
            print("Warning: 'image_proj' key not found in ipadapter_state_dict. Image projection weights not loaded.")

    def _comfy_clip_vision_preprocess_pil(self, clip_vision_obj, pil_images_list: List[Image.Image]):
        """
        Helper to preprocess PIL images using ComfyUI's CLIPVision object.
        This is a placeholder. Actual implementation depends on how ComfyUI's CLIP_VISION
        objects expose their preprocessing (e.g., if they are callable or have a specific method).
        For now, assumes it might have a 'preprocess' method or can be called.
        It should return a pixel_values tensor.
        """
        # This is a simplified placeholder. ComfyUI's `clip_preprocess` is more complex.
        # It typically involves converting PIL to tensor (0-1 range), then normalizing.
        # Example:
        # comfy_images = []
        # for pil_image in pil_images_list:
        #     np_image = np.array(pil_image).astype(np.float32) / 255.0
        #     comfy_images.append(torch.from_numpy(np_image))
        # comfy_image_tensor = torch.stack(comfy_images).permute(0, 3, 1, 2) # B, C, H, W

        # if hasattr(clip_vision_obj, 'preprocess'):
        #     # Assuming clip_vision_obj.preprocess takes a list of PIL images
        #     # and returns a tensor of pixel_values
        #     return clip_vision_obj.preprocess(pil_images_list)
        # elif callable(clip_vision_obj):
        #     # This is a guess; ComfyUI CLIP_VISION objects might not be directly callable for preprocessing
        #     # Or they might expect a ComfyUI-formatted tensor input
        #     # This part needs to align with how ComfyUI's CLIPLoader makes CLIPVision objects work.
        #     # For now, let's assume it can take PIL images and return processed tensors.
        #     # This is highly dependent on the actual ComfyUI CLIP_VISION wrapper.
        #     # A common pattern is that the CLIPVision object from ComfyUI might have the processor
        #     # embedded, e.g. clip_vision_obj.processor(images=pil_images_list, return_tensors="pt").pixel_values
        #     # This matches the old pipeline's direct use of HF processors.
        if hasattr(clip_vision_obj, 'processor') and callable(clip_vision_obj.processor):
             return clip_vision_obj.processor(images=pil_images_list, return_tensors="pt").pixel_values.to(self.device, self.dtype)
        elif hasattr(clip_vision_obj, 'image_processor') and callable(clip_vision_obj.image_processor): # common in some comfy wrappers
             return clip_vision_obj.image_processor(images=pil_images_list, return_tensors="pt").pixel_values.to(self.device, self.dtype)
        else:
            # Fallback: if the CLIP_VISION object *is* the processor (like HF processors)
            try:
                return clip_vision_obj(images=pil_images_list, return_tensors="pt").pixel_values.to(self.device, self.dtype)
            except Exception as e:
                print(f"Warning: Could not preprocess with clip_vision_obj directly: {e}. Preprocessing might fail.")
                # Return a dummy tensor of expected shape if possible, or raise error
                # This is a critical part that needs to work with ComfyUI's CLIP_VISION type.
                # For now, let's assume the object itself is a processor-like callable.
                # This matches the structure of the original code using HF processors.
                raise NotImplementedError(
                    "Preprocessing with ComfyUI CLIP_VISION object needs concrete implementation "
                    "or verification of its callable nature for preprocessing."
                )


    @torch.inference_mode()
    def encode_siglip_image_emb(self, siglip_pixel_values, device, dtype):
        # siglip_pixel_values are preprocessed tensors.
        self.siglip_image_encoder_model.to(device, dtype=dtype)
        # siglip_pixel_values already moved to device by _comfy_clip_vision_preprocess_pil
        res = self.siglip_image_encoder_model(siglip_pixel_values, output_hidden_states=True)
        siglip_image_embeds = res.last_hidden_state
        # Verify hidden state indices [7, 13, 26] for the specific SigLIP model
        siglip_image_shallow_embeds = torch.cat([res.hidden_states[i] for i in [7, 13, 26]], dim=1)
        return siglip_image_embeds, siglip_image_shallow_embeds

    @torch.inference_mode()
    def encode_dinov2_image_emb(self, dinov2_pixel_values, device, dtype):
        # dinov2_pixel_values are preprocessed tensors.
        self.dinov2_image_encoder_model.to(device, dtype=dtype)
        # dinov2_pixel_values already moved to device by _comfy_clip_vision_preprocess_pil
        res = self.dinov2_image_encoder_model(dinov2_pixel_values, output_hidden_states=True)
        # Verify hidden state indices [9, 19, 29] for the specific DINOv2 model
        dinov2_image_embeds = res.last_hidden_state[:, 1:] # Exclude CLS token
        dinov2_image_shallow_embeds = torch.cat([res.hidden_states[i][:, 1:] for i in [9, 19, 29]], dim=1)
        return dinov2_image_embeds, dinov2_image_shallow_embeds

    @torch.inference_mode()
    def encode_image_emb(self, subject_image_pil: Image.Image, device, dtype):
        # Cropping and resizing logic for low_res and high_res PIL images
        object_image_pil_low_res = [subject_image_pil.resize((384, 384))]
        object_image_pil_high_res_orig = subject_image_pil.resize((768, 768))
        object_image_pil_high_res_crops = [
            object_image_pil_high_res_orig.crop((0, 0, 384, 384)),
            object_image_pil_high_res_orig.crop((384, 0, 768, 384)),
            object_image_pil_high_res_orig.crop((0, 384, 384, 768)),
            object_image_pil_high_res_orig.crop((384, 384, 768, 768)),
        ]
        nb_split_image = len(object_image_pil_high_res_crops)

        # Preprocess PIL images using the ComfyUI CLIP_VISION objects
        siglip_low_res_pixels = self._comfy_clip_vision_preprocess_pil(self.siglip_image_processor, object_image_pil_low_res)
        dinov2_low_res_pixels = self._comfy_clip_vision_preprocess_pil(self.dinov2_image_processor, object_image_pil_low_res)

        siglip_embeds_tuple = self.encode_siglip_image_emb(siglip_low_res_pixels, device, dtype)
        dinov2_embeds_tuple = self.encode_dinov2_image_emb(dinov2_low_res_pixels, device, dtype)

        image_embeds_low_res_deep = torch.cat([siglip_embeds_tuple[0], dinov2_embeds_tuple[0]], dim=2)
        image_embeds_low_res_shallow = torch.cat([siglip_embeds_tuple[1], dinov2_embeds_tuple[1]], dim=2)

        # High-resolution processing
        siglip_high_res_pixels = self._comfy_clip_vision_preprocess_pil(self.siglip_image_processor, object_image_pil_high_res_crops)
        # siglip_high_res_pixels = siglip_high_res_pixels[None] # Assuming preprocess handles batching correctly
        siglip_high_res_pixels = rearrange(siglip_high_res_pixels, '(b n) c h w -> (b n) c h w', b=1, n=nb_split_image) # No-op if already batched by preprocess
        
        siglip_high_res_embeds_tuple = self.encode_siglip_image_emb(siglip_high_res_pixels, device, dtype)
        siglip_image_high_res_deep = rearrange(siglip_high_res_embeds_tuple[0], '(b n) l c -> b (n l) c', n=nb_split_image)

        dinov2_high_res_pixels = self._comfy_clip_vision_preprocess_pil(self.dinov2_image_processor, object_image_pil_high_res_crops)
        # dinov2_high_res_pixels = dinov2_high_res_pixels[None]
        dinov2_high_res_pixels = rearrange(dinov2_high_res_pixels, '(b n) c h w -> (b n) c h w', b=1, n=nb_split_image)

        dinov2_high_res_embeds_tuple = self.encode_dinov2_image_emb(dinov2_high_res_pixels, device, dtype)
        dinov2_image_high_res_deep = rearrange(dinov2_high_res_embeds_tuple[0], '(b n) l c -> b (n l) c', n=nb_split_image)
        
        image_embeds_high_res_deep = torch.cat([siglip_image_high_res_deep, dinov2_image_high_res_deep], dim=2)

        image_embeds_dict = dict(
            image_embeds_low_res_shallow=image_embeds_low_res_shallow.to(device=device, dtype=dtype),
            image_embeds_low_res_deep=image_embeds_low_res_deep.to(device=device, dtype=dtype),
            image_embeds_high_res_deep=image_embeds_high_res_deep.to(device=device, dtype=dtype),
        )
        return image_embeds_dict

    # Removed init_ccp_and_attn_processor
    # Removed init_adapter

    # encode_prompt method (simplified, adapted from Diffusers' CLIP and T5 handling)
    # This is a complex part and needs to be accurate for FLUX's dual text encoder setup.
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]],
        device,
        num_images_per_prompt: int,
        max_sequence_length: int,
        # Other params like do_classifier_free_guidance, negative_prompt etc. are handled by __call__ logic
    ):
        if self.tokenizer is None or self.text_encoder is None or \
           self.tokenizer_2 is None or self.text_encoder_2 is None:
            raise ValueError("Tokenizers and Text Encoders must be initialized.")

        if prompt_2 is None:
            prompt_2 = prompt

        # Tokenize and encode with first text encoder (e.g., CLIP L for FLUX)
        text_inputs_one = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length, # self.tokenizer.model_max_length often 77 for CLIP
            truncation=True,
            return_tensors="pt",
        )
        text_ids_one = text_inputs_one.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_ids_one.shape[-1] and not torch.equal(text_ids_one, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            print(f"Warning: The following part of your input was truncated because CLIP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}")
        
        prompt_embeds_one = self.text_encoder(text_ids_one.to(device), output_hidden_states=True)
        # FLUX might use specific hidden states or pooled output, this is a general CLIP approach
        # pooled_prompt_embeds_one = prompt_embeds_one.pooled_output
        prompt_embeds_one = prompt_embeds_one.last_hidden_state


        # Tokenize and encode with second text encoder (e.g., CLIP G or T5-xxl for FLUX)
        text_inputs_two = self.tokenizer_2(
            prompt_2,
            padding="max_length",
            max_length=max_sequence_length, # self.tokenizer_2.model_max_length (e.g. 256 or 512 for T5)
            truncation=True,
            return_tensors="pt",
        )
        text_ids_two = text_inputs_two.input_ids
        untruncated_ids_two = self.tokenizer_2(prompt_2, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids_two.shape[-1] >= text_ids_two.shape[-1] and not torch.equal(text_ids_two, untruncated_ids_two):
            removed_text_two = self.tokenizer_2.batch_decode(untruncated_ids_two[:, self.tokenizer_2.model_max_length - 1 : -1])
            print(f"Warning: The following part of your input was truncated because Text Encoder 2 can only handle sequences up to {self.tokenizer_2.model_max_length} tokens: {removed_text_two}")

        prompt_embeds_two = self.text_encoder_2(text_ids_two.to(device), output_hidden_states=True)
        # FLUX uses specific outputs from its text encoders.
        # For FLUX.1-dev, text_encoder_2 (CLIP-G based) provides pooled_output and last_hidden_state
        # The `pooled_prompt_embeds` for FLUX usually comes from text_encoder_2
        pooled_prompt_embeds = prompt_embeds_two.pooled_output # This is typical for FLUX
        prompt_embeds_two = prompt_embeds_two.last_hidden_state

        # Concatenate or combine embeddings as FLUX expects
        # FLUX.1-dev concatenates the hidden states from both encoders
        prompt_embeds = torch.cat((prompt_embeds_one, prompt_embeds_two), dim=-1)
        
        # Duplicate embeddings for each image per prompt
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        bs_embed_pooled, _ = pooled_prompt_embeds.shape
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed_pooled * num_images_per_prompt, -1)
        
        # For FLUX, text_ids are also passed to the transformer. Usually from the larger text encoder.
        # This needs to match what the FLUX transformer expects.
        # Let's assume text_ids_two are used, or a combined version if necessary.
        text_ids_for_transformer = text_ids_two.repeat(1, num_images_per_prompt).view(bs_embed * num_images_per_prompt, -1)


        return prompt_embeds.to(device=device, dtype=self.dtype), \
               pooled_prompt_embeds.to(device=device, dtype=self.dtype), \
               text_ids_for_transformer.to(device=device) # Or appropriate text_ids for FLUX transformer

    # prepare_latents method (simplified, from Diffusers)
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            height // getattr(self.vae.config, "downscale_factor", 8), # Use VAE's scale factor
            width // getattr(self.vae.config, "downscale_factor", 8),
        )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # FLUX uses specific latent image IDs
        latent_image_ids = torch.tensor([list(range(shape[2] * shape[3]))], device=device).expand(
            batch_size, shape[2] * shape[3]
        )
        # scale the latents by the scheduler's init_sigma
        # latents = latents * self.scheduler.init_sigma # If scheduler has init_sigma
        return latents, latent_image_ids

    # _unpack_latents (simplified from FLUX)
    def _unpack_latents(self, latents, height, width, vae_scale_factor):
        # This is a placeholder. FLUX has specific logic for unpacking B C H W latents
        # if they are packed differently by the VAE.
        # For a standard VAE, this might not be needed or might be simpler.
        return latents # Assuming standard VAE output that doesn't need complex unpacking

    # image_processor.postprocess (simplified placeholder if not using Diffusers ImageProcessor)
    def _postprocess_image(self, image, output_type="pil"):
        if output_type == "pil":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image * 255).round().astype("uint8")
            if image.shape[0] == 1:
                return [Image.fromarray(image[0])]
            return [Image.fromarray(img) for img in image]
        elif output_type == "latent":
            return image # No processing for latent output
        else: # np
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            return image


    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING) # This decorator might need removal if base class changes
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None, # FLUX uses prompt_2
        negative_prompt: Union[str, List[str]] = None, # Added for CFG
        negative_prompt_2: Optional[Union[str, List[str]]] = None, # Added for CFG
        true_cfg_scale: float = 1.0, # For FLUX-like CFG
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        num_inference_steps: int = 20, # Changed default from 28
        sigmas: Optional[List[float]] = None, # For schedulers supporting custom sigmas
        guidance_scale: float = 7.5, # Changed default from 3.5
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None, # Allow pre-computed embeds
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None, # Allow pre-computed embeds
        # ip_adapter_image and ip_adapter_image_embeds are removed as per design for internal handling
        negative_prompt_embeds: Optional[torch.FloatTensor] = None, # Allow pre-computed negative embeds
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None, # Allow pre-computed negative embeds
        output_type: Optional[str] = "pil", # "pil", "latent", "np"
        return_dict: bool = True,
        # joint_attention_kwargs: Optional[Dict[str, Any]] = None, # Handled internally by IP-Adapter
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None, # For progress
        callback_on_step_end_tensor_inputs: List[str] = ["latents"], # For progress
        max_sequence_length: int = 512, # Max length for tokenizers
        subject_image: Image.Image = None, # Main subject image input
        subject_scale: float = 1.0, # Scale for subject IP-Adapter
    ):
        # This __call__ method is a significant adaptation.
        # It tries to follow FLUX logic using the components now available in `self`.
        # Many safety checks and utility functions from Diffusers' base Pipeline are missing.

        device = self.device
        dtype = self.dtype

        if height is None: height = 1024 # Default if not provided
        if width is None: width = 1024   # Default if not provided

        # 1. Check inputs (simplified)
        if prompt is None and prompt_embeds is None:
            raise ValueError("Either `prompt` or `prompt_embeds` must be provided.")
        if subject_image is None:
            raise ValueError("`subject_image` must be provided for InstantCharacter.")

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            batch_size = 1 # Should not happen if input check passes

        # 3. Encode prompt
        if prompt_embeds is None:
            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        else: # Assume text_ids also pre-computed or not strictly needed by this transformer version
            # This part is tricky if text_ids are essential for the transformer and not provided
            # For now, if prompt_embeds are given, we might not have text_ids.
            # FLUX transformer does take text_ids.
            print("Warning: Using pre-computed prompt_embeds. Ensure text_ids are handled if needed by the transformer.")
            # Placeholder for text_ids if prompt_embeds are pre-supplied
            # This needs a robust solution if pre-computed embeds are to be fully supported with FLUX.
            # For now, let's assume if prompt_embeds are passed, text_ids might be dummy or derived.
            # A simple approach: tokenize prompt to get text_ids even if embeds are passed.
            if prompt:
                 _, _, text_ids = self.encode_prompt(prompt, prompt_2, device, num_images_per_prompt, max_sequence_length)
            else: # Cannot derive text_ids if prompt is also None
                raise ValueError("If prompt_embeds are provided, prompt must also be provided to derive text_ids for FLUX transformer.")


        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                if negative_prompt is None: # Create unconditional guidance
                    negative_prompt = ""
                    if prompt_2 is not None: # If prompt_2 exists, uncond should also have two parts
                        negative_prompt_2 = ""
                
                uncond_prompt_embeds, uncond_pooled_prompt_embeds, uncond_text_ids = self.encode_prompt(
                    prompt=negative_prompt,
                    prompt_2=negative_prompt_2,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                )
            else:
                uncond_prompt_embeds = negative_prompt_embeds
                uncond_pooled_prompt_embeds = negative_pooled_prompt_embeds
                # Similar issue with uncond_text_ids if negative_prompt_embeds are pre-supplied
                if negative_prompt:
                    _, _, uncond_text_ids = self.encode_prompt(negative_prompt, negative_prompt_2, device, num_images_per_prompt, max_sequence_length)
                else:
                    raise ValueError("If negative_prompt_embeds are provided, negative_prompt must also be provided for uncond_text_ids.")


            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and conditional embeddings.
            prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds])
            pooled_prompt_embeds = torch.cat([uncond_pooled_prompt_embeds, pooled_prompt_embeds])
            text_ids = torch.cat([uncond_text_ids, text_ids])


        # 3.1 Prepare subject image embeddings
        subject_image_pil = subject_image.resize((max(subject_image.size), max(subject_image.size))) # Simple resize
        subject_image_embeds_dict = self.encode_image_emb(subject_image_pil, device, dtype)

        # 4. Prepare latent variables
        # VAE scale factor might be on vae.config.scale_factor or similar
        vae_scale_factor = getattr(self.vae.config, "downscale_factor", 8) # Common for /8 VAEs
        num_channels_latents = self.transformer.config.in_channels // 4 # FLUX specific latent channels
        
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype, # Use dtype of embeddings
            device,
            generator,
            latents,
        )
        if do_classifier_free_guidance: # Latents need to be duplicated for CFG
            latents = torch.cat([latents] * 2)
            latent_image_ids = torch.cat([latent_image_ids]*2)


        # 5. Prepare timesteps and scheduler
        if self.scheduler is None:
            raise ValueError("Scheduler is not initialized.")
        
        # FLUX specific mu calculation for scheduler
        image_seq_len = latents.shape[-2] * latents.shape[-1] # H * W of latent
        # Scheduler config access needs to be robust
        base_image_seq_len = getattr(self.scheduler.config, "base_image_seq_len", 256*256) # Example
        max_image_seq_len = getattr(self.scheduler.config, "max_image_seq_len", 256*256) # Example
        base_shift = getattr(self.scheduler.config, "base_shift", 0.25) # Example
        max_shift = getattr(self.scheduler.config, "max_shift", 0.25) # Example

        mu = calculate_shift(image_seq_len, base_image_seq_len, max_image_seq_len, base_shift, max_shift)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )
        
        # Guidance embedding for FLUX transformer
        if self.transformer.config.guidance_embeds:
            guidance_emb = torch.full([1], guidance_scale if not do_classifier_free_guidance else 0.0, device=device, dtype=torch.float32) # Uncond guidance is 0
            guidance_emb = guidance_emb.expand(batch_size * num_images_per_prompt)
            if do_classifier_free_guidance:
                 # Conditional part of CFG uses the actual guidance_scale
                cond_guidance_emb = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                cond_guidance_emb = cond_guidance_emb.expand(batch_size * num_images_per_prompt)
                guidance_emb = torch.cat([guidance_emb, cond_guidance_emb])
        else:
            guidance_emb = None
            
        # Prepare joint_attention_kwargs for IP-Adapter
        joint_attention_kwargs = {}

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance and i==0 and not (latents.shape[0] == batch_size * num_images_per_prompt * 2) else latents
                # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) # If scheduler needs this

                timestep_expanded = t.expand(latent_model_input.shape[0]).to(dtype=dtype) # Ensure dtype match

                # Subject IP-Adapter injection
                self.subject_image_proj_model.to(device, dtype=dtype) # Ensure device
                
                # Prepare subject embeddings for the current batch size (considering CFG)
                current_bs = latent_model_input.shape[0]
                
                # Tile subject embeds if batch size of latents is larger (e.g. due to CFG)
                # Original subject_image_embeds_dict is for batch_size * num_images_per_prompt
                # If CFG, latents are 2 * (batch_size * num_images_per_prompt)
                # So, subject embeds also need to be tiled for the unconditional part.
                # For unconditional, IP adapter effect should ideally be neutral or scaled down.
                # Simplest: repeat. More advanced: use zero embeds for uncond pass or scale down.
                
                # For now, let's assume the IP-Adapter is applied to both cond and uncond passes if not handled by scale=0 for uncond.
                # The scale parameter in FluxIPAttnProcessor can handle this if set to 0 for uncond.
                
                # We need to ensure subject_image_embeds_dict tensors are correctly batched for CFG
                # If latent_model_input.shape[0] is 2 * original_batch_size, then tile embeds
                if latent_model_input.shape[0] > subject_image_embeds_dict['image_embeds_low_res_shallow'].shape[0]:
                    factor = latent_model_input.shape[0] // subject_image_embeds_dict['image_embeds_low_res_shallow'].shape[0]
                    s_low_shallow = subject_image_embeds_dict['image_embeds_low_res_shallow'].repeat(factor, 1, 1)
                    s_low_deep = subject_image_embeds_dict['image_embeds_low_res_deep'].repeat(factor, 1, 1)
                    s_high_deep = subject_image_embeds_dict['image_embeds_high_res_deep'].repeat(factor, 1, 1)
                else:
                    s_low_shallow = subject_image_embeds_dict['image_embeds_low_res_shallow']
                    s_low_deep = subject_image_embeds_dict['image_embeds_low_res_deep']
                    s_high_deep = subject_image_embeds_dict['image_embeds_high_res_deep']

                subject_image_projected_embeds = self.subject_image_proj_model(
                    low_res_shallow=s_low_shallow.to(device, dtype),
                    low_res_deep=s_low_deep.to(device, dtype),
                    high_res_deep=s_high_deep.to(device, dtype),
                    timesteps=timestep_expanded.to(device), # Ensure timestep is on correct device
                    need_temb=True
                )[0]

                joint_attention_kwargs['emb_dict'] = dict(
                    length_encoder_hidden_states=prompt_embeds.shape[1] # Length of the text embeds
                )
                joint_attention_kwargs['subject_emb_dict'] = dict(
                    ip_hidden_states=subject_image_projected_embeds,
                    scale=subject_scale, # Apply scale here
                )
                
                # For CFG, the unconditional pass should ideally have scale=0 for IP-Adapter
                # This needs to be handled by FluxIPAttnProcessor or by modifying subject_emb_dict for the uncond part
                if do_classifier_free_guidance:
                    # Create a scale tensor: 0 for uncond, subject_scale for cond
                    cfg_ip_scale = torch.tensor([0.0] * (current_bs // 2) + [subject_scale] * (current_bs // 2), device=device, dtype=dtype).view(-1,1,1)
                    # This assumes FluxIPAttnProcessor can take a scale tensor. If not, this logic needs adjustment.
                    # Or, pass two different subject_emb_dicts if the processor is called per CFG half.
                    # For now, let's assume the processor handles a single call with batched CFG inputs.
                    # The 'scale' in subject_emb_dict might need to be a tensor if it varies per batch item.
                    # For simplicity, the current FluxIPAttnProcessor takes a single float scale.
                    # A more robust way: call transformer twice if IP-adapter scale needs to differ for CFG.
                    # Or, modify FluxIPAttnProcessor to accept scale tensor.
                    # For now, the `subject_scale` will apply to both if not handled inside attn_proc.
                    # This is a simplification.
                    pass


                # Predict the noise residual
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep_expanded / 1000, # FLUX scales timestep
                    guidance=guidance_emb,
                    pooled_projections=pooled_prompt_embeds, # Already CFG-batched if needed
                    encoder_hidden_states=prompt_embeds,   # Already CFG-batched if needed
                    txt_ids=text_ids,                      # Already CFG-batched if needed
                    img_ids=latent_image_ids,              # Already CFG-batched if needed
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if true_cfg_scale > 1.0: # FLUX specific true_cfg
                         # This part needs careful adaptation from original FLUX true_cfg logic
                         # It might involve another model call or specific scaling of noise_pred_text
                         # For now, this is a simplified CFG.
                         pass


                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    # Simplified callback handling
                    callback_kwargs = {"latents": latents}
                    callback_on_step_end(self, i, t, callback_kwargs)

                if i == len(timesteps) - 1 or ((i + 1) >= num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
            
        # 7. Post-processing
        if output_type == "latent":
            image = latents
        else:
            # Ensure VAE is on the correct device
            self.vae.to(device, dtype=dtype)
            # FLUX VAE might have specific scaling/shifting factors
            vae_scaling_factor = getattr(self.vae.config, "scaling_factor", 0.13025) # Example
            vae_shift_factor = getattr(self.vae.config, "shift_factor", 0.0)       # Example
            
            latents = self._unpack_latents(latents, height, width, vae_scale_factor) # vae_scale_factor here is for unpack, not decode
            latents = (latents / vae_scaling_factor) + vae_shift_factor
            image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0] # Ensure latents match VAE dtype
            image = self._postprocess_image(image, output_type=output_type)


        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image) # Ensure FluxPipelineOutput is defined or imported


    def with_style_lora(self, lora_file_path, lora_weight=1.0, trigger='', *args, **kwargs):
        # This method might need significant changes if flux_load_lora relies on Diffusers pipeline structure
        print("Applying LoRA. Ensure flux_load_lora is compatible with the new pipeline structure.")
        flux_load_lora(self, lora_file_path, lora_weight) # self here is InstantCharacterFluxPipeline
        
        current_prompt = kwargs.get("prompt", "")
        if isinstance(current_prompt, list):
            kwargs['prompt'] = [f"{trigger}, {p}" for p in current_prompt]
        else:
            kwargs['prompt'] = f"{trigger}, {current_prompt}"
            
        res = self.__call__(*args, **kwargs)
        
        print("Reverting LoRA.")
        flux_load_lora(self, lora_file_path, -lora_weight) # Attempt to revert
        return res

