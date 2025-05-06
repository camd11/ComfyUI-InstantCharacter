import os
import sys
import torch
import folder_paths
from PIL import Image
import numpy as np
import copy # For deep copying the model

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from InstantCharacter.models.attn_processor import FluxIPAttnProcessor
from InstantCharacter.models.resampler import CrossLayerCrossScaleProjector
# NOTE: Pipeline import might be unnecessary if logic moves into the Apply node or helper functions
# NOTE: Pipeline import might be unnecessary if logic moves into the Apply node or helper functions
# from InstantCharacter.pipeline import InstantCharacterFluxPipeline
# from huggingface_hub import login


if "ipadapter" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "ipadapter")]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ipadapter"]
# Ensure IPADAPTER type is recognized (might need adjustment based on actual IP-Adapter extension used)
folder_paths.folder_names_and_paths["ipadapter"] = (current_paths, list(folder_paths.supported_pt_extensions) + [".bin"])


# --- Apply Node (Refactored from Generate) ---
class InstantCharacterApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Model Inputs
                "base_model": ("MODEL",),
                "image_encoder_1": ("CLIP_VISION",),
                "image_encoder_2": ("CLIP_VISION",),
                "ip_adapter": ("IPADAPTER",), # Placeholder type, adjust if needed based on loader output
                # Generation Parameters
                "subject_scale": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                 # Subject image is still needed to get embeddings
                "subject_image": ("IMAGE",),
            }
        }

    # Outputting a modified MODEL for KSampler
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_instant_character"
    CATEGORY = "InstantCharacter"
    DESCRIPTION = "Applies InstantCharacter conditioning (IP-Adapter + Encoders) to a base model."

    def apply_instant_character(self, base_model, image_encoder_1, image_encoder_2, ip_adapter,
                                subject_scale, subject_image=None):

        # 1. Get the underlying diffusion model (transformer/DiT)
        # Clone the model object to avoid modifying the original in the graph
        model_patcher = base_model.clone()
        model = model_patcher.model # Access the actual nn.Module
        device = model.device
        # Get dtype from model parameters instead of model directly
        try:
            dtype = next(model.parameters()).dtype
        except StopIteration:
            # Handle case where model has no parameters
            print("Warning: Model has no parameters. Falling back to default dtype float32.")
            dtype = torch.float32

        # 2. Get IP-Adapter components (projector and attention weights)
        proj_module = None
        attn_weights = None

        if hasattr(ip_adapter, "image_proj_model") and ip_adapter.image_proj_model is not None:
            proj_module = ip_adapter.image_proj_model
            print("Using pre-built projector from IP Adapter object.")
        else:
            print("Warning: IP Adapter object does not have 'image_proj_model' or it is None.")
            # Potentially try to load from a state_dict if that's an alternative path,
            # but the research suggests the module itself should be present.
            # For now, if no proj_module, image conditioning will be skipped.

        if hasattr(ip_adapter, "processor") and hasattr(ip_adapter.processor, "state_dict"):
            attn_weights = ip_adapter.processor.state_dict()
            print("Extracted attention processor weights from IP Adapter object.")
        else:
            print("Warning: Could not get attention processor weights from IP Adapter object.")
            # If no attn_weights, IP-Adapter effect won't be applied.

        if proj_module is not None:
            proj_module = proj_module.to(device=device, dtype=dtype).eval()
        else:
            print("No projector module available. Image conditioning will be skipped.")


        # 4. Encode subject image if provided
        image_cond = None
        if subject_image is not None and proj_module is not None: # Ensure projector is available
            print("Encoding subject image...")
            if not hasattr(image_encoder_1, "encode_image") or not hasattr(image_encoder_2, "encode_image"):
                print("Warning: Provided image encoders lack 'encode_image' method. Cannot encode subject image.")
            else:
                # Ensure image is 4D tensor [B, C, H, W] on correct device/dtype
                img = subject_image
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                # Move image tensor to the main model's device
                img = img.to(device=device, dtype=torch.float32) # Encoders might expect float32

                # Encode using both encoders
                # Ensure encoders are on the main model's device
                if hasattr(image_encoder_1, "model") and isinstance(image_encoder_1.model, torch.nn.Module):
                    image_encoder_1.model.to(device)
                elif isinstance(image_encoder_1, torch.nn.Module): # If the object itself is the module
                    image_encoder_1.to(device)
                else:
                    print("Warning: Could not move image_encoder_1 to device, .model attribute not found or not a module.")

                if hasattr(image_encoder_2, "model") and isinstance(image_encoder_2.model, torch.nn.Module):
                    image_encoder_2.model.to(device)
                elif isinstance(image_encoder_2, torch.nn.Module): # If the object itself is the module
                    image_encoder_2.to(device)
                else:
                    print("Warning: Could not move image_encoder_2 to device, .model attribute not found or not a module.")

                with torch.no_grad():
                    emb1 = image_encoder_1.encode_image(img)
                    emb2 = image_encoder_2.encode_image(img)

                if emb1 is None or emb2 is None:
                     print("Warning: Image encoding failed for one or both encoders.")
                else:
                    emb1 = emb1.to(device=device, dtype=dtype) # Move embeddings to model device/dtype
                    emb2 = emb2.to(device=device, dtype=dtype)

                    # Basic CLS token removal heuristic (needs verification for specific encoders)
                    if emb1.shape[1] > 1 and emb1.shape[1] == emb2.shape[1] + 1:
                        emb1 = emb1[:, 1:]
                    elif emb2.shape[1] > 1 and emb2.shape[1] == emb1.shape[1] + 1:
                        emb2 = emb2[:, 1:]
                    elif emb1.shape[1] != emb2.shape[1]:
                         print(f"Warning: Encoder output token lengths differ significantly ({emb1.shape[1]} vs {emb2.shape[1]}). Check encoder compatibility.")
                         # Attempt to proceed if shapes are plausible, otherwise skip concat

                    if emb1.shape[1] == emb2.shape[1]:
                        print(f"Concatenating encoder features: {emb1.shape} + {emb2.shape}")
                        image_tokens = torch.cat([emb1, emb2], dim=-1) # Concat feature dim

                        # Project concatenated tokens - requires timestep, use dummy 0 for patching setup
                        # The actual timestep will be used during sampling by the patched attn processor
                        dummy_timestep = torch.zeros(1, device=device, dtype=dtype)
                        with torch.no_grad():
                             # Projector might return tuple (embeds, time_embeds), take first
                             proj_output = proj_module(image_tokens, dummy_timestep, need_temb=False)
                             if isinstance(proj_output, tuple):
                                 image_cond = proj_output[0]
                             else:
                                 image_cond = proj_output
                        print(f"Projected image condition shape: {image_cond.shape}")
                    else:
                         print("Skipping feature concatenation due to mismatched token lengths.")

        # 5. Set up custom attention processors
        if attn_weights is not None and hasattr(model, "set_attn_processor"):
            print("Setting custom attention processors...")
            attn_procs = {}
            ip_hidden_states_dim = proj_module.output_dim if proj_module else 4096 # Fallback dim

            # Iterate through existing processors to find target modules
            for name in model.attn_processors.keys():
                # Dynamically find the attention module corresponding to the processor name
                module_path = name.split('.')
                current_module = model
                try:
                    for part in module_path:
                        if part.isdigit():
                            current_module = current_module[int(part)]
                        else:
                            current_module = getattr(current_module, part)
                    # Infer hidden_size from the attention module's query projection
                    if hasattr(current_module, 'to_q'):
                         hidden_size = current_module.to_q.in_features
                    elif hasattr(model.config, 'hidden_size'): # Fallback to model config
                         hidden_size = model.config.hidden_size
                    else: # Further fallback
                         hidden_size = current_module.processor.to_q.in_features # Access original processor
                    print(f"  - Processor '{name}': hidden_size={hidden_size}, ip_dim={ip_hidden_states_dim}")
                    attn_procs[name] = FluxIPAttnProcessor(
                        hidden_size=hidden_size,
                        ip_hidden_states_dim=ip_hidden_states_dim
                    ).to(device, dtype=dtype)
                except Exception as e:
                    print(f"Warning: Could not get module or hidden_size for processor '{name}': {e}")


            # Load weights into the new processors
            try:
                # Convert processor dict keys if necessary (sometimes saved with 'processor.' prefix)
                converted_attn_weights = {}
                for k, v in attn_weights.items():
                    # Example conversion: remove potential prefix if loader added one
                    new_key = k.replace("processor.", "")
                    converted_attn_weights[new_key] = v

                # Create a temporary ModuleList to load state dict into processors
                temp_module_list = torch.nn.ModuleList(attn_procs.values())
                missing_keys, unexpected_keys = temp_module_list.load_state_dict(converted_attn_weights, strict=False)
                if missing_keys: print(f"Warning: Missing IP-Adapter attn weights: {missing_keys}")
                if unexpected_keys: print(f"Warning: Unexpected IP-Adapter attn weights: {unexpected_keys}")

                # Set the processors on the model
                model.set_attn_processor(attn_procs)
                print("Custom attention processors set and weights loaded.")
            except Exception as e:
                print(f"Error loading attention processor weights: {e}")
                # Optionally revert to original processors or return unmodified model
                # return (base_model,)
        elif attn_weights is None:
             print("No attention weights found in IP Adapter object. Skipping attention processor patching.")
        else:
             print("Warning: Model does not support 'set_attn_processor'. Cannot apply IP-Adapter via processors.")


        # 6. Store conditioning info on the model patcher for the sampling process to use
        # The custom FluxIPAttnProcessor needs access to this during its forward pass.
        # KSampler or a custom sampler needs to be aware of this structure.
        if image_cond is not None:
            model_patcher.instant_character_cond = {
                'image_embeds': image_cond,
                'scale': subject_scale
            }
            print(f"Stored InstantCharacter conditioning on model patcher (scale: {subject_scale}).")
        else:
            # Ensure attribute exists even if no image provided, maybe with scale 0?
            model_patcher.instant_character_cond = {
                 'image_embeds': None,
                 'scale': 0.0 # Apply no effect if no image
            }
            print("No subject image provided or encoding failed; setting scale to 0.")


        # 7. Return the modified model_patcher (which is a ComfyUI MODEL object)
        print("InstantCharacterApply finished.")
        return (model_patcher,)


# --- Node Mappings ---

NODE_CLASS_MAPPINGS = {
    "InstantCharacterApply": InstantCharacterApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantCharacterApply": "Apply InstantCharacter",
}
