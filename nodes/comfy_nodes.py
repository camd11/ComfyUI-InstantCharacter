import os
import sys
import torch
import folder_paths
from PIL import Image
import numpy as np
import copy # For deep copying the model
from einops import rearrange
from transformers import SiglipImageProcessor, AutoImageProcessor # Needed for image processing

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


        # 4. Encode subject image using multi-resolution approach if provided
        image_embeds_dict = None # Initialize
        if subject_image is not None and proj_module is not None:
            print("Encoding subject image using multi-resolution approach...")
            # --- Helper functions to replicate original pipeline's encoding ---
            @torch.inference_mode()
            def _encode_siglip_helper(encoder_model, image_tensor, device, dtype):
                encoder_model.to(device, dtype=dtype)
                image_tensor = image_tensor.to(device, dtype=dtype)
                try:
                    res = encoder_model(image_tensor, output_hidden_states=True)
                    last_hidden = res.last_hidden_state
                    shallow_indices = [7, 13, 26] # 0-based indices from original pipeline
                    if res.hidden_states is None or len(res.hidden_states) <= max(shallow_indices):
                         print("Warning: SigLIP hidden_states not available or too short for shallow features.")
                         return last_hidden, None # Return only deep if shallow fails
                    shallow_hidden = torch.cat([res.hidden_states[i] for i in shallow_indices], dim=1)
                    return last_hidden, shallow_hidden
                except Exception as e:
                    print(f"Error during SigLIP encoding: {e}")
                    return None, None

            @torch.inference_mode()
            def _encode_dinov2_helper(encoder_model, image_tensor, device, dtype):
                encoder_model.to(device, dtype=dtype)
                image_tensor = image_tensor.to(device, dtype=dtype)
                try:
                    res = encoder_model(image_tensor, output_hidden_states=True)
                    last_hidden = res.last_hidden_state[:, 1:] # Remove CLS token
                    shallow_indices = [9, 19, 29] # 0-based indices from original pipeline
                    if res.hidden_states is None or len(res.hidden_states) <= max(shallow_indices):
                         print("Warning: DINOv2 hidden_states not available or too short for shallow features.")
                         return last_hidden, None # Return only deep if shallow fails
                    shallow_hidden = torch.cat([res.hidden_states[i][:, 1:] for i in shallow_indices], dim=1) # Remove CLS
                    return last_hidden, shallow_hidden
                except Exception as e:
                    print(f"Error during DINOv2 encoding: {e}")
                    return None, None
            # --- End Helper Functions ---

            # Get underlying HF models (assuming .model attribute exists)
            siglip_hf_model = getattr(image_encoder_1, "model", image_encoder_1)
            dinov2_hf_model = getattr(image_encoder_2, "model", image_encoder_2)

            # Instantiate image processors (temporary solution)
            try:
                # TODO: Ideally get processors associated with the loaded models instead of hardcoding names
                siglip_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
                dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
                dino_processor.crop_size = dict(height=384, width=384)
                dino_processor.size = dict(shortest_edge=384)
            except Exception as e:
                print(f"Warning: Could not instantiate default image processors: {e}. Cannot encode image.")
                siglip_processor = None
                dino_processor = None

            if siglip_processor and dino_processor:
                # Convert input tensor to PIL Image
                if subject_image.dim() == 4: img_tensor = subject_image[0]
                else: img_tensor = subject_image
                # Ensure tensor is on CPU for numpy conversion
                img_pil = Image.fromarray((img_tensor.cpu().numpy() * 255).astype(np.uint8))
                # Resize to square aspect ratio before further processing
                img_pil = img_pil.resize((max(img_pil.size), max(img_pil.size)))

                # Prepare low-res and high-res versions
                img_pil_low_res = [img_pil.resize((384, 384))]
                img_pil_high_res_base = img_pil.resize((768, 768))
                img_pil_high_res_crops = [
                    img_pil_high_res_base.crop((0, 0, 384, 384)),
                    img_pil_high_res_base.crop((384, 0, 768, 384)),
                    img_pil_high_res_base.crop((0, 384, 384, 768)),
                    img_pil_high_res_base.crop((384, 384, 768, 768)),
                ]
                nb_split_image = len(img_pil_high_res_crops)

                # Process images to tensors
                siglip_pixels_low = siglip_processor(images=img_pil_low_res, return_tensors="pt").pixel_values
                dino_pixels_low = dino_processor(images=img_pil_low_res, return_tensors="pt").pixel_values
                siglip_pixels_high = siglip_processor(images=img_pil_high_res_crops, return_tensors="pt").pixel_values
                dino_pixels_high = dino_processor(images=img_pil_high_res_crops, return_tensors="pt").pixel_values

                # Encode low-res
                siglip_deep_low, siglip_shallow_low = _encode_siglip_helper(siglip_hf_model, siglip_pixels_low, device, dtype)
                dino_deep_low, dino_shallow_low = _encode_dinov2_helper(dinov2_hf_model, dino_pixels_low, device, dtype)

                # Encode high-res crops
                siglip_pixels_high = rearrange(siglip_pixels_high, 'n c h w -> n c h w') # Ensure 4D for batch processing
                dino_pixels_high = rearrange(dino_pixels_high, 'n c h w -> n c h w')   # Ensure 4D for batch processing
                siglip_deep_high_crops, _ = _encode_siglip_helper(siglip_hf_model, siglip_pixels_high, device, dtype)
                dino_deep_high_crops, _ = _encode_dinov2_helper(dinov2_hf_model, dino_pixels_high, device, dtype)

                # Combine results if all encodings succeeded
                if all(e is not None for e in [siglip_deep_low, siglip_shallow_low, dino_deep_low, dino_shallow_low, siglip_deep_high_crops, dino_deep_high_crops]):
                    # Ensure results are on the correct device/dtype before concatenation
                    siglip_deep_low = siglip_deep_low.to(device=device, dtype=dtype)
                    siglip_shallow_low = siglip_shallow_low.to(device=device, dtype=dtype)
                    dino_deep_low = dino_deep_low.to(device=device, dtype=dtype)
                    dino_shallow_low = dino_shallow_low.to(device=device, dtype=dtype)
                    siglip_deep_high_crops = siglip_deep_high_crops.to(device=device, dtype=dtype)
                    dino_deep_high_crops = dino_deep_high_crops.to(device=device, dtype=dtype)

                    # Combine deep low-res
                    image_embeds_low_res_deep = torch.cat([siglip_deep_low, dino_deep_low], dim=2)
                    # Combine shallow low-res
                    image_embeds_low_res_shallow = torch.cat([siglip_shallow_low, dino_shallow_low], dim=2)

                    # Rearrange and combine deep high-res
                    # Need batch dimension for rearrange: add it if missing (e.g., if batch size was 1)
                    if siglip_deep_high_crops.dim() == 3: siglip_deep_high_crops = siglip_deep_high_crops.unsqueeze(0)
                    if dino_deep_high_crops.dim() == 3: dino_deep_high_crops = dino_deep_high_crops.unsqueeze(0)
                    # The rearrange logic assumes the batch dim was added *before* encoding if nb_split_image > 1
                    # Let's adjust based on the output shape directly
                    # Expected shape after encoding crops: (nb_split_image, seq_len, channels)
                    # We need to reshape to (1, nb_split_image * seq_len, channels)
                    siglip_deep_high = siglip_deep_high_crops.view(1, -1, siglip_deep_high_crops.shape[-1])
                    dino_deep_high = dino_deep_high_crops.view(1, -1, dino_deep_high_crops.shape[-1])
                    image_embeds_high_res_deep = torch.cat([siglip_deep_high, dino_deep_high], dim=2)

                    image_embeds_dict = dict(
                        image_embeds_low_res_shallow=image_embeds_low_res_shallow,
                        image_embeds_low_res_deep=image_embeds_low_res_deep,
                        image_embeds_high_res_deep=image_embeds_high_res_deep,
                    )
                    print("Successfully generated multi-resolution image embeddings dict.")
                else:
                    print("Warning: Failed to generate all required embeddings.")
                    image_embeds_dict = None
            else:
                print("Warning: Cannot proceed with image encoding without processors.")
                image_embeds_dict = None

        # 5. Set up custom attention processors (FluxIPAttnProcessor) and load weights
        if attn_weights is not None and hasattr(model, "set_attn_processor"):
            print("Setting custom FluxIPAttnProcessors and loading weights...")
            attn_procs = {}
            # ip_hidden_states_dim should match the *output* of the projector
            ip_hidden_states_dim = proj_module.output_dim if proj_module and hasattr(proj_module, 'output_dim') else 4096

            # Ensure attn_weights keys don't have unexpected prefixes
            converted_attn_weights = {}
            for k, v in attn_weights.items():
                new_key = k.replace("processor.", "") # Remove potential prefix
                converted_attn_weights[new_key] = v

            for name in model.attn_processors.keys():
                current_attention_module = model
                try:
                    # Navigate to the actual attention module
                    for part in name.split('.'):
                        current_attention_module = getattr(current_attention_module, part) if not part.isdigit() else current_attention_module[int(part)]

                    # Infer hidden_size
                    hidden_size = getattr(current_attention_module, 'to_q', {}).in_features if hasattr(current_attention_module, 'to_q') else model.config.hidden_size

                    # Instantiate OUR custom processor
                    current_processor = FluxIPAttnProcessor(
                        hidden_size=hidden_size,
                        ip_hidden_states_dim=ip_hidden_states_dim
                        # We will modify FluxIPAttnProcessor later to accept model/projector refs if needed
                    ).to(device, dtype=dtype)

                    # Load weights from the converted state dict into this specific processor instance
                    # We assume the state dict from ip_adapter.processor contains weights compatible
                    # with FluxIPAttnProcessor's layers (e.g., to_k_ip, to_v_ip)
                    missing, unexpected = current_processor.load_state_dict(converted_attn_weights, strict=False)
                    if missing: print(f"Warning: Missing IP-Adapter attn weights for '{name}': {missing}")
                    if unexpected: print(f"Warning: Unexpected IP-Adapter attn weights for '{name}': {unexpected}")

                    attn_procs[name] = current_processor
                except Exception as e:
                    print(f"Error setting up processor for '{name}': {e}")

            if attn_procs:
                model.set_attn_processor(attn_procs)
                print("Custom FluxIPAttnProcessors set and weights loaded.")
        elif attn_weights is None:
             print("No attention weights found in IP Adapter object. Skipping attention processor patching.")
        else:
             print("Warning: Model does not support 'set_attn_processor'. Cannot apply IP-Adapter via processors.")


        # 6. Store conditioning info for the KSampler (IPAdapter+) or patched forward
        # Store unprojected tokens and the projector module itself.
        # Store the generated dict (or None) and other info for the attn processors
        if subject_image is not None and image_embeds_dict is not None and proj_module is not None:
            model_patcher._ip_kwargs = {
                "image_embeds": image_embeds_dict, # Store the dict with multi-res embeddings
                "projector": proj_module,
                "scale": subject_scale,
            }
            print(f"Stored multi-res embeddings and projector in _ip_kwargs (scale: {subject_scale}).")
        else:
            # Ensure attribute exists but indicates no effect if encoding failed or no image
            model_patcher._ip_kwargs = {
                 "image_embeds": None,
                 "projector": None,
                 "scale": 0.0
            }
            print("No subject image, tokens, or projector; IP-Adapter effect will be nullified (scale 0).")


        # 7. Return the modified model_patcher (which is a ComfyUI MODEL object containing the patched model)
        print("InstantCharacterApply finished.")
        return (model_patcher,)


# --- Node Mappings ---

NODE_CLASS_MAPPINGS = {
    "InstantCharacterApply": InstantCharacterApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantCharacterApply": "Apply InstantCharacter",
}
