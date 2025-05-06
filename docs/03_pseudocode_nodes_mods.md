# Phase 5: Pseudocode for `nodes/comfy_nodes.py` Modifications

This document outlines the pseudocode for refactoring the ComfyUI nodes, primarily by creating a new loader node that accepts pre-loaded model components.

```python
# PSEUDOCODE for nodes/comfy_nodes.py

# --- Imports ---
import torch
import folder_paths # Keep
from PIL import Image # Keep
# ... other necessary comfy imports (e.g., for comfy.utils if needed)

# Ensure InstantCharacter.pipeline can be imported
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Keep if structure demands
from InstantCharacter.pipeline import InstantCharacterFluxPipeline # Keep

# REMOVE: from huggingface_hub import login (if InstantCharacterLoadModel is removed)

# --- Node Definitions ---

# To be REMOVED:
# class InstantCharacterLoadModel:
#     # This class uses hardcoded Hugging Face downloads and will be replaced.
#     pass

# NEW or REFACTORED Loader Node:
class InstantCharacterLoader: # Can rename from InstantCharacterLoadModelFromLocal or be a new class
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_unet_model": ("MODEL", {}),        # From GGUF Unet Loader (or other UNet loader)
                "siglip_vision_model": ("CLIP_VISION", {}), # From CLIP Vision Loader
                "dinov2_vision_model": ("CLIP_VISION", {}), # From CLIP Vision Loader
                "ipadapter_model_data": ("IPADAPTER", {}), # From IPAdapter Model Loader (expects dict of weights)
                "cpu_offload": ("BOOLEAN", {"default": False}),
                # Optional: If VAE and Text Encoders are not part of `flux_unet_model`
                # "flux_vae": ("VAE", {}),
                # "flux_clip_l": ("CLIP", {}), # For text_encoder_one & tokenizer_one
                # "flux_clip_g": ("CLIP", {}), # For text_encoder_two & tokenizer_two
            }
        }

    RETURN_TYPES = ("INSTANTCHAR_PIPE",)
    FUNCTION = "load_pipe_from_models"
    CATEGORY = "InstantCharacter"
    DESCRIPTION = "Loads InstantCharacter pipeline from pre-loaded model components."

    def load_pipe_from_models(self,
                              flux_unet_model,
                              siglip_vision_model,
                              dinov2_vision_model,
                              ipadapter_model_data,
                              cpu_offload,
                              # flux_vae=None, flux_clip_l=None, flux_clip_g=None # If VAE/CLIP are separate
                             ):

        # Device and dtype considerations
        # device = comfy.model_management.get_torch_device() # ComfyUI preferred way
        # dtype = torch.bfloat16 # Or infer from loaded models

        # Instantiate the refactored pipeline
        # This assumes InstantCharacterFluxPipeline.__init__ is updated as per
        # docs/02_pseudocode_pipeline_mods.md.
        pipe = InstantCharacterFluxPipeline(
            flux_unet_model_object=flux_unet_model,
            siglip_vision_model_object=siglip_vision_model,
            dinov2_vision_model_object=dinov2_vision_model,
            ipadapter_model_data_dict=ipadapter_model_data,
            # flux_vae_object=flux_vae, # Pass if separate
            # flux_clip_l_object=flux_clip_l, # Pass if separate
            # flux_clip_g_object=flux_clip_g, # Pass if separate
            # dtype=dtype
        )

        # CPU Offload Handling:
        # ComfyUI's model management might handle offloading if models are registered.
        # If the pipeline itself needs specific offloading logic (like Diffusers' enable_sequential_cpu_offload),
        # it should be called here or handled within the pipeline's __init__.
        if cpu_offload:
            # print("Attempting CPU offload for InstantCharacter pipeline...")
            # pipe.enable_sequential_cpu_offload() # If using Diffusers-like offload
            # OR: Rely on ComfyUI's default behavior for model components on CPU.
            # The pipeline's internal components (like custom attn processors) should be
            # initialized on the correct device or moved.
            pass

        # Note: Individual model components (flux_unet_model, siglip_vision_model, etc.)
        # are already managed by ComfyUI regarding their device placement.
        # The `pipe` object itself doesn't need a top-level `.to(device)` if its internal
        # components are correctly handled during its __init__.

        return (pipe,)

# InstantCharacterGenerate Node:
# This class should remain largely UNCHANGED.
# It consumes the `INSTANTCHAR_PIPE` from the new loader.
class InstantCharacterGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("INSTANTCHAR_PIPE",),
                "prompt": ("STRING", {"multiline": True}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "subject_scale": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "subject_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "InstantCharacter"

    def generate(self, pipe: InstantCharacterFluxPipeline, prompt, height, width, guidance_scale,
                 num_inference_steps, seed, subject_scale, subject_image=None):

        # Image conversion logic (tensor to PIL for subject_image) remains the same.
        subject_image_pil = None
        if subject_image is not None:
            # ... (conversion logic as in original)
            pass

        # Call the pipeline
        output = pipe( # This now calls the refactored pipeline instance
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(seed), # Consider device for generator
            subject_image=subject_image_pil,
            subject_scale=subject_scale,
        )

        # Image conversion logic (PIL to tensor for output) remains the same.
        # final_image_tensor = ...
        # return (final_image_tensor,)
        pass # Placeholder for original conversion logic

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "InstantCharacterLoader": InstantCharacterLoader,
    "InstantCharacterGenerate": InstantCharacterGenerate,
    # Remove "InstantCharacterLoadModel"
    # Decide whether to keep "InstantCharacterLoadModelFromLocal" for backward compatibility
    # or remove it if "InstantCharacterLoader" fully supersedes it.
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantCharacterLoader": "Load InstantCharacter (Modular)",
    "InstantCharacterGenerate": "Generate with InstantCharacter",
}

```

**Key Considerations for `nodes/comfy_nodes.py`:**

1.  **Clarity of `InstantCharacterLoader` Inputs:**
    *   The `flux_unet_model` input of type `MODEL` needs to be well-understood. If the GGUF Unet Loader (or any other UNet loader used for FLUX) only provides the UNet part, then the VAE and the two FLUX Text Encoders (and their tokenizers) must be loaded separately and passed as additional inputs (e.g., `flux_vae` (VAE type), `flux_clip_l` (CLIP type), `flux_clip_g` (CLIP type)). The pseudocode includes commented-out optional inputs for this scenario.
    *   The `ipadapter_model_data` of type `IPADAPTER` is assumed to be the dictionary of weights loaded by `IPAdapterModelLoader`.

2.  **Device Handling in `load_pipe_from_models`:**
    *   ComfyUI generally manages model device placement. The input models (`flux_unet_model`, `siglip_vision_model`, etc.) will already be on a device.
    *   The `InstantCharacterFluxPipeline.__init__` method should ensure any *new* PyTorch modules it creates (like `FluxIPAttnProcessor` or `CrossLayerCrossScaleProjector`) are moved to the correct device, likely inferred from the input `flux_unet_model`.
    *   The `cpu_offload` flag should ideally trigger ComfyUI's standard offloading mechanisms if applicable to custom pipelines, or the pipeline itself should implement `enable_sequential_cpu_offload` if it's structured like a Diffusers pipeline.

3.  **Error Handling:** Robust error handling should be added to `load_pipe_from_models` to check if the provided model objects are of the expected types or have the necessary components.

This pseudocode focuses on the structural changes for modular loading.