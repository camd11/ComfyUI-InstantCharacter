# Phase 4: Define New Input Slots for Refactored InstantCharacter Node

The existing `InstantCharacterLoadModelFromLocal` node in `nodes/comfy_nodes.py` currently accepts string paths for all models. The refactoring aims to replace these path inputs with direct model object inputs from dedicated ComfyUI loader nodes.

## Current `InstantCharacterLoadModelFromLocal` Inputs (for reference):
```python
"required": {
    "base_model_path": ("STRING", {"default": "models/FLUX.1-dev"}),
    "image_encoder_path": ("STRING", {"default": "models/google/siglip-so400m-patch14-384"}),
    "image_encoder_2_path": ("STRING", {"default": "models/facebook/dinov2-giant"}),
    "ip_adapter_path": ("STRING", {"default": "models/InstantCharacter/instantcharacter_ip-adapter.bin"}),
    "cpu_offload": ("BOOLEAN", {"default": False}),
}
```

## Proposed New Inputs for Refactored `InstantCharacterLoaderNode`:

A new node, let's call it `InstantCharacterLoaderNode` (or we can refactor `InstantCharacterLoadModelFromLocal`), will be created/modified. It will accept the following inputs:

```python
"required": {
    "flux_unet_model": ("MODEL", {}),        // Output from GGUF Unet Loader
    "siglip_vision_model": ("CLIP_VISION", {}), // Output from a CLIP Vision Loader (for SigLIP)
    "dinov2_vision_model": ("CLIP_VISION", {}), // Output from a CLIP Vision Loader (for DINOv2)
    "ipadapter_model_data": ("IPADAPTER", {}), // Output from IPAdapter Model Loader
    "cpu_offload": ("BOOLEAN", {"default": False}), // Existing functionality
}
```

## Rationale for New Inputs:

*   **`flux_unet_model` (MODEL):** This will be the UNet/transformer component of the FLUX model, loaded by a `GGUF Unet Loader` (or a standard ComfyUI UNet loader if a non-GGUF FLUX model is used in the future). This replaces the `base_model_path` and the direct `FluxPipeline.from_pretrained()` call for the UNet.
*   **`siglip_vision_model` (CLIP_VISION):** This will be a loaded `ClipVisionModel` object (specifically SigLIP), provided by a standard ComfyUI "Load CLIP Vision" node. This replaces `image_encoder_path`.
*   **`dinov2_vision_model` (CLIP_VISION):** This will be a loaded `ClipVisionModel` object (specifically DINOv2), also provided by a "Load CLIP Vision" node. This replaces `image_encoder_2_path`.
*   **`ipadapter_model_data` (IPADAPTER):** This will be the loaded IPAdapter model data (likely a dictionary of tensors), provided by the `IPAdapterModelLoader` from `ComfyUI_IPAdapter_plus`. This replaces `ip_adapter_path`.
*   **`cpu_offload` (BOOLEAN):** This existing option will be retained to allow users to offload the pipeline to the CPU if VRAM is limited.

## Output of the Refactored Node:

The refactored loader node will continue to output:
*   `RETURN_TYPES = ("INSTANTCHAR_PIPE",)`

This `INSTANTCHAR_PIPE` will be an instance of the `InstantCharacterFluxPipeline` class, but initialized differently to use the provided model objects instead of loading them from paths or Hugging Face identifiers.