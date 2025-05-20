# 2. Pseudocode: Node Modifications for Local Model Loading

## 2.1. Introduction
This document provides pseudocode for modifying the `InstantCharacterNode` (assumed class name, actual name might vary) within [`nodes/comfy_nodes.py`](nodes/comfy_nodes.py) to support loading models from local paths.

## 2.2. Target File
[`nodes/comfy_nodes.py`](nodes/comfy_nodes.py)

## 2.3. Node Class Definition (Conceptual)

```python
# Conceptual structure of the node class in comfy_nodes.py
# Actual class name and methods might differ.

class InstantCharacterNode: # Or existing name like InstantCharacterLoaderNode
    # ... (existing class attributes and methods) ...

    @classmethod
    def INPUT_TYPES(s):
        # ... (existing input types) ...
        # ADD new input types for local paths
        # REVISE existing model selection if it's mutually exclusive with local paths

    # ... (other methods like IS_CHANGED if applicable) ...

    def load_instant_character(self, base_model_path, siglip_encoder_path, dinov2_encoder_path, ip_adapter_path, # ... other existing parameters ...
                              ):
        # MODIFIED logic to handle local paths
        # ...
        pass
```

## 2.4. New Input Type Definitions

**Location**: Inside the `INPUT_TYPES` class method of the InstantCharacter node.

**Pseudocode**:
```python
@classmethod
def INPUT_TYPES(s):
    return {
        "required": {
            # ... (keep existing required inputs like seed, steps, cfg, etc.)
            # ... (original model selection, e.g., dropdown for HuggingFace ID, might need to be made optional or removed if local paths are always prioritized)

            "base_model_path": ("STRING", {
                "multiline": False,
                "default": "ComfyUI/models/flux/", # Or an empty string, or a more specific placeholder like "/path/to/your/flux/model_directory_or_file"
                # Consider adding "forceInput": True if this should always be a direct input rather than a widget
            }),
            "siglip_encoder_path": ("STRING", {
                "multiline": False,
                "default": "ComfyUI/models/clip_vision/siglip_encoder/", # Or "/path/to/your/siglip_encoder_directory_or_file"
            }),
            "dinov2_encoder_path": ("STRING", {
                "multiline": False,
                "default": "ComfyUI/models/clip_vision/dinov2_encoder/", # Or "/path/to/your/dinov2_encoder_directory_or_file"
            }),
            "ip_adapter_path": ("STRING", {
                "multiline": False,
                "default": "ComfyUI/models/ipadapter/ip_adapter.bin", # Or "/path/to/your/ip_adapter.bin"
            }),
            # Add other existing inputs like image, prompt, etc.
        },
        "optional": {
            # ... (existing optional inputs)
            # If the original HuggingFace ID selection is kept, it might move here
            # or be handled with conditional logic. For this spec, we assume new paths take precedence.
            "original_model_id_selector": (["list_of_hf_ids"], ), # Example if keeping old selector
        }
    }
# TEST: FR5.1 New input fields are present in the node UI
# TEST: FR5.2 Node correctly interprets provided paths (tested via successful loading)
```

**Notes**:
- The `default` values are illustrative. They could be empty strings or common paths.
- Consideration: If an original HuggingFace ID selector exists (e.g., `model_name`), decide its interaction with these new path inputs.
    - Option A: Prioritize local paths. If `base_model_path` is non-empty, use it; otherwise, fall back to `model_name` (this spec assumes local paths are primary if provided).
    - Option B: Make them mutually exclusive (e.g., a boolean switch "load_local").
    - Option C: Remove `model_name` if local paths become the sole method.
    - For this pseudocode, we assume if `base_model_path` (and others) are provided, they are used.

## 2.5. Modification of Model Loading Logic

**Location**: Inside the main execution method of the node (e.g., `load_instant_character`, `execute`, or similar).

**Pseudocode**:
```python
# Assuming a method like 'load_instant_character' or 'execute'
# Parameters will include all INPUT_TYPES
def load_instant_character(self, base_model_path, siglip_encoder_path, dinov2_encoder_path, ip_adapter_path, 
                           # other_parameters like prompt, image, seed, etc.
                           original_model_id_selector=None # if kept
                           ):
    
    # --- 1. Input Validation and Path Handling ---
    # TEST: EC1 User provides an empty string for a model path
    if not base_model_path:
        raise ValueError("Base model path cannot be empty when loading locally.")
    if not siglip_encoder_path:
        raise ValueError("SigLIP encoder path cannot be empty when loading locally.")
    if not dinov2_encoder_path:
        raise ValueError("DINOv2 encoder path cannot be empty when loading locally.")
    if not ip_adapter_path:
        raise ValueError("IP-adapter path cannot be empty when loading locally.")

    # Basic path validation (existence checks are often handled by the from_pretrained/load methods themselves,
    # but an early check can be useful)
    # For example:
    # if not os.path.exists(base_model_path):
    #     raise FileNotFoundError(f"Base model path not found: {base_model_path}")
    # Similar checks for other paths, or rely on HuggingFace's local_files_only to error out.
    # TEST: FR1.2 System errors if base model path is invalid or file not found
    # TEST: FR2.3 System errors if SigLIP encoder path is invalid or file not found
    # TEST: FR2.4 System errors if DINOv2 encoder path is invalid or file not found
    # TEST: FR3.2 System errors if IP-adapter path is invalid or file not found

    # --- 2. Load Base Model ---
    try:
        # // TEST: FR1.1 Base model loads successfully from valid local path
        # // TEST: FR1.3 No download attempt occurs when local base model path is specified
        # // TEST: FR4.1 Node execution fails if a required local model is missing, without attempting download
        # // TEST: FR4.2 HuggingFace calls use local_files_only=True or equivalent
        
        # Ensure local_files_only is True to prevent downloads.
        # The exact parameter name might vary based on the HuggingFace Diffusers version.
        # Common parameters: local_files_only=True, resume_download=False, offline=True
        # For from_pretrained, if the path is local, it often doesn't download,
        # but local_files_only makes this explicit.
        
        # If InstantCharacterFluxPipeline.from_pretrained directly supports local_files_only:
        pipe = InstantCharacterFluxPipeline.from_pretrained(
            base_model_path,  # This is now the local path
            # torch_dtype=torch.bfloat16, # Or other relevant dtypes
            local_files_only=True # CRITICAL: Prevents downloads
        )
        # If from_pretrained doesn't directly take local_files_only, but the underlying
        # snapshot_download or hf_hub_download does, the pipeline might need to be
        # instantiated differently or the download function called manually with this flag
        # before from_pretrained. However, most modern diffusers pipelines respect this.

    except Exception as e:
        # Log error, provide informative message
        raise RuntimeError(f"Failed to load base model from {base_model_path}. Error: {e}")

    # --- 3. Initialize Adapter with Local Encoders and IP-Adapter ---
    try:
        # // TEST: FR2.1 SigLIP encoder loads successfully from valid local path
        # // TEST: FR2.2 DINOv2 encoder loads successfully from valid local path
        # // TEST: FR3.1 IP-adapter loads successfully from valid local path
        # // TEST: FR2.5 No download attempt occurs for encoders when local paths are specified
        # // TEST: FR3.3 No download attempt occurs for IP-adapter when local path is specified
        
        # The init_adapter method in InstantCharacter/pipeline.py might need to be
        # checked or modified if it doesn't respect local_files_only for its sub-component loading.
        # Assuming init_adapter itself will use these paths directly or pass them to
        # sub-loaders that can be configured for local_files_only.

        pipe.init_adapter(
            image_encoder_path=siglip_encoder_path,       # Local path for SigLIP
            image_encoder_2_path=dinov2_encoder_path,     # Local path for DINOv2
            subject_ip_adapter_path=ip_adapter_path,  # Local path for IP-Adapter
            # any other necessary parameters for init_adapter
            # CRITICAL: Ensure that init_adapter or its underlying calls also use local_files_only=True
            # This might require inspecting/modifying InstantCharacter/pipeline.py if init_adapter
            # itself tries to download. For now, assume it passes paths correctly.
            # If init_adapter calls from_pretrained for encoders, those calls need local_files_only=True.
        )
        # If init_adapter doesn't have a local_files_only flag, but loads components using
        # from_pretrained, the modification might be needed inside init_adapter in pipeline.py
        # (See docs/03_pseudocode_pipeline_mods.md if created)

    except Exception as e:
        # Log error, provide informative message
        raise RuntimeError(f"Failed to initialize adapter with local paths. Error: {e}." +
                           f" Paths used: SigLIP='{siglip_encoder_path}', DINOv2='{dinov2_encoder_path}', IP-Adapter='{ip_adapter_path}'")

    # --- 4. Move to device (example) ---
    # pipe.to(device) # device would be determined by ComfyUI context

    # --- 5. Return the loaded pipe and other necessary outputs ---
    # The return tuple structure must match what ComfyUI expects from this node.
    # Example: return (pipe, other_outputs_if_any)
    # Typically, for a loader node, it might return a MODEL, CLIP, VAE, etc.
    # For this pipeline, it likely returns a "MODEL" object that is the pipe itself,
    # and potentially "CLIP_VISION" if encoders are returned separately.
    # Adjust based on how InstantCharacter integrates.
    
    # Assuming the pipe object is what's needed downstream:
    return (pipe, ) # Or (pipe, pipe.image_encoder, pipe.image_encoder_2) if needed separately

```

## 2.6. Considerations for `InstantCharacter/pipeline.py`

While the primary changes are in [`nodes/comfy_nodes.py`](nodes/comfy_nodes.py), the `pipe.init_adapter(...)` method within [`InstantCharacter/pipeline.py`](InstantCharacter/pipeline.py) is a key area.
- If `init_adapter` itself calls `from_pretrained` (or similar HuggingFace utilities) for the image encoders or IP-adapter without a `local_files_only=True` (or equivalent) mechanism, or if it doesn't correctly handle direct file paths for these components, then `pipeline.py` might also need modifications.
- **Assumption for this document**: `init_adapter` can correctly use the provided local paths, or if it calls `from_pretrained` internally, those calls are made in a way that respects local paths without downloading (e.g., by implicitly handling local paths or if `local_files_only` is passed down).
- If `pipeline.py` needs changes, a separate pseudocode section for it would be required (e.g., `docs/03_pseudocode_pipeline_mods.md`). The current focus is on the node.

**Key check for `init_adapter` in `pipeline.py`**:
- Does it load `image_encoder_path`, `image_encoder_2_path`, `subject_ip_adapter_path` using `CLIPVisionModelWithProjection.from_pretrained`, `AutoImageProcessor.from_pretrained`, `IPAdapterPlusXLInstantIDImageProjModel.from_pretrained`, etc.?
- If so, ensure these calls within `init_adapter` also use `local_files_only=True` when local paths are provided. This might involve passing a `local_files_only` flag into `init_adapter`.

Example (Conceptual modification if `init_adapter` needs it):
```python
# In InstantCharacter/pipeline.py
# def init_adapter(self, ..., image_encoder_path, ..., local_files_only_override=True): # New param
#    self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path, local_files_only=local_files_only_override)
#    ...
```
And then in `nodes/comfy_nodes.py`:
```python
# pipe.init_adapter(..., local_files_only_override=True)
```

This ensures the "no download" policy propagates.

## 2.7. Error Handling and Validation
- Paths should be validated (e.g., not empty).
- Errors during model loading should be caught and re-raised with informative messages.
- The `local_files_only=True` flag (or equivalent like `offline=True` depending on the specific Hugging Face library version and function) is critical to prevent fallback to downloads.

This pseudocode provides a blueprint for the necessary modifications. The implementer will need to adapt it to the exact structure and naming conventions within [`nodes/comfy_nodes.py`](nodes/comfy_nodes.py) and verify the behavior of `init_adapter` in [`InstantCharacter/pipeline.py`](InstantCharacter/pipeline.py).