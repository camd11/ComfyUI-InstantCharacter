# 3. Pseudocode: Pipeline Modifications for Local Model Loading

## 3.1. Introduction
This document provides pseudocode for potential modifications to the `InstantCharacterFluxPipeline` class within [`InstantCharacter/pipeline.py`](InstantCharacter/pipeline.py), specifically focusing on the `init_adapter` method. These changes ensure that when local paths are provided for image encoders and the IP-adapter, no network downloads are attempted for these components.

## 3.2. Target File
[`InstantCharacter/pipeline.py`](InstantCharacter/pipeline.py)

## 3.3. `InstantCharacterFluxPipeline` Class Overview

The `InstantCharacterFluxPipeline` class contains the `init_adapter` method, which is responsible for setting up image encoders and the IP-adapter.

```python
# Conceptual structure in InstantCharacter/pipeline.py
# from diffusers import ...
# from .models.ip_adapter import IPAdapterPlusXLInstantIDImageProjModel # Example import

class InstantCharacterFluxPipeline(...): # Existing base classes
    # ... existing methods ...

    def init_adapter(
        self,
        image_encoder_path: str,
        image_encoder_2_path: str,
        subject_ip_adapter_path: str,
        # ... other existing parameters ...
        # ADD local_files_only parameter
    ):
        # MODIFIED logic to handle local paths and prevent downloads
        pass

    # ... other existing methods ...
```

## 3.4. Modification of `init_adapter` Method

**Location**: Inside the `InstantCharacterFluxPipeline` class in [`InstantCharacter/pipeline.py`](InstantCharacter/pipeline.py).

**Objective**: Ensure that if `init_adapter` loads components (like image encoders or the IP-adapter model itself) using HuggingFace utilities (e.g., `CLIPVisionModelWithProjection.from_pretrained`, `AutoImageProcessor.from_pretrained`, `IPAdapterPlusXLInstantIDImageProjModel.from_pretrained`), these calls respect the `local_files_only` principle.

**Pseudocode**:

```python
# In class InstantCharacterFluxPipeline:

def init_adapter(
    self,
    image_encoder_path: str,         # Expected to be a local path from the node
    image_encoder_2_path: str,       # Expected to be a local path from the node
    subject_ip_adapter_path: str,  # Expected to be a local path from the node
    # --- Add a new parameter to control download behavior ---
    # This allows the calling node to enforce local-only loading.
    # Default to True if this method is primarily called by the modified node.
    # Or, default to False to maintain original behavior if called from elsewhere,
    # but the node should explicitly pass True. For this spec, assume node passes True.
    local_files_only_for_components: bool = True, 
    # ... other existing parameters like device, dtype, num_tokens, etc.
):
    # // TEST: FR4.2 HuggingFace calls use local_files_only=True or equivalent (within pipeline)

    # --- 1. Load Image Encoder 1 (e.g., SigLIP) ---
    try:
        # // TEST: FR2.1 SigLIP encoder loads successfully from valid local path (pipeline part)
        # // TEST: FR2.5 No download attempt occurs for encoders when local paths are specified (pipeline part)
        self.image_processor = AutoImageProcessor.from_pretrained(
            image_encoder_path,
            local_files_only=local_files_only_for_components # Pass the flag
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path,
            local_files_only=local_files_only_for_components # Pass the flag
            # torch_dtype=self.torch_dtype # Or other relevant params
        ).to(self.device) # Assuming self.device and self.torch_dtype are set
    except Exception as e:
        raise RuntimeError(f"Pipeline: Failed to load image_encoder (SigLIP) from {image_encoder_path}. Error: {e}")

    # --- 2. Load Image Encoder 2 (e.g., DINOv2) ---
    try:
        # // TEST: FR2.2 DINOv2 encoder loads successfully from valid local path (pipeline part)
        self.image_processor_2 = AutoImageProcessor.from_pretrained(
            image_encoder_2_path,
            local_files_only=local_files_only_for_components # Pass the flag
        )
        self.image_encoder_2 = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_2_path,
            local_files_only=local_files_only_for_components # Pass the flag
            # torch_dtype=self.torch_dtype
        ).to(self.device)
    except Exception as e:
        raise RuntimeError(f"Pipeline: Failed to load image_encoder_2 (DINOv2) from {image_encoder_2_path}. Error: {e}")

    # --- 3. Load IP-Adapter Model ---
    # The IP-Adapter might be loaded differently, e.g., not always via from_pretrained
    # if it's a custom model class. Check how IPAdapterPlusXLInstantIDImageProjModel is loaded.
    # If it's a simple state_dict load, local_files_only is less relevant for that specific step,
    # as the path is already local. But if it internally uses from_pretrained for sub-parts,
    # the flag would be needed there.

    # Assuming IPAdapterPlusXLInstantIDImageProjModel.from_pretrained exists and is used:
    try:
        # // TEST: FR3.1 IP-adapter loads successfully from valid local path (pipeline part)
        # // TEST: FR3.3 No download attempt occurs for IP-adapter when local path is specified (pipeline part)
        self.image_proj_model = IPAdapterPlusXLInstantIDImageProjModel.from_pretrained(
            subject_ip_adapter_path, # This might be a directory containing the model or a direct .bin file
            # If it's a directory, from_pretrained handles it. If it's a file,
            # it might need a different loading mechanism or specific config.
            # For now, assume from_pretrained can handle the provided path.
            local_files_only=local_files_only_for_components, # Pass the flag
            # cross_attention_dim=self.transformer.config.cross_attention_dim, # Example param
            # id_embeddings_dim=512, # Example param
            # num_tokens=num_tokens # Example param
        ).to(self.device) #, dtype=self.torch_dtype)
    except Exception as e:
        # If from_pretrained is not the method, adjust accordingly.
        # Example: if it's loading a state_dict from a .bin file directly:
        # if subject_ip_adapter_path.endswith(".bin"):
        #     state_dict = torch.load(subject_ip_adapter_path, map_location="cpu")
        #     # Initialize model and load state_dict
        #     self.image_proj_model = IPAdapterPlusXLInstantIDImageProjModel(...)
        #     self.image_proj_model.load_state_dict(state_dict)
        #     self.image_proj_model.to(self.device)
        # else: # Assume it's a pretrained directory
        #     self.image_proj_model = IPAdapterPlusXLInstantIDImageProjModel.from_pretrained(...)
        raise RuntimeError(f"Pipeline: Failed to load IP-Adapter from {subject_ip_adapter_path}. Error: {e}")

    # --- 4. Set other attributes ---
    # self.adapter_modules = ... (existing logic)
    # self.set_adapter_layers() (existing logic)
    # ... (any other setup logic within init_adapter) ...

    # No explicit return value, modifies 'self'.
```

## 3.5. Calling `init_adapter` from the Node

In [`nodes/comfy_nodes.py`](nodes/comfy_nodes.py), the call to `init_adapter` should now include the `local_files_only_for_components` parameter:

```python
# In nodes/comfy_nodes.py, inside the node's execution method:

# ... (pipe = InstantCharacterFluxPipeline.from_pretrained(...) call) ...

pipe.init_adapter(
    image_encoder_path=siglip_encoder_path,
    image_encoder_2_path=dinov2_encoder_path,
    subject_ip_adapter_path=ip_adapter_path,
    local_files_only_for_components=True, # Explicitly pass True
    # ... other necessary parameters ...
)
```

## 3.6. Important Considerations:
- **`from_pretrained` Behavior**: The `local_files_only=True` flag is standard in HuggingFace `diffusers` and `transformers` libraries. If any custom loading functions are used that wrap or bypass these, they must also be modified to prevent network access when local paths are intended.
- **Path Types**: `from_pretrained` can typically handle paths to directories containing model files and configuration, or sometimes direct paths to specific checkpoint files (though directory paths are more common for full models). The exact nature of `image_encoder_path`, `image_encoder_2_path`, and `subject_ip_adapter_path` (file vs. directory) needs to be consistent with how these components are saved and loaded. The pseudocode assumes `from_pretrained` handles them appropriately.
- **Error Handling**: Robust error handling is crucial to inform the user if a local file is missing, corrupted, or of the wrong type, especially since downloads are disabled.
- **Alternative for IP-Adapter**: If `IPAdapterPlusXLInstantIDImageProjModel` is typically loaded via `torch.load` for a `.bin` file and then `load_state_dict`, the `local_files_only` flag is less relevant for that specific step, as `torch.load` inherently works with local files. However, if `IPAdapterPlusXLInstantIDImageProjModel` itself has sub-components loaded via `from_pretrained`, those would need the flag. The pseudocode above includes a comment for this alternative. The primary goal is that no part of `init_adapter` attempts a download.

This pseudocode ensures that the `local_files_only` policy is enforced not just for the main pipeline but also for components loaded within `init_adapter`.