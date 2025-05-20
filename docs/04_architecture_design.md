# Architecture Design: Local Model Loading and LoRA Support for InstantCharacter

## 1. Introduction

This document outlines the architectural modifications required for the ComfyUI-InstantCharacter custom node to support:
1.  Loading all necessary models (base model, image encoders, IP-adapter) from local file paths.
2.  Applying a style LoRA during the character generation process.

The design prioritizes minimal changes to the existing codebase, primarily affecting [`nodes/comfy_nodes.py`](nodes/comfy_nodes.py) and requiring minor adjustments to [`InstantCharacter/pipeline.py`](InstantCharacter/pipeline.py).

## 2. Scope of Changes

The architectural changes address two distinct functional enhancements:

*   **Local Model Loading**: Enables users to specify local paths for all dependent models, bypassing automatic downloads and giving explicit control over model sources. This involves adding new input fields to the node and ensuring Hugging Face library calls use `local_files_only=True` (or equivalent).
*   **LoRA Support**: Allows users to optionally specify a LoRA file path and trigger words to apply a stylistic effect. This involves adding new optional input fields and conditionally calling an existing pipeline method.

## 3. Affected Components

The primary components affected by these changes are:

*   **[`nodes/comfy_nodes.py`](nodes/comfy_nodes.py)**: This file contains the ComfyUI node definition. It will be modified to:
    *   Include new input types for local model paths and LoRA parameters.
    *   Update the node's execution logic to handle these new inputs.
*   **[`InstantCharacter/pipeline.py`](InstantCharacter/pipeline.py)**: This file contains the `InstantCharacterFluxPipeline` class. It will be modified to:
    *   Ensure its `init_adapter` method can enforce local-only file loading for its components.
    *   Utilize its existing `with_style_lora` method for LoRA application.

## 4. Architectural Modifications - Local Model Loading

### 4.1. Node Inputs (`nodes/comfy_nodes.py`)

The `INPUT_TYPES` class method in the `InstantCharacterNode` (or similarly named class) within [`nodes/comfy_nodes.py`](nodes/comfy_nodes.py) will be updated to include the following new **required** string inputs:

*   `base_model_path`: Path to the local base InstantCharacter (Flux) model directory/file.
*   `siglip_encoder_path`: Path to the local SigLIP image encoder directory/file.
*   `dinov2_encoder_path`: Path to the local DINOv2 image encoder directory/file.
*   `ip_adapter_path`: Path to the local IP-adapter model file (e.g., `.bin`) or directory.

These inputs will replace or take precedence over any existing HuggingFace ID-based model selection if local paths are provided.

**Reference**: [`docs/02_pseudocode_node_modifications.md`](docs/02_pseudocode_node_modifications.md) (Section 2.4)

### 4.2. Node Logic (`nodes/comfy_nodes.py`)

The main execution/loading method of the node (e.g., `load_instant_character` or `execute`) will be modified:

1.  **Input Reception**: Receive the new path strings from the node inputs.
2.  **Path Validation**: Basic validation (e.g., ensuring paths are not empty) will be performed.
3.  **Base Model Loading**:
    *   The `InstantCharacterFluxPipeline.from_pretrained()` method will be called with `base_model_path`.
    *   Crucially, `local_files_only=True` (or an equivalent parameter like `offline=True` depending on the library version) will be passed to `from_pretrained` to prevent any network download attempts.
4.  **Adapter Initialization**:
    *   The `pipe.init_adapter()` method will be called, passing `siglip_encoder_path`, `dinov2_encoder_path`, and `ip_adapter_path`.
    *   A new boolean parameter, `local_files_only_for_components=True`, will be passed to `init_adapter` to ensure its internal component loading also adheres to the local-only policy.

**Reference**: [`docs/02_pseudocode_node_modifications.md`](docs/02_pseudocode_node_modifications.md) (Section 2.5)

### 4.3. Pipeline Interaction (`init_adapter` in `InstantCharacter/pipeline.py`)

The `init_adapter` method within the `InstantCharacterFluxPipeline` class in [`InstantCharacter/pipeline.py`](InstantCharacter/pipeline.py) will be modified:

1.  **New Parameter**: It will accept a new boolean parameter, e.g., `local_files_only_for_components: bool = True`.
2.  **Internal Loading**: When `init_adapter` loads its constituent components (SigLIP encoder, DINOv2 encoder, IP-adapter model) using Hugging Face utilities like `CLIPVisionModelWithProjection.from_pretrained`, `AutoImageProcessor.from_pretrained`, or `IPAdapterPlusXLInstantIDImageProjModel.from_pretrained`, it will pass this `local_files_only_for_components` flag (as `local_files_only=...`) to these calls. This ensures that these sub-loadings also do not attempt network downloads.

**Reference**: [`docs/03_pseudocode_pipeline_modifications.md`](docs/03_pseudocode_pipeline_modifications.md) (Section 3.4)

### 4.4. Data Flow (Local Model Loading)

```
ComfyUI Node Inputs:
  - base_model_path
  - siglip_encoder_path
  - dinov2_encoder_path
  - ip_adapter_path
      |
      v
nodes/comfy_nodes.py (Execution Method)
  1. pipe = InstantCharacterFluxPipeline.from_pretrained(base_model_path, local_files_only=True)
  2. pipe.init_adapter(
         image_encoder_path=siglip_encoder_path,
         image_encoder_2_path=dinov2_encoder_path,
         subject_ip_adapter_path=ip_adapter_path,
         local_files_only_for_components=True
     )
      |
      v
InstantCharacter/pipeline.py (InstantCharacterFluxPipeline)
  - init_adapter(..., local_files_only_for_components):
    - self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path, local_files_only=local_files_only_for_components)
    - self.image_encoder_2 = CLIPVisionModelWithProjection.from_pretrained(image_encoder_2_path, local_files_only=local_files_only_for_components)
    - self.image_proj_model = IPAdapterPlusXLInstantIDImageProjModel.from_pretrained(subject_ip_adapter_path, local_files_only=local_files_only_for_components)
      |
      v
Loaded Pipeline Instance (pipe)
```

## 5. Architectural Modifications - LoRA Support

### 5.1. Node Inputs (`nodes/comfy_nodes.py`)

The `INPUT_TYPES` class method in [`nodes/comfy_nodes.py`](nodes/comfy_nodes.py) will be updated to include the following new **optional** string inputs:

*   `lora_path`: Path to the LoRA `.safetensors` file. Defaults to `""`.
*   `lora_trigger`: Trigger word(s) for the LoRA. Defaults to `""`.

**Reference**: [`docs/spec_01_node_input_definitions.md`](docs/spec_01_node_input_definitions.md) (Section 3)

### 5.2. Node Logic (`nodes/comfy_nodes.py`)

The main generation method of the node (e.g., `generate_image` or `execute`) will be modified:

1.  **Input Reception**: Receive `lora_path` and `lora_trigger` from the node inputs.
2.  **Conditional Call**:
    *   If `lora_path` is provided (i.e., not `None` and not an empty string after stripping whitespace):
        *   The node will call `pipe.with_style_lora(...)`, passing `lora_path`, `lora_trigger`, the main `prompt`, and all other necessary generation parameters (e.g., `subject_image`, `num_inference_steps`, etc.).
        *   A `try-except` block will wrap this call. If `pipe.with_style_lora` fails, the node will fall back to standard generation.
    *   Else (no `lora_path` provided):
        *   The node will call the standard pipeline generation method (e.g., `pipe(...)` or `pipe.__call__(...)`) with the original parameters.

**Reference**: [`docs/spec_02_conditional_pipeline_logic.md`](docs/spec_02_conditional_pipeline_logic.md) (Section 2)

### 5.3. Pipeline Interaction (`with_style_lora` in `InstantCharacter/pipeline.py`)

The existing `with_style_lora` method in `InstantCharacterFluxPipeline` (located in [`InstantCharacter/pipeline.py`](InstantCharacter/pipeline.py)) is suitable and **requires no modifications** for this feature. Its current behavior is:

1.  Accepts `lora_file_path`, `lora_weight` (defaults to `1.0`), and `trigger`.
2.  Calls an internal helper (`flux_load_lora`) to load the LoRA.
3.  Prepends the `trigger` to the `prompt` in `kwargs`.
4.  Executes the main generation call (`self.__call__`) with modified `kwargs`.
5.  Calls `flux_load_lora` again with a negative weight to unload/reverse the LoRA.
6.  Returns the generation result.

This aligns with the requirements, as no node-level LoRA weight input is planned for this iteration.

**Reference**: [`docs/spec_03_pipeline_method_with_style_lora.md`](docs/spec_03_pipeline_method_with_style_lora.md) (Sections 1, 2)

### 5.4. Data Flow (LoRA Application)

```
ComfyUI Node Inputs:
  - prompt
  - subject_image
  - ... (other generation params)
  - lora_path (optional)
  - lora_trigger (optional)
      |
      v
nodes/comfy_nodes.py (Generation Method)
  IF lora_path is valid:
    generated_image = pipe.with_style_lora(
        lora_file_path=lora_path,
        trigger=lora_trigger,
        prompt=prompt,
        subject_image=subject_image,
        ...
    )
  ELSE:
    generated_image = pipe(
        prompt=prompt,
        subject_image=subject_image,
        ...
    )
      |
      v
InstantCharacter/pipeline.py (InstantCharacterFluxPipeline)
  - with_style_lora(lora_file_path, trigger, ...):
    1. flux_load_lora(lora_file_path, weight=1.0)
    2. modified_prompt = trigger + ", " + prompt
    3. result = self.__call__(prompt=modified_prompt, ...)
    4. flux_load_lora(lora_file_path, weight=-1.0)
    5. RETURN result
      |
      v
Generated Image (Output from Node)
```

## 6. Integration Points Summary

*   **[`nodes/comfy_nodes.py`](nodes/comfy_nodes.py)**:
    *   `INPUT_TYPES` class method: Add new inputs for local paths and LoRA.
    *   Main execution/loading method:
        *   Incorporate logic for `InstantCharacterFluxPipeline.from_pretrained(..., local_files_only=True)`.
        *   Call `pipe.init_adapter(..., local_files_only_for_components=True)`.
        *   Implement conditional logic to call `pipe.with_style_lora(...)` or `pipe(...)` based on `lora_path`.
*   **[`InstantCharacter/pipeline.py`](InstantCharacter/pipeline.py)**:
    *   `InstantCharacterFluxPipeline.init_adapter` method: Add `local_files_only_for_components` parameter and pass it to internal `from_pretrained` calls.
    *   `InstantCharacterFluxPipeline.with_style_lora` method: No changes required.

## 7. Error Handling Strategy

*   **Node Level ([`nodes/comfy_nodes.py`](nodes/comfy_nodes.py))**:
    *   Basic validation for new path inputs (e.g., check for empty strings).
    *   If `pipe.with_style_lora` raises an exception, catch it, log an informative message, and fall back to standard generation.
    *   Other exceptions during model loading or generation will propagate, causing the ComfyUI node to error out, which is standard behavior.
*   **Pipeline Level ([`InstantCharacter/pipeline.py`](InstantCharacter/pipeline.py))**:
    *   `from_pretrained` calls (with `local_files_only=True`) will raise errors if local files are missing, paths are invalid, or models are corrupted. These errors will propagate to the node.
    *   The `flux_load_lora` helper (used by `with_style_lora`) is expected to handle errors related to LoRA file loading (e.g., file not found, invalid format) by raising exceptions.
*   **User Feedback**: Errors will manifest as standard ComfyUI node errors, displaying exception messages. Print statements can be added for more detailed logging during development/debugging.

## 8. Modularity and Impact

*   **Minimal Changes**: The design adheres to the "minimal changes" principle.
    *   Local model loading primarily involves passing new parameters and flags.
    *   LoRA support leverages an existing, suitable pipeline method.
*   **Containment**:
    *   Changes in [`nodes/comfy_nodes.py`](nodes/comfy_nodes.py) are localized to the InstantCharacter node.
    *   The change in [`InstantCharacter/pipeline.py`](InstantCharacter/pipeline.py) (`init_adapter`) is minor and controlled by a new parameter, maintaining backward compatibility if the new parameter defaults appropriately or is not used by other callers.
*   **Preservation of Existing Functionality**:
    *   If local model path inputs are not used (assuming the original model selection mechanism is retained as a fallback, or if the node is refactored to *require* local paths), the node's original download behavior would be superseded. The specification implies these new local paths are the primary mechanism.
    *   If `lora_path` is not provided, the node functions exactly as before, without LoRA application.

## 9. Configuration Management

*   **Direct Path Inputs**: Model and LoRA paths are provided directly as string inputs to the node. There is no new global configuration file or mechanism introduced by this design.
*   **Path Resolution**: The paths provided (e.g., `base_model_path`, `lora_path`) are expected to be absolute paths or paths relative to a location understood by the underlying Hugging Face libraries or ComfyUI's typical model directory structure (e.g., `ComfyUI/models/`). The node itself will not perform complex path resolution beyond what the libraries offer. Users will need to ensure these paths are correct and accessible. The default values suggested in pseudocode (e.g., `ComfyUI/models/flux/`) hint at this convention.

## 10. Conclusion

This architectural design provides a clear plan for integrating local model loading and LoRA support into the ComfyUI-InstantCharacter node. It focuses on leveraging existing pipeline capabilities where possible and making targeted modifications to ensure functionality, maintainability, and adherence to user requirements. This document serves as a blueprint for the subsequent implementation phase.