# Specification: New Node Inputs for LoRA Support

This document outlines the new input types required for the ComfyUI-InstantCharacter node to support LoRA (Low-Rank Adaptation) styles.

## 1. Overview

To enable LoRA support, two new optional string inputs will be added to the node:
- `lora_path`: Specifies the file path to the LoRA model (`.safetensors` format).
- `lora_trigger`: Specifies the trigger phrase(s) associated with the LoRA model.

These inputs will allow users to optionally apply a LoRA style during character generation. If `lora_path` is not provided, the node will function as it currently does, without attempting to apply any LoRA.

## 2. Input Definitions

### 2.1. `lora_path`

-   **Type**: `STRING`
-   **Purpose**: To provide the full file system path to the LoRA `.safetensors` file.
-   **Optional**: Yes
-   **Default Value**: `""` (empty string)
-   **Behavior**:
    -   If a valid path is provided, the node will attempt to load and apply the specified LoRA.
    -   If empty or invalid, LoRA processing will be skipped.
-   **TDD Anchor**: `// TEST: lora_path input accepts valid .safetensors path`
-   **TDD Anchor**: `// TEST: lora_path input defaults to empty string`

### 2.2. `lora_trigger`

-   **Type**: `STRING`
-   **Purpose**: To provide the trigger word(s) or phrase(s) necessary to activate the LoRA's style. This phrase will be prepended to the user's main prompt.
-   **Optional**: Yes
-   **Default Value**: `""` (empty string)
-   **Behavior**:
    -   If `lora_path` is provided and this field contains a trigger, it will be used in conjunction with the LoRA.
    -   If `lora_path` is provided but this field is empty, the LoRA might still be applied, but its effect might be suboptimal if it relies on specific triggers. (The `with_style_lora` method in the pipeline should handle this gracefully).
-   **TDD Anchor**: `// TEST: lora_trigger input accepts string value`
-   **TDD Anchor**: `// TEST: lora_trigger input defaults to empty string`

## 3. Implementation in `INPUT_TYPES`

These new inputs will be added to the `INPUT_TYPES` class method within the relevant node definition in `nodes/comfy_nodes.py`.

**Pseudocode for `INPUT_TYPES` modification in `nodes/comfy_nodes.py`:**

```python
class InstantCharacterNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # ... existing required inputs ...
                "prompt": ("STRING", {"multiline": True, "default": "photo of a character"}),
                "subject_image": ("IMAGE",),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0}),
                "subject_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
                # ... other existing inputs ...
            },
            "optional": {
                # ... existing optional inputs ...
                "lora_path": ("STRING", {"multiline": False, "default": ""}), # NEW INPUT
                # TEST: lora_path is present in INPUT_TYPES
                "lora_trigger": ("STRING", {"multiline": True, "default": ""}), # NEW INPUT
                # TEST: lora_trigger is present in INPUT_TYPES
            }
        }

    # ... rest of the node class ...
```

## 4. Constraints and Considerations

-   **No LoRA Weight Input**: For simplicity in this iteration, a LoRA weight input is not being added. The pipeline method (`with_style_lora`) is assumed to use a default weight (e.g., 1.0).
-   **No Separate LoRA Loader**: The LoRA loading and unloading will be handled within the pipeline, not by a separate ComfyUI loader node.
-   **Error Handling**: Basic validation (e.g., checking if `lora_path` is non-empty) will occur in the node. More detailed validation (file existence, format correctness) should ideally be handled by the pipeline method.
    -   `// TEST: Node handles empty lora_path by falling back to default pipeline`