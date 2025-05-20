# Specification: Conditional Pipeline Call Logic for LoRA Support

This document outlines the pseudocode for the execution logic within the ComfyUI-InstantCharacter node, specifically how it handles the conditional invocation of a LoRA-enhanced pipeline.

## 1. Overview

The node's primary execution method (e.g., `generate`, `sample`, or a similar method responsible for image generation) will be modified to check for the presence of a `lora_path`.
- If `lora_path` is provided and is not an empty string, the node will attempt to use a specialized pipeline method designed for LoRA integration (assumed to be `pipe.with_style_lora`).
- Otherwise, it will fall back to the standard generation pipeline method (assumed to be `pipe`).

This ensures that the node maintains its original functionality if LoRA parameters are not supplied.

## 2. Node Execution Method Pseudocode

The following pseudocode illustrates the modified logic within the main execution method of the `InstantCharacterNode` class in `nodes/comfy_nodes.py`. The exact method name (`generate_image` is used here as a placeholder) should match the actual implementation.

**Pseudocode for execution method in `nodes/comfy_nodes.py`:**

```python
class InstantCharacterNode:
    # ... (INPUT_TYPES and other class members) ...

    # Assuming the execution method is named 'generate_image'
    # The actual parameters will depend on the node's INPUT_TYPES
    def generate_image(self, prompt, subject_image, num_inference_steps, guidance_scale, subject_scale, # ... other_params ...,
                       lora_path=None, lora_trigger=None): # New optional LoRA params

        # TEST: generate_image method exists and accepts lora_path and lora_trigger

        # Initialize or retrieve the generation pipeline (pipe)
        # This step is assumed to be part of the existing node setup
        # pipe = self.get_pipeline_instance() # Example

        # --- Conditional LoRA Pipeline Invocation ---
        # TEST: lora_path is checked for non-empty value
        if lora_path and lora_path.strip() != "":
            # A LoRA path is provided.
            # TEST: Non-empty lora_path triggers with_style_lora call
            print(f"Attempting to use LoRA: {lora_path} with trigger: '{lora_trigger}'")

            # Call the LoRA-enhanced pipeline method.
            # The exact parameters passed to with_style_lora will depend on its definition
            # in InstantCharacter/pipeline.py. We assume it takes at least these:
            try:
                # TEST: with_style_lora is called with correct parameters when lora_path is valid
                generated_image = pipe.with_style_lora(
                    prompt=prompt,
                    lora_file_path=lora_path,
                    trigger_phrase=lora_trigger, # Pass trigger phrase
                    subject_image=subject_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    subject_scale=subject_scale,
                    # ... any other necessary parameters from other_params ...
                )
                # TEST: Successful generation with LoRA returns an image
            except Exception as e:
                # TEST: Error during with_style_lora call is handled (e.g., logs error, returns fallback)
                print(f"Error during LoRA-enhanced generation: {e}. Falling back to standard generation.")
                # Fallback to standard generation if LoRA pipeline fails
                generated_image = pipe(
                    prompt=prompt,
                    subject_image=subject_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    subject_scale=subject_scale,
                    # ... any other necessary parameters from other_params ...
                )
        else:
            # No LoRA path provided, or it's an empty string. Use standard pipeline.
            # TEST: Empty or None lora_path triggers standard pipe call
            print("No LoRA path provided. Using standard generation pipeline.")
            generated_image = pipe(
                prompt=prompt,
                subject_image=subject_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                subject_scale=subject_scale,
                # ... any other necessary parameters from other_params ...
            )
            # TEST: Successful standard generation returns an image

        # Return the generated image in the format expected by ComfyUI
        # (e.g., a tuple containing the image tensor)
        # TEST: Node returns image in expected ComfyUI format
        return (generated_image,)

    # ... (other methods like IS_CHANGED if necessary) ...
```

## 3. Key Assumptions

-   **Pipeline Instance**: The node has a mechanism to access an instance of the generation pipeline (referred to as `pipe`).
-   **Pipeline Methods**:
    -   The standard pipeline call is `pipe(...)`.
    -   The LoRA-enhanced pipeline call is `pipe.with_style_lora(...)`.
-   **Parameter Consistency**: The parameters required by `pipe(...)` and `pipe.with_style_lora(...)` (excluding LoRA-specific ones for the latter) are largely consistent or can be adapted.
-   **Error Handling in `with_style_lora`**: The `with_style_lora` method itself is expected to handle internal errors related to LoRA loading (e.g., file not found, invalid format). The node's execution logic includes a basic try-except block as a fallback.

## 4. TDD Anchors Summary

-   `// TEST: generate_image method exists and accepts lora_path and lora_trigger`
-   `// TEST: lora_path is checked for non-empty value`
-   `// TEST: Non-empty lora_path triggers with_style_lora call`
-   `// TEST: with_style_lora is called with correct parameters when lora_path is valid`
-   `// TEST: Successful generation with LoRA returns an image`
-   `// TEST: Error during with_style_lora call is handled (e.g., logs error, returns fallback)`
-   `// TEST: Empty or None lora_path triggers standard pipe call`
-   `// TEST: Successful standard generation returns an image`
-   `// TEST: Node returns image in expected ComfyUI format`

This logic ensures that the integration of LoRA support is conditional and does not alter the node's default behavior when LoRA is not used.