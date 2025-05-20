# Specification: Pipeline Method `with_style_lora`

This document addresses the `with_style_lora` method within the `InstantCharacterFluxPipeline` class, located in `InstantCharacter/pipeline.py`, in the context of adding LoRA support to the ComfyUI-InstantCharacter node.

## 1. Confirmation of Existence and Suitability

A method named `with_style_lora` already exists in [`InstantCharacter/pipeline.py`](InstantCharacter/pipeline.py:552) within the `InstantCharacterFluxPipeline` class. Its signature is:

```python
def with_style_lora(self, lora_file_path, lora_weight=1.0, trigger='', *args, **kwargs):
```

This existing method is suitable for the current LoRA integration requirements for the following reasons:
-   It directly accepts `lora_file_path` (which corresponds to the node's `lora_path` input) and `trigger` (corresponding to the node's `lora_trigger` input).
-   It includes a `lora_weight` parameter that defaults to `1.0`. This aligns with the requirement that no LoRA weight input is needed at the node level for this iteration, and a default weight of 1.0 will be used.
-   It is designed to wrap the main generation call (`self.__call__`), passing through other necessary generation parameters via `*args` and `**kwargs`.

## 2. Existing Behavior of `with_style_lora`

The current implementation of [`with_style_lora(self, lora_file_path, lora_weight=1.0, trigger='', *args, **kwargs)`](InstantCharacter/pipeline.py:552) performs the following steps:

1.  **Load LoRA**: It calls an internal helper function, [`flux_load_lora(self, lora_file_path, lora_weight)`](InstantCharacter/pipeline.py:12), to load the specified LoRA model into the pipeline with the given weight.
    -   `// TEST: flux_load_lora is called with correct path and weight (1.0 by default)`
2.  **Prepend Trigger to Prompt**: It modifies the `prompt` within the `kwargs` by prepending the `trigger` phrase. If the `trigger` is empty, the prompt remains largely unchanged (potentially with an added comma and space if the original prompt exists).
    -   `kwargs['prompt'] = f"{trigger}, {kwargs['prompt']}"`
    -   `// TEST: Trigger phrase is correctly prepended to the prompt`
    -   `// TEST: Empty trigger phrase results in minimal change to prompt structure`
3.  **Execute Generation**: It calls the main pipeline generation method, `self.__call__(*args, **kwargs)`, passing all original and modified arguments. This step performs the actual image generation.
    -   `// TEST: self.__call__ is invoked with LoRA active and modified prompt`
4.  **Unload LoRA**: After generation, it calls [`flux_load_lora(self, lora_file_path, -lora_weight)`](InstantCharacter/pipeline.py:12) again, but with a negative `lora_weight`. This is a common technique to effectively unload or reverse the effect of the LoRA from the model.
    -   `// TEST: flux_load_lora is called with negative weight to unload LoRA`
5.  **Return Result**: It returns the result from `self.__call__` (the generated image).

## 3. Implementation Status

**No new implementation or significant modification of the `with_style_lora` method itself is required for the current scope of adding LoRA support as defined.**

The existing method already fulfills the specified behaviors:
-   Loading the LoRA model.
-   Prepending the trigger phrase to the prompt.
-   Performing the generation.
-   Unloading the LoRA model.

The primary implementation effort will be within the ComfyUI node (`nodes/comfy_nodes.py`) to correctly call this existing pipeline method with the appropriate parameters, as detailed in `docs/spec_02_conditional_pipeline_logic.md`.

## 4. Assumptions

-   The helper function [`flux_load_lora()`](InstantCharacter/pipeline.py:12) (imported from `.models.utils`) correctly handles the loading and unloading (via negative weight) of `.safetensors` LoRA files for the FLUX pipeline.
    -   `// TEST: flux_load_lora successfully loads a valid .safetensors LoRA file`
    -   `// TEST: flux_load_lora successfully unloads/reverses LoRA with negative weight`
    -   `// TEST: flux_load_lora handles invalid LoRA file path gracefully (e.g., raises error)`
-   The main generation method `self.__call__` correctly utilizes the LoRA when it's loaded into the model components (e.g., transformer/UNet).

## 5. TDD Anchors Summary (for `with_style_lora` context)

-   `// TEST: flux_load_lora is called with correct path and weight (1.0 by default)`
-   `// TEST: Trigger phrase is correctly prepended to the prompt`
-   `// TEST: Empty trigger phrase results in minimal change to prompt structure`
-   `// TEST: self.__call__ is invoked with LoRA active and modified prompt`
-   `// TEST: flux_load_lora is called with negative weight to unload LoRA`
-   `// TEST: flux_load_lora successfully loads a valid .safetensors LoRA file` (Relates to `flux_load_lora` itself)
-   `// TEST: flux_load_lora successfully unloads/reverses LoRA with negative weight` (Relates to `flux_load_lora` itself)
-   `// TEST: flux_load_lora handles invalid LoRA file path gracefully (e.g., raises error)` (Relates to `flux_load_lora` itself)

This confirms that the pipeline side of the LoRA integration is largely in place, simplifying the overall task.