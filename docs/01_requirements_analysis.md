# 1. Requirements Analysis: Local Model Loading for InstantCharacter Node

## 1.1. Introduction
This document outlines the functional requirements, edge cases, and acceptance criteria for modifying the ComfyUI InstantCharacter node to support loading all required models from local file paths. The modification aims to eliminate reliance on automatic downloads from HuggingFace and provide users with explicit control over model sources.

## 1.2. Goals
- Allow users to specify local paths for the main model, SigLIP image encoder, DINOv2 image encoder, and IP-adapter.
- Prevent any automatic model downloads from HuggingFace or other remote sources.
- Provide clear error messages if specified local model files are not found or are invalid.
- Ensure the node functions correctly using the locally loaded models.

## 1.3. Functional Requirements

### FR1: Local Base Model Loading
- The node MUST provide an input field for the user to specify the local file path or directory path to the main InstantCharacter (Flux) model.
- The node MUST load the base model using `InstantCharacterFluxPipeline.from_pretrained` (or an equivalent method) by passing the user-provided local path.
- The system MUST NOT attempt to download the base model if the local path is provided.
- **TDD Anchor**: `// TEST: FR1.1 Base model loads successfully from valid local path`
- **TDD Anchor**: `// TEST: FR1.2 System errors if base model path is invalid or file not found`
- **TDD Anchor**: `// TEST: FR1.3 No download attempt occurs when local base model path is specified`

### FR2: Local Image Encoder Loading
- The node MUST provide input fields for the user to specify local file paths or directory paths for:
    - SigLIP image encoder
    - DINOv2 image encoder
- The `pipe.init_adapter(...)` method (or equivalent logic) MUST be modified to use these local paths for `image_encoder_path` (SigLIP) and `image_encoder_2_path` (DINOv2).
- The system MUST NOT attempt to download these image encoders if local paths are provided.
- **TDD Anchor**: `// TEST: FR2.1 SigLIP encoder loads successfully from valid local path`
- **TDD Anchor**: `// TEST: FR2.2 DINOv2 encoder loads successfully from valid local path`
- **TDD Anchor**: `// TEST: FR2.3 System errors if SigLIP encoder path is invalid or file not found`
- **TDD Anchor**: `// TEST: FR2.4 System errors if DINOv2 encoder path is invalid or file not found`
- **TDD Anchor**: `// TEST: FR2.5 No download attempt occurs for encoders when local paths are specified`

### FR3: Local IP-Adapter Loading
- The node MUST provide an input field for the user to specify the local file path for the IP-adapter model.
- The `pipe.init_adapter(...)` method (or equivalent logic) MUST be modified to use this local path for `subject_ip_adapter_path`.
- The system MUST NOT attempt to download the IP-adapter if the local path is provided.
- **TDD Anchor**: `// TEST: FR3.1 IP-adapter loads successfully from valid local path`
- **TDD Anchor**: `// TEST: FR3.2 System errors if IP-adapter path is invalid or file not found`
- **TDD Anchor**: `// TEST: FR3.3 No download attempt occurs for IP-adapter when local path is specified`

### FR4: Disable Automatic Downloads
- All automatic download functionalities related to these models within the node's execution path MUST be disabled or bypassed when local paths are utilized.
- If a local path is provided, the system MUST use `local_files_only=True` (or equivalent mechanism) in HuggingFace library calls, or directly pass the path if the library supports it for local-only loading.
- If a required local model file is missing or the path is invalid, the node MUST raise an error and NOT fall back to downloading.
- **TDD Anchor**: `// TEST: FR4.1 Node execution fails if a required local model is missing, without attempting download`
- **TDD Anchor**: `// TEST: FR4.2 HuggingFace calls use local_files_only=True or equivalent when local paths are given`

### FR5: Node Input Configuration
- The node's `INPUT_TYPES` definition MUST be updated to include new string input fields for:
    - `base_model_path`
    - `siglip_encoder_path`
    - `dinov2_encoder_path`
    - `ip_adapter_path`
- These inputs should accept string paths (absolute or relative to a configurable base, e.g., `ComfyUI/models/`). The node should clearly document how relative paths are resolved. For simplicity in this initial specification, we will assume paths are either absolute or directly usable by the underlying libraries.
- **TDD Anchor**: `// TEST: FR5.1 New input fields are present in the node UI`
- **TDD Anchor**: `// TEST: FR5.2 Node correctly interprets provided paths`

## 1.4. Edge Cases
- **EC1**: User provides an empty string for a model path. (System should error or treat as missing).
- **EC2**: User provides a path to a directory when a specific file is expected (e.g., for IP-adapter `.bin` file). (System should handle this gracefully, possibly by looking for a default filename within the directory, or error).
- **EC3**: User provides a path to an incorrect type of model file (e.g., a text file instead of a model checkpoint). (Loading will likely fail; error should be informative).
- **EC4**: File permissions prevent reading the model files. (System should error).
- **EC5**: Paths contain unusual characters or are malformed. (System should error).
- **EC6**: Partial paths provided - some local, some not. (The design is to make all specified paths local-only. If a path is not provided for a model that *could* be local, the original behavior might persist for that specific model unless explicitly disabled globally. This spec focuses on the case where *all* these models are loaded locally via new inputs).

## 1.5. Acceptance Criteria
- **AC1**: The InstantCharacter node can successfully generate an image when all required models (base, SigLIP, DINOv2, IP-adapter) are loaded from valid, user-specified local paths.
- **AC2**: The node execution fails with a clear error message if any of the specified local model paths are invalid, files are missing, or files are corrupted, without attempting any network requests for those models.
- **AC3**: No HuggingFace model downloads are initiated by the node for the base model, image encoders, or IP-adapter when local paths are provided and used.
- **AC4**: The new input fields for local paths are visible and configurable in the ComfyUI interface for the InstantCharacter node.

## 1.6. Constraints
- **C1**: Changes should be primarily within [`nodes/comfy_nodes.py`](nodes/comfy_nodes.py), minimizing modifications to the core [`InstantCharacter/pipeline.py`](InstantCharacter/pipeline.py) if possible, unless `pipeline.py` inherently handles path resolution in a way that needs adjustment for local-only loading.
- **C2**: The solution must be compatible with the existing ComfyUI node structure and practices.
- **C3**: Security considerations for file path handling (e.g., preventing path traversal) are important but will be more deeply addressed in a separate security review. For this specification, assume valid, user-intended paths.

## 1.7. Out of Scope
- Automatic searching for models in predefined ComfyUI directories (e.g., `ComfyUI/models/checkpoints/`) if a simple name is given. The current specification assumes the user provides a direct, usable path. (This could be a future enhancement).
- UI for browsing/selecting local paths (ComfyUI typically uses string inputs for paths).