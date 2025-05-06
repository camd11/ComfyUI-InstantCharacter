# Architectural Design: Refactoring InstantCharacter ComfyUI Nodes

## 1. Overview

This document outlines the architectural changes for refactoring the `InstantCharacter` ComfyUI custom nodes. The primary goal is to enhance modularity by accepting pre-loaded model components (UNet, CLIP Vision models, IPAdapter weights) from standard ComfyUI loader nodes, rather than relying on hardcoded paths or internal download mechanisms.

This involves:
- Modifying the `InstantCharacterFluxPipeline` class in `InstantCharacter/pipeline.py` to accept these pre-loaded model objects in its constructor.
- Creating a new ComfyUI node, `InstantCharacterLoader`, in `nodes/comfy_nodes.py` to gather these model objects and instantiate the pipeline.
- Removing the old `InstantCharacterLoadModel` node that handled model loading via paths/downloads.
- Ensuring the `InstantCharacterGenerate` node remains compatible with the refactored pipeline.

## 2. Class Structures and Method Signatures

### 2.1. `InstantCharacterFluxPipeline` (in `InstantCharacter/pipeline.py`)

The `InstantCharacterFluxPipeline` will be modified to initialize with pre-loaded model components.

**Assumptions:**
- The `flux_unet_model_object` (ComfyUI `MODEL` type) is expected to contain the UNet/transformer, VAE, and potentially the FLUX text encoders and tokenizers. If not, these would need to be separate inputs.
- `siglip_vision_model_object` and `dinov2_vision_model_object` (ComfyUI `CLIP_VISION` type) provide both the vision model and its associated preprocessor.
- `ipadapter_model_data_dict` (ComfyUI `IPADAPTER` type) is a dictionary containing the IPAdapter weights.

```python
# File: InstantCharacter/pipeline.py
import torch
from PIL import Image
# from diffusers import DiffusionPipeline # Or a more suitable base class
# from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput # If used

from .models.attn_processor import FluxIPAttnProcessor
from .models.resampler import CrossLayerCrossScaleProjector
# ... other necessary model imports

class InstantCharacterFluxPipeline(torch.nn.Module): # Or BaseDiffusionPipeline
    def __init__(self,
                 flux_unet_model_object,      # ComfyUI MODEL object (UNet, VAE, TextEncoders, Scheduler config)
                 siglip_vision_model_object,  # ComfyUI CLIP_VISION object for SigLIP
                 dinov2_vision_model_object,  # ComfyUI CLIP_VISION object for DINOv2
                 ipadapter_model_data_dict,   # Dict of IPAdapter weights (e.g., from IPAdapterModelLoader)
                 dtype=torch.bfloat16):
        """
        Initializes the InstantCharacterFluxPipeline with pre-loaded model components.

        Args:
            flux_unet_model_object: A ComfyUI MODEL object. Expected to contain:
                - .diffusion_model (the UNet/transformer)
                - .first_stage_model (the VAE)
                - .text_encoder_one, .tokenizer_one (for FLUX text encoder 1) - TBC
                - .text_encoder_two, .tokenizer_two (for FLUX text encoder 2) - TBC
                - .scheduler (or scheduler config) - TBC
                If these are not part of the MODEL object, they need to be passed separately.
            siglip_vision_model_object: A ComfyUI CLIP_VISION object for SigLIP.
                - .model (the vision transformer)
                - (provides preprocessing capabilities)
            dinov2_vision_model_object: A ComfyUI CLIP_VISION object for DINOv2.
                - .model (the vision transformer)
                - (provides preprocessing capabilities)
            ipadapter_model_data_dict: A dictionary containing the state_dicts for
                                       'ip_adapter' and 'image_proj' for the IPAdapter.
            dtype: The torch dtype for the pipeline.
        """
        super().__init__()

        # Assign FLUX components (UNet, VAE, Text Encoders, Scheduler)
        # Example:
        self.transformer = flux_unet_model_object.diffusion_model
        self.vae = flux_unet_model_object.first_stage_model
        # self.text_encoder = flux_unet_model_object.text_encoder_one # Placeholder
        # self.tokenizer = flux_unet_model_object.tokenizer_one     # Placeholder
        # self.text_encoder_2 = flux_unet_model_object.text_encoder_two # Placeholder
        # self.tokenizer_2 = flux_unet_model_object.tokenizer_two   # Placeholder
        # self.scheduler = flux_unet_model_object.scheduler # Or initialize a FluxScheduler

        # Assign Image Encoders
        self.siglip_image_encoder_model = siglip_vision_model_object.model
        self.siglip_image_processor = siglip_vision_model_object # Used for its preprocessing info

        self.dinov2_image_encoder_model = dinov2_vision_model_object.model
        self.dinov2_image_processor = dinov2_vision_model_object # Used for its preprocessing info

        self.dtype = dtype
        self._initialize_ip_adapter_components(ipadapter_model_data_dict, self.dtype)

        # TODO: Register modules if necessary for Diffusers compatibility or ComfyUI management
        # self.register_modules(...)

    def _initialize_ip_adapter_components(self, ipadapter_state_dict, dtype):
        """
        Initializes IPAdapter attention processors and image projection model
        using the provided state dictionary.
        """
        device = self.transformer.device # Assuming transformer is already on the correct device

        # Initialize and load FluxIPAttnProcessor
        attn_procs = {}
        # These dimensions need to be accurate for FLUX.1-dev or configurable
        flux_transformer_hidden_size = 4096 # Example, verify from FLUX config
        flux_text_encoder_dim = 4096    # Example, verify (e.g., text_encoder_2.config.d_model)

        for name in self.transformer.attn_processors.keys():
            attn_procs[name] = FluxIPAttnProcessor(
                hidden_size=flux_transformer_hidden_size,
                ip_hidden_states_dim=flux_text_encoder_dim,
            ).to(device, dtype=dtype)
        self.transformer.set_attn_processor(attn_procs)
        # Load weights for IP Attn Layers
        # Assuming ipadapter_state_dict["ip_adapter"] contains the state_dict for these ModuleList
        tmp_ip_layers = torch.nn.ModuleList(self.transformer.attn_processors.values())
        tmp_ip_layers.load_state_dict(ipadapter_state_dict["ip_adapter"], strict=False)


        # Initialize and load CrossLayerCrossScaleProjector (image_proj_model)
        # Parameters need to be accurate for the pre-trained IPAdapter model.
        self.subject_image_proj_model = CrossLayerCrossScaleProjector(
            inner_dim=1152 + 1536, num_attention_heads=42, attention_head_dim=64,
            cross_attention_dim=1152 + 1536, num_layers=4, dim=1280, depth=4,
            dim_head=64, heads=20, num_queries=1024, # nb_token
            embedding_dim=1152 + 1536, output_dim=flux_transformer_hidden_size, # output_dim should match transformer's input for IP
            ff_mult=4,
            timestep_in_dim=320, timestep_flip_sin_to_cos=True, timestep_freq_shift=0,
        ).to(device, dtype=dtype)
        self.subject_image_proj_model.eval()
        self.subject_image_proj_model.load_state_dict(ipadapter_state_dict["image_proj"], strict=False)

    @torch.inference_mode()
    def encode_siglip_image_emb(self, siglip_pixel_values, device, dtype):
        """ Encodes preprocessed SigLIP pixel values. """
        self.siglip_image_encoder_model.to(device, dtype=dtype)
        siglip_pixel_values = siglip_pixel_values.to(device, dtype=dtype)
        # Ensure output_hidden_states=True is compatible with how ComfyUI wraps CLIPVisionModel
        res = self.siglip_image_encoder_model(siglip_pixel_values, output_hidden_states=True)
        # Verify hidden state indices [7, 13, 26] for the specific SigLIP model
        siglip_image_embeds = res.last_hidden_state
        siglip_image_shallow_embeds = torch.cat([res.hidden_states[i] for i in [7, 13, 26]], dim=1)
        return siglip_image_embeds, siglip_image_shallow_embeds

    @torch.inference_mode()
    def encode_dinov2_image_emb(self, dinov2_pixel_values, device, dtype):
        """ Encodes preprocessed DINOv2 pixel values. """
        self.dinov2_image_encoder_model.to(device, dtype=dtype)
        dinov2_pixel_values = dinov2_pixel_values.to(device, dtype=dtype)
        res = self.dinov2_image_encoder_model(dinov2_pixel_values, output_hidden_states=True)
        # Verify hidden state indices [9, 19, 29] for the specific DINOv2 model
        dinov2_image_embeds = res.last_hidden_state[:, 1:] # Exclude CLS token
        dinov2_image_shallow_embeds = torch.cat([res.hidden_states[i][:, 1:] for i in [9, 19, 29]], dim=1)
        return dinov2_image_embeds, dinov2_image_shallow_embeds

    def _comfy_clip_vision_preprocess_pil(self, clip_vision_obj, pil_images_list):
        """
        Helper to preprocess PIL images using ComfyUI's CLIPVision object's expected input format
        and its internal preprocessing utilities.
        This is a conceptual placeholder; actual implementation will depend on comfy.clip_vision.clip_preprocess
        and how CLIP_VISION objects expose their target size, mean, and std.
        """
        # Convert PIL list to batched ComfyUI image tensor (B, H, W, C), range 0-1
        # This step needs careful implementation based on ComfyUI tensor conventions.
        # Example:
        # comfy_images = []
        # for pil_image in pil_images_list:
        #     np_image = np.array(pil_image).astype(np.float32) / 255.0
        #     comfy_images.append(torch.from_numpy(np_image))
        # comfy_image_tensor = torch.stack(comfy_images)

        # Access preprocessing parameters from clip_vision_obj
        # target_size = clip_vision_obj.image_size # e.g., (384, 384)
        # mean = clip_vision_obj.image_mean
        # std = clip_vision_obj.image_std
        # pixel_values = comfy.clip_vision.clip_preprocess(comfy_image_tensor, size=target_size, mean=mean, std=std)
        # return pixel_values
        raise NotImplementedError("Preprocessing with ComfyUI CLIP_VISION object needs concrete implementation.")


    @torch.inference_mode()
    def encode_image_emb(self, subject_image_pil: Image.Image, device, dtype):
        """
        Encodes the subject PIL image using pre-loaded SigLIP and DINOv2 models.
        Relies on the CLIP_VISION objects for preprocessing information.
        """
        # Cropping and resizing logic for low_res and high_res PIL images (remains similar to original)
        object_image_pil_low_res = [subject_image_pil.resize((384, 384))] # Example size
        # ... (high_res cropping as in original, e.g., 4 crops of 512x512)
        # object_image_pil_high_res = [...] # list of 4 cropped PIL images

        # Preprocess PIL images
        # siglip_low_res_pixels = self._comfy_clip_vision_preprocess_pil(self.siglip_image_processor, object_image_pil_low_res)
        # dinov2_low_res_pixels = self._comfy_clip_vision_preprocess_pil(self.dinov2_image_processor, object_image_pil_low_res)
        # siglip_high_res_pixels = self._comfy_clip_vision_preprocess_pil(self.siglip_image_processor, object_image_pil_high_res)
        # dinov2_high_res_pixels = self._comfy_clip_vision_preprocess_pil(self.dinov2_image_processor, object_image_pil_high_res)

        # For now, using placeholder direct PIL-to-tensor and then model's processor if available,
        # or assuming CLIP_VISION objects can take PIL directly or have a .preprocess() method.
        # This part needs careful integration with ComfyUI's CLIPVision preprocessing.
        # The pseudocode in docs/02_pseudocode_pipeline_mods.md suggests a helper like
        # `comfy_clip_vision_preprocess_pil`. This needs to be implemented correctly.

        # Placeholder for actual preprocessing and encoding:
        # siglip_embeds_tuple = self.encode_siglip_image_emb(siglip_low_res_pixels, device, dtype)
        # dinov2_embeds_tuple = self.encode_dinov2_image_emb(dinov2_low_res_pixels, device, dtype)
        # ... and for high_res

        # This method's implementation details for preprocessing are critical and depend on
        # how ComfyUI's CLIP_VISION objects are best utilized for their preprocessing steps.
        # The original pipeline had its own image processing logic.
        # The goal is to leverage the preprocessing associated with the loaded CLIP_VISION objects.

        # Returning dummy data for structure
        return {
            "image_embeds_low_res_shallow": torch.zeros((1,1,1), device=device, dtype=dtype), # Placeholder
            "image_embeds_low_res_deep": torch.zeros((1,1,1), device=device, dtype=dtype),    # Placeholder
            "image_embeds_high_res_deep": torch.zeros((1,1,1), device=device, dtype=dtype)   # Placeholder
        }


    @torch.no_grad()
    def __call__(self,
                 prompt: str,
                 subject_image: Image.Image,
                 height: int = 1024,
                 width: int = 1024,
                 num_inference_steps: int = 20,
                 guidance_scale: float = 7.5,
                 negative_prompt: str = None, # Add if FLUX supports it
                 generator: torch.Generator = None,
                 subject_scale: float = 1.0,
                 output_type: str = "pil", # "pil", "latent"
                 return_dict: bool = True,
                 # ... other FLUX specific parameters
                ):
        """
        Main generation call. Uses pre-initialized components.
        The logic for prompt encoding, latent preparation, denoising loop, and image decoding
        will largely follow the original FLUX pipeline's __call__ method, adapted to use
        the instance's `self.transformer`, `self.vae`, `self.scheduler`, `self.text_encoder`, etc.
        """
        device = self.transformer.device # Or comfy.model_management.get_torch_device()

        # 1. Encode prompt (using self.tokenizer, self.text_encoder, self.text_encoder_2)
        #    This needs to be implemented based on how FLUX handles its dual text encoders.
        #    prompt_embeds = self._encode_prompt(prompt, device, num_images_per_prompt, ...)

        # 2. Encode subject image
        image_embeds_dict = self.encode_image_emb(subject_image, device, self.dtype)
        # These embeds are then passed to the IP-Adapter attention processors via `self.transformer.set_attn_processor_state`
        # or directly used by custom `FluxIPAttnProcessor` if it's designed to take them in `__call__`.
        # The `subject_image_proj_model` will process these further.

        # 3. Prepare timesteps and latents (using self.scheduler, self.vae)

        # 4. Denoising loop (using self.transformer, self.scheduler)
        #    - IP-Adapter injection happens here, using `image_embeds_dict` processed by `self.subject_image_proj_model`
        #      and fed into `FluxIPAttnProcessor` instances.

        # 5. Decode latents (using self.vae)

        # 6. Post-process and return
        # if output_type == "pil":
        #     images = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        #     images = self.image_processor.postprocess(images, output_type=output_type) # If diffusers ImageProcessor is used
        # else: # "latent"
        #     images = latents

        # if not return_dict:
        #     return (images,)

        # return FluxPipelineOutput(images=images) # Or similar output structure
        raise NotImplementedError("Full __call__ method needs to be implemented based on FLUX logic.")

```

### 2.2. `InstantCharacterLoader` (in `nodes/comfy_nodes.py`)

This new node will take pre-loaded model components as inputs and output the initialized `InstantCharacterFluxPipeline`.

```python
# File: nodes/comfy_nodes.py
import torch
import folder_paths
# from comfy import model_management # For device and dtype
from InstantCharacter.pipeline import InstantCharacterFluxPipeline

class InstantCharacterLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_unet_model": ("MODEL", {}),        # From GGUF Unet Loader or standard UNet Loader
                "siglip_vision_model": ("CLIP_VISION", {}), # From CLIP Vision Loader (for SigLIP)
                "dinov2_vision_model": ("CLIP_VISION", {}), # From CLIP Vision Loader (for DINOv2)
                "ipadapter_model_data": ("IPADAPTER", {}), # From IPAdapter Model Loader (expects dict: {"ip_adapter": state_dict, "image_proj": state_dict})
                "cpu_offload": ("BOOLEAN", {"default": False}), # Retained functionality
            },
            # Optional, if not bundled in `flux_unet_model` (MODEL type):
            # "flux_vae": ("VAE", {}),
            # "flux_text_encoder_1": ("CLIP", {}), # Or appropriate type for FLUX text encoder 1
            # "flux_text_encoder_2": ("CLIP", {}), # Or appropriate type for FLUX text encoder 2
        }

    RETURN_TYPES = ("INSTANTCHAR_PIPE",)
    FUNCTION = "load_pipe_from_models"
    CATEGORY = "InstantCharacter"
    # DESCRIPTION = "Loads InstantCharacter pipeline from pre-loaded model components." # Optional

    def load_pipe_from_models(self,
                              flux_unet_model,
                              siglip_vision_model,
                              dinov2_vision_model,
                              ipadapter_model_data,
                              cpu_offload,
                              # flux_vae=None, flux_text_encoder_1=None, flux_text_encoder_2=None # If passed separately
                             ):

        # Determine device and dtype, e.g., from an input model or ComfyUI's model_management
        # device = model_management.get_torch_device()
        # dtype = model_management.VAE_DTYPE # Or infer, e.g. flux_unet_model.model.dtype

        # TODO: Add robust checks for the structure of ipadapter_model_data
        # It should be a dictionary like:
        # {
        #   "ip_adapter": state_dict_for_attn_procs,
        #   "image_proj": state_dict_for_image_proj_model
        # }
        if not isinstance(ipadapter_model_data, dict) or \
           "ip_adapter" not in ipadapter_model_data or \
           "image_proj" not in ipadapter_model_data:
            raise ValueError("IPAdapter model data is not in the expected dictionary format with 'ip_adapter' and 'image_proj' keys.")


        # Instantiate the refactored pipeline
        pipe = InstantCharacterFluxPipeline(
            flux_unet_model_object=flux_unet_model,
            siglip_vision_model_object=siglip_vision_model,
            dinov2_vision_model_object=dinov2_vision_model,
            ipadapter_model_data_dict=ipadapter_model_data,
            # dtype=dtype # Pass determined dtype
        )

        # CPU Offload:
        # ComfyUI's model management typically handles device placement for MODEL, CLIP_VISION, etc.
        # If the pipeline itself has specific offloading methods (like Diffusers' enable_sequential_cpu_offload),
        # they could be called here based on `cpu_offload`.
        # For now, assuming individual components are managed by ComfyUI and the pipeline
        # correctly initializes its own new modules on the device of `flux_unet_model`.
        if cpu_offload:
            print("InstantCharacter: CPU offload requested. Note: Pipeline components are initialized on the device of the input UNet. ComfyUI manages individual model offloading.")
            # If pipe had a method like pipe.to("cpu") or pipe.enable_sequential_cpu_offload()
            # pipe.enable_sequential_cpu_offload() # Example

        return (pipe,)

```

### 2.3. `InstantCharacterLoadModel` (in `nodes/comfy_nodes.py`)

This node is to be **REMOVED**. Its functionality of loading models from paths/Hugging Face Hub is superseded by using dedicated ComfyUI loader nodes (GGUF Unet Loader, CLIP Vision Loader, IPAdapter Model Loader) whose outputs are fed into the new `InstantCharacterLoader`.

### 2.4. `InstantCharacterGenerate` (in `nodes/comfy_nodes.py`)

This node's class structure and method signatures remain largely **UNCHANGED**. It will continue to accept an `INSTANTCHAR_PIPE` object, which will now be an instance of the refactored `InstantCharacterFluxPipeline`.

```python
# File: nodes/comfy_nodes.py (relevant part)
# class InstantCharacterGenerate:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "pipe": ("INSTANTCHAR_PIPE",),
#                 "prompt": ("STRING", {"multiline": True}),
#                 # ... other existing inputs (height, width, guidance_scale, etc.)
#                 "subject_scale": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.1}),
#             },
#             "optional": {
#                 "subject_image": ("IMAGE",),
#             }
#         }
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "generate"
#     CATEGORY = "InstantCharacter"

#     def generate(self, pipe: InstantCharacterFluxPipeline, prompt, height, width, guidance_scale,
#                  num_inference_steps, seed, subject_scale, subject_image=None):
#         # ... (PIL conversion for subject_image)
#         # output = pipe(...) # Call the pipeline instance
#         # ... (PIL conversion for output image)
#         # return (final_image_tensor,)
```
The implementation of `generate` will call the `__call__` method of the refactored `pipe` object.

## 3. Data Flow

### 3.1. External Loaders to `InstantCharacterLoader`

The data flows from various standard ComfyUI loader nodes into the new `InstantCharacterLoader`:

```
+-----------------------+      +---------------------------+      +-------------------------+
| GGUF Unet Loader      |----->| flux_unet_model (MODEL)   |----->|                         |
| (or other UNet Loader)|      +---------------------------+      |                         |
+-----------------------+                                        |                         |
                                                                 |                         |
+-----------------------+      +-------------------------------+  |                         |
| CLIP Vision Loader    |----->| siglip_vision_model (CLIP_VISION)|----->| InstantCharacterLoader  |
| (for SigLIP)          |      +-------------------------------+  |                         |
+-----------------------+                                        |                         |
                                                                 |                         |
+-----------------------+      +-------------------------------+  |                         |
| CLIP Vision Loader    |----->| dinov2_vision_model (CLIP_VISION)|----->|                         |
| (for DINOv2)          |      +-------------------------------+  |                         |
+-----------------------+                                        |                         |
                                                                 |                         |
+-----------------------+      +--------------------------------+|  +-------------------------+
| IPAdapter Model Loader|----->| ipadapter_model_data (IPADAPTER)|----->|      (Output: INSTANTCHAR_PIPE)
+-----------------------+      +---------------------------------+      +-------------------------+
```

### 3.2. `InstantCharacterLoader` to `InstantCharacterFluxPipeline`

The `InstantCharacterLoader` node takes the model objects and instantiates the `InstantCharacterFluxPipeline`:

```
InstantCharacterLoader.load_pipe_from_models(
    flux_unet_model,      // MODEL object
    siglip_vision_model,  // CLIP_VISION object
    dinov2_vision_model,  // CLIP_VISION object
    ipadapter_model_data  // IPADAPTER object (dictionary of weights)
)
  |
  | Instantiates
  V
InstantCharacterFluxPipeline.__init__(
    flux_unet_model_object=flux_unet_model,
    siglip_vision_model_object=siglip_vision_model,
    dinov2_vision_model_object=dinov2_vision_model,
    ipadapter_model_data_dict=ipadapter_model_data
)
  |
  | Returns
  V
INSTANTCHAR_PIPE (instance of InstantCharacterFluxPipeline)
```

This `INSTANTCHAR_PIPE` is then consumed by the `InstantCharacterGenerate` node.

## 4. Alignment with ComfyUI Modularity

These changes significantly improve alignment with ComfyUI's modular design principles:

1.  **Separation of Concerns:** Model loading is now delegated to dedicated loader nodes (GGUF Unet Loader, CLIP Vision Loader, IPAdapter Model Loader). The `InstantCharacterLoader` focuses solely on assembling these pre-loaded components into the specialized `InstantCharacterFluxPipeline`. The pipeline itself focuses on the generation logic.
2.  **Reusability and Flexibility:** Users can leverage any compatible ComfyUI loader node for each component. For example, if a new type of UNet loader becomes available for FLUX models, it can be easily swapped in without modifying the `InstantCharacter` nodes.
3.  **Standardized Inputs/Outputs:** The `InstantCharacterLoader` consumes standard ComfyUI types (`MODEL`, `CLIP_VISION`, `IPADAPTER`) and produces a custom pipe object (`INSTANTCHAR_PIPE`). This makes it integrate smoothly into ComfyUI workflows.
4.  **User Experience:** Users familiar with ComfyUI's loader patterns will find the new workflow intuitive. They can manage their models using their preferred methods and loaders.
5.  **Reduced Maintenance for InstantCharacter:** By relying on external loaders, the `InstantCharacter` custom node no longer needs to manage file paths, downloads, or model caching logic for these core components, simplifying its maintenance.

## 5. Node Mappings Update

The `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in `nodes/comfy_nodes.py` will be updated:

```python
NODE_CLASS_MAPPINGS = {
    "InstantCharacterLoader": InstantCharacterLoader,
    "InstantCharacterGenerate": InstantCharacterGenerate,
    # "InstantCharacterLoadModel": REMOVED
    # "InstantCharacterLoadModelFromLocal": Decide if this is also removed or aliased to InstantCharacterLoader if it was the precursor.
    # Based on docs/01_new_inputs.md, InstantCharacterLoadModelFromLocal is the one being refactored/replaced.
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantCharacterLoader": "Load InstantCharacter Pipeline", # Or "InstantCharacter Loader"
    "InstantCharacterGenerate": "Generate with InstantCharacter",
}
```

## 6. Key Implementation Notes & TODOs

-   **`InstantCharacterFluxPipeline.__init__`:**
    -   Carefully verify the structure of the `flux_unet_model_object` (ComfyUI `MODEL` type) to correctly access the UNet, VAE, text encoders, and scheduler. If these are not bundled, the `__init__` signature and `InstantCharacterLoader` inputs must be adjusted to accept them as separate ComfyUI objects (e.g., `VAE`, `CLIP`).
    -   Ensure the hidden sizes and dimensions used in `_initialize_ip_adapter_components` (e.g., `flux_transformer_hidden_size`, `flux_text_encoder_dim`) are accurate for the FLUX.1-dev model and the IPAdapter weights. These might need to be configurable or inferred from the loaded models if possible.
    -   The `output_dim` for `CrossLayerCrossScaleProjector` must match the `ip_hidden_states_dim` expected by `FluxIPAttnProcessor`.
-   **Image Preprocessing in `InstantCharacterFluxPipeline`:**
    -   The `_comfy_clip_vision_preprocess_pil` helper method needs a concrete implementation. It should leverage the preprocessing capabilities or parameters (target size, mean, std) exposed by the input `siglip_vision_model_object` and `dinov2_vision_model_object` (ComfyUI `CLIP_VISION` objects). This is crucial for correct image encoding.
    -   Verify hidden state indices for SigLIP and DINOv2 in `encode_siglip_image_emb` and `encode_dinov2_image_emb`.
-   **`InstantCharacterFluxPipeline.__call__`:**
    -   The full generation logic needs to be implemented, mirroring the original FLUX pipeline's behavior but using the pre-initialized components. This includes prompt encoding, latent preparation, the denoising loop with IP-Adapter injection, and image decoding.
-   **`InstantCharacterLoader.load_pipe_from_models`:**
    -   Add robust error checking for input types and the structure of `ipadapter_model_data`.
    -   Determine the best way to handle `dtype` and `device` for the pipeline, likely inferring from input models or using `comfy.model_management`.
-   **Device Handling:** Ensure all PyTorch modules created within `InstantCharacterFluxPipeline` (e.g., `FluxIPAttnProcessor`, `CrossLayerCrossScaleProjector`) are moved to the correct device, typically the device of the main `flux_unet_model`. ComfyUI's model management should handle the device for the input model objects.
-   **CPU Offload:** The `cpu_offload` flag's implementation needs consideration. If the pipeline is structured like a Diffusers pipeline, `enable_sequential_cpu_offload()` might be applicable. Otherwise, rely on ComfyUI's management of individual components.

This architectural design provides a blueprint for the refactoring work, emphasizing modularity and integration with the ComfyUI ecosystem.