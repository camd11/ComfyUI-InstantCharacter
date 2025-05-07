# Copyright 2025 Tencent InstantX Team. All rights reserved.
#

from PIL import Image
from einops import rearrange
import torch
import torch.nn as nn # Added
import torchvision.transforms.functional as TF # Added for preprocessing
import numpy as np # Added for __call__
from typing import Union, List, Optional, Dict, Any, Callable # Added for __call__
import inspect # Added for detailed debugging

# Removed: from diffusers.pipelines.flux.pipeline_flux import *
# Removed: from transformers import SiglipVisionModel, SiglipImageProcessor, AutoModel, AutoImageProcessor
# Import for FluxPipelineOutput if used, or define a similar structure
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput # Keep if __call__ returns this
from diffusers.utils import replace_example_docstring # Keep if EXAMPLE_DOC_STRING is used
from diffusers.utils.torch_utils import randn_tensor # For latent generation if not from model object

from .models.attn_processor import FluxIPAttnProcessor
from .models.resampler import CrossLayerCrossScaleProjector
from .models.utils import flux_load_lora # Keep for with_style_lora, may need adaptation


# TODO: This EXAMPLE_DOC_STRING might need to be adapted if FluxPipeline is not the base
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> # pipe = InstantCharacterFluxPipeline(...) # Initialization will change
        >>> # pipe.to("cuda")
        >>> # prompt = "A cat holding a sign that says hello world"
        >>> # image = pipe(prompt, subject_image=...).images[0] # Call will change
        >>> # image.save("flux_instant_character.png")
        ```
"""

# Helper functions that might be needed if not inheriting from FluxPipeline
# These are simplified placeholders and would need robust implementation
def calculate_shift(image_seq_len, base_image_seq_len, max_image_seq_len, base_shift, max_shift):
    # Placeholder for shift calculation logic from FLUX
    return base_shift + (max_shift - base_shift) * (image_seq_len - base_image_seq_len) / (
        max_image_seq_len - base_image_seq_len
    )

def retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=None, mu=0.0):
    # Placeholder for timestep retrieval
    if hasattr(scheduler, "set_timesteps_mu"):
        scheduler.set_timesteps_mu(num_inference_steps, sigmas=sigmas, mu=mu, device=device)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    return timesteps, len(timesteps)


class InstantCharacterFluxPipeline(nn.Module): # Changed base class
    def __init__(self,
                 flux_unet_model_object,      # ComfyUI MODEL object
                 vae_module,                  # VAE nn.Module
                 siglip_vision_model_object,  # ComfyUI CLIP_VISION object for SigLIP
                 dinov2_vision_model_object,  # ComfyUI CLIP_VISION object for DINOv2
                 ipadapter_model_data_dict,   # Dict of IPAdapter weights
                 dtype=torch.bfloat16):
        """
        Initializes the InstantCharacterFluxPipeline with pre-loaded model components.
        """
        super().__init__()

        self.dtype = dtype
        self.device = flux_unet_model_object.model.device # Assuming model object has a device attribute or a model attribute with device

        # Assign FLUX components from flux_unet_model_object
        # These attributes are based on typical ComfyUI MODEL object structure and FLUX needs
        self.transformer = flux_unet_model_object.model
        if not hasattr(self.transformer, 'attn_processors'):
            self.transformer.attn_processors = {}
        self.vae = vae_module
        
        # Text encoders, tokenizers, and scheduler are expected to be part of flux_unet_model_object
        # or need to be passed separately if not.
        # For FLUX, there are typically two text encoders.
        # Ensure attribute names are text_encoder_1, tokenizer_1, text_encoder_2, tokenizer_2
        if hasattr(flux_unet_model_object, 'text_encoder_1') and hasattr(flux_unet_model_object, 'tokenizer_1'):
            self.text_encoder_1 = flux_unet_model_object.text_encoder_1
            self.tokenizer_1 = flux_unet_model_object.tokenizer_1
            print("InstantCharacterPipeline: Assigned self.text_encoder_1 and self.tokenizer_1.")
        else:
            # Fallback or error if not found, as these are crucial for prompt encoding
            print("Warning: Text Encoder One (text_encoder_1) / Tokenizer One (tokenizer_1) not found directly on flux_unet_model_object.")
            self.text_encoder_1 = None # Placeholder, will cause issues if not properly set
            self.tokenizer_1 = None  # Placeholder

        if hasattr(flux_unet_model_object, 'text_encoder_2') and hasattr(flux_unet_model_object, 'tokenizer_2'):
            self.text_encoder_2 = flux_unet_model_object.text_encoder_2
            self.tokenizer_2 = flux_unet_model_object.tokenizer_2
            print("InstantCharacterPipeline: Assigned self.text_encoder_2 and self.tokenizer_2.")
        else:
            print("Warning: Text Encoder Two (text_encoder_2) / Tokenizer Two (tokenizer_2) not found directly on flux_unet_model_object.")
            self.text_encoder_2 = None # Placeholder
            self.tokenizer_2 = None    # Placeholder

        if hasattr(flux_unet_model_object, 'scheduler'):
            self.scheduler = flux_unet_model_object.scheduler
        else:
            # Fallback: try to get config and initialize a scheduler
            # from diffusers.schedulers import FluxScheduler # Example
            # self.scheduler = FluxScheduler.from_config(flux_unet_model_object.model_config.scheduler)
            print("Warning: Scheduler not found directly on flux_unet_model_object. Needs manual setup or to be included in MODEL.")
            self.scheduler = None # Placeholder

        # Assign Image Encoders from CLIP_VISION objects
        # Assign Image Encoders from CLIP_VISION objects
        self.siglip_image_processor = siglip_vision_model_object # This is the wrapper, passed to _comfy_clip_vision_preprocess_pil
        if siglip_vision_model_object is not None and hasattr(siglip_vision_model_object, 'model'):
            self.siglip_image_encoder_model = siglip_vision_model_object.model # This is the actual nn.Module for encoding
            print("InstantCharacterPipeline: Assigned self.siglip_image_encoder_model from siglip_vision_model_object.model.")
        elif siglip_vision_model_object is not None:
            # Fallback if .model is not present, but the object itself might be the encoder.
            # This is less likely for ComfyUI CLIP_VISION wrappers which usually encapsulate a model.
            print("Warning: siglip_vision_model_object.model not found. Attempting to use siglip_vision_model_object directly as siglip_image_encoder_model.")
            self.siglip_image_encoder_model = siglip_vision_model_object
        else:
            self.siglip_image_encoder_model = None
            print("Warning: siglip_vision_model_object is None. self.siglip_image_encoder_model set to None.")

        self.dinov2_image_processor = dinov2_vision_model_object # This is the wrapper
        if dinov2_vision_model_object is not None and hasattr(dinov2_vision_model_object, 'model'):
            self.dinov2_image_encoder_model = dinov2_vision_model_object.model # Actual nn.Module
            print("InstantCharacterPipeline: Assigned self.dinov2_image_encoder_model from dinov2_vision_model_object.model.")
        elif dinov2_vision_model_object is not None:
            print("Warning: dinov2_vision_model_object.model not found. Attempting to use dinov2_vision_model_object directly as dinov2_image_encoder_model.")
            self.dinov2_image_encoder_model = dinov2_vision_model_object
        else:
            self.dinov2_image_encoder_model = None
            print("Warning: dinov2_vision_model_object is None. self.dinov2_image_encoder_model set to None.")

        self._initialize_ip_adapter_components(ipadapter_model_data_dict, self.dtype)

        # For progress bar in __call__ if not inheriting from DiffusionPipeline
        self.progress_bar = lambda x: x # Simple placeholder

        # Attributes that were previously inherited from FluxPipeline/DiffusionPipeline
        # These might need to be set if __call__ logic relies on them.
        # self.vae_scale_factor = getattr(self.vae.config, "scale_factor", 0.13025) # Example, verify FLUX VAE scale
        # self.default_sample_size = getattr(self.transformer.config, "sample_size", 128) # Example for FLUX unet sample size (latent)
        # self._execution_device = self.device # Already set as self.device

    def _initialize_ip_adapter_components(self, ipadapter_state_dict, dtype):
        """
        Initializes IPAdapter attention processors and image projection model.
        """
        device = self.device # Use device from __init__

        # Initialize and load FluxIPAttnProcessor
        attn_procs = {}
        # Dimensions need to be accurate for FLUX.1-dev or configurable.
        # Try to get from transformer/text_encoder config if available
        try:
            flux_transformer_hidden_size = self.transformer.config.hidden_size # Or equivalent for FLUX UNet
            # flux_transformer_hidden_size = self.transformer.config.attention_head_dim * self.transformer.config.num_attention_heads
        except AttributeError:
            print("Warning: Could not infer flux_transformer_hidden_size from transformer.config. Using default 4096.")
            flux_transformer_hidden_size = 4096 # Default from design doc

        try:
            # FLUX often uses text_encoder_2 for IP-Adapter conditioning
            flux_text_encoder_dim = self.text_encoder_2.config.hidden_size # Or d_model
        except AttributeError:
            print("Warning: Could not infer flux_text_encoder_dim from text_encoder_2.config. Using default 4096.")
            flux_text_encoder_dim = 4096    # Default from design doc

        for name in self.transformer.attn_processors.keys():
            attn_procs[name] = FluxIPAttnProcessor(
                hidden_size=flux_transformer_hidden_size,
                ip_hidden_states_dim=flux_text_encoder_dim,
            ).to(device, dtype=dtype)
        self.transformer.attn_processors = attn_procs
        
        tmp_ip_layers = torch.nn.ModuleList(list(self.transformer.attn_processors.values())) # Ensure it's a list for ModuleList
        if "ip_adapter" in ipadapter_state_dict:
            tmp_ip_layers.load_state_dict(ipadapter_state_dict["ip_adapter"], strict=False)
            print("=> IP Adapter Attention Processor weights loaded.")
        else:
            print("Warning: 'ip_adapter' key not found in ipadapter_state_dict. Attn processor weights not loaded.")


        # Initialize and load CrossLayerCrossScaleProjector (image_proj_model)
        self.subject_image_proj_model = CrossLayerCrossScaleProjector(
            inner_dim=1152 + 1536, num_attention_heads=42, attention_head_dim=64,
            cross_attention_dim=1152 + 1536, num_layers=4, dim=1280, depth=4,
            dim_head=64, heads=20, num_queries=1024, # nb_token
            embedding_dim=1152 + 1536, output_dim=flux_transformer_hidden_size, # output_dim should match transformer's input for IP
            ff_mult=4,
            timestep_in_dim=320, timestep_flip_sin_to_cos=True, timestep_freq_shift=0,
        ).to(device, dtype=dtype)
        self.subject_image_proj_model.eval()
        if "image_proj" in ipadapter_state_dict:
            self.subject_image_proj_model.load_state_dict(ipadapter_state_dict["image_proj"], strict=False)
            print("=> IP Adapter Image Projection weights loaded.")
        else:
            print("Warning: 'image_proj' key not found in ipadapter_state_dict. Image projection weights not loaded.")

    def _comfy_clip_vision_preprocess_pil(self, image_processor_obj, pil_image: Image.Image):
        """
        Manually preprocesses a single PIL image using parameters from a ComfyUI CLIP_VISION object.
        """
        if image_processor_obj is None:
            raise ValueError("image_processor_obj cannot be None for preprocessing.")
        if pil_image is None:
            raise ValueError("pil_image cannot be None for preprocessing.")

        try:
            image_size = image_processor_obj.image_size
            image_mean = image_processor_obj.image_mean
            image_std = image_processor_obj.image_std
        except AttributeError as e:
            raise AttributeError(
                f"Missing one or more required attributes (image_size, image_mean, image_std) "
                f"on image_processor_obj (type: {type(image_processor_obj)}). Error: {e}"
            )

        # 1. Convert to RGB
        image = pil_image.convert("RGB")

        # 2. Resize and Crop
        if isinstance(image_size, int):
            target_size = image_size
            # Resize shorter side to target_size
            w, h = image.size
            if w < h:
                new_w = target_size
                new_h = int(h * target_size / w)
            else:
                new_h = target_size
                new_w = int(w * target_size / h)
            image = image.resize((new_w, new_h), Image.BICUBIC)

            # Center crop to (target_size, target_size)
            left = (new_w - target_size) / 2
            top = (new_h - target_size) / 2
            right = (new_w + target_size) / 2
            bottom = (new_h + target_size) / 2
            image = image.crop((left, top, right, bottom))
            target_height, target_width = target_size, target_size
        elif isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            target_height, target_width = image_size
            # PIL resize uses (width, height)
            image = image.resize((target_width, target_height), Image.BICUBIC)
        else:
            raise ValueError(
                f"image_size must be an int or a tuple/list of two ints. Got {image_size}"
            )

        # 3. Convert to Tensor and scale to [0, 1]
        # Using TF.to_tensor which handles PIL to (C, H, W) tensor and scales to [0,1]
        image_tensor = TF.to_tensor(image) # Shape: (C, H, W)

        # 4. Normalize
        if not isinstance(image_mean, (list, tuple)) or not isinstance(image_std, (list, tuple)):
            raise ValueError("image_mean and image_std must be lists or tuples.")
        if len(image_mean) != image_tensor.shape[0] or len(image_std) != image_tensor.shape[0]:
             raise ValueError(f"image_mean/std length ({len(image_mean)}) must match image channels ({image_tensor.shape[0]})")

        # Ensure mean and std are tensors with shape (C, 1, 1) for TF.normalize
        # TF.normalize expects list/tuple for mean/std, it handles conversion internally.
        normalized_tensor = TF.normalize(image_tensor, mean=image_mean, std=image_std)

        # 5. Add Batch Dimension
        pixel_values = normalized_tensor.unsqueeze(0) # Shape: (1, C, H, W)

        return pixel_values.to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def encode_siglip_image_emb(self, siglip_pixel_values, device, dtype):
        # --- BEGIN DEBUG PRINTS ---
        print(f"[encode_siglip_image_emb DEBUG] Type of self.siglip_image_encoder_model: {type(self.siglip_image_encoder_model)}")
        # --- END DEBUG PRINTS ---
        
        # Determine device and dtype from the model if possible, otherwise use input
        try:
            model_device = next(self.siglip_image_encoder_model.parameters()).device
            model_dtype = next(self.siglip_image_encoder_model.parameters()).dtype
        except StopIteration:
            print("[encode_siglip_image_emb WARNING] self.siglip_image_encoder_model has no parameters. Using input device/dtype.")
            model_device = device
            model_dtype = dtype
        except AttributeError: # Should not happen if model is nn.Module
            print("[encode_siglip_image_emb WARNING] Could not get parameters from self.siglip_image_encoder_model. Using input device/dtype.")
            model_device = device
            model_dtype = dtype

        self.siglip_image_encoder_model.to(device=model_device, dtype=model_dtype) # Ensure model is on correct device/dtype

        required_indices = [7, 13, 26]
        intermediate_outputs = []
        last_hidden_state = None
        pooled_output = None

        # Ensure model is on the correct device/dtype (already done before this block)
        # self.siglip_image_encoder_model.to(device=model_device, dtype=model_dtype)

        for index in required_indices:
            print(f"[encode_siglip_image_emb DEBUG] Calling siglip_image_encoder_model for intermediate_output={index}...")
            # The siglip_image_encoder_model is likely CLIPVisionModelProjection from ComfyUI
            # Its forward method, when intermediate_output is specified, typically returns a tuple:
            # (last_hidden_state_from_encoder, requested_intermediate_layer, projected_pooled_output, potentially_multimodal_projector_output)
            # We are interested in the first three elements primarily.
            res_tuple = self.siglip_image_encoder_model(
                pixel_values=siglip_pixel_values.to(device=model_device, dtype=model_dtype),
                intermediate_output=index
            )
            print(f"[encode_siglip_image_emb DEBUG] Call for index {index} returned tuple of type: {type(res_tuple)}, length: {len(res_tuple) if isinstance(res_tuple, tuple) else 'N/A'}")

            if not isinstance(res_tuple, tuple) or len(res_tuple) < 2: # Need at least last_hidden_state and intermediate_output
                raise ValueError(
                    f"Unexpected output tuple structure for intermediate_output={index}. "
                    f"Expected at least 2 elements, got {len(res_tuple) if isinstance(res_tuple, tuple) else type(res_tuple)}"
                )

            # Intermediate output is typically the second element
            intermediate_outputs.append(res_tuple[1])

            # Capture final outputs from the last iteration (or any, they should be consistent from the encoder itself)
            # The projected_pooled_output might change if the projection depends on the intermediate layer,
            # but last_hidden_state from the vision_model part should be the same.
            # Let's take them from the last specified index call for consistency.
            if index == required_indices[-1]:
                last_hidden_state = res_tuple[0]
                if len(res_tuple) >= 3:
                    pooled_output = res_tuple[2] # Projected pooled output
                else:
                    # This case might occur if the tuple structure is shorter than expected (e.g. no multimodal projector output)
                    # or if the pooled output is not the third element.
                    # For safety, one might make a separate call without intermediate_output if pooled_output is critical and missing.
                    # However, ComfyUI's CLIPVisionModelProjection usually provides it.
                    print(f"[encode_siglip_image_emb WARNING] Pooled output (res_tuple[2]) not found for index {index}. Length: {len(res_tuple)}. Setting to None.")
                    pooled_output = None


        if not intermediate_outputs or len(intermediate_outputs) != len(required_indices):
            raise RuntimeError(f"Failed to retrieve all required intermediate hidden states. Expected {len(required_indices)}, got {len(intermediate_outputs)}.")
        if last_hidden_state is None:
            # This could happen if required_indices is empty or loop logic fails
            raise RuntimeError("Failed to retrieve last_hidden_state from SigLIP model.")
        # pooled_output can be None if not found, handle accordingly in consuming code if critical

        # Combine shallow embeddings
        # Assuming intermediate outputs are [Batch, SeqLen, Dim_layer], concatenate along feature dim
        siglip_image_shallow_embeds = torch.cat(intermediate_outputs, dim=-1)

        siglip_image_embeds = last_hidden_state
        # pooled_output is already set

        if siglip_image_embeds is None: # Redundant check if previous one passes, but good for safety
            raise ValueError("SigLIP last_hidden_state (siglip_image_embeds) is None after processing.")
        if siglip_image_shallow_embeds is None:
             raise ValueError("Combined siglip_image_shallow_embeds is None after processing.")
        # siglip_pooled_output can be None, so no strict check here unless always required

        return siglip_image_embeds, pooled_output, siglip_image_shallow_embeds

    @torch.inference_mode()
    def encode_dinov2_image_emb(self, dinov2_pixel_values, device, dtype):
        # dinov2_pixel_values are preprocessed tensors.
        
        # Determine device and dtype from the model if possible, otherwise use input
        try:
            model_device = next(self.dinov2_image_encoder_model.parameters()).device
            model_dtype = next(self.dinov2_image_encoder_model.parameters()).dtype
        except StopIteration:
            print("[encode_dinov2_image_emb WARNING] self.dinov2_image_encoder_model has no parameters. Using input device/dtype.")
            model_device = device
            model_dtype = dtype
        except AttributeError:
            print("[encode_dinov2_image_emb WARNING] Could not get parameters from self.dinov2_image_encoder_model. Using input device/dtype.")
            model_device = device
            model_dtype = dtype

        self.dinov2_image_encoder_model.to(device=model_device, dtype=model_dtype)
        
        required_indices = [9, 19, 29] # Specific layers for DINOv2
        dinov2_intermediate_layers = []
        dinov2_image_embeds = None # Will be set from the first call

        for i, layer_idx in enumerate(required_indices):
            print(f"[encode_dinov2_image_emb DEBUG] Calling self.dinov2_image_encoder_model ({type(self.dinov2_image_encoder_model)}) with intermediate_output={layer_idx}...")
            comfy_output = self.dinov2_image_encoder_model(
                pixel_values=dinov2_pixel_values.to(device=model_device, dtype=model_dtype),
                intermediate_output=layer_idx
            )

            if not isinstance(comfy_output, tuple) or len(comfy_output) < 2: # Need at least last_hidden_state and intermediate_output
                raise TypeError(f"Unexpected output type from dinov2_image_encoder_model for layer {layer_idx}: {type(comfy_output)}. Expected a tuple of at least 2 elements.")

            if i == 0: # Get last_hidden_state from the first call
                dinov2_image_embeds = comfy_output[0]
                if dinov2_image_embeds is None:
                    raise ValueError(f"Failed to retrieve last_hidden_state from DINOv2 model output for layer {layer_idx}.")
                # DINOv2 typically excludes CLS token for image features
                if dinov2_image_embeds.shape[1] > 1: # Check if there's more than one token (CLS + patches)
                    dinov2_image_embeds = dinov2_image_embeds[:, 1:]


            intermediate_layer = comfy_output[1]
            if intermediate_layer is None:
                raise ValueError(f"Failed to retrieve intermediate layer {layer_idx} from DINOv2 model output.")
            # DINOv2 typically excludes CLS token for image features
            if intermediate_layer.shape[1] > 1: # Check if there's more than one token
                 intermediate_layer = intermediate_layer[:, 1:]
            dinov2_intermediate_layers.append(intermediate_layer)

        if not dinov2_intermediate_layers:
            raise ValueError("No intermediate layers were extracted for DINOv2.")
            
        dinov2_image_shallow_embeds = torch.cat(dinov2_intermediate_layers, dim=1)
        
        if dinov2_image_embeds is None:
             raise ValueError("dinov2_image_embeds (last_hidden_state) was not properly set.")

        return dinov2_image_embeds, dinov2_image_shallow_embeds

    @torch.inference_mode()
    def encode_image_emb(self, subject_image_pil: Image.Image, device, dtype):
        # Input handling for subject_image_pil
        if isinstance(subject_image_pil, torch.Tensor):
            # Assuming ComfyUI IMAGE tensor [B, H, W, C], range [0, 1]
            # Select the first image in the batch
            if subject_image_pil.ndim == 4 and subject_image_pil.shape[0] > 0:
                image_tensor_slice = subject_image_pil[0]
            elif subject_image_pil.ndim == 3: # Handle case if batch dim is missing
                image_tensor_slice = subject_image_pil
            else:
                raise ValueError(f"Unsupported tensor shape for subject_image_pil: {subject_image_pil.shape}")
            # Convert to numpy HxWxC, range [0, 255], uint8
            image_np = (image_tensor_slice.cpu().numpy() * 255).astype(np.uint8)
            # Convert to PIL Image
            pil_image_to_process = Image.fromarray(image_np, 'RGB')
        elif isinstance(subject_image_pil, list):
            # Assuming a list of PIL Images, take the first one
            if len(subject_image_pil) > 0 and isinstance(subject_image_pil[0], Image.Image):
                pil_image_to_process = subject_image_pil[0]
            else:
                # Handle case where list might contain non-PIL Images or be empty.
                raise ValueError("Input subject_image_pil is a list, but does not contain a PIL Image at the first position or is empty.")
        elif isinstance(subject_image_pil, Image.Image):
            # Input is already a PIL Image
            pil_image_to_process = subject_image_pil
        else:
            raise ValueError(f"Unsupported type for subject_image_pil: {type(subject_image_pil)}")

        # Now use pil_image_to_process for subsequent operations
        # Cropping and resizing logic for low_res and high_res PIL images
        object_image_pil_low_res = pil_image_to_process.resize((384, 384)) # Ensure this is a single PIL image
        object_image_pil_high_res_orig = pil_image_to_process.resize((768, 768))
        object_image_pil_high_res_crops = [
            object_image_pil_high_res_orig.crop((0, 0, 384, 384)),
            object_image_pil_high_res_orig.crop((384, 0, 768, 384)),
            object_image_pil_high_res_orig.crop((0, 384, 384, 768)),
            object_image_pil_high_res_orig.crop((384, 384, 768, 768)),
        ]
        nb_split_image = len(object_image_pil_high_res_crops)

        # Preprocess PIL images using the ComfyUI CLIP_VISION objects
        siglip_low_res_pixels = self._comfy_clip_vision_preprocess_pil(self.siglip_image_processor, object_image_pil_low_res)
        dinov2_low_res_pixels = self._comfy_clip_vision_preprocess_pil(self.dinov2_image_processor, object_image_pil_low_res)

        siglip_embeds_tuple = self.encode_siglip_image_emb(siglip_low_res_pixels, device, dtype)
        dinov2_embeds_tuple = self.encode_dinov2_image_emb(dinov2_low_res_pixels, device, dtype)

        image_embeds_low_res_deep = torch.cat([siglip_embeds_tuple[0], dinov2_embeds_tuple[0]], dim=2)
        image_embeds_low_res_shallow = torch.cat([siglip_embeds_tuple[1], dinov2_embeds_tuple[1]], dim=2)

        # High-resolution processing
        siglip_high_res_pixels = self._comfy_clip_vision_preprocess_pil(self.siglip_image_processor, object_image_pil_high_res_crops)
        # siglip_high_res_pixels = siglip_high_res_pixels[None] # Assuming preprocess handles batching correctly
        siglip_high_res_pixels = rearrange(siglip_high_res_pixels, '(b n) c h w -> (b n) c h w', b=1, n=nb_split_image) # No-op if already batched by preprocess
        
        siglip_high_res_embeds_tuple = self.encode_siglip_image_emb(siglip_high_res_pixels, device, dtype)
        siglip_image_high_res_deep = rearrange(siglip_high_res_embeds_tuple[0], '(b n) l c -> b (n l) c', n=nb_split_image)

        dinov2_high_res_pixels = self._comfy_clip_vision_preprocess_pil(self.dinov2_image_processor, object_image_pil_high_res_crops)
        # dinov2_high_res_pixels = dinov2_high_res_pixels[None]
        dinov2_high_res_pixels = rearrange(dinov2_high_res_pixels, '(b n) c h w -> (b n) c h w', b=1, n=nb_split_image)

        dinov2_high_res_embeds_tuple = self.encode_dinov2_image_emb(dinov2_high_res_pixels, device, dtype)
        dinov2_image_high_res_deep = rearrange(dinov2_high_res_embeds_tuple[0], '(b n) l c -> b (n l) c', n=nb_split_image)
        
        image_embeds_high_res_deep = torch.cat([siglip_image_high_res_deep, dinov2_image_high_res_deep], dim=2)

        image_embeds_dict = dict(
            image_embeds_low_res_shallow=image_embeds_low_res_shallow.to(device=device, dtype=dtype),
            image_embeds_low_res_deep=image_embeds_low_res_deep.to(device=device, dtype=dtype),
            image_embeds_high_res_deep=image_embeds_high_res_deep.to(device=device, dtype=dtype),
        )
        return image_embeds_dict

    # Removed init_ccp_and_attn_processor
    # Removed init_adapter

    # encode_prompt method (simplified, adapted from Diffusers' CLIP and T5 handling)
    # This is a complex part and needs to be accurate for FLUX's dual text encoder setup.
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]],
        device,
        num_images_per_prompt: int,
        max_sequence_length: int, # Max sequence length for T5/common padding
    ):
        missing_components = []
        if not self.tokenizer_1:
            missing_components.append("Tokenizer One (self.tokenizer_1)")
        if not self.text_encoder_1:
            missing_components.append("Text Encoder One (self.text_encoder_1)")
        if not self.tokenizer_2:
            missing_components.append("Tokenizer Two (self.tokenizer_2)")
        if not self.text_encoder_2:
            missing_components.append("Text Encoder Two (self.text_encoder_2)")

        if missing_components:
            raise ValueError(
                f"The following MANDATORY text processing components are missing or not initialized: {', '.join(missing_components)}. "
                "Please ensure a valid and fully loaded composite FLUX CLIP model (providing both clip_l and t5xxl components with their respective tokenizers and text encoders) "
                "is connected to the flux_text_encoder_one input of the InstantCharacterLoader node."
            )

        # Batch handling for prompts
        if isinstance(prompt, str):
            batch_size = 1
            prompts = [prompt]
            if prompt_2 is None:
                prompts_2 = [prompt] # Use main prompt if prompt_2 is None
            elif isinstance(prompt_2, str):
                prompts_2 = [prompt_2]
            else: # prompt_2 is a list
                 raise ValueError("Prompt is a string, but prompt_2 is a list. Mismatched batching.")
        elif isinstance(prompt, list):
            batch_size = len(prompt)
            prompts = prompt
            if prompt_2 is None:
                prompts_2 = prompts
            elif isinstance(prompt_2, list) and len(prompt_2) == batch_size:
                prompts_2 = prompt_2
            else:
                raise ValueError("If prompt is a list, prompt_2 must be None or a list of the same length.")
        else:
            raise TypeError("Prompt must be a string or a list of strings.")

        # Initialize lists to store results for each batch item
        batch_prompt_embeds_1_list = []
        batch_pooled_embeds_1_list = []
        batch_text_ids_1_padded_list = []

        batch_prompt_embeds_2_list = []
        batch_pooled_embeds_2_list = []
        batch_text_ids_2_padded_list = []

        # Define FLUX standard lengths and pad token IDs based on research
        clip_l_max_len = 77
        clip_l_pad_token_id = 49407  # EOS token ID for CLIP-L
        
        t5xxl_max_len = 256  # Effective max sequence length for T5XXL in FLUX
        t5xxl_pad_token_id = 0 # Pad token ID for T5XXL

        for i in range(batch_size):
            current_prompt_1 = prompts[i]
            current_prompt_2 = prompts_2[i]

            # --- Encoder 1 (e.g., CLIP L) ---
            token_data_one = self.tokenizer_1.tokenize_with_weights(current_prompt_1)
            
            encoder_output_1 = self.text_encoder_1.encode_token_weights(token_data_one)
            prompt_embeds_1_single = None
            pooled_embeds_1_single = None
            if isinstance(encoder_output_1, tuple) and len(encoder_output_1) == 2:
                prompt_embeds_1_single, pooled_embeds_1_single = encoder_output_1
            elif torch.is_tensor(encoder_output_1):
                prompt_embeds_1_single = encoder_output_1
            else:
                raise ValueError("Text Encoder 1 returned unexpected output type.")
            
            if prompt_embeds_1_single is None:
                 raise ValueError(f"Text Encoder 1 failed to produce embeddings for prompt: {current_prompt_1}")

            batch_prompt_embeds_1_list.append(prompt_embeds_1_single)
            if pooled_embeds_1_single is not None:
                batch_pooled_embeds_1_list.append(pooled_embeds_1_single)

            # Padded text_ids for UNet from tokenizer_1 (derived from token_data_one)
            raw_text_ids_one_single = []
            if token_data_one: # token_data_one is List[List[Tuple[int, float]]]
                for chunk in token_data_one:
                    raw_text_ids_one_single.extend([item[0] for item in chunk]) # item is (token_id, weight)
            
            # Use fixed values for CLIP-L
            # max_len_1 = clip_l_max_len (implicitly used below)
            # pad_id_1 = clip_l_pad_token_id (implicitly used below)

            # Truncate if necessary
            truncated_ids_one = raw_text_ids_one_single[:clip_l_max_len]
            # Pad if necessary
            padded_ids_1 = truncated_ids_one + [clip_l_pad_token_id] * (clip_l_max_len - len(truncated_ids_one))
            
            text_ids_1_single_padded = torch.tensor([padded_ids_1], dtype=torch.long, device=device)
            batch_text_ids_1_padded_list.append(text_ids_1_single_padded)

            # Truncation warning for tokenizer_1
            if len(raw_text_ids_one_single) > clip_l_max_len:
                try:
                    removed_tokens_one = raw_text_ids_one_single[clip_l_max_len:]
                    if hasattr(self.tokenizer_1, 'decode') and callable(self.tokenizer_1.decode):
                        removed_text_one = self.tokenizer_1.decode(removed_tokens_one)
                        print(f"Warning (Tokenizer 1): Part of your input was truncated: \"{removed_text_one}\"")
                    else:
                        print(f"Warning (Tokenizer 1): Input was truncated. Decoder not available for removed part.")
                except Exception as e:
                    print(f"Warning (Tokenizer 1): Input was truncated, but could not decode removed part. Error: {e}")


            # --- Encoder 2 (e.g., T5XXL) ---
            token_data_two = self.tokenizer_2.tokenize_with_weights(current_prompt_2)

            encoder_output_2 = self.text_encoder_2.encode_token_weights(token_data_two)
            prompt_embeds_2_single = None
            pooled_embeds_2_single = None
            if isinstance(encoder_output_2, tuple) and len(encoder_output_2) == 2:
                prompt_embeds_2_single, pooled_embeds_2_single = encoder_output_2
            elif torch.is_tensor(encoder_output_2):
                prompt_embeds_2_single = encoder_output_2
            else:
                raise ValueError("Text Encoder 2 returned unexpected output type.")

            if prompt_embeds_2_single is None:
                 raise ValueError(f"Text Encoder 2 failed to produce embeddings for prompt: {current_prompt_2}")

            batch_prompt_embeds_2_list.append(prompt_embeds_2_single)
            if pooled_embeds_2_single is not None:
                batch_pooled_embeds_2_list.append(pooled_embeds_2_single)
            else: # Fallback for pooled if T5 didn't provide it (should be rare)
                pooled_dim_2 = 4096 # Standard d_model for T5XXL in FLUX
                batch_pooled_embeds_2_list.append(torch.zeros(1, pooled_dim_2, device=device, dtype=self.dtype))


            # Padded text_ids for UNet from tokenizer_2 (derived from token_data_two)
            raw_text_ids_two_single = []
            if token_data_two: # token_data_two is List[List[Tuple[int, float]]]
                for chunk in token_data_two:
                    raw_text_ids_two_single.extend([item[0] for item in chunk]) # item is (token_id, weight)

            # Use fixed values for T5XXL
            # max_len_2 = t5xxl_max_len (implicitly used below)
            # pad_id_2 = t5xxl_pad_token_id (implicitly used below)

            # Truncate if necessary
            truncated_ids_two = raw_text_ids_two_single[:t5xxl_max_len]
            # Pad if necessary
            padded_ids_2 = truncated_ids_two + [t5xxl_pad_token_id] * (t5xxl_max_len - len(truncated_ids_two))
            
            text_ids_2_single_padded = torch.tensor([padded_ids_2], dtype=torch.long, device=device)
            batch_text_ids_2_padded_list.append(text_ids_2_single_padded)
            
            # Truncation warning for tokenizer_2
            if len(raw_text_ids_two_single) > t5xxl_max_len:
                try:
                    removed_tokens_two = raw_text_ids_two_single[t5xxl_max_len:]
                    if hasattr(self.tokenizer_2, 'decode') and callable(self.tokenizer_2.decode):
                        removed_text_two = self.tokenizer_2.decode(removed_tokens_two)
                        print(f"Warning (Tokenizer 2): Part of your input was truncated: \"{removed_text_two}\"")
                    else:
                        print(f"Warning (Tokenizer 2): Input was truncated. Decoder not available for removed part.")
                except Exception as e:
                    print(f"Warning (Tokenizer 2): Input was truncated, but could not decode removed part. Error: {e}")

        # Consolidate batched results
        prompt_embeds_1 = torch.cat(batch_prompt_embeds_1_list, dim=0)
        text_ids_1_padded = torch.cat(batch_text_ids_1_padded_list, dim=0)
        if batch_pooled_embeds_1_list: # Only if pooled output was consistently available
            pooled_prompt_embeds_1 = torch.cat(batch_pooled_embeds_1_list, dim=0)
        else: # Fallback if text_encoder_1 never gave pooled output
            pooled_dim_1 = 768 # Standard hidden_size/pooled_dimension for CLIP-L
            pooled_prompt_embeds_1 = torch.zeros(batch_size, pooled_dim_1, device=device, dtype=self.dtype)


        prompt_embeds_2 = torch.cat(batch_prompt_embeds_2_list, dim=0)
        text_ids_2_padded = torch.cat(batch_text_ids_2_padded_list, dim=0)
        pooled_prompt_embeds_2 = torch.cat(batch_pooled_embeds_2_list, dim=0)


        # FLUX requires prompt_embeds_1 and prompt_embeds_2 to have the same sequence length
        # for feature-wise concatenation. Pad the shorter one (usually prompt_embeds_1 from CLIP).
        # Target sequence length is t5xxl_max_len (256 for FLUX T5).
        s1_len = prompt_embeds_1.shape[1]
        s2_len = prompt_embeds_2.shape[1]
        
        # Align embeddings to t5xxl_max_len.
        # text_encoder_1 (CLIP) might output shorter sequences (e.g. 77).
        # text_encoder_2 (T5) might output up to t5xxl_max_len.
        
        if s1_len < t5xxl_max_len:
            padding_shape = (prompt_embeds_1.shape[0], t5xxl_max_len - s1_len, prompt_embeds_1.shape[2])
            padding_tensor = torch.zeros(padding_shape, device=prompt_embeds_1.device, dtype=prompt_embeds_1.dtype)
            prompt_embeds_1 = torch.cat([prompt_embeds_1, padding_tensor], dim=1)
        elif s1_len > t5xxl_max_len:
            prompt_embeds_1 = prompt_embeds_1[:, :t5xxl_max_len, :]

        if s2_len < t5xxl_max_len:
            padding_shape = (prompt_embeds_2.shape[0], t5xxl_max_len - s2_len, prompt_embeds_2.shape[2])
            padding_tensor = torch.zeros(padding_shape, device=prompt_embeds_2.device, dtype=prompt_embeds_2.dtype)
            prompt_embeds_2 = torch.cat([prompt_embeds_2, padding_tensor], dim=1)
        elif s2_len > t5xxl_max_len:
            prompt_embeds_2 = prompt_embeds_2[:, :t5xxl_max_len, :]
            
        # Combine embeddings: (B, S, D1+D2)
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        
        # Pooled embeddings: use from encoder 2 if available, else from encoder 1
        pooled_prompt_embeds = pooled_prompt_embeds_2 if pooled_prompt_embeds_2 is not None else pooled_prompt_embeds_1
        if pooled_prompt_embeds is None: # Should not happen with fallbacks
            raise ValueError("Failed to obtain pooled prompt embeddings from either encoder.")

        # Combine text_ids for UNet: (B, S1_padded + S2_padded)
        # text_ids_1_padded is (B, 77), text_ids_2_padded is (B, 512)
        # Resulting text_ids will be (B, 77+512)
        text_ids = torch.cat([text_ids_1_padded, text_ids_2_padded], dim=1) # Concatenate along sequence dim

        # Duplicate for num_images_per_prompt
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        text_ids = text_ids.repeat_interleave(num_images_per_prompt, dim=0)

        return prompt_embeds.to(device=device, dtype=self.dtype), \
               pooled_prompt_embeds.to(device=device, dtype=self.dtype), \
               text_ids.to(device=device) # text_ids are already torch.long

    # prepare_latents method (simplified, from Diffusers)
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            height // getattr(self.vae.config, "downscale_factor", 8), # Use VAE's scale factor
            width // getattr(self.vae.config, "downscale_factor", 8),
        )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # FLUX uses specific latent image IDs
        latent_image_ids = torch.tensor([list(range(shape[2] * shape[3]))], device=device).expand(
            batch_size, shape[2] * shape[3]
        )
        # scale the latents by the scheduler's init_sigma
        # latents = latents * self.scheduler.init_sigma # If scheduler has init_sigma
        return latents, latent_image_ids

    # _unpack_latents (simplified from FLUX)
    def _unpack_latents(self, latents, height, width, vae_scale_factor):
        # This is a placeholder. FLUX has specific logic for unpacking B C H W latents
        # if they are packed differently by the VAE.
        # For a standard VAE, this might not be needed or might be simpler.
        return latents # Assuming standard VAE output that doesn't need complex unpacking

    # image_processor.postprocess (simplified placeholder if not using Diffusers ImageProcessor)
    def _postprocess_image(self, image, output_type="pil"):
        if output_type == "pil":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image * 255).round().astype("uint8")
            if image.shape[0] == 1:
                return [Image.fromarray(image[0])]
            return [Image.fromarray(img) for img in image]
        elif output_type == "latent":
            return image # No processing for latent output
        else: # np
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            return image


    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING) # This decorator might need removal if base class changes
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None, # FLUX uses prompt_2
        negative_prompt: Union[str, List[str]] = None, # Added for CFG
        negative_prompt_2: Optional[Union[str, List[str]]] = None, # Added for CFG
        true_cfg_scale: float = 1.0, # For FLUX-like CFG
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        num_inference_steps: int = 20, # Changed default from 28
        sigmas: Optional[List[float]] = None, # For schedulers supporting custom sigmas
        guidance_scale: float = 7.5, # Changed default from 3.5
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None, # Allow pre-computed embeds
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None, # Allow pre-computed embeds
        # ip_adapter_image and ip_adapter_image_embeds are removed as per design for internal handling
        negative_prompt_embeds: Optional[torch.FloatTensor] = None, # Allow pre-computed negative embeds
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None, # Allow pre-computed negative embeds
        output_type: Optional[str] = "pil", # "pil", "latent", "np"
        return_dict: bool = True,
        # joint_attention_kwargs: Optional[Dict[str, Any]] = None, # Handled internally by IP-Adapter
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None, # For progress
        callback_on_step_end_tensor_inputs: List[str] = ["latents"], # For progress
        max_sequence_length: int = 512, # Max length for tokenizers
        subject_image: Image.Image = None, # Main subject image input
        subject_scale: float = 1.0, # Scale for subject IP-Adapter
    ):
        # This __call__ method is a significant adaptation.
        # It tries to follow FLUX logic using the components now available in `self`.
        # Many safety checks and utility functions from Diffusers' base Pipeline are missing.

        device = self.device
        dtype = self.dtype

        if height is None: height = 1024 # Default if not provided
        if width is None: width = 1024   # Default if not provided

        # 1. Check inputs (simplified)
        if prompt is None and prompt_embeds is None:
            raise ValueError("Either `prompt` or `prompt_embeds` must be provided.")
        if subject_image is None:
            raise ValueError("`subject_image` must be provided for InstantCharacter.")

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            batch_size = 1 # Should not happen if input check passes

        # 3. Encode prompt
        if prompt_embeds is None:
            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        else: # Assume text_ids also pre-computed or not strictly needed by this transformer version
            # This part is tricky if text_ids are essential for the transformer and not provided
            # For now, if prompt_embeds are given, we might not have text_ids.
            # FLUX transformer does take text_ids.
            print("Warning: Using pre-computed prompt_embeds. Ensure text_ids are handled if needed by the transformer.")
            # Placeholder for text_ids if prompt_embeds are pre-supplied
            # This needs a robust solution if pre-computed embeds are to be fully supported with FLUX.
            # For now, let's assume if prompt_embeds are passed, text_ids might be dummy or derived.
            # A simple approach: tokenize prompt to get text_ids even if embeds are passed.
            if prompt:
                 _, _, text_ids = self.encode_prompt(prompt, prompt_2, device, num_images_per_prompt, max_sequence_length)
            else: # Cannot derive text_ids if prompt is also None
                raise ValueError("If prompt_embeds are provided, prompt must also be provided to derive text_ids for FLUX transformer.")


        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                if negative_prompt is None: # Create unconditional guidance
                    negative_prompt = ""
                    if prompt_2 is not None: # If prompt_2 exists, uncond should also have two parts
                        negative_prompt_2 = ""
                
                uncond_prompt_embeds, uncond_pooled_prompt_embeds, uncond_text_ids = self.encode_prompt(
                    prompt=negative_prompt,
                    prompt_2=negative_prompt_2,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                )
            else:
                uncond_prompt_embeds = negative_prompt_embeds
                uncond_pooled_prompt_embeds = negative_pooled_prompt_embeds
                # Similar issue with uncond_text_ids if negative_prompt_embeds are pre-supplied
                if negative_prompt:
                    _, _, uncond_text_ids = self.encode_prompt(negative_prompt, negative_prompt_2, device, num_images_per_prompt, max_sequence_length)
                else:
                    raise ValueError("If negative_prompt_embeds are provided, negative_prompt must also be provided for uncond_text_ids.")


            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and conditional embeddings.
            prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds])
            pooled_prompt_embeds = torch.cat([uncond_pooled_prompt_embeds, pooled_prompt_embeds])
            text_ids = torch.cat([uncond_text_ids, text_ids])


        # 3.1 Prepare subject image embeddings
        subject_image_pil = subject_image.resize((max(subject_image.size), max(subject_image.size))) # Simple resize
        subject_image_embeds_dict = self.encode_image_emb(subject_image_pil, device, dtype)

        # 4. Prepare latent variables
        # VAE scale factor might be on vae.config.scale_factor or similar
        vae_scale_factor = getattr(self.vae.config, "downscale_factor", 8) # Common for /8 VAEs
        num_channels_latents = self.transformer.config.in_channels // 4 # FLUX specific latent channels
        
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype, # Use dtype of embeddings
            device,
            generator,
            latents,
        )
        if do_classifier_free_guidance: # Latents need to be duplicated for CFG
            latents = torch.cat([latents] * 2)
            latent_image_ids = torch.cat([latent_image_ids]*2)


        # 5. Prepare timesteps and scheduler
        if self.scheduler is None:
            raise ValueError("Scheduler is not initialized.")
        
        # FLUX specific mu calculation for scheduler
        image_seq_len = latents.shape[-2] * latents.shape[-1] # H * W of latent
        # Scheduler config access needs to be robust
        base_image_seq_len = getattr(self.scheduler.config, "base_image_seq_len", 256*256) # Example
        max_image_seq_len = getattr(self.scheduler.config, "max_image_seq_len", 256*256) # Example
        base_shift = getattr(self.scheduler.config, "base_shift", 0.25) # Example
        max_shift = getattr(self.scheduler.config, "max_shift", 0.25) # Example

        mu = calculate_shift(image_seq_len, base_image_seq_len, max_image_seq_len, base_shift, max_shift)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu
        )
        
        # Guidance embedding for FLUX transformer
        if self.transformer.config.guidance_embeds:
            guidance_emb = torch.full([1], guidance_scale if not do_classifier_free_guidance else 0.0, device=device, dtype=torch.float32) # Uncond guidance is 0
            guidance_emb = guidance_emb.expand(batch_size * num_images_per_prompt)
            if do_classifier_free_guidance:
                 # Conditional part of CFG uses the actual guidance_scale
                cond_guidance_emb = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                cond_guidance_emb = cond_guidance_emb.expand(batch_size * num_images_per_prompt)
                guidance_emb = torch.cat([guidance_emb, cond_guidance_emb])
        else:
            guidance_emb = None
            
        # Prepare joint_attention_kwargs for IP-Adapter
        joint_attention_kwargs = {}

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance and i==0 and not (latents.shape[0] == batch_size * num_images_per_prompt * 2) else latents
                # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) # If scheduler needs this

                timestep_expanded = t.expand(latent_model_input.shape[0]).to(dtype=dtype) # Ensure dtype match

                # Subject IP-Adapter injection
                self.subject_image_proj_model.to(device, dtype=dtype) # Ensure device
                
                # Prepare subject embeddings for the current batch size (considering CFG)
                current_bs = latent_model_input.shape[0]
                
                # Tile subject embeds if batch size of latents is larger (e.g. due to CFG)
                # Original subject_image_embeds_dict is for batch_size * num_images_per_prompt
                # If CFG, latents are 2 * (batch_size * num_images_per_prompt)
                # So, subject embeds also need to be tiled for the unconditional part.
                # For unconditional, IP adapter effect should ideally be neutral or scaled down.
                # Simplest: repeat. More advanced: use zero embeds for uncond pass or scale down.
                
                # For now, let's assume the IP-Adapter is applied to both cond and uncond passes if not handled by scale=0 for uncond.
                # The scale parameter in FluxIPAttnProcessor can handle this if set to 0 for uncond.
                
                # We need to ensure subject_image_embeds_dict tensors are correctly batched for CFG
                # If latent_model_input.shape[0] is 2 * original_batch_size, then tile embeds
                if latent_model_input.shape[0] > subject_image_embeds_dict['image_embeds_low_res_shallow'].shape[0]:
                    factor = latent_model_input.shape[0] // subject_image_embeds_dict['image_embeds_low_res_shallow'].shape[0]
                    s_low_shallow = subject_image_embeds_dict['image_embeds_low_res_shallow'].repeat(factor, 1, 1)
                    s_low_deep = subject_image_embeds_dict['image_embeds_low_res_deep'].repeat(factor, 1, 1)
                    s_high_deep = subject_image_embeds_dict['image_embeds_high_res_deep'].repeat(factor, 1, 1)
                else:
                    s_low_shallow = subject_image_embeds_dict['image_embeds_low_res_shallow']
                    s_low_deep = subject_image_embeds_dict['image_embeds_low_res_deep']
                    s_high_deep = subject_image_embeds_dict['image_embeds_high_res_deep']

                subject_image_projected_embeds = self.subject_image_proj_model(
                    low_res_shallow=s_low_shallow.to(device, dtype),
                    low_res_deep=s_low_deep.to(device, dtype),
                    high_res_deep=s_high_deep.to(device, dtype),
                    timesteps=timestep_expanded.to(device), # Ensure timestep is on correct device
                    need_temb=True
                )[0]

                joint_attention_kwargs['emb_dict'] = dict(
                    length_encoder_hidden_states=prompt_embeds.shape[1] # Length of the text embeds
                )
                joint_attention_kwargs['subject_emb_dict'] = dict(
                    ip_hidden_states=subject_image_projected_embeds,
                    scale=subject_scale, # Apply scale here
                )
                
                # For CFG, the unconditional pass should ideally have scale=0 for IP-Adapter
                # This needs to be handled by FluxIPAttnProcessor or by modifying subject_emb_dict for the uncond part
                if do_classifier_free_guidance:
                    # Create a scale tensor: 0 for uncond, subject_scale for cond
                    cfg_ip_scale = torch.tensor([0.0] * (current_bs // 2) + [subject_scale] * (current_bs // 2), device=device, dtype=dtype).view(-1,1,1)
                    # This assumes FluxIPAttnProcessor can take a scale tensor. If not, this logic needs adjustment.
                    # Or, pass two different subject_emb_dicts if the processor is called per CFG half.
                    # For now, let's assume the processor handles a single call with batched CFG inputs.
                    # The 'scale' in subject_emb_dict might need to be a tensor if it varies per batch item.
                    # For simplicity, the current FluxIPAttnProcessor takes a single float scale.
                    # A more robust way: call transformer twice if IP-adapter scale needs to differ for CFG.
                    # Or, modify FluxIPAttnProcessor to accept scale tensor.
                    # For now, the `subject_scale` will apply to both if not handled inside attn_proc.
                    # This is a simplification.
                    pass


                # Predict the noise residual
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep_expanded / 1000, # FLUX scales timestep
                    guidance=guidance_emb,
                    pooled_projections=pooled_prompt_embeds, # Already CFG-batched if needed
                    encoder_hidden_states=prompt_embeds,   # Already CFG-batched if needed
                    txt_ids=text_ids,                      # Already CFG-batched if needed
                    img_ids=latent_image_ids,              # Already CFG-batched if needed
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if true_cfg_scale > 1.0: # FLUX specific true_cfg
                         # This part needs careful adaptation from original FLUX true_cfg logic
                         # It might involve another model call or specific scaling of noise_pred_text
                         # For now, this is a simplified CFG.
                         pass


                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    # Simplified callback handling
                    callback_kwargs = {"latents": latents}
                    callback_on_step_end(self, i, t, callback_kwargs)

                if i == len(timesteps) - 1 or ((i + 1) >= num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
            
        # 7. Post-processing
        if output_type == "latent":
            image = latents
        else:
            # Ensure VAE is on the correct device
            self.vae.to(device, dtype=dtype)
            # FLUX VAE might have specific scaling/shifting factors
            vae_scaling_factor = getattr(self.vae.config, "scaling_factor", 0.13025) # Example
            vae_shift_factor = getattr(self.vae.config, "shift_factor", 0.0)       # Example
            
            latents = self._unpack_latents(latents, height, width, vae_scale_factor) # vae_scale_factor here is for unpack, not decode
            latents = (latents / vae_scaling_factor) + vae_shift_factor
            image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0] # Ensure latents match VAE dtype
            image = self._postprocess_image(image, output_type=output_type)


        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image) # Ensure FluxPipelineOutput is defined or imported


    def with_style_lora(self, lora_file_path, lora_weight=1.0, trigger='', *args, **kwargs):
        # This method might need significant changes if flux_load_lora relies on Diffusers pipeline structure
        print("Applying LoRA. Ensure flux_load_lora is compatible with the new pipeline structure.")
        flux_load_lora(self, lora_file_path, lora_weight) # self here is InstantCharacterFluxPipeline
        
        current_prompt = kwargs.get("prompt", "")
        if isinstance(current_prompt, list):
            kwargs['prompt'] = [f"{trigger}, {p}" for p in current_prompt]
        else:
            kwargs['prompt'] = f"{trigger}, {current_prompt}"
            
        res = self.__call__(*args, **kwargs)
        
        print("Reverting LoRA.")
        flux_load_lora(self, lora_file_path, -lora_weight) # Attempt to revert
        return res
