import os
import sys
import torch
import folder_paths
from PIL import Image
import numpy as np

# Add the parent directory to the Python path so we can import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from InstantCharacter.pipeline import InstantCharacterFluxPipeline
# Removed: from huggingface_hub import login

# Ensure 'ipadapter' path is registered if not already (though IPAdapterModelLoader should handle this)
if "ipadapter" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "ipadapter")]
    folder_paths.folder_names_and_paths["ipadapter"] = (current_paths, folder_paths.supported_pt_extensions)


class InstantCharacterLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_unet_model": ("MODEL", {}),
                "flux_text_encoder_one": ("CLIP", {}),
                "flux_vae": ("VAE", {}),
                "siglip_vision_model": ("CLIP_VISION", {}),
                "dinov2_vision_model": ("CLIP_VISION", {}),
                "ipadapter_model_data": ("IPADAPTER", {}),
                "sampler": ("SAMPLER", ), # Added SAMPLER input
                "cpu_offload": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("INSTANTCHAR_PIPE",)
    FUNCTION = "load_pipe_from_models"
    CATEGORY = "InstantCharacter"
    DESCRIPTION = "Loads InstantCharacter pipeline from pre-loaded model components."

    def load_pipe_from_models(self,
                              flux_unet_model,
                              flux_text_encoder_one, # flux_text_encoder_two removed
                              flux_vae,
                              siglip_vision_model,
                              dinov2_vision_model,
                              ipadapter_model_data,
                              sampler, # Added sampler argument
                              cpu_offload,
                             ):

        # Validate ipadapter_model_data structure (remains important)
        if not isinstance(ipadapter_model_data, dict) or \
           "ip_adapter" not in ipadapter_model_data or \
           "image_proj" not in ipadapter_model_data:
            raise ValueError(
                "IPAdapter model data is not in the expected dictionary format "
                "with 'ip_adapter' and 'image_proj' keys containing state_dicts."
            )

        # Extract underlying nn.Module components
        actual_unet = flux_unet_model.model
        actual_vae = flux_vae.first_stage_model # Preserved from original block

        # Get CLIP Input (from method argument)
        clip_input_one = flux_text_encoder_one # This is the key input now
        
        # Corrected logic for extracting text encoder and tokenizer components
        # from the composite CLIP object (flux_text_encoder_one).

        text_encoder_1_module = None
        tokenizer_1_instance = None
        text_encoder_2_module = None
        tokenizer_2_instance = None

        if clip_input_one: # clip_input_one is flux_text_encoder_one from inputs
            # Check for tokenizer attribute
            if hasattr(clip_input_one, 'tokenizer') and clip_input_one.tokenizer is not None:
                # Extract CLIP-L tokenizer
                if hasattr(clip_input_one.tokenizer, 'clip_l'):
                    tokenizer_1_instance = clip_input_one.tokenizer.clip_l
                
                # Extract T5XXL tokenizer
                if hasattr(clip_input_one.tokenizer, 't5xxl'):
                    tokenizer_2_instance = clip_input_one.tokenizer.t5xxl
            
            # Check for cond_stage_model attribute
            if hasattr(clip_input_one, 'cond_stage_model') and clip_input_one.cond_stage_model is not None:
                # Extract CLIP-L text encoder module
                if hasattr(clip_input_one.cond_stage_model, 'clip_l'):
                    text_encoder_1_module = clip_input_one.cond_stage_model.clip_l
                
                # Extract T5XXL text encoder module
                if hasattr(clip_input_one.cond_stage_model, 't5xxl'):
                    text_encoder_2_module = clip_input_one.cond_stage_model.t5xxl
        
        # Attach the extracted (or None) components to the flux_unet_model object.
        # The pipeline (InstantCharacterFluxPipeline) expects to find these attributes on flux_unet_model.
        if flux_unet_model:
            flux_unet_model.text_encoder_1 = text_encoder_1_module
            flux_unet_model.tokenizer_1 = tokenizer_1_instance
            flux_unet_model.text_encoder_2 = text_encoder_2_module
            flux_unet_model.tokenizer_2 = tokenizer_2_instance
            print("InstantCharacterLoader: Assigned extracted/default text encoders and tokenizers to flux_unet_model.")
        else:
            # This case should ideally be prevented by ComfyUI's input validation if MODEL is required.
            print("InstantCharacterLoader WARNING: flux_unet_model is None. Cannot attach text encoder/tokenizer components.")
        
        pipe = InstantCharacterFluxPipeline(
            flux_unet_model_object=flux_unet_model, # Now contains text_encoders/tokenizers
            vae_module=actual_vae,
            siglip_vision_model_object=siglip_vision_model,
            dinov2_vision_model_object=dinov2_vision_model,
            ipadapter_model_data_dict=ipadapter_model_data,
            sampler_object=sampler # Pass sampler to pipeline
            # dtype is handled by the pipeline's __init__
        )

        # CPU Offload:
        # ComfyUI's model management handles device placement for input MODEL, CLIP_VISION.
        # The pipeline initializes its new components (attn_procs, image_proj) on the device
        # of the input UNet. If specific offloading methods were on the pipeline (like Diffusers),
        # they could be called here. For now, we rely on ComfyUI's management of input models.
        if cpu_offload:
            print("InstantCharacter: CPU offload requested for input models is managed by ComfyUI loaders.")
            print("InstantCharacterPipeline initializes its internal components on the UNet's device.")
            # If `pipe` had a method like `pipe.enable_sequential_cpu_offload()` and it was desired:
            # try:
            #     pipe.enable_sequential_cpu_offload()
            #     print("InstantCharacterPipeline: Attempted sequential CPU offload.")
            # except AttributeError:
            #     print("InstantCharacterPipeline: Does not have enable_sequential_cpu_offload method.")
            pass # Primary offload is via ComfyUI's handling of the input model objects

        return (pipe,)


class InstantCharacterGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("INSTANTCHAR_PIPE",),
                "prompt": ("STRING", {"multiline": True, "default": "A photo of a character"}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "subject_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                # Added negative_prompt based on typical pipeline usage
                "negative_prompt": ("STRING", {"multiline": True, "default": "ugly, disfigured, low quality, blurry, nsfw"}),
            },
            "optional": {
                "subject_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "InstantCharacter"

    def generate(self, pipe: InstantCharacterFluxPipeline, prompt, height, width, guidance_scale, 
                 num_inference_steps, seed, subject_scale, negative_prompt="", subject_image=None):
        
        subject_image_pil = None
        if subject_image is not None:
            if isinstance(subject_image, torch.Tensor):
                # Assuming subject_image is a ComfyUI IMAGE tensor: (batch, H, W, C)
                if subject_image.dim() == 4 and subject_image.shape[0] == 1:
                    img_np = subject_image[0].cpu().numpy() # Select first image in batch
                elif subject_image.dim() == 3: # H, W, C
                    img_np = subject_image.cpu().numpy()
                else:
                    raise ValueError("subject_image tensor has unexpected dimensions.")
                subject_image_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            elif isinstance(subject_image, np.ndarray): # Should not happen with ComfyUI IMAGE type
                subject_image_pil = Image.fromarray((subject_image * 255).astype(np.uint8))
            elif isinstance(subject_image, Image.Image): # Already PIL
                 subject_image_pil = subject_image
            else:
                raise TypeError("subject_image must be a ComfyUI IMAGE tensor or PIL.Image.")
        
        if subject_image_pil is None:
            raise ValueError("Subject image is required for InstantCharacterGenerate.")

        # Generate image using the pipeline's __call__ method
        # The pipeline's __call__ method is expected to handle device placement internally
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt, # Pass negative_prompt
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device=pipe.device).manual_seed(seed), # Use pipeline's device for generator
            subject_image=subject_image_pil,
            subject_scale=subject_scale,
            # prompt_2 and negative_prompt_2 can be added if pipeline supports them explicitly
        )
        
        # output.images is expected to be a list of PIL Images from the pipeline
        if not isinstance(output.images, list) or not all(isinstance(img, Image.Image) for img in output.images):
            raise TypeError("Pipeline did not return a list of PIL Images.")

        # Convert PIL image to ComfyUI IMAGE tensor format (batch, H, W, C)
        output_images_np = [(np.array(img).astype(np.float32) / 255.0) for img in output.images]
        output_images_torch = [torch.from_numpy(img_np) for img_np in output_images_np]
        
        # Stack if multiple images, otherwise add batch dim
        if len(output_images_torch) > 1:
            final_image_tensor = torch.stack(output_images_torch, dim=0)
        elif output_images_torch:
            final_image_tensor = output_images_torch[0].unsqueeze(0)
        else:
            raise ValueError("Pipeline returned no images.")
            
        return (final_image_tensor,)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "InstantCharacterLoader": InstantCharacterLoader,
    "InstantCharacterGenerate": InstantCharacterGenerate,
    # "InstantCharacterLoadModel": REMOVED
    # "InstantCharacterLoadModelFromLocal": REMOVED
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantCharacterLoader": "Load InstantCharacter Pipeline",
    "InstantCharacterGenerate": "Generate with InstantCharacter",
}

# Commented out old classes:
# class InstantCharacterLoadModelFromLocal: ...
# class InstantCharacterLoadModel: ...
