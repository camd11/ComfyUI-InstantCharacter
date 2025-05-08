import os
import sys
import torch
import folder_paths
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModel

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
                "flux_text_encoder_one": ("CLIP", {}), # Expects composite CLIP (CLIP-L + T5XXL) OR just CLIP-L
                "flux_text_encoder_two": ("CLIP", {}), # Expects T5XXL CLIP object
                "flux_vae": ("VAE", {}),
                "siglip_model_name_or_path": ("STRING", {"default": "google/siglip-base-patch16-384"}),
                "dinov2_model_name_or_path": ("STRING", {"default": "facebook/dinov2-base"}),
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
                              flux_text_encoder_one, # Should be CLIP-L part
                              flux_text_encoder_two, # Add argument for the second encoder (T5XXL part)
                              flux_vae,
                              siglip_model_name_or_path, # Changed from siglip_vision_model
                              dinov2_model_name_or_path, # Changed from dinov2_vision_model
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

       # Logic to extract components from potentially separate CLIP objects
       
       text_encoder_1_module = None
       tokenizer_1_instance = None
       text_encoder_2_module = None
       tokenizer_2_instance = None

       # --- Process Encoder 1 (Expected: CLIP-L) ---
       if flux_text_encoder_one:
           # Assume flux_text_encoder_one directly provides CLIP-L components
           if hasattr(flux_text_encoder_one, 'tokenizer'):
                # If it's composite, try getting clip_l, else assume it IS clip_l tokenizer
               if hasattr(flux_text_encoder_one.tokenizer, 'clip_l'):
                   tokenizer_1_instance = flux_text_encoder_one.tokenizer.clip_l
               else: # Assume flux_text_encoder_one.tokenizer *is* the CLIP-L tokenizer
                   tokenizer_1_instance = flux_text_encoder_one.tokenizer
                   print("InstantCharacterLoader: Assuming flux_text_encoder_one.tokenizer is CLIP-L tokenizer.")
           else:
                print("InstantCharacterLoader WARNING: flux_text_encoder_one missing 'tokenizer' attribute.")

           if hasattr(flux_text_encoder_one, 'cond_stage_model'):
                # If it's composite, try getting clip_l, else assume it IS clip_l encoder
               if hasattr(flux_text_encoder_one.cond_stage_model, 'clip_l'):
                   text_encoder_1_module = flux_text_encoder_one.cond_stage_model.clip_l
               else: # Assume flux_text_encoder_one.cond_stage_model *is* the CLIP-L encoder
                   text_encoder_1_module = flux_text_encoder_one.cond_stage_model
                   print("InstantCharacterLoader: Assuming flux_text_encoder_one.cond_stage_model is CLIP-L encoder.")
           else:
                print("InstantCharacterLoader WARNING: flux_text_encoder_one missing 'cond_stage_model' attribute.")
       else:
            print("InstantCharacterLoader WARNING: flux_text_encoder_one is None.")


       # --- Process Encoder 2 (Expected: T5XXL) ---
       if flux_text_encoder_two:
           # Assume flux_text_encoder_two directly provides T5XXL components
           if hasattr(flux_text_encoder_two, 'tokenizer'):
                # If it's composite, try getting t5xxl, else assume it IS t5xxl tokenizer
               if hasattr(flux_text_encoder_two.tokenizer, 't5xxl'):
                    tokenizer_2_instance = flux_text_encoder_two.tokenizer.t5xxl
               else: # Assume flux_text_encoder_two.tokenizer *is* the T5XXL tokenizer
                   tokenizer_2_instance = flux_text_encoder_two.tokenizer
                   print("InstantCharacterLoader: Assuming flux_text_encoder_two.tokenizer is T5XXL tokenizer.")
           else:
                print("InstantCharacterLoader WARNING: flux_text_encoder_two missing 'tokenizer' attribute.")

           if hasattr(flux_text_encoder_two, 'cond_stage_model'):
                # If it's composite, try getting t5xxl, else assume it IS t5xxl encoder
               if hasattr(flux_text_encoder_two.cond_stage_model, 't5xxl'):
                   text_encoder_2_module = flux_text_encoder_two.cond_stage_model.t5xxl
               else: # Assume flux_text_encoder_two.cond_stage_model *is* the T5XXL encoder
                   text_encoder_2_module = flux_text_encoder_two.cond_stage_model
                   print("InstantCharacterLoader: Assuming flux_text_encoder_two.cond_stage_model is T5XXL encoder.")
           else:
                print("InstantCharacterLoader WARNING: flux_text_encoder_two missing 'cond_stage_model' attribute.")
       else:
            print("InstantCharacterLoader WARNING: flux_text_encoder_two is None.")

       # --- Final Checks ---
       if text_encoder_1_module is None or tokenizer_1_instance is None:
            print("InstantCharacterLoader WARNING: Failed to extract Text Encoder 1 or Tokenizer 1.")
       if text_encoder_2_module is None or tokenizer_2_instance is None:
            print("InstantCharacterLoader WARNING: Failed to extract Text Encoder 2 or Tokenizer 2.")
       
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

       # Load SigLIP and DINOv2 models and processors using transformers
       print(f"InstantCharacterLoader: Loading SigLIP model from: {siglip_model_name_or_path}")
       siglip_processor = AutoProcessor.from_pretrained(siglip_model_name_or_path)
       siglip_hf_model = AutoModel.from_pretrained(siglip_model_name_or_path)
       print(f"InstantCharacterLoader: SigLIP model loaded: {type(siglip_hf_model)}")

       print(f"InstantCharacterLoader: Loading DINOv2 model from: {dinov2_model_name_or_path}")
       dinov2_processor = AutoProcessor.from_pretrained(dinov2_model_name_or_path)
       dinov2_hf_model = AutoModel.from_pretrained(dinov2_model_name_or_path)
       print(f"InstantCharacterLoader: DINOv2 model loaded: {type(dinov2_hf_model)}")
       
       pipe = InstantCharacterFluxPipeline(
           flux_unet_model_object=flux_unet_model, # Now contains text_encoders/tokenizers
           vae_module=actual_vae,
           siglip_hf_processor=siglip_processor, # Pass loaded HF processor
           siglip_hf_model=siglip_hf_model,       # Pass loaded HF model
           dinov2_hf_processor=dinov2_processor, # Pass loaded HF processor
           dinov2_hf_model=dinov2_hf_model,       # Pass loaded HF model
           ipadapter_model_data_dict=ipadapter_model_data,
           sampler_object=sampler # Pass sampler to pipeline
           # dtype is handled by the pipeline's __init__
       )

       # CPU Offload:
       # ComfyUI's model management handles device placement for input MODEL.
       # The Hugging Face models (SigLIP, DINOv2) loaded here will be on the default device (usually CPU)
       # or GPU if CUDA is available and transformers decides so.
       # The pipeline will move them to the UNet's device during its initialization.
       # The pipeline initializes its new components (attn_procs, image_proj) on the device
       # of the input UNet.
       if cpu_offload:
           print("InstantCharacter: CPU offload for input ComfyUI models (UNet, VAE, CLIPs) is managed by their respective loaders.")
           print("InstantCharacter: For Hugging Face models (SigLIP, DINOv2), they are loaded by transformers and then moved to the UNet's device by the pipeline.")
           # If the pipeline itself had an explicit offload mechanism for its components, it could be called here.
           # For example, if pipe.enable_model_cpu_offload() existed:
           # try:
           #     pipe.enable_model_cpu_offload() # This would be a Diffusers-like method
           #     print("InstantCharacterPipeline: Attempted model CPU offload on the pipeline.")
           # except AttributeError:
           #     print("InstantCharacterPipeline: Does not have a direct CPU offload method. Relies on component device placement.")
           pass

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
