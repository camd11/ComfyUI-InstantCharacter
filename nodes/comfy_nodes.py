import os
import sys
import torch
import folder_paths
from PIL import Image
import numpy as np


# Add the parent directory to the Python path so we can import from easycontrol
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from InstantCharacter.pipeline import InstantCharacterFluxPipeline
from huggingface_hub import login


if "ipadapter" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "ipadapter")]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ipadapter"]
folder_paths.folder_names_and_paths["ipadapter"] = (current_paths, folder_paths.supported_pt_extensions)


class InstantCharacterLoadModelFromLocal:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "path/to/your/local/flux_model_directory_or_file", "tooltip": "Path to the local FLUX.1-dev model directory or .safetensors file"}),
                "image_encoder_path": ("STRING", {"default": "siglip-so400m-patch14-384", "tooltip": "Name of the CLIP Vision model folder (e.g., siglip-so400m-patch14-384) within ComfyUI/models/clip_vision/"}),
                "image_encoder_2_path": ("STRING", {"default": "dinov2-giant", "tooltip": "Name of the DINOv2 model folder (e.g., dinov2-giant) within ComfyUI/models/clip_vision/"}),
                "ip_adapter_path": ("STRING", {"default": "instantcharacter_ip-adapter.bin", "tooltip": "Filename of the IP-Adapter model (e.g., instantcharacter_ip-adapter.bin) within ComfyUI/models/ipadapter/"}),
                "cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "Enable CPU offload to save VRAM"}),
            }
        }

    RETURN_TYPES = ("INSTANTCHAR_PIPE",)
    FUNCTION = "load_model"
    CATEGORY = "InstantCharacter"
    DESCRIPTION = "Loads InstantCharacter models and components from local file paths, disabling HuggingFace downloads."
    
    def load_model(self, model_path, image_encoder_path, image_encoder_2_path, ip_adapter_path, cpu_offload):
        if not model_path or not model_path.strip():
            raise RuntimeError("InstantCharacter: Local model path (model_path) is required. Please specify the path to your FLUX.1-dev (or compatible) model files.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- TEMPORARY DIAGNOSTIC WORKAROUND ---
        # The following paths are constructed manually because folder_paths.get_full_path()
        # was returning None in the user's environment. This is not a permanent solution.
        # The user should investigate their ComfyUI model path configuration (e.g., extra_model_paths.yaml)
        # to ensure 'clip_vision' and 'ipadapter' types are correctly recognized.
        # This workaround assumes a standard ComfyUI directory structure.
        assumed_comfyui_models_base_path = "/home/ryan/ComfyUI2025/ComfyUI/models/"

        # These names come from the input parameters, which have defaults matching the required names
        # For the diagnostic, we use the default names directly.
        image_encoder_name_1 = "siglip-so400m-patch14-384" # Default from INPUT_TYPES
        image_encoder_name_2 = "dinov2-giant" # Default from INPUT_TYPES
        ip_adapter_filename = "instantcharacter_ip-adapter.bin" # Default from INPUT_TYPES

        image_encoder_path_resolved = os.path.join(assumed_comfyui_models_base_path, "clip_vision", image_encoder_name_1)
        image_encoder_2_path_resolved = os.path.join(assumed_comfyui_models_base_path, "clip_vision", image_encoder_name_2)
        ip_adapter_path_resolved = os.path.join(assumed_comfyui_models_base_path, "ipadapter", ip_adapter_filename)

        # Check paths and raise specific errors
        if not os.path.exists(image_encoder_path_resolved):
            raise FileNotFoundError(
                f"DIAGNOSTIC: Image Encoder 1 ({image_encoder_name_1}) not found at manually constructed path: '{image_encoder_path_resolved}'. "
                f"This path was built assuming ComfyUI models are at '{assumed_comfyui_models_base_path}'. "
                f"Please verify the path and file existence. If this diagnostic works, the underlying issue is likely with ComfyUI's folder_paths configuration."
            )

        if not os.path.exists(image_encoder_2_path_resolved):
            raise FileNotFoundError(
                f"DIAGNOSTIC: Image Encoder 2 ({image_encoder_name_2}) not found at manually constructed path: '{image_encoder_2_path_resolved}'. "
                f"This path was built assuming ComfyUI models are at '{assumed_comfyui_models_base_path}'. "
                f"Please verify the path and file existence. If this diagnostic works, the underlying issue is likely with ComfyUI's folder_paths configuration."
            )

        if not os.path.exists(ip_adapter_path_resolved):
            raise FileNotFoundError(
                f"DIAGNOSTIC: IP-Adapter ({ip_adapter_filename}) not found at manually constructed path: '{ip_adapter_path_resolved}'. "
                f"This path was built assuming ComfyUI models are at '{assumed_comfyui_models_base_path}'. "
                f"Please verify the path and file existence. If this diagnostic works, the underlying issue is likely with ComfyUI's folder_paths configuration."
            )
        
        pipe = InstantCharacterFluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True # Ensure no download attempts for the base model
        )

        # Initialize adapter first
        pipe.init_adapter(
            image_encoder_path=image_encoder_path_resolved, # Use resolved path
            image_encoder_2_path=image_encoder_2_path_resolved, # Use resolved path
            subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path_resolved, nb_token=1024), # Use resolved path
            # The pipeline.py init_adapter will need to use local_files_only=True for these encoders
        )

        # Then move to device or enable offloading
        if cpu_offload:
            print("Enabling CPU offload for InstantCharacter pipeline...")
            pipe.enable_sequential_cpu_offload()
            print("CPU offload enabled.")
        else:
            pipe.to(device)

        return (pipe,)


class InstantCharacterLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hf_token": ("STRING", {"default": "", "multiline": True}),
                "ip_adapter_name": (folder_paths.get_filename_list("ipadapter"), ),
                "cpu_offload": ("BOOLEAN", {"default": False})
            }
        }
    
    RETURN_TYPES = ("INSTANTCHAR_PIPE",)
    FUNCTION = "load_model"
    CATEGORY = "InstantCharacter"

    def load_model(self, hf_token, ip_adapter_name, cpu_offload):
        login(token=hf_token)
        base_model = "black-forest-labs/FLUX.1-dev"
        image_encoder_path = "google/siglip-so400m-patch14-384"
        image_encoder_2_path = "facebook/dinov2-giant"
        cache_dir = folder_paths.get_folder_paths("diffusers")[0]
        image_encoder_cache_dir = folder_paths.get_folder_paths("clip_vision")[0]
        image_encoder_2_cache_dir = folder_paths.get_folder_paths("clip_vision")[0]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ip_adapter_path = folder_paths.get_full_path("ipadapter", ip_adapter_name)
        
        pipe = InstantCharacterFluxPipeline.from_pretrained(
            base_model, 
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )

        # Initialize adapter first
        pipe.init_adapter(
            image_encoder_path=image_encoder_path,
            cache_dir=image_encoder_cache_dir,
            image_encoder_2_path=image_encoder_2_path,
            cache_dir_2=image_encoder_2_cache_dir,
            subject_ipadapter_cfg=dict(
                subject_ip_adapter_path=ip_adapter_path,
                nb_token=1024
            ),
        )

        # Then move to device or enable offloading
        if cpu_offload:
            print("Enabling CPU offload for InstantCharacter pipeline...")
            pipe.enable_sequential_cpu_offload()
            print("CPU offload enabled.")
        else:
            pipe.to(device)

        return (pipe,)


class InstantCharacterGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("INSTANTCHAR_PIPE",),
                "prompt": ("STRING", {"multiline": True}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "subject_scale": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "subject_image": ("IMAGE",),
                "lora_path": ("STRING", {"default": "", "tooltip": "Path to the LoRA (.safetensors) file"}),
                "lora_trigger": ("STRING", {"default": "", "tooltip": "Trigger keyword(s) for the LoRA"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "InstantCharacter"

    def generate(self, pipe, prompt, height, width, guidance_scale,
                num_inference_steps, seed, subject_scale, subject_image=None, lora_path=None, lora_trigger=None):
        
        # Convert subject image from tensor to PIL if provided
        subject_image_pil = None
        if subject_image is not None:
            if isinstance(subject_image, torch.Tensor):
                if subject_image.dim() == 4:  # [batch, height, width, channels]
                    img = subject_image[0].cpu().numpy()
                else:  # [height, width, channels]
                    img = subject_image.cpu().numpy()
                subject_image_pil = Image.fromarray((img * 255).astype(np.uint8))
            elif isinstance(subject_image, np.ndarray):
                subject_image_pil = Image.fromarray((subject_image * 255).astype(np.uint8))
        
        lora_path_input = lora_path.strip() if lora_path else ""
        lora_trigger_input = lora_trigger.strip() if lora_trigger else ""

        # Generate image
        if lora_path_input:
            output = pipe.with_style_lora(
                lora_file_path=lora_path_input,
                trigger=lora_trigger_input,
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator("cpu").manual_seed(seed),
                subject_image=subject_image_pil,
                subject_scale=subject_scale,
            )
        else:
            output = pipe(
                prompt=prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator("cpu").manual_seed(seed),
                subject_image=subject_image_pil,
                subject_scale=subject_scale,
            )
        
        # Convert PIL image to tensor format
        image = np.array(output.images[0]) / 255.0
        image = torch.from_numpy(image).float()
        
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        return (image,)

