# Phase 5: Pseudocode for `InstantCharacter/pipeline.py` Modifications

This document outlines the pseudocode for refactoring `InstantCharacterFluxPipeline` to accept pre-loaded model components.

```python
# PSEUDOCODE for InstantCharacter/pipeline.py

# --- Imports ---
# Keep existing PIL, torch, einops, etc.
# from .models.attn_processor import FluxIPAttnProcessor # Keep
# from .models.resampler import CrossLayerCrossScaleProjector # Keep
# from .models.utils import flux_load_lora # Keep

# Critical: Determine base class. FluxPipeline from diffusers might be too tied to from_pretrained.
# Consider a more generic diffusers.DiffusionPipeline or a custom base.
# For this pseudocode, using a placeholder:
# from diffusers import DiffusionPipeline as BaseDiffusionPipeline # Placeholder
# from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput # If needed

# --- Class Definition ---
class InstantCharacterFluxPipeline(BaseDiffusionPipeline): # Or appropriate base

    def __init__(self,
                 flux_unet_model_object,      # ComfyUI MODEL object (UNet, VAE, TextEncoders, Scheduler config)
                 siglip_vision_model_object,  # ComfyUI CLIP_VISION object for SigLIP
                 dinov2_vision_model_object,  # ComfyUI CLIP_VISION object for DINOv2
                 ipadapter_model_data_dict,   # Dict of IPAdapter weights from IPAdapterModelLoader
                 dtype=torch.bfloat16):

        super().__init__() # Initialize base pipeline

        # --- Assign Core FLUX Components ---
        # These assignments depend on the structure of ComfyUI's MODEL object.
        # The MODEL object should encapsulate the UNet, VAE, Text Encoders, and Scheduler config
        # that were previously loaded by FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev").
        # If GGUF Unet Loader only provides the UNet, VAE and Text Encoders must be separate inputs.
        # For this pseudocode, assume `flux_unet_model_object` is comprehensive.

        self.transformer = flux_unet_model_object.diffusion_model # UNet
        self.vae = flux_unet_model_object.first_stage_model    # VAE
        # Text Encoders & Tokenizers (example access, may vary)
        # self.text_encoder = flux_unet_model_object.text_encoder_one
        # self.tokenizer = flux_unet_model_object.tokenizer_one
        # self.text_encoder_2 = flux_unet_model_object.text_encoder_two
        # self.tokenizer_2 = flux_unet_model_object.tokenizer_two
        # self.scheduler = flux_unet_model_object.scheduler # Or initialize a FluxScheduler

        # --- Assign Image Encoders (from CLIP_VISION objects) ---
        self.siglip_image_encoder_model = siglip_vision_model_object.model # The actual transformer/vision model
        self.siglip_image_processor = siglip_vision_model_object # The ComfyUI CLIP_VISION object for preprocessing

        self.dinov2_image_encoder_model = dinov2_vision_model_object.model
        self.dinov2_image_processor = dinov2_vision_model_object

        # --- Initialize IP Adapter Components ---
        self._initialize_ip_adapter_components(ipadapter_model_data_dict, dtype)

        self.dtype = dtype
        # Register modules if necessary for Diffusers compatibility or ComfyUI management
        # self.register_modules(transformer=self.transformer, vae=self.vae, ...)

    def _initialize_ip_adapter_components(self, ipadapter_state_dict, dtype):
        # This method reuses logic from the original init_ccp_and_attn_processor
        # but uses the pre-loaded ipadapter_state_dict.
        device = self.transformer.device # Or a managed device from ComfyUI

        # Attn Processor Initialization
        attn_procs = {}
        # transformer_config = self.transformer.config # Assumes UNet has config
        # text_encoder_2_config = self.text_encoder_2.config # Assumes text encoder has config
        # These configs are needed for hidden_size, ip_hidden_states_dim.
        # If not directly available, these dimensions might need to be hardcoded or inferred.
        # Example dimensions (MUST BE VERIFIED):
        flux_transformer_hidden_size = 4096 # Example
        flux_text_encoder_dim = 4096    # Example for text_encoder_2.config.d_model

        for name in self.transformer.attn_processors.keys(): # Iterate expected processor names
            attn_procs[name] = FluxIPAttnProcessor(
                hidden_size=flux_transformer_hidden_size, # Placeholder
                ip_hidden_states_dim=flux_text_encoder_dim, # Placeholder
            ).to(device, dtype=dtype)
        self.transformer.set_attn_processor(attn_procs)
        tmp_ip_layers = torch.nn.ModuleList(self.transformer.attn_processors.values())
        tmp_ip_layers.load_state_dict(ipadapter_state_dict["ip_adapter"], strict=False)

        # Image Projection Model Initialization
        # Parameters for CrossLayerCrossScaleProjector need to be accurate.
        # nb_token was 1024 in original.
        image_proj_model = CrossLayerCrossScaleProjector(
            inner_dim=1152 + 1536, num_attention_heads=42, attention_head_dim=64,
            cross_attention_dim=1152 + 1536, num_layers=4, dim=1280, depth=4,
            dim_head=64, heads=20, num_queries=1024, # nb_token
            embedding_dim=1152 + 1536, output_dim=4096, ff_mult=4,
            timestep_in_dim=320, timestep_flip_sin_to_cos=True, timestep_freq_shift=0,
        ).to(device, dtype=dtype)
        image_proj_model.eval()
        image_proj_model.load_state_dict(ipadapter_state_dict["image_proj"], strict=False)
        self.subject_image_proj_model = image_proj_model

    # --- Image Encoding Methods ---
    # These methods now use the pre-loaded image encoder models and processors.

    @torch.inference_mode()
    def encode_siglip_image_emb(self, siglip_pixel_values, device, dtype):
        # siglip_pixel_values are preprocessed tensors.
        # self.siglip_image_encoder_model is the raw vision transformer.
        # Accessing specific hidden states (7, 13, 26) requires output_hidden_states=True.
        # This needs to be supported by the way the CLIP_VISION object exposes its model.
        self.siglip_image_encoder_model.to(device, dtype=dtype)
        siglip_pixel_values = siglip_pixel_values.to(device, dtype=dtype)
        res = self.siglip_image_encoder_model(siglip_pixel_values, output_hidden_states=True)
        # TODO: Verify hidden state indices [7, 13, 26] are correct for the loaded SigLIP model.
        siglip_image_embeds = res.last_hidden_state
        siglip_image_shallow_embeds = torch.cat([res.hidden_states[i] for i in [7, 13, 26]], dim=1)
        return siglip_image_embeds, siglip_image_shallow_embeds

    @torch.inference_mode()
    def encode_dinov2_image_emb(self, dinov2_pixel_values, device, dtype):
        # dinov2_pixel_values are preprocessed tensors.
        self.dinov2_image_encoder_model.to(device, dtype=dtype)
        dinov2_pixel_values = dinov2_pixel_values.to(device, dtype=dtype)
        res = self.dinov2_image_encoder_model(dinov2_pixel_values, output_hidden_states=True)
        # TODO: Verify hidden state indices [9, 19, 29] are correct for the loaded DINOv2 model.
        dinov2_image_embeds = res.last_hidden_state[:, 1:]
        dinov2_image_shallow_embeds = torch.cat([res.hidden_states[i][:, 1:] for i in [9, 19, 29]], dim=1)
        return dinov2_image_embeds, dinov2_image_shallow_embeds

    @torch.inference_mode()
    def encode_image_emb(self, subject_image_pil, device, dtype): # subject_image_pil is PIL.Image
        # Cropping and resizing logic for low_res and high_res PIL images (remains similar to original)
        object_image_pil_low_res = [subject_image_pil.resize((384, 384))]
        # ... (high_res cropping as in original)
        object_image_pil_high_res = # ... list of 4 cropped PIL images

        # Preprocess PIL images using the ComfyUI CLIP_VISION object's capabilities
        # The `self.siglip_image_processor` (a CLIP_VISION object) should handle this.
        # It might have a method like `preprocess_pil(pil_list)` or require conversion to tensor first.
        # For pseudocode, assume a helper:
        # `pixel_values = comfy_clip_vision_preprocess_pil(clip_vision_obj, pil_list, target_size)`

        siglip_low_res_pixels = comfy_clip_vision_preprocess_pil(self.siglip_image_processor, object_image_pil_low_res)
        siglip_embeds_tuple = self.encode_siglip_image_emb(siglip_low_res_pixels, device, dtype)

        dinov2_low_res_pixels = comfy_clip_vision_preprocess_pil(self.dinov2_image_processor, object_image_pil_low_res)
        dinov2_embeds_tuple = self.encode_dinov2_image_emb(dinov2_low_res_pixels, device, dtype)

        image_embeds_low_res_deep = torch.cat([siglip_embeds_tuple[0], dinov2_embeds_tuple[0]], dim=2)
        image_embeds_low_res_shallow = torch.cat([siglip_embeds_tuple[1], dinov2_embeds_tuple[1]], dim=2)

        # High-resolution processing
        siglip_high_res_pixels = comfy_clip_vision_preprocess_pil(self.siglip_image_processor, object_image_pil_high_res, batched=True)
        siglip_high_res_embeds_tuple = self.encode_siglip_image_emb(siglip_high_res_pixels, device, dtype)
        # ... (rearrange logic as in original)
        siglip_image_high_res_deep = # ... rearranged

        dinov2_high_res_pixels = comfy_clip_vision_preprocess_pil(self.dinov2_image_processor, object_image_pil_high_res, batched=True)
        dinov2_high_res_embeds_tuple = self.encode_dinov2_image_emb(dinov2_high_res_pixels, device, dtype)
        # ... (rearrange logic as in original)
        dinov2_image_high_res_deep = # ... rearranged

        image_embeds_high_res_deep = torch.cat([siglip_image_high_res_deep, dinov2_image_high_res_deep], dim=2)

        image_embeds_dict = dict(
            image_embeds_low_res_shallow=image_embeds_low_res_shallow,
            image_embeds_low_res_deep=image_embeds_low_res_deep,
            image_embeds_high_res_deep=image_embeds_high_res_deep,
        )
        return image_embeds_dict

    # --- Main __call__ method ---
    # The __call__ method's core denoising loop, prompt encoding, latent preparation, etc.,
    # should largely remain the same as it uses the models initialized in `__init__`.
    # Ensure all member variables like self.scheduler, self.text_encoder, self.tokenizer, etc.,
    # are correctly available.
    @torch.no_grad()
    def __call__(self, prompt, subject_image: Image.Image, # ... other parameters as original
                 # Remove ip_adapter_image, ip_adapter_image_embeds if not used by FLUX custom IP-Adapter
                ):
        # ... (Input checking, device setup, batch_size derivation)
        # ... (Prompt encoding using self.text_encoder, self.tokenizer, etc.)
        # ... (Subject image embedding using self.encode_image_emb)
        # ... (Latent preparation using self.vae, self.scheduler)
        # ... (Timestep preparation using self.scheduler)
        # ... (Denoising loop using self.transformer and self.subject_image_proj_model for IP-Adapter injection)
        # ... (Image decoding using self.vae)
        # ... (Return FluxPipelineOutput or image tuple)
        pass # Detailed logic largely follows original __call__

# Conceptual helper for preprocessing PIL images with ComfyUI CLIPVision objects
def comfy_clip_vision_preprocess_pil(clip_vision_obj, pil_images_list, batched=False):
    # 1. Convert PIL list to a single batched ComfyUI image tensor (B, H, W, C), range 0-1.
    #    (Handle single image vs. list for `batched` logic if needed by underlying processor)
    #    comfy_image_tensor = convert_pil_to_comfy_tensor_batch(pil_images_list)
    # 2. Use the clip_vision_obj's internal preprocessing.
    #    The `clip_vision_obj.encode_image(comfy_image_tensor)` itself calls `clip_preprocess`.
    #    We need the *pixel_values* that go into the vision model, not the final embeddings here.
    #    So, we'd call the `clip_preprocess` function from `comfy.clip_vision` directly.
    #    pixel_values = clip_preprocess(comfy_image_tensor,
    #                                   size=clip_vision_obj.image_size,
    #                                   mean=clip_vision_obj.image_mean,
    #                                   std=clip_vision_obj.image_std)
    # return pixel_values
    pass # Placeholder