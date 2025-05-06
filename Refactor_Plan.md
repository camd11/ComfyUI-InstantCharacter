Existing Loader Nodes for InstantCharacter Components

InstantCharacter’s pipeline requires four model components: the FLUX diffusion model, two image encoders (CLIP Vision models), and an IP-Adapter. In ComfyUI, each of these can be loaded via dedicated loader nodes (either built-in or from common extensions) that produce the appropriate model objects:

    Base FLUX Model (Diffusion Transformer): The FLUX.1 diffusion model (e.g. FLUX.1-dev) can be loaded as a Model in ComfyUI using the standard Load Checkpoint node. The FLUX weights are distributed as a .safetensors checkpoint (including its text encoders) that ComfyUI’s checkpoint loader can handle
    huggingface.co
    . For example, placing flux1-dev.safetensors in the ComfyUI/models/diffusion_models/ folder allows the Load Checkpoint node to load it and output a MODEL object representing the base diffusion model (analogous to a UNet+VAE+text encoders for FLUX).

    Image Encoder 1 (CLIP Vision): InstantCharacter uses a CLIP Vision model (e.g. google/siglip-so400m-patch14-384) as an image encoder. ComfyUI provides a Load CLIP Vision node that loads a specified CLIP vision model and outputs a CLIP_VISION model object
    blenderneko.github.io
    . This node can be used to load the first image encoder by name (if the model is in models/clip_vision or a known HuggingFace identifier). The output is a CLIP Vision model instance that can encode the reference image into image embeddings.

    Image Encoder 2 (Second Vision Model): The pipeline’s second image encoder (e.g. facebook/dinov2-giant) should likewise be loaded as a CLIP_VISION (or similar) model object. If the second encoder is a standard CLIP model, the same Load CLIP Vision node can load it. In InstantCharacter’s case, the second encoder is a DINOv2 model (not a CLIP-based architecture), which may not be supported by the default CLIP loader. In practice, this can be addressed by either using a custom/extended vision loader (for example, the Advanced Vision Model Loader extension that supports additional vision backbones) or by creating a new loader node for DINOv2. The key is that the second encoder should be loaded as a model object (of type CLIP_VISION or a similar interface) rather than referenced by file path. This ensures it can be passed into the pipeline node just like any other CLIP vision model.

    IP-Adapter Model: For the InstantCharacter IP-Adapter (the .bin file that injects the character’s features), an IP-Adapter model loader node should be used. ComfyUI’s IP-Adapter extensions provide such nodes. For example, the Load IPAdapter Model node from the ComfyUI-IPAdapter extension (or specifically the IPAdapterFluxLoader from InstantX’s FLUX plugin) will load the .bin checkpoint and output an IP-Adapter model object
    runcomfy.com
    . In the FLUX context, the Load IPAdapter Flux Model node takes the IP-Adapter checkpoint (placed in models/ipadapter-flux) and the associated vision encoder info, and returns an ipadapterFlux object representing the loaded adapter
    runcomfy.com
    . This object contains the initialized adapter weights/config and is ready to be applied in the diffusion pipeline. In short, rather than providing a file path for the IP-Adapter, the workflow should include a loader node that produces an IPADAPTER model object.

Each of these loader nodes outputs a proper ComfyUI model datatype (e.g. MODEL, CLIP_VISION, or a custom IP-Adapter type) that can be wired into the InstantCharacter node graph instead of using raw file paths. This modular design is analogous to how ControlNet or IP-Adapter are incorporated in ComfyUI: the base model is loaded separately, the conditioning models (image encoders/adapters) are loaded separately, and then a combination node applies them.
Refactoring Plan for InstantCharacter Nodes

To align InstantCharacter with ComfyUI’s modular pipeline conventions, we will refactor the custom nodes to accept model objects as inputs instead of file path strings. The main changes involve the InstantCharacterLoadModelFromLocal node (and the InstantCharacter generation node) as follows:

1. Introduce Proper Model Inputs:
Replace all string-based inputs (base_model_path, image_encoder_path, image_encoder_2_path, ip_adapter_path) with model-type inputs that correspond to the outputs of the loader nodes described above. In the node definitions, update the INPUT_TYPES to use the appropriate ComfyUI types:

    base_model should expect a MODEL (the loaded FLUX diffusion model)
    huggingface.co
    .

    image_encoder_1 should expect a CLIP_VISION model (first image encoder)
    blenderneko.github.io
    .

    image_encoder_2 should expect a CLIP_VISION model (second image encoder).

    ip_adapter should expect an IP-Adapter model object (e.g. the output of the IP-Adapter loader). For the FLUX IP-Adapter, this might be a custom type (we’ll call it IPADAPTER here) matching the loader’s output
    runcomfy.com
    .

Concretely, in the Python class definitions, the INPUT_TYPES = {"required": {...}} should be changed from ("STRING", ...) entries to the corresponding model types. For example:

INPUT_TYPES = {
    "required": {
        "base_model": ("MODEL",),
        "image_encoder_1": ("CLIP_VISION",),
        "image_encoder_2": ("CLIP_VISION",),
        "ip_adapter": ("IPADAPTER",),  # or the exact type name used by the IP-Adapter loader
        ... (other inputs like prompt, image, etc.)
    }
}

This change means the node will no longer present text boxes for file paths, but instead ports to connect the outputs from the model loader nodes.

2. Utilize Existing Loader Nodes:
With the above input types in place, the InstantCharacter workflow will use the standard loader nodes to supply these inputs. For example, a Checkpoint Loader node can feed the base_model input by loading the FLUX model (placed in the proper folder) into a MODEL
huggingface.co
. Two Load CLIP Vision nodes can feed the image_encoder_1 and image_encoder_2 inputs with the respective vision models
blenderneko.github.io
. An IP-Adapter Model Loader (from a common extension) will provide the ip_adapter input as an IPAdapter model object
runcomfy.com
. The InstantCharacter nodes should not attempt to load files internally – instead, they will receive already-loaded model objects from these upstream nodes. This leverages ComfyUI’s existing model-loading mechanism (e.g. dropdown selection of models in those loader nodes) and avoids hard-coding file paths. It also ensures that if the models are not already in memory, the loader nodes handle downloading or loading from disk (as they do for other pipelines).

3. Refactor InstantCharacterLoadModelFromLocal:
The current InstantCharacterLoadModelFromLocal node, which likely takes file paths and internally loads models, can be refactored or repurposed. One approach is to remove this node entirely in favor of the dedicated loaders as described. Another approach is to transform it into an “assembly” node that accepts the loaded model objects and perhaps combines or verifies them. In either case, its internal code for reading files (via HuggingFace or disk) should be removed. Instead, the node’s execute method can assume that it receives ready-to-use objects – for example, a diffusers Pipeline or UNet model in base_model, CLIP model instances in the encoders, and the IP-Adapter module in ip_adapter. This dramatically simplifies the node logic (no download or file I/O needed at runtime) and defers model management to ComfyUI’s standard process.

If the InstantCharacter pipeline requires an initialization step (similar to pipe.init_adapter(...) in the original code) to attach the IP-Adapter to the base model, that logic can be executed using the passed-in objects. For example, the node can call whatever integration function is needed, using the base_model and ip_adapter objects. The two CLIP vision models can be used to encode the reference image inside the pipeline node or integrated into the adapter as needed. The key point is that the node will use the already loaded weights from inputs instead of loading them by path. (Any auto-download features can be deprecated, since the loader nodes handle model availability.)

4. Update the Generation Node (InstantCharacter Pipeline Node):
Aside from the loading node, the actual generation node (which performs inference) must also be updated to accept these model objects. In some designs, the loading and generation might be combined into one node – but the refactored design favors separation. We will ensure the main InstantCharacter node (which likely takes a reference image, prompt, etc. and produces an output) now includes the model inputs in its INPUT_TYPES. For instance, an InstantCharacterGenerator node would declare required inputs for base_model (MODEL), image_enc1 (CLIP_VISION), image_enc2 (CLIP_VISION), and ip_adapter (IPAdapter model), in addition to the textual prompt, reference image, guidance scales, etc. This way, the outputs from the loader nodes can be directly connected into the generator node in the ComfyUI graph.

5. Pass Models into the Pipeline Logic:
Inside the InstantCharacter generator node’s implementation, use the provided model objects to run the generation. For example, if previously the code did:

pipe = InstantCharacterFluxPipeline.from_pretrained(base_model_path, ...)
pipe.init_adapter(image_encoder_path=..., image_encoder_2_path=..., subject_ip_adapter_path=...)
image = pipe(prompt=..., subject_image=..., ...)

…now it should skip from_pretrained and init_adapter that load from paths. Instead, it can use the base_model input (which could be a Diffusers pipeline or a ComfyUI Model containing the diffusion model and VAE) and directly incorporate the adapter. If using an approach similar to ComfyUI’s IP-Adapter nodes, this may involve merging the adapter into the base model’s conditioning. For instance, the generator node could take the base_model (MODEL) and the ip_adapter object and apply a method to inject the adapter (e.g. patching the model’s attention layers or using an API call if provided by an extension). The two CLIP_VISION encoders can be used to encode the reference image into embeddings (the node can call something like enc1.encode_image(ref_image) and same for enc2) and feed those embeddings to the adapter or model as needed. Since these are all Python objects in memory, the pipeline can use them just like the original HuggingFace pipeline code did, but without any file loading.

6. Output and Integration with Sampler:
Decide what the output of the InstantCharacter node should be after refactoring. There are two possible patterns:

    Option A: The InstantCharacter node itself produces the final image (or latent) output. In this case, it would internally perform the diffusion sampling (given the prompt, model, adapter, etc.) and output an image. This monolithic approach is how the original pipeline code works (calling pipe(...) directly). If we keep this, we simply ensure the node runs the diffusion process using the provided models.

    Option B: The InstantCharacter node produces a combined Model (with the adapter applied) that can be fed into a standard KSampler node for generation. This approach is more in line with ComfyUI’s modular design. For example, the IP-Adapter Flux extension splits the process into loading and applying – the Apply IPAdapter Flux Model node takes a base MODEL and an IPAdapter and outputs a modified MODEL with the adapter integrated
    runcomfy.com
    . That model can then go into a KSampler (along with a prompt and seed) to generate images in the usual way. Adopting this pattern, we could have the InstantCharacter node output a MODEL (the FLUX model with the character’s adapter/encoders applied) as its result instead of an image. The user would then connect this MODEL to a KSampler (or similar sampler node), input the text prompt there, and get the image output.

Recommended: Implement Option B for consistency. It makes the InstantCharacter adapter act like an add-on to the base model, analogous to ControlNet or IP-Adapter usage in ComfyUI. For instance, the refactored InstantCharacterApply node would have outputs: MODEL (the adapted diffusion model). The node’s RETURN_TYPES can be set to return a MODEL type. The documentation for Apply IPAdapter Flux shows that its output is indeed a modified MODEL ready for sampling
runcomfy.com
. We would achieve the same: the output model contains the FLUX diffusion model with the InstantCharacter conditioning ready, so that when it’s used in a sampler, it will generate the character-specific image.

7. Update Node Definitions and UI:
After changing the input/output types as above, the node JSON/UI will automatically reflect that the InstantCharacter node expects model connections instead of text boxes. We should remove any now-unused GUI elements (like file path dropdowns) from the UI_COMPONENTS definitions. Instead, ensure the model inputs are marked as required and properly labeled (e.g. “Base Model”, “Image Encoder 1”, etc.). The node’s category and name can remain in the InstantCharacter section, but now it will plug into the broader ComfyUI graph like other model-modifying nodes.

8. Ensure Compatibility and Testing:
Finally, we must test the refactored nodes in a ComfyUI workflow. A typical usage after refactoring would be:

    Load Checkpoint (select FLUX model) → outputs MODEL (FLUX).

    Load CLIP Vision (select SigLIP model) → outputs CLIP_VISION.

    Load CLIP Vision (select DINO model or alternate loader) → outputs CLIP_VISION.

    Load IP-Adapter Model (select instantcharacter_ip-adapter.bin) → outputs IPAdapter model.

    These four outputs connect into the InstantCharacter Apply/Generate node’s inputs. The reference image and any other controls (strength, etc.) also go into this node.

    The InstantCharacter node outputs a MODEL (with adapter applied). Connect this to a KSampler (or the scheduler of choice) along with the text prompt and seed to produce the final image. The KSampler uses the adapted model to sample, yielding an image of the personalized character.

This refactoring aligns InstantCharacter’s integration with how other ComfyUI pipelines are constructed. We eliminate hard-coded file path handling in favor of reusable loader nodes, making the system more flexible. The user can now manage models via ComfyUI’s model directories and UI (no manual path input needed, avoiding errors). By passing proper model objects between nodes, we ensure the InstantCharacter components behave just like built-in nodes – the FLUX model is a drop-in replacement for a Stable Diffusion UNet model (handled by the checkpoint loader) and the IP-Adapter is applied through a node rather than hidden inside a path string.

Overall, this plan brings InstantCharacter’s nodes up to par with ComfyUI’s design philosophy:

    Modularity: Each model is loaded by a dedicated node and can be swapped/changed without editing the InstantCharacter node itself.

    Clarity: The node graph will explicitly show the Base Model, both Image Encoders, and the IP-Adapter feeding into the InstantCharacter pipeline, which is easier to understand and debug.

    Reusability: If in the future other pipelines want to use the FLUX model or the image encoders, those are available as separate nodes. Likewise, the InstantCharacter node could potentially accept different image encoders or adapters if needed, since it’s not locked to specific file paths.

By implementing these changes in the node code (adjusting INPUT_TYPES/RETURN_TYPES and removing internal loading logic), we ensure that InstantCharacter can be seamlessly used within ComfyUI with no string or path inputs required, fully leveraging the existing model loader nodes for UNet, CLIP Vision, and IP-Adapter models
blenderneko.github.io
runcomfy.com
. This refactoring will make the InstantCharacter extension more robust and consistent with the rest of the ComfyUI ecosystem.

Implementing apply_instant_character in a ComfyUI Node

Implementing the InstantCharacterApply node involves integrating a reference image's conditioning (via an IP-Adapter) into a diffusion model (FLUX/DiT or UNet). Below we address each sub-question with guidance and code snippets.
1. Accessing the Underlying Transformer (UNet/DiT) from the MODEL

In ComfyUI, the MODEL object (output of a CheckpointLoader) is typically a wrapper that holds the actual diffusion model (UNet or transformer). You should safely retrieve the inner model. Usually this is available as an attribute (commonly .model). For example, custom nodes often do:

# Assume base_model is the MODEL object input
if hasattr(base_model, "model"):
    diff_model = base_model.model 
else:
    diff_model = base_model  # base_model might already be the nn.Module

This pattern ensures you get the actual torch.nn.Module (the UNet or DiT) without errors
github.com
. For instance, one ComfyUI extension uses base_model = model.model to get the UNet, then operates on it
github.com
. Always check with hasattr to avoid exceptions if the structure differs.
2. Structure of the IPADAPTER Object (Projector & Weights)

The IP-Adapter model (loaded from a .bin) is usually provided as a single IPADAPTER object. Internally, this contains two parts:

    Projector module – a neural network (often a “resampler” or cross-attention projector). In the InstantCharacter context, this is the CrossLayerCrossScaleProjector used to process image features
    huggingface.co
    . In other IPAdapter implementations, it might be called ImageProjModel or similar. This module takes the encoded image tokens and produces a conditioning vector/tokens for the UNet’s attention layers.

    Attention processor weights – a set of weights (matrices) that modify the cross-attention layers (often called to_k_ip, to_v_ip, etc.). These are analogous to LoRA weights applied to the diffusion model’s attention. They inject the image prompt information into the keys/values of cross-attention.

In ComfyUI-IPAdapter-Plus (and the FLUX IP-Adapter extension), the loader loads a state dict with keys like "ip_adapter" and "image_proj". For example, Tencent’s IP-Adapter saving splits the checkpoint into {"ip_adapter": ip_sd, "image_proj": proj_sd}
github.com
github.com
. The ComfyUI loader likely returns an object (or dict) where you can access these components.

Accessing them: If the ip_adapter object is a dictionary, you might do:

proj_state = ip_adapter["image_proj"]    # state dict for projector
attn_state = ip_adapter["ip_adapter"]    # state dict for attention weights

If the loader already constructed the projector model, it may instead provide an object with attributes, e.g. ip_adapter.projector (the CrossLayerCrossScaleProjector instance) and perhaps an internal dict of attention weights. In that case, use those attributes directly (e.g. ip_adapter.projector) to get the module, and something like ip_adapter.attn_weights or a state dict property for the weights.

Example: In a FLUX pipeline, after loading, they instantiate the projector and load its weights, and load the adapter weights into custom attention processors
huggingface.co
huggingface.co
. Our node will mirror that: use the provided projector (or create it) and apply the weights for to_q/k/v etc. per layer.
3. Encoding the Subject Image with CLIP Vision Encoders

The node receives two CLIP vision encoders (e.g. one SigLIP model and one DINOv2 model). We must feed the subject_image into both to get image embeddings. Key steps:

    Prepare the image tensor: ComfyUI’s IMAGE input is usually a PyTorch tensor (shape [C,H,W] or [1,C,H,W]) in 0–1 float range. Ensure it’s 4D [B,C,H,W] (batch dimension) and move it to the same device/dtype as the encoders. For example:

img = subject_image
if isinstance(img, PIL.Image.Image):
    img = transforms.ToTensor()(img)  # convert PIL to tensor [C,H,W]
if img.dim() == 3:
    img = img.unsqueeze(0)  # add batch dimension
img = img.to(device=encoder.device, dtype=encoder.dtype)

Apply the appropriate preprocessing: CLIP vision models expect normalized pixels. If the ComfyUI CLIP encoders expose an encode_image method (likely they do), it will handle preprocessing internally. For example, the IPAdapter Plus code calls clip_embed = clip_vision.encode_image(image)
stackoverflow.com
– here clip_vision is the CLIP_VISION model and image is the tensor. This suggests you can call image_encoder_1.encode_image(img_tensor) directly to get an embedding
stackoverflow.com
. If such a method isn't available, use the transformer’s processor: e.g., for SigLIP use SiglipImageProcessor (as in InstantCharacter pipeline) to resize/crop to 384x384 and normalize
huggingface.co
, then feed into the model’s forward.

Obtain embeddings: After encoding, you’ll get an output (probably a model output object or tensor). For HuggingFace CLIP models, .encode_image() might return a pooled feature or you may need model(img_tensor).last_hidden_state. The InstantCharacter pipeline, for instance, calls the SigLIP model with output_hidden_states=True and presumably takes the final hidden states
huggingface.co
. We likely want the last hidden state or pooled output as the image representation. For simplicity, you can use the final hidden states of each encoder and (if needed) remove the CLS token.

Combine features: The projector expects a joint image feature. The InstantCharacter pipeline concatenates features from SigLIP and DINOv2 along the channel dimension (SigLIP output 1152-dim and DINOv2 output 1536-dim, totaling 2688)
huggingface.co
huggingface.co
. We can do:

    feats1 = encoder1.encode_image(img)   # e.g. shape [1, N, 1152]
    feats2 = encoder2.encode_image(img)   # e.g. shape [1, M, 1536]
    # Align token count N vs M (drop CLS token if present)
    if feats1.shape[1] == feats2.shape[1] + 1:  # one has extra token
        feats1 = feats1[:, 1:]  # drop first token
    elif feats2.shape[1] == feats1.shape[1] + 1:
        feats2 = feats2[:, 1:]
    image_tokens = torch.cat([feats1, feats2], dim=-1)  # concat channels -> [B, T, 2688]

    Now image_tokens is the combined image embedding sequence.

4. Injecting Conditioning into the Transformer During Sampling

To use the image conditioning, we must modify the diffusion model so that during sampling (KSampler) the model’s cross-attention layers incorporate the image. ComfyUI’s standard practice is to attach the conditioning to the model and/or use custom attention processors:

    Using custom attention processors: The FLUX model (DiT) supports the HuggingFace AttnProcessor mechanism. InstantCharacter introduces a FluxIPAttnProcessor class that replaces the standard Cross-Attention behavior
    huggingface.co
    . We will create and set these processors on the model. Each processor will hold learnable weights (to_k_ip, to_v_ip, etc.) and will expect the image embedding as an extra input. In diffusers, one can pass joint_attention_kwargs when calling the model, which gets forwarded to the attention processors
    github.com
    huggingface.co
    . In ComfyUI, the KSampler (especially for FLUX models) likely passes this automatically if the model has AttnProcessor set. Our job is to load the IP-Adapter weights into these processors.

    Setting up joint conditioning: For the processors to know about the image, we have two options:

        Via arguments: If the model is a diffusers Transformer (FLUX), we can call it with joint_attention_kwargs={'image_emb': image_cond} (or similar) so that the processors receive image_emb each step. ComfyUI’s KSampler Advanced node has provisions for “joint conditioning” which aligns with this approach. (In code, the FLUX pipeline does transformer(attention_kwargs=...)). Ensure to attach your image embedding to the model or pipeline so it can be passed.

        Via model attributes: Another approach is to store the processed image conditioning in the model itself (e.g., set model.ip_attention_embeds = image_cond). If you use a custom patched forward or a monkey-patched CrossAttention, it can retrieve this attribute. The IPAdapter Plus extension, for example, patches the CrossAttention to use the stored image tokens if available (instead of requiring changes in the sampler).

ComfyUI convention: When possible, prefer using the built-in conditioning pathways. For FLUX models, attaching through joint_attention_kwargs is ideal, as diffusers intended
github.com
. If implementing for a standard SD UNet (which doesn’t natively support joint attention), you might have to manually patch the cross-attention forward pass (replacing it with a function that adds the image-based keys/values). In summary, yes, you can attach custom data to the MODEL object – either by setting a property or using the model’s own hooks – and configure the KSampler to use it. (For example, you might attach base_model.image_cond = image_cond and ensure the sampler calls model with that context.)
5. Cloning the Base Model Before Modification

It’s recommended to clone the base_model before applying modifications like IP-Adapter, especially if the original model might be used elsewhere in the graph. This prevents side-effects (e.g., applying IP-Adapter weights to a model that another branch expects to be unmodified). In ComfyUI, many “apply” nodes do clone under the hood or advise it.

How to clone: A direct approach is using Python’s copy.deepcopy on the model object. PyTorch modules don’t have a built-in .clone() for full models, so deepcopy is the standard way (which duplicates all weights)
discuss.pytorch.org
. For example:

import copy
new_model = copy.deepcopy(base_model)  # deep copy the MODEL object and its .model

Be mindful that this will use additional VRAM equal to the model size. If you know the base_model won’t be reused, you could skip cloning and modify in-place. However, to be safe, cloning is good practice in custom nodes to avoid unexpected interactions. (Some ComfyUI official nodes may not clone to save memory, but in a custom node it's safer to do so explicitly.)
6. Implementation Code Snippets

Below is a sketch of how the apply_instant_character could be implemented, incorporating the above points:

def apply_instant_character(self, base_model, image_encoder_1, image_encoder_2, ip_adapter, subject_image=None):
    # 1. Retrieve the underlying model (UNet or DiT)
    model = base_model.model if hasattr(base_model, "model") else base_model
    # Optionally, clone the model to avoid side-effects
    model = copy.deepcopy(model)
    
    # 2. Get IP-Adapter weights and projector
    if isinstance(ip_adapter, dict):
        proj_weights = ip_adapter.get("image_proj")
        attn_weights = ip_adapter.get("ip_adapter")
    else:
        # e.g., ip_adapter might be a custom object
        proj_module = getattr(ip_adapter, "projector", None)
        attn_weights = getattr(ip_adapter, "attn_weights", None) or \
                       getattr(ip_adapter, "state_dict", lambda: {})().get("ip_adapter")
        proj_weights = getattr(ip_adapter, "state_dict", lambda: {})().get("image_proj")
    # If the projector module is not already built, instantiate it:
    if proj_module is None:
        # Determine dimensions (for FLUX, these may be known constants or derivable from model config)
        # For example, using InstantCharacter defaults:contentReference[oaicite:18]{index=18}:
        inner_dim = attn_weights[next(iter(attn_weights))].shape[1] if attn_weights else (1152+1536)
        proj_module = CrossLayerCrossScaleProjector(
            inner_dim=inner_dim, num_attention_heads=42, attention_head_dim=64,
            cross_attention_dim=inner_dim, num_layers=4, dim=1280, depth=4,
            dim_head=64, heads=20, num_queries=128, embedding_dim=inner_dim,
            output_dim=4096, ff_mult=4, 
            timestep_in_dim=getattr(model, 'config', {}).get('block_out_channels', [320])[0],
            timestep_flip_sin_to_cos=True, timestep_freq_shift=0
        )
    if proj_weights:
        proj_module.load_state_dict(proj_weights, strict=False)
    proj_module = proj_module.to(device=model.device, dtype=model.dtype).eval()
    
    # 3. Encode the subject image with the two encoders (if provided)
    image_cond = None
    if subject_image is not None:
        # Ensure tensor format
        img = subject_image
        if hasattr(image_encoder_1, "encode_image"):
            # Use the model's encode function (handles preprocessing)
            emb1 = image_encoder_1.encode_image(img)        # :contentReference[oaicite:19]{index=19}
        else:
            emb1 = image_encoder_1(img)  # assume it returns a ModelOutput
            emb1 = getattr(emb1, "last_hidden_state", emb1)
        if hasattr(image_encoder_2, "encode_image"):
            emb2 = image_encoder_2.encode_image(img)
        else:
            emb2 = image_encoder_2(img)
            emb2 = getattr(emb2, "last_hidden_state", emb2)
        # Concatenate features
        if emb1 is not None and emb2 is not None:
            # Remove CLS tokens if present
            if emb1.shape[1] == emb2.shape[1] + 1: 
                emb1 = emb1[:, 1:]
            elif emb2.shape[1] == emb1.shape[1] + 1:
                emb2 = emb2[:, 1:]
            image_tokens = torch.cat([emb1, emb2], dim=-1)  # concat channel dim
            # Project the concatenated tokens to conditioning
            image_cond = proj_module(image_tokens)  # shape [B, some_length, 4096] or similar
        else:
            image_cond = None
    
    # 4. Set up custom attention processors on the model
    if hasattr(model, "set_attn_processor"):
        # Replace all attention processors with FluxIPAttnProcessor and load weights
        attn_procs = {}
        for name, proc in model.attn_processors.items():
            # Determine hidden sizes
            hidden_size = getattr(model, "hidden_size", None)
            if hidden_size is None:
                # e.g., compute from existing attn module if available
                hidden_size = proc.to_q.in_features if hasattr(proc, "to_q") else model.config.cross_attention_dim
            ip_dim = image_cond.shape[-1] if image_cond is not None else proj_module.output_dim
            attn_procs[name] = FluxIPAttnProcessor(hidden_size, ip_hidden_states_dim=ip_dim).to(model.device)
        model.set_attn_processor(attn_procs)
        if attn_weights:
            # Load the IP-Adapter weights into the new processors
            model.attn_processors.load_state_dict(attn_weights, strict=False):contentReference[oaicite:20]{index=20}
    else:
        # Fallback: patch cross-attention modules manually for non-transformer models
        patch_cross_attention_layers(model, attn_weights, image_cond)
    
    # 5. Attach the image conditioning to the model for use during sampling
    model.ip_adapter_cond = image_cond  # an attribute that attn processors can use
    # Alternatively, ensure KSampler passes joint_attention_kwargs:
    base_model.model = model  # put the modified model back if base_model was a wrapper
    return base_model

In this pseudo-code:

    We access base_model.model safely and clone it.

    We extract the IP-Adapter weights (attn_weights) and instantiate or retrieve the projector module, loading its weights
    github.com
    github.com
    .

    We preprocess and encode the image using encode_image (or model forward) for each vision encoder, then concatenate their outputs
    huggingface.co
    huggingface.co
    . This combined tensor is fed through the projector to get the final image conditioning tokens.

    We replace the model’s attention processors with FluxIPAttnProcessor instances (for each attention layer) and load the adapter weights into them
    huggingface.co
    huggingface.co
    . If the model doesn’t support attn_processors (e.g., an SD1.5 UNet), you'd manually integrate the weights by modifying the CrossAttention layers (not shown above for brevity).

    Finally, we attach the computed image_cond to the model (as an attribute or via the processors). In a diffusers-style DiT, the FluxIPAttnProcessor will automatically use joint_attention_kwargs if provided – ComfyUI’s KSampler (for FLUX) can pass this under the hood
    github.com
    . If not, our storing of model.ip_adapter_cond can be utilized by a patched forward.

The result is a modified MODEL ready for sampling. You can plug this output into a standard KSampler (or equivalent) node. During sampling, the model’s attention layers will incorporate the image prompt conditioning, guiding the generation consistent with the reference image.