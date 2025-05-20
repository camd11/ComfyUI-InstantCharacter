import unittest
from unittest.mock import patch, MagicMock, ANY
import os
import sys

# Mock torch at the global level for this test file *before* any other imports
# that might depend on torch.
MOCK_TORCH = MagicMock()
MOCK_TORCH.bfloat16 = "torch.bfloat16_mocked_for_test"
MOCK_TORCH.cuda.is_available.return_value = False
MOCK_TORCH.Generator.return_value.manual_seed.return_value = "mocked_generator"
MOCK_TORCH.Tensor = MagicMock
MOCK_TORCH.from_numpy = MagicMock(return_value=MagicMock(unsqueeze=MagicMock(return_value="mocked_tensor_final")))
MOCK_TORCH_NN_FUNCTIONAL = MagicMock() # For torch.nn.functional
MOCK_TORCH_NN = MagicMock()
MOCK_TORCH_NN.functional = MOCK_TORCH_NN_FUNCTIONAL # For from torch.nn import functional
MOCK_TORCH.nn = MOCK_TORCH_NN

# Mock folder_paths
MOCK_FOLDER_PATHS = MagicMock()
MOCK_FOLDER_PATHS.folder_names_and_paths = {
    "ipadapter": (["fake/ipadapter/path"], [".bin", ".pt", ".safetensors"])
}
MOCK_FOLDER_PATHS.supported_pt_extensions = [".bin", ".pt", ".safetensors"]
MOCK_FOLDER_PATHS.get_filename_list.return_value = ["mocked_ip_adapter.bin"]
MOCK_FOLDER_PATHS.get_folder_paths.return_value = ["mocked_cache_dir/"]
MOCK_FOLDER_PATHS.get_full_path.return_value = "mocked_cache_dir/mocked_ip_adapter.bin"

# Mock numpy
MOCK_NUMPY = MagicMock()
MOCK_NUMPY.array.return_value = MagicMock(astype=MagicMock(return_value="mocked_numpy_array"))
MOCK_NUMPY.uint8 = "numpy.uint8_mocked"

# Mock einops
MOCK_EINOPS = MagicMock()

# Mock diffusers and its potential submodules
# Mock diffusers and its potential submodules more deeply
# Mock diffusers and its potential submodules more deeply
MOCK_FLUX_PIPELINE_CLASS = MagicMock() # This will be the mock for the FluxPipeline class itself
MOCK_DIFFUSERS_PIPELINES_FLUX_PIPELINE_FLUX = MagicMock()
MOCK_DIFFUSERS_PIPELINES_FLUX_PIPELINE_FLUX.FluxPipeline = MOCK_FLUX_PIPELINE_CLASS # For 'from ... import FluxPipeline'
# Also handle 'from ... import *' by making common names available if needed, e.g.
MOCK_DIFFUSERS_PIPELINES_FLUX_PIPELINE_FLUX.EXAMPLE_DOC_STRING = "mocked_doc_string"
MOCK_DIFFUSERS_PIPELINES_FLUX_PIPELINE_FLUX.replace_example_docstring = MagicMock()
MOCK_DIFFUSERS_PIPELINES_FLUX_PIPELINE_FLUX.PipelineImageInput = MagicMock()
MOCK_DIFFUSERS_PIPELINES_FLUX_PIPELINE_FLUX.FluxPipelineOutput = MagicMock()
MOCK_DIFFUSERS_PIPELINES_FLUX_PIPELINE_FLUX.calculate_shift = MagicMock()
MOCK_DIFFUSERS_PIPELINES_FLUX_PIPELINE_FLUX.retrieve_timesteps = MagicMock()
MOCK_DIFFUSERS_PIPELINES_FLUX_PIPELINE_FLUX.XLA_AVAILABLE = False


MOCK_DIFFUSERS_PIPELINES_FLUX = MagicMock()
MOCK_DIFFUSERS_PIPELINES_FLUX.pipeline_flux = MOCK_DIFFUSERS_PIPELINES_FLUX_PIPELINE_FLUX

MOCK_DIFFUSERS_PIPELINES = MagicMock()
MOCK_DIFFUSERS_PIPELINES.flux = MOCK_DIFFUSERS_PIPELINES_FLUX

MOCK_DIFFUSERS_MODELS_EMBEDDINGS = MagicMock()
MOCK_DIFFUSERS_MODELS_TRANSFORMERS_TRANSFORMER_2D = MagicMock() # For diffusers.models.transformers.transformer_2d
MOCK_DIFFUSERS_MODELS_TRANSFORMERS = MagicMock()
MOCK_DIFFUSERS_MODELS_TRANSFORMERS.transformer_2d = MOCK_DIFFUSERS_MODELS_TRANSFORMERS_TRANSFORMER_2D # For from diffusers.models.transformers import transformer_2d
MOCK_DIFFUSERS_MODELS_ATTENTION_PROCESSOR = MagicMock()
MOCK_DIFFUSERS_MODELS = MagicMock()
MOCK_DIFFUSERS_MODELS.attention_processor = MOCK_DIFFUSERS_MODELS_ATTENTION_PROCESSOR
MOCK_DIFFUSERS_MODELS.embeddings = MOCK_DIFFUSERS_MODELS_EMBEDDINGS
MOCK_DIFFUSERS_MODELS.transformers = MOCK_DIFFUSERS_MODELS_TRANSFORMERS

MOCK_DIFFUSERS_SCHEDULERS = MagicMock()
MOCK_DIFFUSERS_UTILS = MagicMock()
MOCK_DIFFUSERS_UTILS.logging = MagicMock()

MOCK_DIFFUSERS = MagicMock()
MOCK_DIFFUSERS.pipelines = MOCK_DIFFUSERS_PIPELINES
MOCK_DIFFUSERS.models = MOCK_DIFFUSERS_MODELS
MOCK_DIFFUSERS.schedulers = MOCK_DIFFUSERS_SCHEDULERS
MOCK_DIFFUSERS.utils = MOCK_DIFFUSERS_UTILS
MOCK_DIFFUSERS.FluxTransformer2DModel = MagicMock()
MOCK_DIFFUSERS.AutoencoderKL = MagicMock()

# Mock timm
MOCK_TIMM_MODELS_VISION_TRANSFORMER = MagicMock() # For timm.models.vision_transformer
MOCK_TIMM_MODELS = MagicMock()
MOCK_TIMM_MODELS.vision_transformer = MOCK_TIMM_MODELS_VISION_TRANSFORMER # For from timm.models import vision_transformer
MOCK_TIMM = MagicMock()
MOCK_TIMM.models = MOCK_TIMM_MODELS

# Mock safetensors
MOCK_SAFETENSORS_TORCH = MagicMock() # For safetensors.torch
MOCK_SAFETENSORS = MagicMock()
MOCK_SAFETENSORS.torch = MOCK_SAFETENSORS_TORCH

# Mock tqdm
MOCK_TQDM = MagicMock()
MOCK_TQDM.tqdm = MagicMock(return_value=iter([])) # Mock tqdm.tqdm to be an iterable

# Mock transformers
MOCK_TRANSFORMERS = MagicMock()
MOCK_TRANSFORMERS_CLIP_IMAGE_PROCESSOR = MagicMock()
MOCK_TRANSFORMERS_AUTO_TOKENIZER = MagicMock()
MOCK_TRANSFORMERS.CLIPImageProcessor = MOCK_TRANSFORMERS_CLIP_IMAGE_PROCESSOR
MOCK_TRANSFORMERS.AutoTokenizer = MOCK_TRANSFORMERS_AUTO_TOKENIZER

# Mock typing for type hints if not available in test environment
MOCK_TYPING = MagicMock()
MOCK_TYPING.Union = MagicMock()
MOCK_TYPING.List = MagicMock()
MOCK_TYPING.Optional = MagicMock()
MOCK_TYPING.Dict = MagicMock()
MOCK_TYPING.Callable = MagicMock()
MOCK_TYPING.Any = MagicMock()

# Mock huggingface_hub
MOCK_HF_HUB = MagicMock()
MOCK_HF_HUB.login = MagicMock()


# Apply sys.modules patches for all critical missing modules *before* importing application code
with patch.dict(sys.modules, {
    'torch': MOCK_TORCH,
    'folder_paths': MOCK_FOLDER_PATHS,
    'numpy': MOCK_NUMPY,
    'einops': MOCK_EINOPS,
    'diffusers': MOCK_DIFFUSERS,
    'diffusers.pipelines': MOCK_DIFFUSERS_PIPELINES,
    'diffusers.pipelines.flux': MOCK_DIFFUSERS_PIPELINES_FLUX,
    'diffusers.pipelines.flux.pipeline_flux': MOCK_DIFFUSERS_PIPELINES_FLUX_PIPELINE_FLUX,
    'diffusers.models': MOCK_DIFFUSERS_MODELS,
    'diffusers.models.attention_processor': MOCK_DIFFUSERS_MODELS_ATTENTION_PROCESSOR,
    'diffusers.models.embeddings': MOCK_DIFFUSERS_MODELS_EMBEDDINGS,
    'diffusers.models.transformers': MOCK_DIFFUSERS_MODELS_TRANSFORMERS,
    'diffusers.models.transformers.transformer_2d': MOCK_DIFFUSERS_MODELS_TRANSFORMERS_TRANSFORMER_2D,
    'diffusers.schedulers': MOCK_DIFFUSERS_SCHEDULERS,
    'diffusers.utils': MOCK_DIFFUSERS_UTILS,
    'diffusers.utils.logging': MOCK_DIFFUSERS_UTILS.logging,
    'transformers': MOCK_TRANSFORMERS,
    'transformers.CLIPImageProcessor': MOCK_TRANSFORMERS_CLIP_IMAGE_PROCESSOR,
    'transformers.AutoTokenizer': MOCK_TRANSFORMERS_AUTO_TOKENIZER,
    'torch.nn': MOCK_TORCH_NN,
    'torch.nn.functional': MOCK_TORCH_NN_FUNCTIONAL,
    'timm': MOCK_TIMM,
    'timm.models': MOCK_TIMM_MODELS,
    'timm.models.vision_transformer': MOCK_TIMM_MODELS_VISION_TRANSFORMER,
    'safetensors': MOCK_SAFETENSORS,
    'safetensors.torch': MOCK_SAFETENSORS_TORCH,
    'tqdm': MOCK_TQDM,
    'tqdm.auto': MOCK_TQDM,
    'typing': MOCK_TYPING,
    'huggingface_hub': MOCK_HF_HUB, # For from huggingface_hub import login
}):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        import nodes.comfy_nodes # Ensure the module is loaded
        from nodes.comfy_nodes import InstantCharacterLoadModelFromLocal, InstantCharacterGenerate
    except ImportError as e:
        print(f"CRITICAL: Error importing node classes for testing: {e}")
        class InstantCharacterLoadModelFromLocal:
            def load_model(self, *args, **kwargs): raise ImportError(f"Dummy class used due to: {e}")
        class InstantCharacterGenerate:
            def generate(self, *args, **kwargs): raise ImportError(f"Dummy class used due to: {e}")


@patch('nodes.comfy_nodes.torch', MOCK_TORCH)
@patch('nodes.comfy_nodes.folder_paths', MOCK_FOLDER_PATHS)
@patch('nodes.comfy_nodes.np', MOCK_NUMPY)
class TestInstantCharacterLoadModelFromLocal(unittest.TestCase):

    @patch('nodes.comfy_nodes.InstantCharacterFluxPipeline')
    @patch('nodes.comfy_nodes.os.path.exists')
    def test_load_model_valid_local_paths(self, mock_os_path_exists, MockPipeline):
        mock_os_path_exists.return_value = True

        # Configure the mock pipeline instance returned by from_pretrained
        mock_pipe_instance = MockPipeline.from_pretrained.return_value
        
        # Mock the 'to' method and 'enable_sequential_cpu_offload' on the instance
        mock_pipe_instance.to = MagicMock()
        mock_pipe_instance.init_adapter = MagicMock()
        mock_pipe_instance.enable_sequential_cpu_offload = MagicMock()

        node = InstantCharacterLoadModelFromLocal()
        
        base_model_path = "fake/path/base_model"
        image_encoder_path = "fake/path/encoder"
        image_encoder_2_path = "fake/path/encoder2"
        ip_adapter_path = "fake/path/ip_adapter.bin"
        cpu_offload = False

        # Execute the node's main function
        # The comma is important if load_model returns a tuple e.g. (pipe,)
        result_pipe_tuple = node.load_model(
            base_model_path,
            image_encoder_path,
            image_encoder_2_path,
            ip_adapter_path,
            cpu_offload
        )
        result_pipe = result_pipe_tuple[0]


        # Verify InstantCharacterFluxPipeline.from_pretrained was called correctly
        MockPipeline.from_pretrained.assert_called_once_with(
            base_model_path,
            torch_dtype=ANY, # Using ANY because torch.bfloat16 might be tricky to assert directly depending on environment
            local_files_only=True
        )

        # Verify pipe.init_adapter was called correctly
        mock_pipe_instance.init_adapter.assert_called_once_with(
            image_encoder_path=image_encoder_path,
            image_encoder_2_path=image_encoder_2_path,
            subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024),
            local_files_only=True # As per spec, assuming init_adapter also supports this
        )
        
        # Verify the returned pipe is the one from from_pretrained
        self.assertEqual(result_pipe, mock_pipe_instance)
        
        # Verify it was moved to device (or offload enabled if cpu_offload was True)
        if not cpu_offload:
            mock_pipe_instance.to.assert_called_once_with(ANY) # "cuda" or "cpu"
        else:
            mock_pipe_instance.enable_sequential_cpu_offload.assert_called_once()

if __name__ == '__main__':
    # This allows running the tests directly from the command line
    # Create a 'tests' directory if it doesn't exist for test discovery
    if not os.path.exists("tests"):
        os.makedirs("tests")
    unittest.main()