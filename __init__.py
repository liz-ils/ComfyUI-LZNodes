from .py.text_utils import StringNode, StringConcatNode, LZTextPreview
from .py.prompts import DualCLIPTextEncode, AdvancedPositivePrompt, AdvancedNegativePrompt
from .py.loaders import EZCheckpointLoader, LZSimpleCheckpointLoader, LZLoRAStacker
from .py.latent import PresetEmptyLatentImage
from .py.pipe import LZPipePack, LZPipeUnpack
from .py.sampling import LZKSamplerDecode
from .py.image_io import LZSaveImageAndLog

NODE_CLASS_MAPPINGS = {
    "StringNode": StringNode,
    "DualCLIPTextEncode": DualCLIPTextEncode,
    "EZCheckpointLoader": EZCheckpointLoader,
    "AdvancedPositivePrompt": AdvancedPositivePrompt,
    "AdvancedNegativePrompt": AdvancedNegativePrompt,
    "PresetEmptyLatentImage": PresetEmptyLatentImage,
    "StringConcatNode": StringConcatNode,
    "LZPipePack": LZPipePack,
    "LZPipeUnpack": LZPipeUnpack,
    "LZKSamplerDecode": LZKSamplerDecode,
    "LZLoRAStacker": LZLoRAStacker,
    "LZTextPreview": LZTextPreview,
    "LZSaveImageAndLog": LZSaveImageAndLog,
    "LZSimpleCheckpointLoader": LZSimpleCheckpointLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringNode": "LZ String Output",
    "DualCLIPTextEncode": "LZ Dual CLIP Text Encode",
    "EZCheckpointLoader": "LZ Checkpoint Loader & Encode",
    "AdvancedPositivePrompt": "LZ Advanced Positive Prompt",
    "AdvancedNegativePrompt": "LZ Advanced Negative Prompt",
    "PresetEmptyLatentImage": "LZ Preset Empty Latent Image",
    "StringConcatNode": "LZ String Concat",
    "LZPipePack": "LZ Pipe Pack",
    "LZPipeUnpack": "LZ Pipe Unpack",
    "LZKSamplerDecode": "LZ KSampler & Decode",
    "LZLoRAStacker": "LZ LoRA Stacker",
    "LZTextPreview": "LZ Text Preview",
    "LZSaveImageAndLog": "LZ Save Image & Log",
    "LZSimpleCheckpointLoader": "LZ Simple Checkpoint Loader",
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']