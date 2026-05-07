from .py.text_utils import StringNode, StringConcatNode, LZTextPreview, LZStringSanitize, LZStringSelect, LZSaveStringToCSV, LZPromptWeight, LZTagEditor
from .py.log_csv import LZAppendLogToCSV
from .py.prompts import DualCLIPTextEncode, AdvancedPositivePrompt, AdvancedNegativePrompt
from .py.loaders import EZCheckpointLoader, LZSimpleCheckpointLoader, LZLoRAStacker
from .py.latent import PresetEmptyLatentImage
from .py.pipe import LZPipePack, LZPipePackXL, LZPipeUnpack
from .py.sampling import LZKSamplerDecode
from .py.image_io import LZSaveImageAndLog
from .py.merge_recipe import LZMergeRecipeRandom, LZMergeRecipeManual, LZMergeRecipeRandomAdvanced
from .py.xy_plot import LZXYPlot, LZXYPlotSampler, LZXYSampler, LZXYGridOutput
from .py.dynamic_prompt import LZPromptReplaceSingle, LZPromptReplaceMulti, LZPromptReplaceString

NODE_CLASS_MAPPINGS = {
    "StringNode": StringNode,
    "DualCLIPTextEncode": DualCLIPTextEncode,
    "EZCheckpointLoader": EZCheckpointLoader,
    "AdvancedPositivePrompt": AdvancedPositivePrompt,
    "AdvancedNegativePrompt": AdvancedNegativePrompt,
    "PresetEmptyLatentImage": PresetEmptyLatentImage,
    "StringConcatNode": StringConcatNode,
    "LZPipePack": LZPipePack,
    "LZPipePackXL": LZPipePackXL,
    "LZPipeUnpack": LZPipeUnpack,
    "LZKSamplerDecode": LZKSamplerDecode,
    "LZLoRAStacker": LZLoRAStacker,
    "LZTextPreview": LZTextPreview,
    "LZStringSanitize": LZStringSanitize,
    "LZStringSelect": LZStringSelect,
    "LZSaveStringToCSV": LZSaveStringToCSV,
    "LZPromptWeight": LZPromptWeight,
    "LZTagEditor": LZTagEditor,
    "LZAppendLogToCSV": LZAppendLogToCSV,
    "LZSaveImageAndLog": LZSaveImageAndLog,
    "LZSimpleCheckpointLoader": LZSimpleCheckpointLoader,
    "LZMergeRecipeRandom": LZMergeRecipeRandom,
    "LZMergeRecipeManual": LZMergeRecipeManual,
    "LZMergeRecipeRandomAdvanced": LZMergeRecipeRandomAdvanced,
    "LZXYPlot": LZXYPlot,
    "LZXYPlotSampler": LZXYPlotSampler,
    "LZXYSampler": LZXYSampler,
    "LZXYGridOutput": LZXYGridOutput,
    "LZPromptReplaceSingle": LZPromptReplaceSingle,
    "LZPromptReplaceMulti": LZPromptReplaceMulti,
    "LZPromptReplaceString": LZPromptReplaceString,
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
    "LZPipePackXL": "LZ Pipe Pack XL",
    "LZPipeUnpack": "LZ Pipe Unpack",
    "LZKSamplerDecode": "LZ KSampler & Decode",
    "LZLoRAStacker": "LZ LoRA Stacker",
    "LZTextPreview": "LZ Text Preview",
    "LZStringSanitize": "LZ String Sanitize",
    "LZStringSelect": "LZ String Select",
    "LZSaveStringToCSV": "LZ Save String to CSV",
    "LZPromptWeight": "LZ Prompt Weight",
    "LZTagEditor": "LZ Tag Editor",
    "LZAppendLogToCSV": "LZ Append Log to CSV",
    "LZSaveImageAndLog": "LZ Save Image & Log",
    "LZSimpleCheckpointLoader": "LZ Simple Checkpoint Loader",
    "LZMergeRecipeRandom": "LZ Merge Recipe Random",
    "LZMergeRecipeManual": "LZ Merge Recipe Manual",
    "LZMergeRecipeRandomAdvanced": "LZ Merge Recipe Random Advanced",
    "LZXYPlot": "LZ XY Plot",
    "LZXYPlotSampler": "LZ XY Plot & Sampler",
    "LZXYSampler": "LZ XY Sampler",
    "LZXYGridOutput": "LZ XY Grid Output",
    "LZPromptReplaceSingle": "LZ Prompt Replace Single",
    "LZPromptReplaceMulti": "LZ Prompt Replace Multi",
    "LZPromptReplaceString": "LZ Prompt Replace String",
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']