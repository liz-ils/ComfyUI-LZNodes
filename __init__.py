from .py.text_utils import StringNode, StringConcatNode, LZTextPreview, LZStringSanitize, LZStringSelect, LZSaveStringToCSV, LZPromptWeight, LZTagEditor
from .py.log_csv import LZAppendLogToCSV
from .py.prompts import DualCLIPTextEncode, AdvancedPositivePrompt, AdvancedNegativePrompt, LZCLIPTextEncode
from .py.loaders import EZCheckpointLoader, LZSimpleCheckpointLoader, LZLoRAStacker, LZLoRALoaderModelOnly
from .py.anima_loader import LZAnimaLoader
from .py.krea2_loader import LZKrea2Loader
from .py.anima_artist_mixer import LZAnimaArtistPack, LZAnimaArtistOptions, LZAnimaArtistCrossAttn, LZAnimaArtistNode
from .py.latent import PresetEmptyLatentImage
from .py.pipe import LZPipePack, LZPipePackXL, LZPipeUnpack
from .py.sampling import LZKSamplerDecode
from .py.image_io import LZSaveImageAndLog
from .py.merge_recipe import LZMergeRecipeRandom, LZMergeRecipeManual, LZMergeRecipeRandomAdvanced
from .py.xy_plot import LZXYPlot, LZXYPlotSampler, LZXYSampler, LZXYGridOutput
from .py.dynamic_prompt import LZPromptReplaceSingle, LZPromptReplaceMulti, LZPromptReplaceString
from .py.pipe_info import LZPipeInfo
from .py.log_reader import LZLogReader
from .py.pipe_merge import LZPipeMerge
from .py.batch_save import LZBatchSaveWithLabels

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
    "LZCLIPTextEncode": LZCLIPTextEncode,
    "LZLoRALoaderModelOnly": LZLoRALoaderModelOnly,
    "LZTextPreview": LZTextPreview,
    "LZStringSanitize": LZStringSanitize,
    "LZStringSelect": LZStringSelect,
    "LZSaveStringToCSV": LZSaveStringToCSV,
    "LZPromptWeight": LZPromptWeight,
    "LZTagEditor": LZTagEditor,
    "LZAppendLogToCSV": LZAppendLogToCSV,
    "LZSaveImageAndLog": LZSaveImageAndLog,
    "LZSimpleCheckpointLoader": LZSimpleCheckpointLoader,
    "LZAnimaLoader": LZAnimaLoader,
    "LZKrea2Loader": LZKrea2Loader,
    "LZAnimaArtistPack": LZAnimaArtistPack,
    "LZAnimaArtistOptions": LZAnimaArtistOptions,
    "LZAnimaArtistCrossAttn": LZAnimaArtistCrossAttn,
    "LZAnimaArtistNode": LZAnimaArtistNode,
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
    "LZPipeInfo": LZPipeInfo,
    "LZLogReader": LZLogReader,
    "LZPipeMerge": LZPipeMerge,
    "LZBatchSaveWithLabels": LZBatchSaveWithLabels,
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
    "LZCLIPTextEncode": "LZ CLIP Text Encode",
    "LZLoRALoaderModelOnly": "LZ LoRA Loader (Model Only)",
    "LZTextPreview": "LZ Text Preview",
    "LZStringSanitize": "LZ String Sanitize",
    "LZStringSelect": "LZ String Select",
    "LZSaveStringToCSV": "LZ Save String to CSV",
    "LZPromptWeight": "LZ Prompt Weight",
    "LZTagEditor": "LZ Tag Editor",
    "LZAppendLogToCSV": "LZ Append Log to CSV",
    "LZSaveImageAndLog": "LZ Save Image & Log",
    "LZSimpleCheckpointLoader": "LZ Simple Checkpoint Loader",
    "LZAnimaLoader": "LZ Anima Loader",
    "LZKrea2Loader": "LZ Krea2 Loader",
    "LZAnimaArtistPack": "LZ Anima Artist Pack (Split + Encode)",
    "LZAnimaArtistOptions": "LZ Anima Artist Options (Advanced)",
    "LZAnimaArtistCrossAttn": "LZ Anima Artist Cross-Attn (v2)",
    "LZAnimaArtistNode": "LZ Anima Artist Node",
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
    "LZPipeInfo": "LZ Pipe Info",
    "LZLogReader": "LZ Log Reader",
    "LZPipeMerge": "LZ Pipe Merge",
    "LZBatchSaveWithLabels": "LZ Batch Save With Labels",
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']