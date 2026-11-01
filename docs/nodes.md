# Node Reference

[Back to README](../README.md) | [日本語](nodes_ja.md)

All 47 nodes in ComfyUI-LZNodes, grouped by their menu category under `MyCustomNodes`. The display name is what you see (and search for) in the ComfyUI UI; the class name is what appears in workflow JSON files.

## Loaders

| Display Name | Class Name |
|---|---|
| LZ Checkpoint Loader & Encode | `LZCheckpointLoader` |
| LZ Checkpoint Loader & Encode (Legacy) | `EZCheckpointLoader` |
| LZ Simple Checkpoint Loader | `LZSimpleCheckpointLoader` |
| LZ LoRA Stacker | `LZLoRAStacker` |
| LZ LoRA Stacker (Model Only) | `LZLoRAStackerModelOnly` |
| LZ LoRA Loader (Model Only) | `LZLoRALoaderModelOnly` |
| LZ Anima Loader | `LZAnimaLoader` |
| LZ Krea2 Loader | `LZKrea2Loader` |

- `LZCheckpointLoader`: Loads a checkpoint and encodes the positive/negative prompts in a single node.
- `EZCheckpointLoader`: The legacy version of `LZCheckpointLoader`, kept so older workflows continue to work.
- `LZSimpleCheckpointLoader`: A lightweight loader that only extracts the model components (MODEL / CLIP / VAE).
- `LZLoRAStacker`: Stacks up to 10 LoRAs, with caching. Outputs `lora_model` / `lora_weight` (comma-joined names/weights, also carried in `lz_pipe`).
- `LZLoRAStackerModelOnly` / `LZLoRALoaderModelOnly`: Variants that apply LoRAs to the MODEL only and do not modify the CLIP.
- `LZAnimaLoader`: Loader for Anima models; loads the diffusion model, text encoder, and VAE separately.
- `LZKrea2Loader`: One-node loader for the Krea 2 architecture (12B SingleStreamDiT + Qwen3-VL-4B text encoder + Qwen-Image VAE). Pipe-ready, with `weight_dtype` selection. With `negative_mode="auto"` (default), the negative conditioning is created by zeroing out the positive conditioning when no negative prompt is used, matching the official template (Krea 2 Turbo runs at CFG 1.0). Requires ComfyUI v0.26.0+. See the [Krea 2 Quick Guide](krea2.md).

## Anima

| Display Name | Class Name |
|---|---|
| LZ Anima Artist Node | `LZAnimaArtistNode` |
| LZ Anima Artist Pack (Split + Encode) | `LZAnimaArtistPack` |
| LZ Anima Artist Cross-Attn (v2) | `LZAnimaArtistCrossAttn` |
| LZ Anima Artist Options (Advanced) | `LZAnimaArtistOptions` |
| LZ Artist Chain Text | `LZArtistChainText` |

- `LZAnimaArtistNode`: Combined node that splits/encodes an artist chain and applies cross-attention injection in one step. Outputs `(MODEL, CONDITIONING, STRING)`, where the STRING is the base prompt text (`positive_text`), ready to connect to the pipe system.
- `LZAnimaArtistPack` / `LZAnimaArtistCrossAttn`: The split-encode and cross-attention injection steps as separate nodes, for advanced workflows. `LZAnimaArtistCrossAttn` also outputs `positive_text` (STRING).
- `LZAnimaArtistOptions`: Advanced options for block ranges, fusion modes, EMA smoothing, low-rank injection, static capture, and anchor Q.
- `LZArtistChainText`: Helper node for the `artist_chain` input (shown in green in the UI). Converts a line-separated artist list into a comma-joined `artist_tag`, and also passes through the original text and the `lz_pipe` unchanged.

## Prompt

| Display Name | Class Name |
|---|---|
| LZ Advanced Positive Prompt | `AdvancedPositivePrompt` |
| LZ Advanced Negative Prompt | `AdvancedNegativePrompt` |

- `AdvancedPositivePrompt` / `AdvancedNegativePrompt`: Enter the base prompt, artist tags, and quality tags in separate fields; they are concatenated and encoded automatically.

## Conditioning

| Display Name | Class Name |
|---|---|
| LZ Dual CLIP Text Encode | `DualCLIPTextEncode` |
| LZ CLIP Text Encode | `LZCLIPTextEncode` |

- `DualCLIPTextEncode`: Encodes the positive and negative prompts in a single node.
- `LZCLIPTextEncode`: Encodes a single text input and outputs `(CONDITIONING, STRING)`.

## Text

| Display Name | Class Name |
|---|---|
| LZ String Output | `StringNode` |
| LZ String Concat | `StringConcatNode` |
| LZ Text Preview | `LZTextPreview` |
| LZ String Sanitize | `LZStringSanitize` |
| LZ String Select | `LZStringSelect` |
| LZ Save String to CSV | `LZSaveStringToCSV` |
| LZ Prompt Weight | `LZPromptWeight` |
| LZ Tag Editor | `LZTagEditor` |
| LZ Prompt Replace Single | `LZPromptReplaceSingle` |
| LZ Prompt Replace Multi | `LZPromptReplaceMulti` |
| LZ Prompt Replace String | `LZPromptReplaceString` |
| LZ Prompt Pick (Multi-source) | `LZPromptPick` |

- `StringNode` / `StringConcatNode`: Basic string output and multi-input concatenation.
- `LZTextPreview`: Display a string value directly in the node UI.
- `LZStringSanitize`: Remove or replace characters that cannot be used in file names.
- `LZStringSelect`: Select one string from multiple inputs by index.
- `LZSaveStringToCSV`: Save or append string rows to a CSV file (the header is written only when the file is created).
- `LZPromptWeight`: Wraps prompt tags with weight syntax `(tag:weight)`.
- `LZTagEditor`: Edit up to 10 tags visually, each with its own strength and on/off toggle.
- `LZPromptReplaceSingle` / `LZPromptReplaceMulti` / `LZPromptReplaceString`: Replace placeholders in a prompt with entries randomly selected from a txt/csv/json file or an inline candidate list. Seeded (`seed`; `-1` = random on every run). See [Notes on File Locations](#notes-on-file-locations).
- `LZPromptPick`: Pick multiple tags from up to two files plus inline candidates, with random weights (`weight_min` / `weight_max`), a no-duplicate mode (`unique`, default ON), and seed control.

## Latent

| Display Name | Class Name |
|---|---|
| LZ Preset Empty Latent Image | `PresetEmptyLatentImage` |

- `PresetEmptyLatentImage`: Creates empty latent images at common SDXL resolutions, and supports both SDXL (4ch) and Anima (16ch) latent formats.

## Pipe

| Display Name | Class Name |
|---|---|
| LZ Pipe Pack | `LZPipePack` |
| LZ Pipe Pack XL | `LZPipePackXL` |
| LZ Pipe Unpack | `LZPipeUnpack` |
| LZ Pipe Info | `LZPipeInfo` |
| LZ Pipe Merge | `LZPipeMerge` |

- The pipe nodes bundle `MODEL`, `CLIP`, `VAE`, `CONDITIONING`, `LATENT`, and generation parameters into a single `lz_pipe` connection: Pack creates the bundle, Unpack extracts its components, Merge combines two pipes, and Info displays what a pipe contains.

## Sampling

| Display Name | Class Name |
|---|---|
| LZ KSampler & Decode | `LZKSamplerDecode` |

- `LZKSamplerDecode`: Combines KSampler and VAE Decode into a single node and reads its inputs from the `lz_pipe`. If no conditioning is connected, it can also encode raw text with the CLIP from the pipe.

## XY

| Display Name | Class Name |
|---|---|
| LZ XY Plot | `LZXYPlot` |
| LZ XY Plot & Sampler | `LZXYPlotSampler` |
| LZ XY Sampler | `LZXYSampler` |
| LZ XY Grid Output | `LZXYGridOutput` |

- The XY plot nodes build grid comparisons over checkpoints, standalone diffusion models (DiT-only models such as Krea 2 / Anima / Flux), LoRAs, samplers, schedulers, and prompt variations. They support replacement keys and automatic grid image generation with labels.

## IO

| Display Name | Class Name |
|---|---|
| LZ Save Image & Log | `LZSaveImageAndLog` |
| LZ Batch Save With Labels | `LZBatchSaveWithLabels` |

- `LZSaveImageAndLog`: Saves the generated image (PNG / WEBP / JPG) together with a detailed `.txt` log file containing the seed, steps, checkpoint hash, full prompt text, and more. For PNG files, the same information is also embedded in the PNG metadata (`parameters`). Note: metadata embedding is PNG only; the `.txt` log is written for all formats.
- `LZBatchSaveWithLabels`: Batch-saves images with XY labels embedded in the file names and metadata, plus an automatic batch log.

## Log

| Display Name | Class Name |
|---|---|
| LZ Append Log to CSV | `LZAppendLogToCSV` |
| LZ Log Reader | `LZLogReader` |

- `LZAppendLogToCSV`: Appends generation parameters to a CSV file for spreadsheet tracking.
- `LZLogReader`: Reads a `.txt` log file written by `LZSaveImageAndLog` and restores the parameters as node outputs.

## Merge

| Display Name | Class Name |
|---|---|
| LZ Merge Recipe Random | `LZMergeRecipeRandom` |
| LZ Merge Recipe Manual | `LZMergeRecipeManual` |
| LZ Merge Recipe Random Advanced | `LZMergeRecipeRandomAdvanced` |

- The merge recipe nodes generate 19-block merge recipes (IN00–IN08, M00, OUT00–OUT08) compatible with model merge tools. See [Merge Recipe Nodes](merge-recipe.md) for inputs and outputs.

## Notes on File Locations

- **Prompt candidate files** (`LZPromptReplaceSingle` / `LZPromptReplaceMulti` / `LZPromptReplaceString` / `LZPromptPick`): place your `.txt` / `.csv` / `.json` files in the `prompt_csv` folder inside ComfyUI's output directory (`ComfyUI/output/prompt_csv/`). The folder is created automatically on first use. Files up to 2 MB are supported.
- **Saved images and logs** (`LZSaveImageAndLog` / `LZBatchSaveWithLabels`): written to ComfyUI's output directory, like the built-in Save Image node.
