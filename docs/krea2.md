# Krea 2 Quick Guide

[Back to README](../README.md) | [日本語](krea2_ja.md)

`LZKrea2Loader` supports the [Krea 2](https://huggingface.co/krea/Krea-2-Turbo) architecture natively (requires **ComfyUI v0.26.0+**).

**Model files** (download the ComfyUI-repackaged versions from [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2)):

| Type | File example | Folder |
|---|---|---|
| Diffusion model (DiT) | `krea2_turbo_fp8_scaled.safetensors` | `models/diffusion_models` |
| Text encoder | `qwen3vl_4b_fp8_scaled.safetensors` | `models/text_encoders` |
| VAE | `qwen_image_vae.safetensors` | `models/vae` |
| Style LoRAs (optional) | `krea2_darkbrush.safetensors` etc. | `models/loras` |

**Recommended settings** (per the official [krea-2](https://github.com/krea-ai/krea-2) repository):

| Variant | Steps | CFG | Sampler | Scheduler |
|---|---|---|---|---|
| Krea 2 **Turbo** | 8 | 1.0 | euler | simple |
| Krea 2 **Raw** | 52 | 3.5 | euler | simple |

- Krea 2 Turbo is guidance-distilled, so negative prompts have no effect at CFG 1.0. `negative_mode="auto"` (default) creates the negative conditioning by zeroing out the positive conditioning, matching the official template's `ConditioningZeroOut` pattern and skipping unnecessary text-encoder work.
- Image sizes from 1K to 2K are supported. Use `PresetEmptyLatentImage` (16ch mode), or connect the built-in `ResolutionSelector` node to its `width` / `height` inputs.
- For XY plots with Krea 2, use the `diffusion_model` axis (the `checkpoint` axis only loads full checkpoints).
- To use negative prompts with Turbo, community nodes such as [ComfyUI-krea2-negpip](https://github.com/blue-pen5805/ComfyUI-krea2-negpip) can be used alongside this pack.
