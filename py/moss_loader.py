# moss_loader.py
# Loader for MOSS-Image (5B single-stream DiT + Qwen3.5-9B + FLUX.2 VAE).
# Requires ComfyUI with native MOSS-Image support (CLIP type "moss_image", 0.42.0+ / nightly).

import folder_paths
import nodes

from .utils import get_checkpoint_hash

MOSS_CLIP_TYPE = "moss_image"

# テンプレート由来の推奨値 (docs: 25 steps / CFG 2.5 / er_sde, 公式Base: 50 steps / CFG 3 / euler)
MOSS_RECOMMENDED = {
    "sft": {"steps": 25, "cfg": 2.5, "sampler_name": "er_sde", "scheduler": "simple"},
    "base": {"steps": 50, "cfg": 3.0, "sampler_name": "euler", "scheduler": "simple"},
}

MOSS_DEFAULT_NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"


def _moss_clip_supported():
    """ComfyUI が CLIP type "moss_image" に対応しているか判定する。"""
    try:
        info = nodes.CLIPLoader.INPUT_TYPES()
        opts = info["required"]["type"][0]
        if isinstance(opts, (list, tuple)):
            return MOSS_CLIP_TYPE in opts
        return MOSS_CLIP_TYPE in str(opts)
    except Exception:
        # 判定できない場合は読み込み時に本体のエラーに任せる
        return True


class LZMossImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusion_model": (folder_paths.get_filename_list("diffusion_models"), ),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {
                    "default": "default",
                    "tooltip": "Model weight precision. Use fp8 variants to reduce VRAM (matches UNETLoader options)."
                }),
                "text_encoder": (folder_paths.get_filename_list("text_encoders"), {
                    "tooltip": "Use the Qwen3.5 9B text encoder (e.g. qwen3.5_9b_w4a8_gptq.safetensors)."
                }),
                "vae": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "Use the FLUX.2 VAE (e.g. flux2-vae_small.safetensors)."
                }),
                "positive": ("STRING", {"multiline": True, "default": ""}),
                "negative": ("STRING", {"multiline": True, "default": MOSS_DEFAULT_NEGATIVE}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "STRING", "STRING", "LZ_PIPE", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "positive", "negative", "positive_text", "negative_text", "lz_pipe", "model_name", "model_hash")
    FUNCTION = "load_and_encode"
    CATEGORY = "MyCustomNodes/Loaders"

    def load_and_encode(self, diffusion_model, weight_dtype, text_encoder, vae, positive, negative):
        if not _moss_clip_supported():
            raise ValueError(
                "LZ MOSS Image Loader Error: This ComfyUI does not support CLIP type 'moss_image'. "
                "MOSS-Image requires ComfyUI v0.42.0 or newer (nightly with native support). Please update ComfyUI."
            )

        # モデル読み込み
        model_path = folder_paths.get_full_path("diffusion_models", diffusion_model)
        model_hash = get_checkpoint_hash(model_path)

        model_loader = nodes.UNETLoader()
        model = model_loader.load_unet(diffusion_model, weight_dtype)[0]

        clip_loader = nodes.CLIPLoader()
        clip = clip_loader.load_clip(text_encoder, MOSS_CLIP_TYPE)[0]

        vae_loader = nodes.VAELoader()
        vae_obj = vae_loader.load_vae(vae)[0]

        # テキストエンコード
        tokens_pos = clip.tokenize(positive)
        positive_cond = clip.encode_from_tokens_scheduled(tokens_pos)

        tokens_neg = clip.tokenize(negative)
        negative_cond = clip.encode_from_tokens_scheduled(tokens_neg)

        lz_pipe = {
            "model": model,
            "clip": clip,
            "vae": vae_obj,
            "positive": positive_cond,
            "negative": negative_cond,
            "positive_text": positive,
            "negative_text": negative,
            "ckpt_name": diffusion_model,
            "ckpt_hash": model_hash,
            "model_family": "moss_image",
        }

        return (model, clip, vae_obj, positive_cond, negative_cond, positive, negative, lz_pipe, diffusion_model, model_hash)
