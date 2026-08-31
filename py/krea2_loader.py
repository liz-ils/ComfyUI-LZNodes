# krea2_loader.py
# Loader for the Krea 2 architecture (SingleStreamDiT + Qwen3-VL-4B + Qwen-Image VAE).
# Requires ComfyUI v0.26.0 or newer (native CLIP type "krea2").

import folder_paths
import nodes

from .utils import get_checkpoint_hash, zero_out_conditioning

KREA2_CLIP_TYPE = "krea2"

# 公式推奨設定 (https://github.com/krea-ai/krea-2)
KREA2_RECOMMENDED = {
    "turbo": {"steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"},
    "raw": {"steps": 52, "cfg": 3.5, "sampler_name": "euler", "scheduler": "simple"},
}


def _krea2_clip_supported():
    """ComfyUI が CLIP type "krea2" に対応しているか判定する。"""
    try:
        info = nodes.CLIPLoader.INPUT_TYPES()
        opts = info["required"]["type"][0]
        if isinstance(opts, (list, tuple)):
            return KREA2_CLIP_TYPE in opts
        return KREA2_CLIP_TYPE in str(opts)
    except Exception:
        # 判定できない場合は読み込み時に本体のエラーに任せる
        return True


class LZKrea2Loader:
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
                    "tooltip": "Use the Qwen3-VL 4B text encoder (e.g. qwen3vl_4b_fp8_scaled.safetensors)."
                }),
                "vae": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "Use the Qwen-Image VAE (qwen_image_vae.safetensors)."
                }),
                "positive": ("STRING", {"multiline": True, "default": ""}),
                "negative": ("STRING", {"multiline": True, "default": ""}),
                "negative_mode": (["auto", "zero_out", "encode"], {
                    "default": "auto",
                    "tooltip": (
                        "auto: if negative is empty, zero-out the positive conditioning instead of a real negative "
                        "(official template behaviour; Krea 2 Turbo runs at CFG 1.0 where negatives have no effect).\n"
                        "zero_out: always zero-out (Turbo).\n"
                        "encode: always encode the negative text (Krea 2 Raw, CFG 3.5)."
                    )
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "STRING", "STRING", "LZ_PIPE", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "positive", "negative", "positive_text", "negative_text", "lz_pipe", "model_name", "model_hash")
    FUNCTION = "load_and_encode"
    CATEGORY = "MyCustomNodes/Loaders"

    def load_and_encode(self, diffusion_model, weight_dtype, text_encoder, vae, positive, negative, negative_mode):
        if not _krea2_clip_supported():
            raise ValueError(
                "LZ Krea2 Loader Error: This ComfyUI does not support CLIP type 'krea2'. "
                "Krea 2 requires ComfyUI v0.26.0 or newer. Please update ComfyUI."
            )

        # モデル読み込み
        model_path = folder_paths.get_full_path("diffusion_models", diffusion_model)
        model_hash = get_checkpoint_hash(model_path)

        model_loader = nodes.UNETLoader()
        model = model_loader.load_unet(diffusion_model, weight_dtype)[0]

        clip_loader = nodes.CLIPLoader()
        clip = clip_loader.load_clip(text_encoder, KREA2_CLIP_TYPE)[0]

        vae_loader = nodes.VAELoader()
        vae_obj = vae_loader.load_vae(vae)[0]

        # テキストエンコード
        tokens_pos = clip.tokenize(positive)
        positive_cond = clip.encode_from_tokens_scheduled(tokens_pos)

        # ネガティブ処理
        #  - zero_out / auto(未入力時): positive をゼロ化して流用 (公式テンプレート準拠)
        #  - encode: ネガティブテキストを通常エンコード (Raw モデル用)
        use_zero_out = (negative_mode == "zero_out") or \
                       (negative_mode == "auto" and not negative.strip())
        if use_zero_out:
            negative_cond = zero_out_conditioning(positive_cond)
        else:
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
            "model_family": "krea2",
        }

        return (model, clip, vae_obj, positive_cond, negative_cond, positive, negative, lz_pipe, diffusion_model, model_hash)
