# sampling.py

import comfy.samplers
import nodes

from .utils import zero_out_conditioning

class LZKSamplerDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "model_preset": (["Auto/SDXL", "Anima", "Krea2 (Turbo)", "Krea2 (Raw)"], {
                    "default": "Auto/SDXL",
                    "tooltip": (
                        "Krea2 (Turbo): recommended 8 steps / CFG 1.0 / euler / simple. "
                        "If no negative is connected, the positive is zeroed-out as negative "
                        "(official template behaviour; CFG 1.0 ignores negatives).\n"
                        "Krea2 (Raw): recommended 52 steps / CFG 3.5. A real negative is required."
                    )
                }),
            },
            "optional": {
                "lz_pipe": ("LZ_PIPE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "positive_text": ("STRING", {"forceInput": True}),
                "negative_text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LZ_PIPE", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "lz_pipe", "positive_text", "negative_text", "width", "height")
    FUNCTION = "sample_and_decode"
    CATEGORY = "MyCustomNodes/Sampling"

    def sample_and_decode(self, seed, steps, cfg, sampler_name, scheduler, denoise, model_preset, **kwargs):
        lz_pipe = kwargs.get("lz_pipe", {})

        # 各種データを取得（個別優先、なければパイプから）
        model = kwargs.get("model", lz_pipe.get("model"))
        clip = kwargs.get("clip", lz_pipe.get("clip"))
        vae = kwargs.get("vae", lz_pipe.get("vae"))
        latent_image = kwargs.get("latent_image", lz_pipe.get("latent"))
        
        positive = kwargs.get("positive", lz_pipe.get("positive"))
        negative = kwargs.get("negative", lz_pipe.get("negative"))
        
        pos_text = kwargs.get("positive_text", lz_pipe.get("positive_text", ""))
        neg_text = kwargs.get("negative_text", lz_pipe.get("negative_text", ""))

        if positive is None and pos_text != "":
            if clip is None:
                raise ValueError("LZ KSampler Error: 'positive' (CONDITIONING) is missing, and 'clip' is required to encode 'positive_text'.")
            tokens_pos = clip.tokenize(pos_text)
            positive = clip.encode_from_tokens_scheduled(tokens_pos)

        if negative is None and neg_text != "":
            if clip is None:
                raise ValueError("LZ KSampler Error: 'negative' (CONDITIONING) is missing, and 'clip' is required to encode 'negative_text'.")
            tokens_neg = clip.tokenize(neg_text)
            negative = clip.encode_from_tokens_scheduled(tokens_neg)

        # Krea2 (Turbo): ネガティブが無い場合は positive をゼロ化して代用 (公式テンプレート準拠)
        if negative is None and positive is not None and model_preset == "Krea2 (Turbo)":
            negative = zero_out_conditioning(positive)

        # 実行に必要なデータが揃っているか最終チェック
        if None in (model, positive, negative, latent_image, vae):
            missing = []
            if model is None: missing.append("model")
            if positive is None: missing.append("positive/positive_text")
            if negative is None: missing.append("negative/negative_text")
            if latent_image is None: missing.append("latent_image")
            if vae is None: missing.append("vae")
            raise ValueError(f"LZ KSampler Error: Missing required data ({', '.join(missing)}).")

        # サンプリング実行
        ksampler = nodes.KSampler()
        sampled_latent = ksampler.sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)[0]

        # VAEデコード実行
        vae_decoder = nodes.VAEDecode()
        image = vae_decoder.decode(vae, sampled_latent)[0]

        # width/height → 入力優先、なければlatentから自動計算
        w = kwargs.get("width") or (lz_pipe or {}).get("width")
        h = kwargs.get("height") or (lz_pipe or {}).get("height")
        if (w is None or h is None) and sampled_latent is not None:
            w = sampled_latent["samples"].shape[3] * 8
            h = sampled_latent["samples"].shape[2] * 8

        new_pipe = lz_pipe.copy()
        new_pipe["model"] = model
        new_pipe["clip"] = clip
        new_pipe["positive"] = positive
        new_pipe["negative"] = negative
        new_pipe["positive_text"] = pos_text
        new_pipe["negative_text"] = neg_text
        new_pipe["latent"] = sampled_latent
        new_pipe["vae"] = vae
        new_pipe["seed"] = seed
        new_pipe["steps"] = steps
        new_pipe["cfg"] = cfg
        new_pipe["sampler_name"] = sampler_name
        new_pipe["scheduler"] = scheduler
        new_pipe["width"] = w
        new_pipe["height"] = h
        new_pipe["ckpt_name"] = (lz_pipe or {}).get("ckpt_name", "Unknown")
        new_pipe["ckpt_hash"] = (lz_pipe or {}).get("ckpt_hash", "Unknown")
        new_pipe["lora_name"] = (lz_pipe or {}).get("lora_name", "")
        new_pipe["lora_strength"] = (lz_pipe or {}).get("lora_strength", "")

        return (image, new_pipe, pos_text, neg_text, w, h)