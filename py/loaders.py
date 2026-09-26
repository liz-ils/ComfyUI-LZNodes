# loaders.py

import folder_paths
import comfy.sd
import comfy.utils
from .utils import get_checkpoint_hash

# LoRAウェイトキャッシュの上限(フルウェイト保持のため無制限にしない)
LORA_CACHE_MAX = 32


def _lora_cache_get(cache, lora_path):
    lora = cache.get(lora_path)
    if lora is not None:
        # LRU的に末尾へ
        cache.pop(lora_path, None)
        cache[lora_path] = lora
    return lora


def _lora_cache_put(cache, lora_path, lora, max_entries=LORA_CACHE_MAX):
    cache[lora_path] = lora
    while len(cache) > max_entries:
        cache.pop(next(iter(cache)))


class LZCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "positive": ("STRING", {"multiline": True, "default": ""}),
                "negative": ("STRING", {"multiline": True, "default": ""}),
            }
        }
        
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "STRING", "STRING", "LZ_PIPE", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "positive", "negative", "positive_text", "negative_text", "lz_pipe", "ckpt_name", "ckpt_hash")
    FUNCTION = "load_and_encode"
    CATEGORY = "MyCustomNodes/Loaders"

    def load_and_encode(self, ckpt_name, positive, negative):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        ckpt_hash = get_checkpoint_hash(ckpt_path)
        
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, 
            output_vae=True, 
            output_clip=True, 
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )
        model, clip, vae = out[:3]
        
        tokens_pos = clip.tokenize(positive)
        positive_cond = clip.encode_from_tokens_scheduled(tokens_pos)
        
        tokens_neg = clip.tokenize(negative)
        negative_cond = clip.encode_from_tokens_scheduled(tokens_neg)
        
        lz_pipe = {
            "model": model,
            "clip": clip,
            "vae": vae,
            "positive": positive_cond,
            "negative": negative_cond,
            "positive_text": positive,
            "negative_text": negative,
            "ckpt_name": ckpt_name,
            "ckpt_hash": ckpt_hash
        }
        
        return (model, clip, vae, positive_cond, negative_cond, positive, negative, lz_pipe, ckpt_name, ckpt_hash)


# 旧名の互換エイリアス
EZCheckpointLoader = LZCheckpointLoader


class LZSimpleCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            }
        }
        
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "LZ_PIPE", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "lz_pipe", "ckpt_name", "ckpt_hash")
    FUNCTION = "load_checkpoint"
    CATEGORY = "MyCustomNodes/Loaders"

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        
        ckpt_hash = get_checkpoint_hash(ckpt_path)
        
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, 
            output_vae=True, 
            output_clip=True, 
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )
        model, clip, vae = out[:3]
        
        lz_pipe = {
            "model": model,
            "clip": clip,
            "vae": vae,
            "ckpt_name": ckpt_name,
            "ckpt_hash": ckpt_hash
        }
        
        return (model, clip, vae, lz_pipe, ckpt_name, ckpt_hash)


class LZLoRALoaderModelOnly:
    def __init__(self):
        self.loaded_loras = {}

    @classmethod
    def INPUT_TYPES(s):
        loras = ["None"] + (folder_paths.get_filename_list("loras") or [])
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (loras,),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "lora_name", "lora_strength", "lora_model", "lora_weight")
    FUNCTION = "load_lora"
    CATEGORY = "MyCustomNodes/Loaders"

    def load_lora(self, model, lora_name, strength_model):
        if lora_name == "None" or strength_model == 0:
            return (model, "None", "0.0", "None", "0.0")

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            # ファイル消失時は適用なし扱いにし、要求名のまま返さない
            return (model, "None", "0.0", "None", "0.0")
        lora = _lora_cache_get(self.loaded_loras, lora_path)
        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            _lora_cache_put(self.loaded_loras, lora_path, lora)
        model, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)

        return (model, lora_name, str(strength_model), lora_name, str(strength_model))


class LZLoRAStacker:
    def __init__(self):
        self.loaded_loras = {}

    @classmethod
    def INPUT_TYPES(s):
        loras = ["None"] + (folder_paths.get_filename_list("loras") or [])
        inputs = {
            "required": {},
            "optional": {
                "lz_pipe": ("LZ_PIPE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
            }
        }
        
        for i in range(1, 11):
            inputs["required"][f"lora_{i}"] = (loras, {"default": "None"})
            inputs["required"][f"model_weight_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["required"][f"clip_weight_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            
        return inputs

    RETURN_TYPES = ("MODEL", "CLIP", "LZ_PIPE", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "lz_pipe", "lora_model", "lora_weight")
    FUNCTION = "load_loras"
    CATEGORY = "MyCustomNodes/Loaders"

    def load_loras(self, **kwargs):
        # lz_pipeが繋がれていない場合の対策
        lz_pipe = kwargs.get("lz_pipe")
        if lz_pipe is None:
            lz_pipe = {}
        
        model = kwargs.get("model", lz_pipe.get("model"))
        clip = kwargs.get("clip", lz_pipe.get("clip"))
        
        if model is None or clip is None:
            raise ValueError("LZ LoRA Stacker Error: MODEL and CLIP must be connected directly or provided via lz_pipe.")

        used_loras = []
        for i in range(1, 11):
            lora_name = kwargs.get(f"lora_{i}", "None")
            if lora_name != "None":
                model_weight = kwargs.get(f"model_weight_{i}", 1.0)
                clip_weight = kwargs.get(f"clip_weight_{i}", 1.0)
                
                if model_weight == 0 and clip_weight == 0:
                    continue

                lora_path = folder_paths.get_full_path("loras", lora_name)
                if lora_path is None:
                    # LoRAが見つからなかった場合は使用済みとして記録しない
                    continue
                lora = _lora_cache_get(self.loaded_loras, lora_path)
                if lora is None:
                    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                    _lora_cache_put(self.loaded_loras, lora_path, lora)

                model, clip = comfy.sd.load_lora_for_models(model, clip, lora, model_weight, clip_weight)
                used_loras.append((lora_name, model_weight, clip_weight))
        
        # 適用された最新のモデルとCLIPでパイプを更新
        new_pipe = lz_pipe.copy() if lz_pipe else {}
        new_pipe["model"] = model
        new_pipe["clip"] = clip

        if used_loras:
            model_names = ", ".join([n for n, _, _ in used_loras])
            # modelとclipで重みが異なる場合は "m:c" 形式で両方残す
            weight_strs = []
            for _, m, c in used_loras:
                if abs(float(m) - float(c)) < 1e-9:
                    weight_strs.append(str(m))
                else:
                    weight_strs.append(f"{m}:{c}")
            model_weights = ", ".join(weight_strs)
            new_pipe["lora_name"] = model_names
            new_pipe["lora_strength"] = model_weights
            new_pipe["lora_model"] = model_names
            new_pipe["lora_weight"] = model_weights

        return (model, clip, new_pipe, new_pipe.get("lora_model", ""), new_pipe.get("lora_weight", ""))
