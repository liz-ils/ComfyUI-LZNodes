# loaders.py

import folder_paths
import comfy.sd
import comfy.utils
import hashlib
import os

CHECKPOINT_HASH_CACHE = {}

def get_checkpoint_hash(file_path):
    if not file_path or not os.path.exists(file_path):
        return "Unknown"
    
    mtime = os.path.getmtime(file_path)
    if file_path in CHECKPOINT_HASH_CACHE:
        cached_mtime, cached_hash = CHECKPOINT_HASH_CACHE[file_path]
        if cached_mtime == mtime:
            return cached_hash
            
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            sha256_hash.update(byte_block)
            
    short_hash = sha256_hash.hexdigest()[:10]
    CHECKPOINT_HASH_CACHE[file_path] = (mtime, short_hash)
    return short_hash


class EZCheckpointLoader:
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
        cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
        positive_cond = [[cond_pos, {"pooled_output": pooled_pos}]]
        
        tokens_neg = clip.tokenize(negative)
        cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
        negative_cond = [[cond_neg, {"pooled_output": pooled_neg}]]
        
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

    RETURN_TYPES = ("MODEL", "CLIP", "LZ_PIPE")
    RETURN_NAMES = ("MODEL", "CLIP", "lz_pipe")
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

        for i in range(1, 11):
            lora_name = kwargs.get(f"lora_{i}", "None")
            if lora_name != "None":
                model_weight = kwargs.get(f"model_weight_{i}", 1.0)
                clip_weight = kwargs.get(f"clip_weight_{i}", 1.0)
                
                if model_weight == 0 and clip_weight == 0:
                    continue
                
                lora_path = folder_paths.get_full_path("loras", lora_name)
                if lora_path is not None:
                    lora = None
                    if lora_path in self.loaded_loras:
                        lora = self.loaded_loras[lora_path]
                    else:
                        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                        self.loaded_loras[lora_path] = lora
                        
                    model, clip = comfy.sd.load_lora_for_models(model, clip, lora, model_weight, clip_weight)
        
        # 適用された最新のモデルとCLIPでパイプを更新
        new_pipe = lz_pipe.copy() if lz_pipe else {}
        new_pipe["model"] = model
        new_pipe["clip"] = clip
        
        return (model, clip, new_pipe)
