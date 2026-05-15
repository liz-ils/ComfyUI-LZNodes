# anima_loader.py

import folder_paths
import nodes
import hashlib
import os

MODEL_HASH_CACHE = {}

def get_model_hash(file_path):
    if not file_path or not os.path.exists(file_path):
        return "Unknown"
    
    mtime = os.path.getmtime(file_path)
    if file_path in MODEL_HASH_CACHE:
        cached_mtime, cached_hash = MODEL_HASH_CACHE[file_path]
        if cached_mtime == mtime:
            return cached_hash
            
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            sha256_hash.update(byte_block)
            
    short_hash = sha256_hash.hexdigest()[:10]
    MODEL_HASH_CACHE[file_path] = (mtime, short_hash)
    return short_hash


class LZAnimaLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusion_model": (folder_paths.get_filename_list("diffusion_models"), ),
                "text_encoder": (folder_paths.get_filename_list("text_encoders"), ),
                "vae": (folder_paths.get_filename_list("vae"), ),
                "positive": ("STRING", {"multiline": True, "default": "masterpiece, best quality, score_9, safe, "}),
                "negative": ("STRING", {"multiline": True, "default": "worst quality, low quality, score_1, score_2, score_3, artist name"}),
            }
        }
        
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "STRING", "STRING", "LZ_PIPE", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "positive", "negative", "positive_text", "negative_text", "lz_pipe", "model_name", "model_hash")
    FUNCTION = "load_and_encode"
    CATEGORY = "MyCustomNodes/Loaders"

    def load_and_encode(self, diffusion_model, text_encoder, vae, positive, negative):
        # モデルパス取得
        model_path = folder_paths.get_full_path("diffusion_models", diffusion_model)
        clip_path = folder_paths.get_full_path("text_encoders", text_encoder)
        vae_path = folder_paths.get_full_path("vae", vae)
        
        # モデルハッシュ計算（diffusion_modelのみ）
        model_hash = get_model_hash(model_path)
        
        # モデル読み込み
        model_loader = nodes.UNETLoader()
        model = model_loader.load_unet(diffusion_model, "default")[0]
        
        clip_loader = nodes.CLIPLoader()
        clip = clip_loader.load_clip(text_encoder, "default")[0]
        
        vae_loader = nodes.VAELoader()
        vae_obj = vae_loader.load_vae(vae)[0]
        
        # テキストエンコード
        tokens_pos = clip.tokenize(positive)
        positive_cond = clip.encode_from_tokens_scheduled(tokens_pos)
        
        tokens_neg = clip.tokenize(negative)
        negative_cond = clip.encode_from_tokens_scheduled(tokens_neg)
        
        # lz_pipe作成
        lz_pipe = {
            "model": model,
            "clip": clip,
            "vae": vae_obj,
            "positive": positive_cond,
            "negative": negative_cond,
            "positive_text": positive,
            "negative_text": negative,
            "ckpt_name": diffusion_model,
            "ckpt_hash": model_hash
        }
        
        return (model, clip, vae_obj, positive_cond, negative_cond, positive, negative, lz_pipe, diffusion_model, model_hash)