# pipe.py

class LZPipePack:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "positive_text": ("STRING", {"forceInput": True}),
                "negative_text": ("STRING", {"forceInput": True}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
                "ckpt_name": ("STRING", {"forceInput": True}),
                "ckpt_hash": ("STRING", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("LZ_PIPE",)
    RETURN_NAMES = ("lz_pipe",)
    FUNCTION = "pack"
    CATEGORY = "MyCustomNodes/Pipe"

    def pack(self, **kwargs):
        return (kwargs,)

class LZPipeUnpack:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "lz_pipe": ("LZ_PIPE",), }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "LATENT", "STRING", "STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "positive", "negative", "LATENT", "positive_text", "negative_text", "width", "height", "ckpt_name", "ckpt_hash")
    FUNCTION = "unpack"
    CATEGORY = "MyCustomNodes/Pipe"

    def unpack(self, lz_pipe):
        return (
            lz_pipe.get("model"), lz_pipe.get("clip"), lz_pipe.get("vae"),
            lz_pipe.get("positive"), lz_pipe.get("negative"), lz_pipe.get("latent"),
            lz_pipe.get("positive_text", ""), lz_pipe.get("negative_text", ""),
            lz_pipe.get("width", 0), lz_pipe.get("height", 0),
            lz_pipe.get("ckpt_name", "Unknown"), lz_pipe.get("ckpt_hash", "Unknown")
        )