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
                "lora_name": ("STRING", {"forceInput": True}),
                "lora_strength": ("STRING", {"forceInput": True}),
                "lora_model": ("STRING", {"forceInput": True}),
                "lora_weight": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("LZ_PIPE",)
    RETURN_NAMES = ("lz_pipe",)
    FUNCTION = "pack"
    CATEGORY = "MyCustomNodes/Pipe"

    def pack(self, **kwargs):
        return (kwargs,)


class LZPipePackXL:
    @classmethod
    def INPUT_TYPES(s):
        param_options = ["none", "checkpoint", "diffusion_model", "lora", "sampler", "scheduler", "positive", "negative"]
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
                "lora_name": ("STRING", {"forceInput": True}),
                "lora_strength": ("STRING", {"forceInput": True}),
                "lora_model": ("STRING", {"forceInput": True}),
                "lora_weight": ("STRING", {"forceInput": True}),
                "x_type": (param_options, {"default": "none"}),
                "x_values": ("STRING", {"multiline": True, "default": ""}),
                "y_type": (param_options, {"default": "none"}),
                "y_values": ("STRING", {"multiline": True, "default": ""}),
                "x_type_str": ("STRING", {"forceInput": True}),
                "y_type_str": ("STRING", {"forceInput": True}),
                "x_replace_key": ("STRING", {"default": ""}),
                "y_replace_key": ("STRING", {"default": ""}),
                "replace_escape": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LZ_PIPE",)
    RETURN_NAMES = ("lz_pipe",)
    FUNCTION = "pack_xl"
    CATEGORY = "MyCustomNodes/Pipe"

    def pack_xl(self, **kwargs):
        x_type = kwargs.get("x_type", "none")
        x_type_str = kwargs.get("x_type_str")
        if x_type_str:
            x_type = x_type_str

        y_type = kwargs.get("y_type", "none")
        y_type_str = kwargs.get("y_type_str")
        if y_type_str:
            y_type = y_type_str

        x_values = kwargs.get("x_values", "")
        y_values = kwargs.get("y_values", "")
        x_replace_key = kwargs.get("x_replace_key", "")
        y_replace_key = kwargs.get("y_replace_key", "")
        replace_escape = kwargs.get("replace_escape", True)

        def parse_values(text):
            if not text:
                return []
            lines = text.strip().split('\n')
            return [line.strip() for line in lines if line.strip()]

        kwargs["_x_type"] = x_type
        kwargs["_x_values"] = parse_values(x_values)
        kwargs["_y_type"] = y_type
        kwargs["_y_values"] = parse_values(y_values)
        kwargs["_x_replace_key"] = x_replace_key
        kwargs["_y_replace_key"] = y_replace_key
        kwargs["_replace_escape"] = replace_escape

        return (kwargs,)


class LZPipeUnpack:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "lz_pipe": ("LZ_PIPE",), }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "LATENT", "STRING", "STRING", "INT", "INT", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "positive", "negative", "LATENT", "positive_text", "negative_text", "width", "height", "ckpt_name", "ckpt_hash", "lora_name", "lora_strength", "lora_model", "lora_weight")
    FUNCTION = "unpack"
    CATEGORY = "MyCustomNodes/Pipe"

    def unpack(self, lz_pipe):
        return (
            lz_pipe.get("model"), lz_pipe.get("clip"), lz_pipe.get("vae"),
            lz_pipe.get("positive"), lz_pipe.get("negative"), lz_pipe.get("latent"),
            lz_pipe.get("positive_text", ""), lz_pipe.get("negative_text", ""),
            lz_pipe.get("width", 0), lz_pipe.get("height", 0),
            lz_pipe.get("ckpt_name", "Unknown"), lz_pipe.get("ckpt_hash", "Unknown"),
            lz_pipe.get("lora_name", ""), lz_pipe.get("lora_strength", ""),
            lz_pipe.get("lora_model", lz_pipe.get("lora_name", "")),
            lz_pipe.get("lora_weight", lz_pipe.get("lora_strength", ""))
        )