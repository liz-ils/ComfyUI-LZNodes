# latent.py

import torch

class PresetEmptyLatentImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "size": (
                    [
                        "1024 x 1024",
                        "832 x 1216",
                        "1216 x 832",
                        "1024 x 1536",
                        "1536 x 1024",
                        "1248 x 1632",
                        "1632 x 1248",
                        "1280 x 1280",
                        "1536 x 1536",
                        "2048 x 2048",
                        "2048 x 1152",
                        "1152 x 2048",
                    ],
                    {"default": "1024 x 1024"}
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "channel_mode": (["SDXL (4ch)", "Anima (16ch)", "Krea2 (16ch)"], {"default": "SDXL (4ch)"}),
            },
            "optional": {
                "width": ("INT", {"forceInput": True, "tooltip": "Optional override (e.g. from ResolutionSelector)."}),
                "height": ("INT", {"forceInput": True, "tooltip": "Optional override (e.g. from ResolutionSelector)."}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("LATENT", "width", "height")
    FUNCTION = "generate"
    CATEGORY = "MyCustomNodes/Latent"

    def generate(self, size, batch_size, channel_mode, width=None, height=None):
        # width/height 入力があればプリセットより優先
        if width is not None and height is not None and int(width) > 0 and int(height) > 0:
            w = int(width)
            h = int(height)
        else:
            width_str, height_str = size.split("x")
            w = int(width_str.strip())
            h = int(height_str.strip())

        channels = 16 if channel_mode in ("Anima (16ch)", "Krea2 (16ch)") else 4
        latent = torch.zeros([batch_size, channels, h // 8, w // 8])

        return ({"samples": latent}, w, h)
