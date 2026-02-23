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
                    ],
                    {"default": "1024 x 1024"}
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("LATENT", "width", "height")
    FUNCTION = "generate"
    CATEGORY = "MyCustomNodes/Latent"

    def generate(self, size, batch_size):
        width_str, height_str = size.split("x")
        width = int(width_str.strip())
        height = int(height_str.strip())

        latent = torch.zeros([batch_size, 4, height // 8, width // 8])

        return ({"samples": latent}, width, height)