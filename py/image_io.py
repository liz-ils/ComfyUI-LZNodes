# image_io.py

import os
import time
import json
import folder_paths
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import random
import string

class LZSaveImageAndLog:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "LZ_Output"}),
                "image_format": (["png", "webp", "jpg"], {"default": "png"}),
                "save_txt_log": ("BOOLEAN", {"default": True}),
                "preview_only": ("BOOLEAN", {"default": False}),
                "add_timestamp": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "lz_pipe": ("LZ_PIPE",),
                "positive_text": ("STRING", {"forceInput": True}),
                "negative_text": ("STRING", {"forceInput": True}),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
                "seed": ("INT", {"forceInput": True}),
                "steps": ("INT", {"forceInput": True}),
                "cfg": ("FLOAT", {"forceInput": True}),
                "sampler_name": ("STRING", {"forceInput": True}),
                "scheduler": ("STRING", {"forceInput": True}),
                "ckpt_name": ("STRING", {"forceInput": True}),
                "ckpt_hash": ("STRING", {"forceInput": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images_and_log"
    OUTPUT_NODE = True
    CATEGORY = "MyCustomNodes/IO"

    def save_images_and_log(self, images, filename_prefix="LZ_Output", image_format="png", save_txt_log=True, preview_only=False, add_timestamp=True, prompt=None, extra_pnginfo=None, **kwargs):
        if preview_only:
            output_dir = self.temp_dir
            out_type = "temp"
            save_txt_log = False
            prefix_append = "_temp_" + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
            filename_prefix += prefix_append
        else:
            output_dir = self.output_dir
            out_type = "output"

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir, images[0].shape[1], images[0].shape[0])
        
        results = list()
        
        lz_pipe = kwargs.get("lz_pipe", {})
        
        # 個別入力があれば優先、なければパイプから取得
        pos_text = kwargs.get("positive_text", lz_pipe.get("positive_text", ""))
        neg_text = kwargs.get("negative_text", lz_pipe.get("negative_text", ""))
        width = kwargs.get("width", lz_pipe.get("width", "Unknown"))
        height = kwargs.get("height", lz_pipe.get("height", "Unknown"))
        seed = kwargs.get("seed", lz_pipe.get("seed", "Unknown"))
        steps = kwargs.get("steps", lz_pipe.get("steps", "Unknown"))
        cfg = kwargs.get("cfg", lz_pipe.get("cfg", "Unknown"))
        sampler = kwargs.get("sampler_name", lz_pipe.get("sampler_name", "Unknown"))
        scheduler = kwargs.get("scheduler", lz_pipe.get("scheduler", "Unknown"))
        ckpt_name = kwargs.get("ckpt_name", lz_pipe.get("ckpt_name", "Unknown"))
        ckpt_hash = kwargs.get("ckpt_hash", lz_pipe.get("ckpt_hash", "Unknown"))

        log_text = ""
        if add_timestamp:
            log_text += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
        if ckpt_name != "Unknown":
            log_text += f"Model: {ckpt_name} (Hash: {ckpt_hash})\n"
            
        log_text += f"Size: {width} x {height}\n"
        if seed != "Unknown":
            log_text += f"Seed: {seed} | Steps: {steps} | CFG: {cfg} | Sampler: {sampler} | Scheduler: {scheduler}\n"
        log_text += f"Positive:\n{pos_text}\n\n"
        log_text += f"Negative:\n{neg_text}\n"

        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            file = f"{filename}_{counter:05}_.{image_format}"
            img_path = os.path.join(full_output_folder, file)
            
            if image_format == "png":
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                metadata.add_text("parameters", log_text)
                
                compress = 1 if preview_only else 4
                img.save(img_path, pnginfo=metadata, compress_level=compress)
                
            elif image_format == "webp":
                img.save(img_path, format="WEBP", lossless=True)
            elif image_format == "jpg":
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(img_path, format="JPEG", quality=95)

            if save_txt_log and not preview_only:
                txt_file = f"{filename}_{counter:05}_.txt"
                txt_path = os.path.join(full_output_folder, txt_file)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(log_text)
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": out_type
            })
            counter += 1

        return { "ui": { "images": results } }