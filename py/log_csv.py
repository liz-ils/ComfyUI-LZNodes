# log_csv.py

import os
import csv
import datetime

class LZAppendLogToCSV:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filepath": ("STRING", {"default": "generation_log.csv"}),
                "mode": (["write", "append"],),
            },
            "optional": {
                "lz_pipe": ("LZ_PIPE",),
                "seed": ("INT", {"default": 0}),
                "steps": ("INT", {"default": 20}),
                "cfg": ("FLOAT", {"default": 8.0}),
                "sampler_name": ("STRING", {"default": "euler"}),
                "scheduler": ("STRING", {"default": "normal"}),
                "positive_prompt": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"default": ""}),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
                "checkpoint_name": ("STRING", {"default": ""}),
                "image_count": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "append_log"
    OUTPUT_NODE = True
    CATEGORY = "MyCustomNodes/Log"

    def append_log(self, filepath, mode, **kwargs):
        filepath = filepath.strip()
        if not filepath:
            filepath = "generation_log.csv"
        
        lz_pipe = kwargs.get("lz_pipe", {})
        
        seed = lz_pipe.get("seed", kwargs.get("seed", 0))
        steps = lz_pipe.get("steps", kwargs.get("steps", 20))
        cfg = lz_pipe.get("cfg", kwargs.get("cfg", 8.0))
        sampler_name = lz_pipe.get("sampler_name", kwargs.get("sampler_name", "euler"))
        scheduler = lz_pipe.get("scheduler", kwargs.get("scheduler", "normal"))
        positive_prompt = lz_pipe.get("positive_text", kwargs.get("positive_prompt", ""))
        negative_prompt = lz_pipe.get("negative_text", kwargs.get("negative_prompt", ""))
        width = lz_pipe.get("width", kwargs.get("width", 512))
        height = lz_pipe.get("height", kwargs.get("height", 512))
        checkpoint_name = lz_pipe.get("ckpt_name", kwargs.get("checkpoint_name", ""))
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_count = kwargs.get("image_count", 1)

        headers = ["timestamp", "seed", "steps", "cfg", "sampler", "scheduler", "width", "height", "checkpoint", "image_count", "positive_prompt", "negative_prompt"]
        row = [timestamp, seed, steps, cfg, sampler_name, scheduler, width, height, checkpoint_name, image_count, positive_prompt, negative_prompt]

        file_exists = os.path.exists(filepath)
        append_mode = (mode == "append" and file_exists)

        with open(filepath, "a" if append_mode else "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            # ヘッダは新規作成時のみ書き込む(既存ファイルへの追記時は書かない)
            if not append_mode:
                writer.writerow(headers)
            writer.writerow(row)

        return (os.path.abspath(filepath),)
