# batch_save.py

import os
import json
import datetime
import numpy as np
from PIL import Image
import PIL.PngImagePlugin
import folder_paths


def parse_values(text):
    if not text:
        return [""]
    lines = text.strip().split('\n')
    result = [line.strip() for line in lines if line.strip()]
    return result if result else [""]


def format_pipe_value(value):
    if hasattr(value, 'shape'):
        return f"tensor {tuple(value.shape)}"
    elif isinstance(value, dict):
        parts = []
        for k, v in sorted(value.items()):
            if k.startswith("_"):
                continue
            parts.append(f"  {k}: {format_pipe_value(v)}")
        return "\n".join(parts) if parts else "<empty dict>"
    elif isinstance(value, list):
        if len(value) == 0:
            return "<empty list>"
        if len(value) > 0 and isinstance(value[0], str):
            items = [str(v)[:80] for v in value[:8]]
            suffix = f"... ({len(value)} items)" if len(value) > 8 else ""
            return f"[{', '.join(items)}{suffix}]"
        else:
            return f"<list {len(value)} items>"
    else:
        return str(value)[:300]


class LZBatchSaveWithLabels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "optional": {
                "x_labels": ("STRING", {"multiline": True, "default": ""}),
                "y_labels": ("STRING", {"multiline": True, "default": ""}),
                "lz_pipe": ("LZ_PIPE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_batch"
    OUTPUT_NODE = True
    CATEGORY = "MyCustomNodes/IO"

    def save_batch(self, images, filename_prefix="ComfyUI", x_labels="", y_labels="",
                   lz_pipe=None, prompt=None, extra_pnginfo=None):
        output_dir = folder_paths.get_output_directory()

        x_list = parse_values(x_labels)
        y_list = parse_values(y_labels)

        # ラベルが空の場合、インデックス番号で代替
        if x_list == [""] and y_list == [""]:
            file_labels = [f"{i:05d}" for i in range(images.shape[0])]
        else:
            file_labels = []
            for y_val in y_list:
                for x_val in x_list:
                    label = filename_prefix
                    label = label.replace("{x}", x_val)
                    label = label.replace("{y}", y_val)
                    if not label:
                        label = "img"
                    file_labels.append(label)

            if len(file_labels) < images.shape[0]:
                for i in range(len(file_labels), images.shape[0]):
                    file_labels.append(f"{filename_prefix or 'img'}_{i:05d}")

        # prefixからディレクトリ部分を分離
        if "/" in filename_prefix:
            subfolder = filename_prefix.rsplit("/", 1)[0]
        else:
            subfolder = ""

        results = []
        save_details = []

        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            file_label = file_labels[batch_number] if batch_number < len(file_labels) else f"{filename_prefix}_{batch_number:05d}"
            file = f"{file_label}_{batch_number:05d}.png"

            full_output_folder, filename, counter, sub, filename_prefix_clean = \
                folder_paths.get_save_image_path(file, output_dir, images[0].shape[1], images[0].shape[0])

            if subfolder:
                full_output_folder = os.path.join(full_output_folder, subfolder)
                os.makedirs(full_output_folder, exist_ok=True)

            img_path = os.path.join(full_output_folder, filename + ".png")

            metadata = PIL.PngImagePlugin.PngInfo()

            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            # lz_pipe情報をメタデータに追加
            if isinstance(lz_pipe, dict):
                pipe_info_lines = []
                for key, value in sorted(lz_pipe.items()):
                    if key.startswith("_"):
                        continue
                    pipe_info_lines.append(f"{key}: {format_pipe_value(value)}")
                metadata.add_text("lz_pipe", "\n".join(pipe_info_lines))

                pos_text = lz_pipe.get("positive_text", "")
                neg_text = lz_pipe.get("negative_text", "")
            else:
                pos_text = ""
                neg_text = ""

            metadata.add_text("parameters", f"Positive:\n{pos_text}\n\nNegative:\n{neg_text}")

            # 保存
            if img.mode == 'RGBA':
                img.save(img_path, pnginfo=metadata, compress_level=4)
            elif img.mode == 'L':
                img.convert('RGB').save(img_path, pnginfo=metadata, compress_level=4)
            else:
                img.save(img_path, pnginfo=metadata, compress_level=4)

            x_label = x_list[batch_number % len(x_list)] if x_list else ""
            y_label = y_list[batch_number // len(x_list)] if y_list else ""

            results.append({
                "filename": filename + ".png",
                "subfolder": subfolder,
                "type": "output"
            })

            save_details.append({
                "index": batch_number,
                "filename": filename + ".png",
                "x": x_label,
                "y": y_label,
            })

        # ログファイル出力（lz_pipeがあれば詳細情報を含む）
        if isinstance(lz_pipe, dict):
            pos_text = lz_pipe.get("positive_text", "")
            neg_text = lz_pipe.get("negative_text", "")
            if pos_text or neg_text:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                log_name = f"batch_log_{timestamp}.txt"
                log_path = os.path.join(output_dir, log_name)

                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"# Batch Save Log - {timestamp}\n")
                    f.write(f"# Prefix: {filename_prefix}\n\n")

                    for entry in save_details:
                        f.write(f"Index: {entry['index']}, File: {entry['filename']}, ")
                        f.write(f"X: {entry['x']}, Y: {entry['y']}\n")

                    ckpt = lz_pipe.get("ckpt_name", "")
                    if ckpt:
                        f.write(f"\nModel: {ckpt}\n")
                    h = lz_pipe.get("ckpt_hash", "")
                    if h:
                        f.write(f"Hash: {h}\n")
                    seed = lz_pipe.get("seed")
                    if seed is not None:
                        f.write(f"Seed: {seed}\n")
                    steps = lz_pipe.get("steps")
                    if steps is not None:
                        f.write(f"Steps: {steps}\n")
                    cfg = lz_pipe.get("cfg")
                    if cfg is not None:
                        f.write(f"CFG: {cfg}\n")
                    sn = lz_pipe.get("sampler_name", "")
                    if sn:
                        f.write(f"Sampler: {sn}\n")
                    sch = lz_pipe.get("scheduler", "")
                    if sch:
                        f.write(f"Scheduler: {sch}\n")
                    if pos_text:
                        f.write(f"\nPositive:\n{pos_text}\n")
                    if neg_text:
                        f.write(f"\nNegative:\n{neg_text}\n")

        return {"ui": {"images": results}}