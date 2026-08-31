# batch_save.py

import os
import json
import datetime
import numpy as np
from PIL import Image
import PIL.PngImagePlugin
import folder_paths

from .utils import sanitize_filename


def parse_labels(text):
    """カンマ区切り・改行区切りの両方に対応したラベル分割"""
    if not text or not text.strip():
        return []
    # カンマと改行の両方で分割
    parts = text.replace(",", "\n").split("\n")
    result = [p.strip() for p in parts if p.strip()]
    return result


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
                "image_format": (["png", "webp", "jpg"], {"default": "png"}),
                "save_txt_log": ("BOOLEAN", {"default": True}),
                "add_timestamp": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_batch"
    OUTPUT_NODE = True
    CATEGORY = "MyCustomNodes/IO"

    def save_batch(self, images, filename_prefix="ComfyUI", x_labels="", y_labels="",
                   lz_pipe=None, image_format="png", save_txt_log=True, add_timestamp=False,
                   prompt=None, extra_pnginfo=None):
        output_dir = folder_paths.get_output_directory()

        x_list = parse_labels(x_labels)
        y_list = parse_labels(y_labels)

        # ラベルが空の場合、インデックス番号で代替
        if not x_list and not y_list:
            file_labels = [f"{i:05d}" for i in range(images.shape[0])]
            x_count = images.shape[0]
            y_count = 1
        else:
            if not x_list:
                x_list = [""]
            if not y_list:
                y_list = [""]

            x_count = len(x_list)
            y_count = len(y_list)

            file_labels = []
            for y_val in y_list:
                for x_val in x_list:
                    label = filename_prefix
                    label = label.replace("{x}", x_val)
                    label = label.replace("{y}", y_val)
                    if not label:
                        label = "img"
                    file_labels.append(label)

            # 画像数が組み合わせ数を超える場合の補完
            if len(file_labels) < images.shape[0]:
                for i in range(len(file_labels), images.shape[0]):
                    file_labels.append(f"{filename_prefix or 'img'}_{i:05d}")

        results = []
        save_details = []

        total_images = images.shape[0]

        # add_timestamp=True の場合はファイル名に時刻を付与(再実行時の上書き防止)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else None

        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            file_label = file_labels[batch_number] if batch_number < len(file_labels) else f"{filename_prefix or 'img'}_{batch_number:05d}"
            # ラベル内のファイル名禁則文字をサニタイズ(サブフォルダ区切りの "/" は維持)
            file_label = "/".join(sanitize_filename(p) for p in file_label.split("/"))
            if timestamp:
                file_label = f"{file_label}_{timestamp}"
            file = f"{file_label}_{batch_number:05d}"

            full_output_folder, filename, counter, sub, filename_prefix_clean = \
                folder_paths.get_save_image_path(file, output_dir, images[0].shape[1], images[0].shape[0])

            img_path = os.path.join(full_output_folder, filename + "." + image_format)

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

            # parametersメタデータ構築
            param_lines = []
            if isinstance(lz_pipe, dict):
                ckpt_name = lz_pipe.get("ckpt_name", "")
                if ckpt_name:
                    param_lines.append(f"Model: {ckpt_name}")
                seed = lz_pipe.get("seed")
                steps = lz_pipe.get("steps")
                cfg = lz_pipe.get("cfg")
                sampler = lz_pipe.get("sampler_name", "")
                scheduler = lz_pipe.get("scheduler", "")
                width = lz_pipe.get("width", "")
                height = lz_pipe.get("height", "")
                if seed is not None:
                    parts = []
                    parts.append(f"Seed: {seed}")
                    if steps is not None:
                        parts.append(f"Steps: {steps}")
                    if cfg is not None:
                        parts.append(f"CFG: {cfg}")
                    if sampler:
                        parts.append(f"Sampler: {sampler}")
                    if scheduler:
                        parts.append(f"Scheduler: {scheduler}")
                    if width and height:
                        parts.append(f"Size: {width} x {height}")
                    param_lines.append(", ".join(parts))
            if pos_text:
                param_lines.append(f"Positive:\n{pos_text}")
            if neg_text:
                param_lines.append(f"Negative:\n{neg_text}")

            if param_lines:
                metadata.add_text("parameters", "\n".join(param_lines))

            # 画像保存
            if image_format == "png":
                if img.mode == 'RGBA':
                    img.save(img_path, pnginfo=metadata, compress_level=4)
                elif img.mode == 'L':
                    img.convert('RGB').save(img_path, pnginfo=metadata, compress_level=4)
                else:
                    img.save(img_path, pnginfo=metadata, compress_level=4)
            elif image_format == "jpg":
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(img_path, format="JPEG", quality=95)
            elif image_format == "webp":
                if img.mode == 'RGBA':
                    img.save(img_path, format="WEBP", lossless=True)
                elif img.mode == 'L':
                    img.convert('RGB').save(img_path, format="WEBP", lossless=True)
                else:
                    img.save(img_path, format="WEBP", lossless=True)

            # ラベルを安全に取得（画像数 > 組み合わせ数の場合に対応）
            if x_count > 0:
                x_idx = batch_number % x_count
            else:
                x_idx = 0
            if y_count > 0:
                y_idx = (batch_number // x_count) % y_count if x_count > 0 else batch_number % y_count
            else:
                y_idx = 0

            x_label = x_list[x_idx] if x_list else ""
            y_label = y_list[y_idx] if y_list else ""

            results.append({
                "filename": filename + "." + image_format,
                "subfolder": sub,
                "type": "output"
            })

            save_details.append({
                "index": batch_number,
                "filename": filename + "." + image_format,
                "x": x_label,
                "y": y_label,
            })

        # ログ出力
        if save_txt_log and isinstance(lz_pipe, dict):
            pos_text = lz_pipe.get("positive_text", "")
            neg_text = lz_pipe.get("negative_text", "")
            if pos_text or neg_text:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                log_name = f"batch_log_{timestamp}.txt"
                log_path = os.path.join(output_dir, log_name)

                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"# Batch Save Log - {timestamp}\n")
                    f.write(f"# Prefix: {filename_prefix}\n")
                    f.write(f"# Format: {image_format}\n")
                    f.write(f"# Total images: {total_images}\n\n")

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