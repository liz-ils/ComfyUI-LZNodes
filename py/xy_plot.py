# xy_plot.py

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import folder_paths
import comfy.sd
import comfy.samplers
import comfy.utils
import nodes
import gc
import os
import hashlib

CHECKPOINT_HASH_CACHE = {}
CHECKPOINT_LOADER_CACHE = {}


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


def parse_values(text):
    if not text:
        return []
    lines = text.strip().split('\n')
    return [line.strip() for line in lines if line.strip()]


def escape_prompt(text, enabled=True):
    if not enabled:
        return text
    text = text.replace("_", " ")
    text = text.replace("(", "\\(").replace(")", "\\)")
    return text


def replace_prompt_text(base_prompt, replace_key, new_value, escape=True):
    if not replace_key:
        return new_value

    replace_key_escaped = escape_prompt(replace_key, escape)
    new_value_escaped = escape_prompt(new_value, escape)

    result = base_prompt.replace(replace_key_escaped, new_value_escaped)
    return result


class LZXYPlot:
    @classmethod
    def INPUT_TYPES(s):
        param_options = ["none", "checkpoint", "lora", "sampler", "scheduler", "positive", "negative"]
        return {
            "required": {
                "x_type": (param_options, {"default": "none"}),
                "x_values": ("STRING", {"multiline": True, "default": ""}),
                "y_type": (param_options, {"default": "none"}),
                "y_values": ("STRING", {"multiline": True, "default": ""}),
                "y_replace_key": ("STRING", {"default": ""}),
                "x_replace_key": ("STRING", {"default": ""}),
                "replace_escape": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "lz_pipe": ("LZ_PIPE",),
            }
        }

    RETURN_TYPES = ("LZ_PIPE", "STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("lz_pipe", "x_labels", "y_labels", "y_replace_key", "x_replace_key", "total_count")
    FUNCTION = "process_xy"
    CATEGORY = "MyCustomNodes/XY"

    def process_xy(self, x_type, x_values, y_type, y_values, y_replace_key, x_replace_key, replace_escape, lz_pipe=None):
        if lz_pipe is None:
            lz_pipe = {}

        x_list = parse_values(x_values)
        y_list = parse_values(y_values)

        has_x = x_type != "none" and len(x_list) > 0
        has_y = y_type != "none" and len(y_list) > 0

        if not has_x and not has_y:
            raise ValueError("LZXYPlot Error: Either X or Y must have values.")

        if not has_x:
            x_list = ["idx"]
            x_type = "none"
        if not has_y:
            y_list = ["idx"]
            y_type = "none"

        x_labels = [v for v in x_list]
        y_labels = [v for v in y_list]

        total_count = len(x_list) * len(y_list)

        new_pipe = lz_pipe.copy()
        new_pipe["_x_type"] = x_type
        new_pipe["_x_values"] = x_list
        new_pipe["_y_type"] = y_type
        new_pipe["_y_values"] = y_list
        new_pipe["_y_replace_key"] = y_replace_key
        new_pipe["_x_replace_key"] = x_replace_key
        new_pipe["_replace_escape"] = replace_escape
        new_pipe["_x_labels"] = ",".join(x_labels)
        new_pipe["_y_labels"] = ",".join(y_labels)

        return (new_pipe, ",".join(x_labels), ",".join(y_labels), y_replace_key if y_replace_key else "", x_replace_key if x_replace_key else "", total_count)


class LZXYPlotSampler:
    def __init__(self):
        self.loaded_loras = {}

    @classmethod
    def INPUT_TYPES(s):
        param_options = ["none", "checkpoint", "lora", "sampler", "scheduler", "positive", "negative"]
        return {
            "required": {
                "x_type": (param_options, {"default": "none"}),
                "x_values": ("STRING", {"multiline": True, "default": ""}),
                "y_type": (param_options, {"default": "none"}),
                "y_values": ("STRING", {"multiline": True, "default": ""}),
                "y_replace_key": ("STRING", {"default": ""}),
                "x_replace_key": ("STRING", {"default": ""}),
                "replace_escape": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "lz_pipe": ("LZ_PIPE",),
                "latent_image": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE", "LZ_PIPE", "STRING", "STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("images", "lz_pipe", "x_labels", "y_labels", "y_replace_key", "x_replace_key", "width", "height")
    FUNCTION = "xy_sample"
    CATEGORY = "MyCustomNodes/XY"

    def xy_sample(self, x_type, x_values, y_type, y_values, y_replace_key, x_replace_key, replace_escape,
                  seed, steps, cfg, sampler_name, scheduler, denoise,
                  lz_pipe=None, latent_image=None):

        if lz_pipe is None:
            lz_pipe = {}

        x_list = parse_values(x_values)
        y_list = parse_values(y_values)

        has_x = x_type != "none" and len(x_list) > 0
        has_y = y_type != "none" and len(y_list) > 0

        if not has_x and not has_y:
            raise ValueError("LZXYPlotSampler Error: Either X or Y must have values.")

        if not has_x:
            x_list = ["idx"]
            x_type = "none"
        if not has_y:
            y_list = ["idx"]
            y_type = "none"

        images_list = []
        width = lz_pipe.get("width", 512)
        height = lz_pipe.get("height", 512)

        if latent_image is None:
            latent_image = lz_pipe.get("latent")
        if latent_image is None:
            latent_w = width // 8
            latent_h = height // 8
            latent_image = {"samples": torch.zeros([1, 4, latent_h, latent_w])}

        base_model = lz_pipe.get("model")
        base_clip = lz_pipe.get("clip")
        base_vae = lz_pipe.get("vae")
        base_positive = lz_pipe.get("positive")
        base_negative = lz_pipe.get("negative")
        base_pos_text = lz_pipe.get("positive_text", "")
        base_neg_text = lz_pipe.get("negative_text", "")

        current_model = base_model
        current_clip = base_clip
        current_vae = base_vae

        for y_idx, y_val in enumerate(y_list):
            for x_idx, x_val in enumerate(x_list):
                model = current_model
                clip = current_clip
                vae = current_vae

                pipe_model = lz_pipe.get("model")
                pipe_clip = lz_pipe.get("clip")
                pipe_vae = lz_pipe.get("vae")

                if x_type == "checkpoint":
                    ckpt_name = x_val
                    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
                    if ckpt_path is None:
                        raise ValueError(f"LZXYPlotSampler Error: Checkpoint not found: {ckpt_name}")

                    if model is not None:
                        del model
                        del clip
                        del vae
                        gc.collect()

                    out = comfy.sd.load_checkpoint_guess_config(
                        ckpt_path,
                        output_vae=True,
                        output_clip=True,
                        embedding_directory=folder_paths.get_folder_paths("embeddings")
                    )
                    model, clip, vae = out[:3]

                    ckpt_hash = get_checkpoint_hash(ckpt_path)
                    lz_pipe["ckpt_name"] = ckpt_name
                    lz_pipe["ckpt_hash"] = ckpt_hash

                elif x_type == "lora":
                    parts = x_val.split(":")
                    lora_name = parts[0]
                    model_weight = float(parts[1]) if len(parts) > 1 else 1.0
                    clip_weight = float(parts[2]) if len(parts) > 2 else 1.0

                    if model_weight == 0 and clip_weight == 0:
                        pass
                    else:
                        lora_path = folder_paths.get_full_path("loras", lora_name)
                        if lora_path is None:
                            raise ValueError(f"LZXYPlotSampler Error: LoRA not found: {lora_name}")

                        if lora_path in self.loaded_loras:
                            lora = self.loaded_loras[lora_path]
                        else:
                            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                            self.loaded_loras[lora_path] = lora

                        model, clip = comfy.sd.load_lora_for_models(model, clip, lora, model_weight, clip_weight)

                if y_type == "checkpoint":
                    ckpt_name = y_val
                    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
                    if ckpt_path is None:
                        raise ValueError(f"LZXYPlotSampler Error: Checkpoint not found: {ckpt_name}")

                    if model is not None:
                        del model
                        del clip
                        del vae
                        gc.collect()

                    out = comfy.sd.load_checkpoint_guess_config(
                        ckpt_path,
                        output_vae=True,
                        output_clip=True,
                        embedding_directory=folder_paths.get_folder_paths("embeddings")
                    )
                    model, clip, vae = out[:3]

                    ckpt_hash = get_checkpoint_hash(ckpt_path)
                    lz_pipe["ckpt_name"] = ckpt_name
                    lz_pipe["ckpt_hash"] = ckpt_hash

                elif y_type == "lora":
                    parts = y_val.split(":")
                    lora_name = parts[0]
                    model_weight = float(parts[1]) if len(parts) > 1 else 1.0
                    clip_weight = float(parts[2]) if len(parts) > 2 else 1.0

                    if model_weight == 0 and clip_weight == 0:
                        pass
                    else:
                        lora_path = folder_paths.get_full_path("loras", lora_name)
                        if lora_path is None:
                            raise ValueError(f"LZXYPlotSampler Error: LoRA not found: {lora_name}")

                        if lora_path in self.loaded_loras:
                            lora = self.loaded_loras[lora_path]
                        else:
                            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                            self.loaded_loras[lora_path] = lora

                        model, clip = comfy.sd.load_lora_for_models(model, clip, lora, model_weight, clip_weight)

                if y_type == "positive":
                    if clip is None:
                        raise ValueError("LZXYPlotSampler Error: CLIP is required for positive prompt encoding.")
                    replaced_text = replace_prompt_text(base_pos_text, y_replace_key, y_val, replace_escape)
                    tokens_pos = clip.tokenize(replaced_text)
                    cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
                    positive = [[cond_pos, {"pooled_output": pooled_pos}]]
                    lz_pipe["positive_text"] = replaced_text

                elif y_type == "negative":
                    if clip is None:
                        raise ValueError("LZXYPlotSampler Error: CLIP is required for negative prompt encoding.")
                    replaced_text = replace_prompt_text(base_neg_text, y_replace_key, y_val, replace_escape)
                    tokens_neg = clip.tokenize(replaced_text)
                    cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
                    negative = [[cond_neg, {"pooled_output": pooled_neg}]]
                    lz_pipe["negative_text"] = replaced_text

                elif y_type == "sampler":
                    lz_pipe["sampler_name"] = y_val

                elif y_type == "scheduler":
                    lz_pipe["scheduler"] = y_val

                if y_type == "none" or y_type not in ["positive", "negative"]:
                    positive = base_positive
                    negative = base_negative

                if y_type == "none" or y_type == "sampler":
                    sampler_name = lz_pipe.get("sampler_name", sampler_name)
                if y_type == "none" or y_type == "scheduler":
                    scheduler = lz_pipe.get("scheduler", scheduler)

                if model is None or positive is None or negative is None:
                    raise ValueError("LZXYPlotSampler Error: Missing required data (model, positive, negative).")

                if vae is None:
                    raise ValueError("LZXYPlotSampler Error: VAE is required.")

                current_seed = seed + (y_idx * len(x_list) + x_idx)

                ksampler = nodes.KSampler()
                sampled_latent = ksampler.sample(model, current_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)[0]

                vae_decoder = nodes.VAEDecode()
                image = vae_decoder.decode(vae, sampled_latent)[0]

                images_list.append(image)

                current_model = model
                current_clip = clip
                current_vae = vae

        if len(images_list) == 0:
            raise ValueError("LZXYPlotSampler Error: No images generated.")

        images_tensor = torch.cat(images_list, dim=0)

        x_labels = [v for v in x_list]
        y_labels = [v for v in y_list]

        new_pipe = lz_pipe.copy()
        new_pipe["model"] = current_model
        new_pipe["clip"] = current_clip
        new_pipe["vae"] = current_vae
        new_pipe["latent"] = latent_image

        return (images_tensor, new_pipe, ",".join(x_labels), ",".join(y_labels), y_replace_key if y_replace_key else "", x_replace_key if x_replace_key else "", width, height)


class LZXYSampler:
    def __init__(self):
        self.loaded_loras = {}

    @classmethod
    def INPUT_TYPES(s):
        param_options = ["none", "checkpoint", "lora", "sampler", "scheduler", "positive", "negative"]
        inputs = {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "lz_pipe": ("LZ_PIPE",),
                "x_type": (param_options, {"default": "none"}),
                "x_values": ("STRING", {"multiline": True, "default": ""}),
                "y_type": (param_options, {"default": "none"}),
                "y_values": ("STRING", {"multiline": True, "default": ""}),
                "y_replace_key": ("STRING", {"default": ""}),
                "x_replace_key": ("STRING", {"default": ""}),
                "replace_escape": ("BOOLEAN", {"default": True}),
                "x_labels": ("STRING", {"forceInput": True}),
                "y_labels": ("STRING", {"forceInput": True}),
                "replace_key": ("STRING", {"forceInput": True}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "positive_text": ("STRING", {"forceInput": True}),
                "negative_text": ("STRING", {"forceInput": True}),
                "latent_image": ("LATENT",),
            }
        }
        return inputs

    RETURN_TYPES = ("IMAGE", "LZ_PIPE", "INT", "INT")
    RETURN_NAMES = ("images", "lz_pipe", "width", "height")
    FUNCTION = "xy_sample"
    CATEGORY = "MyCustomNodes/XY"

    def xy_sample(self, seed, steps, cfg, sampler_name, scheduler, denoise,
                  lz_pipe=None, x_type="none", x_values="", y_type="none", y_values="",
                  y_replace_key="", x_replace_key="", replace_escape=True,
                  x_labels=None, y_labels=None, replace_key=None,
                  model=None, clip=None, vae=None, positive=None, negative=None,
                  positive_text="", negative_text="", latent_image=None):

        if lz_pipe is None:
            lz_pipe = {}

        x_list = parse_values(x_values)
        y_list = parse_values(y_values)

        has_x = x_type != "none" and len(x_list) > 0
        has_y = y_type != "none" and len(y_list) > 0

        if not has_x and not has_y:
            raise ValueError("LZXYSampler Error: Either X or Y must have values.")

        if not has_x:
            x_list = ["idx"]
            x_type = "none"
        if not has_y:
            y_list = ["idx"]
            y_type = "none"

        if x_labels is None:
            x_labels_out = ",".join(x_list)
        else:
            x_labels_out = x_labels

        if y_labels is None:
            y_labels_out = ",".join(y_list)
        else:
            y_labels_out = y_labels

        replace_key_out = y_replace_key if y_replace_key else x_replace_key

        width = lz_pipe.get("width", 512)
        height = lz_pipe.get("height", 512)

        if latent_image is None:
            latent_image = lz_pipe.get("latent")
        if latent_image is None:
            latent_w = width // 8
            latent_h = height // 8
            latent_image = {"samples": torch.zeros([1, 4, latent_h, latent_w])}

        base_model = model if model is not None else lz_pipe.get("model")
        base_clip = clip if clip is not None else lz_pipe.get("clip")
        base_vae = vae if vae is not None else lz_pipe.get("vae")
        base_positive = positive if positive is not None else lz_pipe.get("positive")
        base_negative = negative if negative is not None else lz_pipe.get("negative")
        base_pos_text = positive_text if positive_text else lz_pipe.get("positive_text", "")
        base_neg_text = negative_text if negative_text else lz_pipe.get("negative_text", "")

        if base_model is None or base_clip is None or base_vae is None:
            raise ValueError("LZXYSampler Error: Model, CLIP, and VAE are required.")

        current_model = base_model
        current_clip = base_clip
        current_vae = base_vae

        images_list = []

        for y_idx, y_val in enumerate(y_list):
            for x_idx, x_val in enumerate(x_list):
                model = current_model
                clip = current_clip
                vae = current_vae

                temp_pipe = lz_pipe.copy()
                temp_pipe["model"] = model
                temp_pipe["clip"] = clip
                temp_pipe["vae"] = vae

                temp_positive = base_positive
                temp_negative = base_negative
                temp_pos_text = base_pos_text
                temp_neg_text = base_neg_text
                temp_sampler = sampler_name
                temp_scheduler = scheduler

                if x_type == "checkpoint":
                    ckpt_name = x_val
                    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
                    if ckpt_path is None:
                        raise ValueError(f"LZXYSampler Error: Checkpoint not found: {ckpt_name}")

                    if model is not None:
                        del model
                        del clip
                        del vae
                        gc.collect()

                    out = comfy.sd.load_checkpoint_guess_config(
                        ckpt_path,
                        output_vae=True,
                        output_clip=True,
                        embedding_directory=folder_paths.get_folder_paths("embeddings")
                    )
                    model, clip, vae = out[:3]
                    temp_pipe["model"] = model
                    temp_pipe["clip"] = clip
                    temp_pipe["vae"] = vae
                    temp_pipe["ckpt_name"] = ckpt_name

                elif x_type == "lora":
                    parts = x_val.split(":")
                    lora_name = parts[0]
                    model_weight = float(parts[1]) if len(parts) > 1 else 1.0
                    clip_weight = float(parts[2]) if len(parts) > 2 else 1.0

                    if model_weight != 0 or clip_weight != 0:
                        lora_path = folder_paths.get_full_path("loras", lora_name)
                        if lora_path is None:
                            raise ValueError(f"LZXYSampler Error: LoRA not found: {lora_name}")

                        if lora_path in self.loaded_loras:
                            lora = self.loaded_loras[lora_path]
                        else:
                            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                            self.loaded_loras[lora_path] = lora

                        model, clip = comfy.sd.load_lora_for_models(model, clip, lora, model_weight, clip_weight)
                        temp_pipe["model"] = model
                        temp_pipe["clip"] = clip

                elif x_type == "positive":
                    if clip is None:
                        raise ValueError("LZXYSampler Error: CLIP is required for positive prompt encoding.")
                    replaced_text = replace_prompt_text(base_pos_text, x_replace_key, x_val, replace_escape)
                    tokens_pos = clip.tokenize(replaced_text)
                    cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
                    temp_positive = [[cond_pos, {"pooled_output": pooled_pos}]]
                    temp_pos_text = replaced_text

                elif x_type == "negative":
                    if clip is None:
                        raise ValueError("LZXYSampler Error: CLIP is required for negative prompt encoding.")
                    replaced_text = replace_prompt_text(base_neg_text, x_replace_key, x_val, replace_escape)
                    tokens_neg = clip.tokenize(replaced_text)
                    cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
                    temp_negative = [[cond_neg, {"pooled_output": pooled_neg}]]
                    temp_neg_text = replaced_text

                elif x_type == "sampler":
                    temp_sampler = x_val

                elif x_type == "scheduler":
                    temp_scheduler = x_val

                if y_type == "checkpoint":
                    ckpt_name = y_val
                    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
                    if ckpt_path is None:
                        raise ValueError(f"LZXYSampler Error: Checkpoint not found: {ckpt_name}")

                    if model is not None:
                        del model
                        del clip
                        del vae
                        gc.collect()

                    out = comfy.sd.load_checkpoint_guess_config(
                        ckpt_path,
                        output_vae=True,
                        output_clip=True,
                        embedding_directory=folder_paths.get_folder_paths("embeddings")
                    )
                    model, clip, vae = out[:3]
                    temp_pipe["model"] = model
                    temp_pipe["clip"] = clip
                    temp_pipe["vae"] = vae
                    temp_pipe["ckpt_name"] = ckpt_name

                elif y_type == "lora":
                    parts = y_val.split(":")
                    lora_name = parts[0]
                    model_weight = float(parts[1]) if len(parts) > 1 else 1.0
                    clip_weight = float(parts[2]) if len(parts) > 2 else 1.0

                    if model_weight != 0 or clip_weight != 0:
                        lora_path = folder_paths.get_full_path("loras", lora_name)
                        if lora_path is None:
                            raise ValueError(f"LZXYSampler Error: LoRA not found: {lora_name}")

                        if lora_path in self.loaded_loras:
                            lora = self.loaded_loras[lora_path]
                        else:
                            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                            self.loaded_loras[lora_path] = lora

                        model, clip = comfy.sd.load_lora_for_models(model, clip, lora, model_weight, clip_weight)
                        temp_pipe["model"] = model
                        temp_pipe["clip"] = clip

                elif y_type == "positive":
                    if clip is None:
                        raise ValueError("LZXYSampler Error: CLIP is required for positive prompt encoding.")
                    replaced_text = replace_prompt_text(base_pos_text, y_replace_key, y_val, replace_escape)
                    tokens_pos = clip.tokenize(replaced_text)
                    cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
                    temp_positive = [[cond_pos, {"pooled_output": pooled_pos}]]
                    temp_pos_text = replaced_text

                elif y_type == "negative":
                    if clip is None:
                        raise ValueError("LZXYSampler Error: CLIP is required for negative prompt encoding.")
                    replaced_text = replace_prompt_text(base_neg_text, y_replace_key, y_val, replace_escape)
                    tokens_neg = clip.tokenize(replaced_text)
                    cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
                    temp_negative = [[cond_neg, {"pooled_output": pooled_neg}]]
                    temp_neg_text = replaced_text

                elif y_type == "sampler":
                    temp_sampler = y_val

                elif y_type == "scheduler":
                    temp_scheduler = y_val

                if temp_positive is None:
                    temp_positive = base_positive
                if temp_negative is None:
                    temp_negative = base_negative

                if model is None or temp_positive is None or temp_negative is None:
                    raise ValueError("LZXYSampler Error: Missing required data (model, positive, negative).")

                if vae is None:
                    raise ValueError("LZXYSampler Error: VAE is required.")

                current_seed = seed + (y_idx * len(x_list) + x_idx)

                ksampler = nodes.KSampler()
                sampled_latent = ksampler.sample(model, current_seed, steps, cfg, temp_sampler, temp_scheduler, temp_positive, temp_negative, latent_image, denoise)[0]

                vae_decoder = nodes.VAEDecode()
                image = vae_decoder.decode(vae, sampled_latent)[0]

                images_list.append(image)

                current_model = model
                current_clip = clip
                current_vae = vae

        if len(images_list) == 0:
            raise ValueError("LZXYSampler Error: No images generated.")

        images_tensor = torch.cat(images_list, dim=0)

        new_pipe = lz_pipe.copy()
        new_pipe["model"] = current_model
        new_pipe["clip"] = current_clip
        new_pipe["vae"] = current_vae
        new_pipe["latent"] = latent_image
        new_pipe["positive"] = temp_positive
        new_pipe["negative"] = temp_negative
        new_pipe["positive_text"] = temp_pos_text
        new_pipe["negative_text"] = temp_neg_text
        new_pipe["sampler_name"] = temp_sampler
        new_pipe["scheduler"] = temp_scheduler
        new_pipe["seed"] = seed
        new_pipe["steps"] = steps
        new_pipe["cfg"] = cfg
        new_pipe["_x_labels"] = x_labels_out
        new_pipe["_y_labels"] = y_labels_out
        new_pipe["_replace_key"] = replace_key_out

        return (images_tensor, new_pipe, width, height)


class LZXYGridOutput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "x_axis_labels": ("STRING", {"multiline": True, "default": ""}),
                "y_axis_labels": ("STRING", {"multiline": True, "default": ""}),
                "columns": ("INT", {"default": 1, "min": 1, "max": 100}),
                "label_font_size": ("INT", {"default": 16, "min": 8, "max": 72}),
                "add_border": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "parameter_text")
    FUNCTION = "create_grid"
    OUTPUT_NODE = True
    CATEGORY = "MyCustomNodes/XY"

    def create_grid(self, images, x_axis_labels, y_axis_labels, columns, label_font_size, add_border):
        x_labels = parse_values(x_axis_labels)
        y_labels = parse_values(y_axis_labels)

        num_images = images.shape[0]
        rows = (num_images + columns - 1) // columns
        cols = min(columns, num_images)

        if len(y_labels) == 1 and y_labels[0] == "idx":
            y_labels = [str(i) for i in range(num_images)]
            rows = num_images
            cols = 1

        if len(x_labels) == 1 and x_labels[0] == "idx":
            if num_images > 1:
                x_labels = [f"idx{i}" for i in range(num_images)]
                cols = num_images
                rows = 1

        if len(x_labels) != cols:
            if len(x_labels) == 1:
                x_labels = [x_labels[0]] * cols
            else:
                x_labels = [f"x{i}" for i in range(cols)]

        if len(y_labels) != rows:
            if len(y_labels) == 1:
                y_labels = [y_labels[0]] * rows
            else:
                y_labels = [f"y{i}" for i in range(rows)]

        if num_images != cols * rows:
            raise ValueError(f"LZXYGridOutput Error: Image count ({num_images}) does not match grid size ({cols}x{rows}={cols*rows}).")

        img_height = images.shape[1]
        img_width = images.shape[2]
        channels = images.shape[3]

        label_height = label_font_size * 2
        label_width = label_font_size * 6

        grid_width = label_width + cols * img_width + (cols + 1) if add_border else cols * img_width
        grid_height = label_height + rows * img_height + (rows + 1) if add_border else rows * img_height

        try:
            font = ImageFont.truetype("arial.ttf", label_font_size)
        except:
            font = ImageFont.load_default()

        if channels == 4:
            grid_img = Image.new("RGBA", (grid_width, grid_height), (255, 255, 255, 255))
        else:
            grid_img = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

        draw = ImageDraw.Draw(grid_img)

        param_text_lines = []

        for row in range(rows):
            y_label = y_labels[row] if row < len(y_labels) else f"y{row}"

            for col in range(cols):
                idx = row * cols + col
                if idx >= num_images:
                    break

                img_data = images[idx].cpu().numpy()
                if channels == 4:
                    img = Image.fromarray((img_data * 255).astype(np.uint8), mode="RGBA")
                else:
                    if channels == 1:
                        img = Image.fromarray((img_data.squeeze() * 255).astype(np.uint8), mode="L")
                    else:
                        img = Image.fromarray((img_data * 255).astype(np.uint8), mode="RGB")

                x_pos = label_width + col * img_width + (col + 1) if add_border else label_width + col * img_width
                y_pos = label_height + row * img_height + (row + 1) if add_border else label_height + row * img_height

                grid_img.paste(img, (x_pos, y_pos))

                if add_border:
                    draw.rectangle([x_pos, y_pos, x_pos + img_width, y_pos + img_height], outline="black", width=1)

                x_label = x_labels[col] if col < len(x_labels) else f"x{col}"
                param_text_lines.append(f"[{row},{col}] X={x_label} Y={y_label}")

                if row == 0:
                    draw.text((x_pos + img_width // 2 - 20, label_font_size // 2), x_label[:15], fill="black", font=font)

            draw.text((label_font_size // 2, y_pos + img_height // 2 - label_font_size // 2), y_label[:15], fill="black", font=font)

        result_img = torch.from_numpy(np.array(grid_img).astype(np.float32) / 255.0)
        if result_img.shape[2] == 4:
            result_img = result_img[:, :, :3]
        result_img = result_img.permute(2, 0, 1).unsqueeze(0)

        param_text = "\n".join(param_text_lines)

        return (result_img, param_text)