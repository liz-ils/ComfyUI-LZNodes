# log_reader.py

import os
import re


class LZLogReader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filepath": ("STRING", {"default": ""}),
                "index": ("INT", {"default": -1, "min": -1, "max": 10000}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "FLOAT", "STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("positive_text", "negative_text", "seed", "steps", "cfg", "sampler_name", "scheduler", "width", "height", "ckpt_name")
    FUNCTION = "read_log"
    CATEGORY = "MyCustomNodes/Log"

    def read_log(self, filepath, index):
        if not filepath or not os.path.exists(filepath):
            return ("", "", 0, 20, 8.0, "euler", "normal", 512, 512, "")

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = self._parse_log_blocks(content)

        if not blocks:
            return ("", "", 0, 20, 8.0, "euler", "normal", 512, 512, "")

        if index == -1 or index >= len(blocks):
            block = blocks[-1]
        else:
            block = blocks[index]

        positive = block.get("positive", "")
        negative = block.get("negative", "")
        seed = self._try_int(block.get("seed", 0))
        steps = self._try_int(block.get("steps", 20))
        cfg = self._try_float(block.get("cfg", 8.0))
        sampler = block.get("sampler", "euler")
        scheduler = block.get("scheduler", "normal")
        width = self._try_int(block.get("width", 512))
        height = self._try_int(block.get("height", 512))
        ckpt_name = block.get("ckpt_name", "")

        return (positive, negative, seed, steps, cfg, sampler, scheduler, width, height, ckpt_name)

    def _parse_log_blocks(self, content):
        blocks = []
        current_block = {}
        
        lines = content.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if line.startswith("Date:"):
                if current_block:
                    blocks.append(current_block)
                current_block = {}
                current_block["date"] = line[5:].strip()
                i += 1
                continue
            
            if line.startswith("Model:"):
                model_part = line[6:].strip()
                hash_match = re.search(r'\(Hash: ([a-f0-9]+)\)', model_part)
                if hash_match:
                    current_block["ckpt_hash"] = hash_match.group(1)
                    model_name = model_part.replace("(Hash:", "").replace(hash_match.group(1) + ")", "").strip()
                else:
                    model_name = model_part
                current_block["ckpt_name"] = model_name
                i += 1
                continue
            
            if line.startswith("Size:"):
                size_str = line[5:].strip()
                match = re.match(r'(\d+)\s*x\s*(\d+)', size_str)
                if match:
                    current_block["width"] = int(match.group(1))
                    current_block["height"] = int(match.group(2))
                i += 1
                continue
            
            if line.startswith("Seed:"):
                seed_match = re.match(r'Seed:\s*(\d+)\s*\|\s*Steps:\s*(\d+)\s*\|\s*CFG:\s*([\d.]+)\s*\|\s*Sampler:\s*(\w+)\s*\|\s*Scheduler:\s*(\w+)', line)
                if seed_match:
                    current_block["seed"] = seed_match.group(1)
                    current_block["steps"] = seed_match.group(2)
                    current_block["cfg"] = seed_match.group(3)
                    current_block["sampler"] = seed_match.group(4)
                    current_block["scheduler"] = seed_match.group(5)
                i += 1
                continue
            
            if line.startswith("Positive:"):
                i += 1
                positive_lines = []
                while i < len(lines) and not lines[i].startswith("Negative:"):
                    positive_lines.append(lines[i])
                    i += 1
                current_block["positive"] = "\n".join(positive_lines).strip()
                continue
            
            if line.startswith("Negative:"):
                i += 1
                negative_lines = []
                while i < len(lines):
                    if lines[i].strip() == "" or (not lines[i].startswith(" ") and not lines[i].startswith("\t")):
                        if lines[i].strip() != "":
                            break
                    negative_lines.append(lines[i])
                    i += 1
                    if i < len(lines) and (lines[i].startswith("Date:") or lines[i].startswith("Model:")):
                        break
                current_block["negative"] = "\n".join(negative_lines).strip()
                continue
            
            i += 1

        if current_block:
            blocks.append(current_block)

        return blocks

    def _try_int(self, value):
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0

    def _try_float(self, value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
