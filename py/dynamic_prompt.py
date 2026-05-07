# dynamic_prompt.py

import os
import random
import folder_paths


def get_csv_files():
    csv_dir = os.path.join(folder_paths.get_output_directory(), "prompt_csv")
    os.makedirs(csv_dir, exist_ok=True)
    if os.path.isdir(csv_dir):
        files = [f for f in os.listdir(csv_dir) if f.endswith(('.csv', '.txt'))]
        return [""] + sorted(files) if files else [""]
    return [""]


def read_csv_options(csv_name):
    if not csv_name:
        return []
    csv_dir = os.path.join(folder_paths.get_output_directory(), "prompt_csv")
    csv_path = os.path.join(csv_dir, csv_name)
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    return []


def replace_placeholder(text, placeholder, replacement):
    if not placeholder:
        return text
    return text.replace(placeholder, replacement)


class LZPromptReplaceSingle:
    @classmethod
    def INPUT_TYPES(s):
        csv_files = get_csv_files()
        return {
            "required": {
                "template": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "placeholder": ("STRING", {"default": "_hair_"}),
                "csv_file": (csv_files, {"default": ""}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "replace_single"
    CATEGORY = "MyCustomNodes/Text"

    def replace_single(self, template, placeholder, csv_file):
        if not csv_file:
            return (template,)

        options = read_csv_options(csv_file)
        if not options:
            return (template,)

        selected = random.choice(options)
        result = replace_placeholder(template, placeholder, selected)
        return (result,)


class LZPromptReplaceMulti:
    @classmethod
    def INPUT_TYPES(s):
        csv_files = get_csv_files()
        inputs = {
            "required": {
                "template": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            },
            "optional": {}
        }

        for i in range(1, 6):
            inputs["required"][f"placeholder{i}"] = ("STRING", {"default": ""})
            inputs["required"][f"csv_file{i}"] = (csv_files, {"default": ""})

        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "replace_multi"
    CATEGORY = "MyCustomNodes/Text"

    def replace_multi(self, template, **kwargs):
        result = template

        for i in range(1, 6):
            placeholder = kwargs.get(f"placeholder{i}", "")
            csv_file = kwargs.get(f"csv_file{i}", "")

            if not placeholder or not csv_file:
                continue

            options = read_csv_options(csv_file)
            if not options:
                continue

            selected = random.choice(options)
            result = replace_placeholder(result, placeholder, selected)

        return (result,)


class LZPromptReplaceString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "template": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "placeholder": ("STRING", {"default": "_hair_"}),
                "candidates": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "replace_string"
    CATEGORY = "MyCustomNodes/Text"

    def replace_string(self, template, placeholder, candidates):
        if not candidates.strip():
            return (template,)

        lines = [line.strip() for line in candidates.split('\n') if line.strip()]
        if not lines:
            return (template,)

        selected = random.choice(lines)
        result = replace_placeholder(template, placeholder, selected)
        return (result,)