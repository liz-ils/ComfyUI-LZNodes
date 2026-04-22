# text_utils.py

class StringNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "output_string"
    CATEGORY = "MyCustomNodes/Text"

    def output_string(self, text):
        return (text,)

class StringConcatNode:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "separator": ("STRING", {"default": ", "}),
            },
            "optional": {}
        }
        
        # 1〜5個までのテキスト入力ボックスを裏側で定義
        for i in range(1, 6):
            inputs["optional"][f"text{i}"] = ("STRING", {"multiline": False, "default": ""})
            
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "concat"
    CATEGORY = "MyCustomNodes/Text"

    def concat(self, separator=", ", **kwargs):
        texts = []
        
        # text1, text2 ... のキーを数字の順番通りに並び替える
        text_keys = [k for k in kwargs.keys() if k.startswith("text")]
        text_keys.sort(key=lambda x: int(x[4:]) if x[4:].isdigit() else 0)

        for key in text_keys:
            val = kwargs[key]
            # 文字列が入っていて、かつ空欄ではないものだけを抽出
            if isinstance(val, str) and val.strip() != "":
                texts.append(val.strip())
        
        # 指定された記号で結合
        result = separator.join(texts)
        return (result,)
        
LZBannedChars = r'\/?"<>\:|*'

class LZTextPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "preview"
    
    OUTPUT_NODE = True
    CATEGORY = "MyCustomNodes/Text"

    def preview(self, text):
        return {"ui": {"text": [text]}, "result": (text,)}


class LZStringSanitize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "mode": (["remove", "replace"],),
            },
            "optional": {
                "replace_with": ("STRING", {"default": "_"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "clean"
    CATEGORY = "MyCustomNodes/Text"

    def clean(self, text, mode, replace_with="_"):
        for char in LZBannedChars:
            if mode == "remove":
                text = text.replace(char, "")
            else:
                text = text.replace(char, replace_with)
        
        if text.endswith("."):
            text = text[:-1]
        
        return (text,)


class LZStringSelect:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                # 使用する入力ボックスの数
                "count": ("INT", {"default": 2, "min": 1, "max": 10}),
                # 出力するインデックス(1始まり)
                "select_index": ("INT", {"default": 1, "min": 1, "max": 10}),
            },
            "optional": {}
        }
        for i in range(1, 11):
            inputs["optional"][f"text{i}"] = ("STRING", {"default": ""})
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "select"
    CATEGORY = "MyCustomNodes/Text"

    def select(self, count, select_index, **kwargs):
        # 範囲外は空文字を返す
        if not isinstance(select_index, int) or not isinstance(count, int):
            return ("",)
        if count < 1:
            return ("",)
        if select_index < 1 or select_index > count:
            return ("",)

        key = f"text{select_index}"
        val = kwargs.get(key, "")
        if isinstance(val, str):
            return (val,)
        return ("",)


class LZSaveStringToCSV:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "filepath": ("STRING", {"default": "output.csv"}),
                "mode": (["write", "append"],),
                "header": ("STRING", {"default": ""}),
                "row_data": ("STRING", {"multiline": False}),
            },
            "optional": {}
        }
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "save_csv"
    OUTPUT_NODE = True
    CATEGORY = "MyCustomNodes/Text"

    def save_csv(self, filepath, mode, header, row_data):
        import os
        import csv
        
        filepath = filepath.strip()
        if not filepath:
            filepath = "output.csv"
        
        rows = []
        write_header = False
        
        if mode == "append" and os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
        else:
            write_header = True
        
        if write_header and header:
            rows.append([h.strip() for h in header.split(",")])
        
        if row_data.strip():
            rows.append([d.strip() for d in row_data.split(",")])
        
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        return (os.path.abspath(filepath),)


class LZPromptWeight:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": False, "default": ""}),
                "default_weight": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.9, "step": 0.05}),
                "separator": ("STRING", {"default": ", "}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "process"
    CATEGORY = "MyCustomNodes/Text"

    def process(self, prompt, default_weight, separator):
        if not prompt or not prompt.strip():
            return ("",)
        
        parts = [p.strip() for p in prompt.split(",")]
        result_parts = []
        
        for part in parts:
            if not part:
                continue
            
            if ":" in part:
                tag, weight_str = part.rsplit(":", 1)
                tag = tag.strip()
                try:
                    weight = float(weight_str)
                except ValueError:
                    weight = default_weight
            else:
                tag = part.strip()
                weight = default_weight
            
            if weight == 1.0:
                result_parts.append(tag)
            elif weight == int(weight):
                result_parts.append(f"({tag}:{int(weight)})")
            else:
                result_parts.append(f"({tag}:{weight})")
        
        result = separator.join(result_parts)
        return (result,)
