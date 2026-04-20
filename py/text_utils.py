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
                "count": ("INT", {"default": 2, "min": 2, "max": 10}),
                "join_mode": (["separator", "newline"],),
                "separator": ("STRING", {"default": ", "}),
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

    def select(self, count, join_mode, separator=", ", **kwargs):
        texts = []
        for i in range(1, count + 1):
            key = f"text{i}"
            if kwargs.get(key, "").strip():
                texts.append(kwargs[key].strip())
        
        if count == 1:
            return ("",)
        
        join_str = "\n" if join_mode == "newline" else separator
        return (join_str.join(texts),)


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
