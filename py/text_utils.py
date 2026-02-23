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
