# set_get.py

SET_GET_STORE = {}


class LZSetNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": ("STRING", {"default": "default"}),
            },
            "optional": {
                "value": ("*",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("trigger",)
    FUNCTION = "set_value"
    CATEGORY = "MyCustomNodes/Utils"
    OUTPUT_NODE = True

    def set_value(self, name, value=None):
        SET_GET_STORE[name] = value
        return (name,)


class LZGetNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": ("STRING", {"default": "default"}),
            },
            "optional": {
                "trigger": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_value"
    CATEGORY = "MyCustomNodes/Utils"

    def get_value(self, name, trigger=None):
        if name not in SET_GET_STORE:
            raise ValueError(f"LZGetNode Error: Value '{name}' not found in store.")
        value = SET_GET_STORE[name]
        if value is None:
            raise ValueError(f"LZGetNode Error: Value '{name}' is None.")
        return (value,)