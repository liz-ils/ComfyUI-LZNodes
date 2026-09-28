# pipe_merge.py

from .utils import get_pipe_loras, set_pipe_loras

class LZPipeMerge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lz_pipe_main": ("LZ_PIPE",),
                "lz_pipe_sub": ("LZ_PIPE",),
            },
        }

    RETURN_TYPES = ("LZ_PIPE",)
    RETURN_NAMES = ("lz_pipe",)
    FUNCTION = "merge_pipes"
    CATEGORY = "MyCustomNodes/Pipe"

    def merge_pipes(self, lz_pipe_main, lz_pipe_sub):
        merged = {}
        
        if lz_pipe_main is not None and isinstance(lz_pipe_main, dict):
            for key, value in lz_pipe_main.items():
                if value is not None:
                    merged[key] = value
        
        if lz_pipe_sub is not None and isinstance(lz_pipe_sub, dict):
            for key, value in lz_pipe_sub.items():
                if key not in merged or merged[key] is None:
                    if value is not None:
                        merged[key] = value

        # 複数 LoRA 対応: 両 pipe の LoRA は結合して文字列も再同期する
        try:
            main_loras = get_pipe_loras(lz_pipe_main if isinstance(lz_pipe_main, dict) else {})
            sub_loras = get_pipe_loras(lz_pipe_sub if isinstance(lz_pipe_sub, dict) else {})
            if main_loras or sub_loras:
                set_pipe_loras(merged, list(main_loras) + list(sub_loras))
        except Exception:
            pass

        return (merged,)
