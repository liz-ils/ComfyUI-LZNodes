# pipe_merge.py

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
        
        return (merged,)
