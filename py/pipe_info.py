# pipe_info.py

class LZPipeInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            },
            "optional": {
                "lz_pipe": ("LZ_PIPE",),
            }
        }

    RETURN_TYPES = ("LZ_PIPE", "STRING")
    RETURN_NAMES = ("lz_pipe", "info_text")
    FUNCTION = "pipe_info"
    OUTPUT_NODE = True
    CATEGORY = "MyCustomNodes/Pipe"

    def pipe_info(self, text, lz_pipe=None):
        lines = []
        
        if text and text.strip():
            lines.append(f"=== {text.strip()} ===")
            lines.append("")
        
        if lz_pipe is not None and isinstance(lz_pipe, dict):
            for key, value in sorted(lz_pipe.items()):
                if key.startswith("_"):
                    continue
                if isinstance(value, dict):
                    lines.append(f"{key}: <dict with {len(value)} keys>")
                elif isinstance(value, list):
                    if len(value) > 0 and isinstance(value[0], str):
                        val_str = ", ".join(str(v)[:50] for v in value[:5])
                        if len(value) > 5:
                            val_str += f"... ({len(value)} items)"
                        lines.append(f"{key}: [{val_str}]")
                    else:
                        lines.append(f"{key}: <list with {len(value)} items>")
                elif hasattr(value, 'shape'):
                    lines.append(f"{key}: <tensor {tuple(value.shape)}>")
                else:
                    val_str = str(value)[:200]
                    lines.append(f"{key}: {val_str}")
        else:
            lines.append("No lz_pipe connected or empty.")
        
        info_text = "\n".join(lines)
        
        return (lz_pipe, info_text)
