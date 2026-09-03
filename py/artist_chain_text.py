# artist_chain_text.py
# String node for the artist_chain input of the LZ Anima Artist nodes.


class LZArtistChainText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "artist_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "One artist per line.\nExample:\nartist 1\nartist a\nartist b\n\n"
                               "artist_tag output: \"artist 1, artist a, artist b\" (connect to artist_chain)\n"
                               "artist_text output: the input text as-is."
                }),
            },
            "optional": {
                "lz_pipe": ("LZ_PIPE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "LZ_PIPE")
    RETURN_NAMES = ("artist_tag", "artist_text", "lz_pipe")
    FUNCTION = "build"
    CATEGORY = "MyCustomNodes/Anima"

    def build(self, artist_text, lz_pipe=None):
        text = artist_text or ""
        # 改行(\r\n含む)で分割し、空行を除いて ", " で連結
        lines = text.replace("\r", "").split("\n")
        tags = [line.strip() for line in lines if line.strip()]
        artist_tag = ", ".join(tags)
        return (artist_tag, text, lz_pipe)
