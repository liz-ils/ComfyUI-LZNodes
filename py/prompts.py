# prompt.py

class AdvancedPositivePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # CLIP入力ピン
                "clip": ("CLIP", ),
                
                # プロンプト（メイン）
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                
                # アーティストタグ
                "artist": ("STRING", {"multiline": True, "default": ""}),
                
                # その他（LoRAトリガーワードなど）
                "others": ("STRING", {"multiline": True, "default": ""}),
                
                # クオリティ
                "quality": ("STRING", {"multiline": True, "default": "masterpiece, best quality, ultra-detailed, highres"}),
            }
        }
        
    # 出力は「CONDITIONING (CLIPエンコード済み)」と「STRING (結合された文字)」
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning (positive)", "prompt_text")
    FUNCTION = "build_and_encode"
    CATEGORY = "MyCustomNodes/Prompt"

    def build_and_encode(self, clip, prompt, artist, others, quality):
        # 1. 各入力をリストにまとめる（前後の余計な空白を削除）
        parts = [
            prompt.strip(),
            artist.strip(),
            others.strip(),
            quality.strip()
        ]
        
        # 2. 空欄（文字が入っていない）の要素を除外する
        valid_parts = [p for p in parts if p != ""]
        
        # 3. カンマとスペースで結合する
        final_prompt = ", ".join(valid_parts)
        
        # 4. 結合した文字列をCLIPでエンコードする
        tokens = clip.tokenize(final_prompt)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        # CONDITIONINGと、確認用の文字列そのものを出力
        return (conditioning, final_prompt)

class AdvancedNegativePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # CLIP入力ピン
                "clip": ("CLIP", ),
                
                # 基本のネガティブプロンプト
                "base_negative": ("STRING", {"multiline": True, "default": "worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, blurry"}),
                
                # 追加ネガティブプロンプト
                "extra_negative": ("STRING", {"multiline": True, "default": ""}),
            }
        }
        
    # 出力は「CONDITIONING (CLIPエンコード済み)」と「STRING (結合された文字)」
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning (negative)", "negative_text")
    FUNCTION = "build_and_encode"
    CATEGORY = "MyCustomNodes/Prompt"

    def build_and_encode(self, clip, base_negative, extra_negative):
        # 1. 各入力をリストにまとめる（前後の余計な空白を削除）
        parts = [
            base_negative.strip(),
            extra_negative.strip()
        ]
        
        # 2. 空欄（文字が入っていない）の要素を除外する
        valid_parts = [p for p in parts if p != ""]
        
        # 3. カンマとスペースで結合する
        final_negative = ", ".join(valid_parts)
        
        # 4. 結合した文字列をCLIPでエンコードする
        tokens = clip.tokenize(final_negative)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        # CONDITIONINGと、確認用の文字列そのものを出力
        return (conditioning, final_negative)

class DualCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "positive": ("STRING", {"multiline": True, "default": "positive prompt"}),
                "negative": ("STRING", {"multiline": True, "default": "negative prompt"}),
            },
            "optional": {
                "positive_cond": ("CONDITIONING",),
                "negative_cond": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "positive_text", "negative_text")
    FUNCTION = "encode"
    CATEGORY = "MyCustomNodes/Conditioning"

    def encode(self, clip, positive, negative, positive_cond=None, negative_cond=None):
        if positive_cond is not None:
            pos_out = positive_cond
        else:
            tokens_pos = clip.tokenize(positive)
            pos_out = clip.encode_from_tokens_scheduled(tokens_pos)

        if negative_cond is not None:
            neg_out = negative_cond
        else:
            tokens_neg = clip.tokenize(negative)
            neg_out = clip.encode_from_tokens_scheduled(tokens_neg)

        return (pos_out, neg_out, positive, negative)


class LZCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "text")
    FUNCTION = "encode"
    CATEGORY = "MyCustomNodes/Conditioning"

    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        cond = clip.encode_from_tokens_scheduled(tokens)
        return (cond, text)
