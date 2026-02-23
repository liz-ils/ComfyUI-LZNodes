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
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        conditioning = [[cond, {"pooled_output": pooled}]]
        
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
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        conditioning = [[cond, {"pooled_output": pooled}]]
        
        # CONDITIONINGと、確認用の文字列そのものを出力
        return (conditioning, final_negative)

class DualCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ), # CLIP入力ピン
                "positive": ("STRING", {"multiline": True, "default": "positive prompt"}),
                "negative": ("STRING", {"multiline": True, "default": "negative prompt"}),
            }
        }
        
    # CONDITIONINGを2つ（ポジティブ用、ネガティブ用）出力する
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "MyCustomNodes/Conditioning"

    def encode(self, clip, positive, negative):
        # --- ポジティブのエンコード処理 ---
        tokens_pos = clip.tokenize(positive)
        cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
        positive_cond = [[cond_pos, {"pooled_output": pooled_pos}]]
        
        # --- ネガティブのエンコード処理 ---
        tokens_neg = clip.tokenize(negative)
        cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
        negative_cond = [[cond_neg, {"pooled_output": pooled_neg}]]
        
        return (positive_cond, negative_cond)
