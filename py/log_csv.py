# log_csv.py

import os
import csv
import datetime

from .utils import get_pipe_loras, lora_weight_str

class LZAppendLogToCSV:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filepath": ("STRING", {"default": "generation_log.csv"}),
                "mode": (["write", "append"],),
            },
            "optional": {
                "lz_pipe": ("LZ_PIPE",),
                "seed": ("INT", {"default": 0}),
                "steps": ("INT", {"default": 20}),
                "cfg": ("FLOAT", {"default": 8.0}),
                "sampler_name": ("STRING", {"default": "euler"}),
                "scheduler": ("STRING", {"default": "normal"}),
                "positive_prompt": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"default": ""}),
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 512}),
                "checkpoint_name": ("STRING", {"default": ""}),
                "lora_model": ("STRING", {"default": ""}),
                "lora_weight": ("STRING", {"default": ""}),
                "image_count": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "append_log"
    OUTPUT_NODE = True
    CATEGORY = "MyCustomNodes/Log"

    def append_log(self, filepath, mode, **kwargs):
        filepath = filepath.strip()
        if not filepath:
            filepath = "generation_log.csv"
        
        lz_pipe = kwargs.get("lz_pipe") or {}
        
        seed = lz_pipe.get("seed", kwargs.get("seed", 0))
        steps = lz_pipe.get("steps", kwargs.get("steps", 20))
        cfg = lz_pipe.get("cfg", kwargs.get("cfg", 8.0))
        sampler_name = lz_pipe.get("sampler_name", kwargs.get("sampler_name", "euler"))
        scheduler = lz_pipe.get("scheduler", kwargs.get("scheduler", "normal"))
        positive_prompt = lz_pipe.get("positive_text", kwargs.get("positive_prompt", ""))
        negative_prompt = lz_pipe.get("negative_text", kwargs.get("negative_prompt", ""))
        width = lz_pipe.get("width", kwargs.get("width", 512))
        height = lz_pipe.get("height", kwargs.get("height", 512))
        checkpoint_name = lz_pipe.get("ckpt_name", kwargs.get("checkpoint_name", ""))
        # 複数 LoRA 対応: 直接入力優先、なければ pipe(構造化 loras 含む)
        _direct_lora_model = kwargs.get("lora_model", "")
        _direct_lora_weight = kwargs.get("lora_weight", "")
        if _direct_lora_model:
            lora_model = _direct_lora_model
            lora_weight = _direct_lora_weight or lz_pipe.get("lora_weight") or lz_pipe.get("lora_strength") or ""
        else:
            _entries = get_pipe_loras(lz_pipe)
            if _entries:
                lora_model = ", ".join(e["name"] for e in _entries)
                lora_weight = ", ".join(lora_weight_str(e.get("model_weight", 1.0), e.get("clip_weight")) for e in _entries)
            else:
                lora_model = lz_pipe.get("lora_model") or lz_pipe.get("lora_name") or ""
                lora_weight = lz_pipe.get("lora_weight") or lz_pipe.get("lora_strength") or ""

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_count = kwargs.get("image_count", 1)

        headers = ["timestamp", "seed", "steps", "cfg", "sampler", "scheduler", "width", "height", "checkpoint", "lora_model", "lora_weight", "image_count", "positive_prompt", "negative_prompt"]
        row = [timestamp, seed, steps, cfg, sampler_name, scheduler, width, height, checkpoint_name, lora_model, lora_weight, image_count, positive_prompt, negative_prompt]

        file_exists = os.path.exists(filepath)
        append_mode = (mode == "append" and file_exists)

        if append_mode:
            # 旧 CSV(LoRA列なし)への追記時はヘッダを移行して列ズレを防ぐ
            try:
                with open(filepath, "r", encoding="utf-8", newline="") as rf:
                    reader = csv.reader(rf)
                    old_header = next(reader, None)
                    old_rows = list(reader)
                if old_header is not None and ("lora_model" not in old_header or "lora_weight" not in old_header):
                    # 旧ヘッダ: [..., checkpoint, image_count, positive_prompt, negative_prompt]
                    # 新ヘッダに合わせて空の LoRA 値を補完して書き直す
                    fixed_rows = []
                    for r in old_rows:
                        if len(r) == len(old_header):
                            try:
                                ci = old_header.index("checkpoint")
                            except ValueError:
                                ci = 8
                            # image_count 以降を後ろにずらす
                            head = r[:ci + 1]
                            tail = r[ci + 1:]
                            fixed_rows.append(head + ["", ""] + tail)
                        else:
                            fixed_rows.append(r)
                    with open(filepath, "w", encoding="utf-8", newline="") as wf:
                        writer = csv.writer(wf)
                        # 旧ヘッダに lora 列を挿入
                        try:
                            ci = old_header.index("checkpoint")
                        except ValueError:
                            ci = 8
                        new_old_header = old_header[:ci + 1] + ["lora_model", "lora_weight"] + old_header[ci + 1:]
                        # 完全一致しない場合は新ヘッダで統一
                        if set(new_old_header) != set(headers):
                            new_old_header = headers
                            # 行の列数も合わせる(足りなければ空埋め)
                            fixed_rows = [(rr + [""] * len(headers))[:len(headers)] if len(rr) != len(headers) else rr for rr in fixed_rows]
                        writer.writerow(new_old_header)
                        writer.writerows(fixed_rows)
            except Exception:
                pass

        with open(filepath, "a" if append_mode else "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            # ヘッダは新規作成時のみ書き込む(既存ファイルへの追記時は書かない)
            if not append_mode:
                writer.writerow(headers)
            writer.writerow(row)

        return (os.path.abspath(filepath),)
