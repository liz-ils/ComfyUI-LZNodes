# dynamic_prompt.py

import csv
import json
import os
import random
import folder_paths

PROMPT_CSV_DIRNAME = "prompt_csv"
SUPPORTED_EXTS = (".txt", ".csv", ".json")
# 巨大ファイルの誤爆読みを避けるための上限
MAX_OPTION_FILE_BYTES = 2 * 1024 * 1024


def _option_dir():
    csv_dir = os.path.join(folder_paths.get_output_directory(), PROMPT_CSV_DIRNAME)
    os.makedirs(csv_dir, exist_ok=True)
    return csv_dir


def get_csv_files():
    """候補ファイル一覧。txt/csv/jsonをファイルのみ対象にする。"""
    csv_dir = _option_dir()
    if os.path.isdir(csv_dir):
        files = [
            f for f in os.listdir(csv_dir)
            if f.lower().endswith(SUPPORTED_EXTS)
            and os.path.isfile(os.path.join(csv_dir, f))
        ]
        return [""] + sorted(files) if files else [""]
    return [""]


def _safe_option_path(csv_name):
    """ディレクトリトラバーサル対策付きで候補ファイルの実パスを返す。"""
    if not csv_name:
        return None
    if os.path.isabs(csv_name):
        return None
    # サブディレクトリ指定・親参照を拒否
    normalized = os.path.normpath(csv_name)
    if normalized != csv_name.strip():
        # "./x" や "a/../b" 等は拒否
        return None
    if os.path.basename(normalized) != normalized:
        return None
    if ".." in normalized.split(os.sep):
        return None
    csv_dir = _option_dir()
    csv_path = os.path.join(csv_dir, normalized)
    if not os.path.isfile(csv_path):
        return None
    try:
        if os.path.getsize(csv_path) > MAX_OPTION_FILE_BYTES:
            return None
    except OSError:
        return None
    return csv_path


def _clean_lines(lines):
    """空行と#コメント行を除く。"""
    out = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _read_txt_options(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return _clean_lines(f)


def _read_csv_options(path):
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        items = []
        for row in reader:
            for cell in row:
                s = cell.strip()
                if not s or s.startswith("#"):
                    continue
                items.append(s)
        return items


def _read_json_options(path, json_key=""):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]
    if isinstance(data, dict):
        if json_key and json_key in data:
            v = data[json_key]
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
            return [str(v).strip()] if str(v).strip() else []
        # キー指定なしは全リスト値をキー順に結合(決定的順序)
        items = []
        for k in sorted(data.keys()):
            v = data[k]
            if isinstance(v, list):
                items.extend(str(x).strip() for x in v if str(x).strip())
            elif isinstance(v, str) and v.strip():
                items.append(v.strip())
        return items
    return []


def read_csv_options(csv_name, json_key=""):
    """後方互換の読み込み関数。txt/csv/jsonに対応。"""
    path = _safe_option_path(csv_name)
    if path is None:
        return []
    try:
        lower = path.lower()
        if lower.endswith(".json"):
            return _read_json_options(path, json_key or "")
        if lower.endswith(".csv"):
            return _read_csv_options(path)
        return _read_txt_options(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return []


def replace_placeholder(text, placeholder, replacement):
    if not placeholder:
        return text
    return text.replace(placeholder, replacement)


def _rng_for_seed(seed):
    """seed<0は毎回ランダム、それ以外は再現性のあるRNGを返す。"""
    if seed is None or int(seed) < 0:
        return random
    return random.Random(int(seed))


def _format_weight(weight):
    if weight == int(weight):
        return str(int(weight))
    # 浮動小数ゴミを避ける
    return f"{weight:.2f}".rstrip("0").rstrip(".")


def _apply_weight(tag, weight):
    if weight is None or abs(float(weight) - 1.0) < 1e-9:
        return tag
    return f"({tag}:{_format_weight(float(weight))})"


def _pick_options(rng, options, pick_count=1, unique=True):
    """候補からpick_count個取り出す。unique標準ON。足りなければ重複許可に緩和。"""
    if not options:
        return []
    pick_count = max(1, int(pick_count))
    if unique and len(options) >= pick_count:
        return rng.sample(list(options), pick_count)
    # 候補不足時は重複を許可して埋める
    return [rng.choice(options) for _ in range(pick_count)]


def _random_weight(rng, weight_min, weight_max):
    lo = float(weight_min)
    hi = float(weight_max)
    if hi < lo:
        lo, hi = hi, lo
    if abs(hi - lo) < 1e-9:
        return lo
    return rng.uniform(lo, hi)


class LZPromptReplaceSingle:
    @classmethod
    def INPUT_TYPES(s):
        csv_files = get_csv_files()
        return {
            "required": {
                "template": ("STRING", {"multiline": True, "default": ""}),
                "placeholder": ("STRING", {"default": "_hair_"}),
                "csv_file": (csv_files, {"default": ""}),
            },
            "optional": {
                "json_key": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "control_after_generate": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "replace_single"
    CATEGORY = "MyCustomNodes/Text"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        seed = kwargs.get("seed", -1)
        if seed is None or int(seed) < 0:
            return float("nan")
        return int(seed)

    def replace_single(self, template, placeholder, csv_file, json_key="", seed=-1):
        if not csv_file:
            return (template,)

        options = read_csv_options(csv_file, json_key)
        if not options:
            return (template,)

        rng = _rng_for_seed(seed)
        selected = rng.choice(options)
        result = replace_placeholder(template, placeholder, selected)
        return (result,)


class LZPromptReplaceMulti:
    @classmethod
    def INPUT_TYPES(s):
        csv_files = get_csv_files()
        inputs = {
            "required": {
                "template": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "unique_across_slots": ("BOOLEAN", {"default": True}),
            }
        }

        for i in range(1, 6):
            inputs["required"][f"placeholder{i}"] = ("STRING", {"default": ""})
            inputs["required"][f"csv_file{i}"] = (csv_files, {"default": ""})

        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "replace_multi"
    CATEGORY = "MyCustomNodes/Text"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        seed = kwargs.get("seed", -1)
        if seed is None or int(seed) < 0:
            return float("nan")
        return int(seed)

    def replace_multi(self, template, seed=-1, unique_across_slots=True, **kwargs):
        result = template
        rng = _rng_for_seed(seed)
        used = set()

        for i in range(1, 6):
            placeholder = kwargs.get(f"placeholder{i}", "")
            csv_file = kwargs.get(f"csv_file{i}", "")

            if not placeholder or not csv_file:
                continue

            options = read_csv_options(csv_file)
            if not options:
                continue

            if unique_across_slots:
                fresh = [o for o in options if o not in used]
                pool = fresh if fresh else options
            else:
                pool = options
            selected = rng.choice(pool)
            used.add(selected)
            result = replace_placeholder(result, placeholder, selected)

        return (result,)


class LZPromptReplaceString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "template": ("STRING", {"multiline": True, "default": ""}),
                "placeholder": ("STRING", {"default": "_hair_"}),
                "candidates": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "control_after_generate": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "replace_string"
    CATEGORY = "MyCustomNodes/Text"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        seed = kwargs.get("seed", -1)
        if seed is None or int(seed) < 0:
            return float("nan")
        return int(seed)

    def replace_string(self, template, placeholder, candidates, seed=-1):
        if not candidates.strip():
            return (template,)

        lines = _clean_lines(candidates.split("\n"))
        if not lines:
            return (template,)

        rng = _rng_for_seed(seed)
        selected = rng.choice(lines)
        result = replace_placeholder(template, placeholder, selected)
        return (result,)


class LZPromptPick:
    """複数ソース(txt/csv/json＋直書き)から複数個ピックし、重み付けもできるノード。

    hair/color等は本ノードを複数並べて使う。重複なしが標準ON。
    """

    @classmethod
    def INPUT_TYPES(s):
        files = get_csv_files()
        return {
            "required": {
                "template": ("STRING", {"multiline": True, "default": ""}),
                "placeholder": ("STRING", {"default": "_tag_"}),
                "source_file_a": (files, {"default": ""}),
                "pick_count": ("INT", {"default": 1, "min": 1, "max": 20}),
                "unique": ("BOOLEAN", {"default": True}),
                "separator": ("STRING", {"default": ", "}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "control_after_generate": True}),
            },
            "optional": {
                "source_file_b": (files, {"default": ""}),
                "json_key": ("STRING", {"default": ""}),
                "candidates": ("STRING", {"multiline": True, "default": ""}),
                "weight_min": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05}),
                "weight_max": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "picked")
    FUNCTION = "pick"
    CATEGORY = "MyCustomNodes/Text"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        seed = kwargs.get("seed", -1)
        if seed is None or int(seed) < 0:
            return float("nan")
        return int(seed)

    def pick(self, template, placeholder, source_file_a, pick_count=1,
             unique=True, separator=", ", seed=-1, source_file_b="",
             json_key="", candidates="", weight_min=1.0, weight_max=1.0):
        pool = []
        for name in (source_file_a, source_file_b):
            if name:
                pool.extend(read_csv_options(name, json_key))
        if candidates and candidates.strip():
            pool.extend(_clean_lines(candidates.split("\n")))
        # 空・重複候補の掃除(順序は保持)
        seen = set()
        options = []
        for o in pool:
            if o not in seen:
                seen.add(o)
                options.append(o)
        if not options:
            return (template, "")

        rng = _rng_for_seed(seed)
        picks = _pick_options(rng, options, pick_count, unique)
        weighted = [_apply_weight(p, _random_weight(rng, weight_min, weight_max)) for p in picks]
        joined = (separator if separator else ", ").join(weighted)

        if placeholder and placeholder in template:
            result = template.replace(placeholder, joined)
        elif placeholder:
            result = template
        else:
            result = f"{template}{separator}{joined}" if template else joined
        return (result, ", ".join(weighted))
