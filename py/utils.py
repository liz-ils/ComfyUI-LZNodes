# utils.py
# Shared helpers used by multiple LZ nodes.

import hashlib
import os

CHECKPOINT_HASH_CACHE = {}

LZBannedChars = r'\/?"<>\:|*'


def get_checkpoint_hash(file_path):
    """Return a short sha256 hash (10 chars) of a model file, cached by mtime."""
    if not file_path or not os.path.exists(file_path):
        return "Unknown"

    mtime = os.path.getmtime(file_path)
    if file_path in CHECKPOINT_HASH_CACHE:
        cached_mtime, cached_hash = CHECKPOINT_HASH_CACHE[file_path]
        if cached_mtime == mtime:
            return cached_hash

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            sha256_hash.update(byte_block)

    short_hash = sha256_hash.hexdigest()[:10]
    CHECKPOINT_HASH_CACHE[file_path] = (mtime, short_hash)
    return short_hash


def parse_values(text):
    """Split a multiline string into a list of non-empty trimmed lines."""
    if not text:
        return []
    lines = text.strip().split('\n')
    return [line.strip() for line in lines if line.strip()]


def sanitize_filename(text, replace_with="_"):
    """Remove characters that are illegal in Windows file/folder names."""
    if not isinstance(text, str):
        return text
    for char in LZBannedChars:
        text = text.replace(char, replace_with)
    if text.endswith("."):
        text = text[:-1]
    return text


def zero_out_conditioning(conditioning):
    """ComfyUI の ConditioningZeroOut と同等の処理。

    Krea2 Turbo 等の CFG=1.0 前提モデルで、ネガティブを無効化するために使用する。
    ComfyUI 本体の ConditioningZeroOut ノードが利用可能ならそれを使う。
    """
    import torch

    zero_node = None
    try:
        import nodes as comfy_nodes
        zero_node = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get("ConditioningZeroOut")
    except Exception:
        zero_node = None

    if zero_node is not None:
        return zero_node().zero_out(conditioning)[0]

    # フォールバック実装(ConditioningZeroOut と同一の挙動)
    out = []
    for t in conditioning:
        d = t[1].copy() if len(t) > 1 and isinstance(t[1], dict) else {}
        if "pooled_output" in d:
            d["pooled_output"] = torch.zeros_like(d["pooled_output"])
        out.append([torch.zeros_like(t[0]), d])
    return out


# --- LoRA / lz_pipe shared helpers (multi-LoRA) ---

def lora_weight_str(model_weight, clip_weight=None):
    """model/clip 重みをログ用文字列化。等しい場合は単値、異なる場合は m:c 形式。"""
    try:
        m = float(model_weight)
    except Exception:
        return str(model_weight)
    if clip_weight is None:
        return str(model_weight)
    try:
        c = float(clip_weight)
    except Exception:
        return str(model_weight)
    if abs(m - c) < 1e-9:
        return str(model_weight)
    return f"{model_weight}:{clip_weight}"


def _split_csv_names(text):
    if not text or not isinstance(text, str):
        return []
    return [p.strip() for p in text.split(",") if p.strip()]


def get_pipe_loras(lz_pipe):
    """lz_pipe から構造化 LoRA リストを取得。無ければ文字列から復元を試みる。"""
    if not isinstance(lz_pipe, dict):
        return []
    loras = lz_pipe.get("loras")
    if isinstance(loras, list) and loras:
        out = []
        for e in loras:
            if isinstance(e, dict) and e.get("name"):
                out.append({
                    "name": str(e.get("name")),
                    "model_weight": e.get("model_weight", 1.0),
                    "clip_weight": e.get("clip_weight", e.get("model_weight", 1.0)),
                })
            elif isinstance(e, (list, tuple)) and len(e) >= 1:
                out.append({
                    "name": str(e[0]),
                    "model_weight": e[1] if len(e) > 1 else 1.0,
                    "clip_weight": e[2] if len(e) > 2 else (e[1] if len(e) > 1 else 1.0),
                })
        if out:
            return out
    # フォールバック: 文字列フィールドから復元
    names = _split_csv_names(lz_pipe.get("lora_model") or lz_pipe.get("lora_name") or "")
    weights = _split_csv_names(lz_pipe.get("lora_weight") or lz_pipe.get("lora_strength") or "")
    out = []
    for idx, name in enumerate(names):
        w = weights[idx] if idx < len(weights) else "1.0"
        if ":" in w:
            parts = w.split(":")
            try:
                m = float(parts[0].strip())
            except Exception:
                m = 1.0
            try:
                c = float(parts[1].strip())
            except Exception:
                c = m
        else:
            try:
                m = float(w)
            except Exception:
                m = 1.0
            c = m
        out.append({"name": name, "model_weight": m, "clip_weight": c})
    return out


def set_pipe_loras(lz_pipe, loras):
    """構造化リストから lz_pipe の LoRA 系フィールドを同期する(後方互換の文字列も維持)。"""
    names = [e["name"] for e in loras]
    weights = [lora_weight_str(e.get("model_weight", 1.0), e.get("clip_weight", e.get("model_weight", 1.0))) for e in loras]
    joined_names = ", ".join(names)
    joined_weights = ", ".join(weights)
    lz_pipe["loras"] = [{"name": e["name"], "model_weight": e.get("model_weight", 1.0), "clip_weight": e.get("clip_weight", e.get("model_weight", 1.0))} for e in loras]
    lz_pipe["lora_name"] = joined_names
    lz_pipe["lora_strength"] = joined_weights
    lz_pipe["lora_model"] = joined_names
    lz_pipe["lora_weight"] = joined_weights
    return lz_pipe


def pipe_lora_strings(lz_pipe):
    """表示/ログ用に (names, weights, count) を返す。loras リスト優先。"""
    loras = get_pipe_loras(lz_pipe if isinstance(lz_pipe, dict) else {})
    if loras:
        names = ", ".join(e["name"] for e in loras)
        weights = ", ".join(lora_weight_str(e.get("model_weight", 1.0), e.get("clip_weight")) for e in loras)
        return names, weights, len(loras)
    if isinstance(lz_pipe, dict):
        names = lz_pipe.get("lora_model") or lz_pipe.get("lora_name") or ""
        weights = lz_pipe.get("lora_weight") or lz_pipe.get("lora_strength") or ""
        if names:
            return names, weights, len(_split_csv_names(names))
    return "", "", 0


def format_lora_log_lines(lz_pipe, prefix="LoRA"):
    """複数 LoRA をログ行リスト化。1件なら従来の1行形式、複数なら件数+列挙。"""
    loras = get_pipe_loras(lz_pipe if isinstance(lz_pipe, dict) else {})
    if not loras:
        # 文字列のみの場合のフォールバック
        if isinstance(lz_pipe, dict):
            names = lz_pipe.get("lora_model") or lz_pipe.get("lora_name") or ""
            weights = lz_pipe.get("lora_weight") or lz_pipe.get("lora_strength") or ""
            if names:
                lines = [f"{prefix}s: {names}"]
                if weights:
                    lines.append(f"{prefix} strengths: {weights}")
                return lines
        return []
    if len(loras) == 1:
        e = loras[0]
        lines = [f"{prefix}s: {e['name']}"]
        lines.append(f"{prefix} strengths: {lora_weight_str(e.get('model_weight', 1.0), e.get('clip_weight'))}")
        return lines
    lines = [f"{prefix}s ({len(loras)}): {', '.join(e['name'] for e in loras)}"]
    for e in loras:
        lines.append(f"  - {e['name']}: {lora_weight_str(e.get('model_weight', 1.0), e.get('clip_weight'))}")
    return lines
