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
