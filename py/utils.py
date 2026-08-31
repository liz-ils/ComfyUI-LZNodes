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
