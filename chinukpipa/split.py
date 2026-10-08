"""Train/test split of Chinook words, shared by the recognizer and the synthetic-data generator.

The split is by *word*: the same word (after removing case, accents and punctuation) is always on the same side,
so a test word never appears in training, neither as a real image nor as a synthetic rendering.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata


def word_key(latin: str) -> str:
    s = unicodedata.normalize("NFD", latin.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z]", "", s)


def split_of(latin: str, test_frac: float = 0.2, seed: str = "v0") -> str:
    """Deterministic split by word: 'test' for about `test_frac` of words, else 'train'."""
    h = int(hashlib.sha1((seed + word_key(latin)).encode()).hexdigest(), 16) % 1000
    return "test" if h < test_frac * 1000 else "train"
