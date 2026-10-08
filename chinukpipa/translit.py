"""Le Jeune Latin spelling  <->  Chinuk Pipa tokens  ->  Unicode Duployan.

Tokens are the recognition/label unit (see data/signs/tokens.yaml): one token per sign Le Jeune wrote,
e.g. ``latin_to_tokens("kamooks") == ["K", "A", "M", "OO", "K", "S"]``.

The Latin->token rules are hypotheses about how Le Jeune's Roman spelling corresponds to his shorthand.
They are being tested against his bilingual word lists (data/gt/).
"""
from __future__ import annotations

import functools
import re
import unicodedata
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
TOKENS_YAML = _ROOT / "data" / "signs" / "tokens.yaml"

# characters that mark stress / syllable division in Le Jeune's Roman spelling (not letters)
_STRIP = "'’‘`´ʼ,.;:!?\"()[]"
_VOWEL_TOKENS = {"A", "O", "OO", "OW", "E", "U"}
_W_DIPHTHONGS = {"WA", "WO", "WE", "WEYIE", "WOW", "WOO", "OWA", "WAY", "WEEYA"}


@functools.lru_cache(maxsize=1)
def load_tokens(path: str | Path = TOKENS_YAML) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["tokens"]


@functools.lru_cache(maxsize=1)
def _latin_table() -> list[tuple[str, str]]:
    """(latin_string, token) pairs, longest first."""
    pairs = []
    for tok, spec in load_tokens().items():
        for lat in spec.get("latin") or []:
            if lat.strip():
                pairs.append((lat, tok))
    pairs.sort(key=lambda p: -len(p[0]))
    return pairs


def normalize_latin(word: str) -> str:
    """Lower-case, drop stress marks and punctuation, keep diaeresis-i as a plain vowel letter."""
    w = word.strip().lower()
    w = "".join(ch for ch in w if ch not in _STRIP)
    # accents on vowels carry stress only; ï (separately pronounced i) -> i
    w = "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn")
    return w


def latin_to_tokens(word: str, *, x_to_ks: bool = True, collapse_doubles: bool = True,
                    w_alone: str = "OO") -> list[str]:
    """Greedy longest-match conversion of one Le Jeune Roman-spelled word into pipa tokens.

    Multi-word input is split on whitespace/hyphens and joined with the word-space token "_".
    Unknown characters raise ValueError so silent errors do not enter training labels.
    """
    words = [w for w in re.split(r"[\s\-]+", word) if w]
    out: list[str] = []
    for wi, raw in enumerate(words):
        if wi:
            out.append("_")
        w = normalize_latin(raw)
        if x_to_ks:
            w = w.replace("x", "ks")
        toks: list[str] = []
        i = 0
        table = _latin_table()
        while i < len(w):
            for lat, tok in table:
                if w.startswith(lat, i):
                    toks.append(tok)
                    i += len(lat)
                    break
            else:
                ch = w[i]
                if ch == "w":
                    toks.append(w_alone)
                    i += 1
                else:
                    raise ValueError(f"no token for {ch!r} in {raw!r}")
        if collapse_doubles:
            dedup = []
            for t in toks:
                if dedup and t == dedup[-1] and t not in _VOWEL_TOKENS and t not in _W_DIPHTHONGS:
                    continue  # doubled consonant letters in Roman spelling = one sign
                dedup.append(t)
            toks = dedup
        out.extend(toks)
    return out


def tokens_to_unicode(tokens: list[str] | str, *, missing: str = "�") -> str:
    """Tokens -> Unicode Duployan string. Signs with no code point become `missing`."""
    if isinstance(tokens, str):
        tokens = tokens.split()
    spec = load_tokens()
    out = []
    for t in tokens:
        cps = spec[t].get("unicode")
        if not cps:
            out.append(missing)
            continue
        out.extend(chr(int(cp[2:], 16)) for cp in cps)
    return "".join(out)


def unicode_to_tokens(text: str) -> list[str]:
    """Inverse of tokens_to_unicode for encoded signs (unencoded signs cannot round-trip)."""
    rev: dict[str, str] = {}
    for tok, spec in load_tokens().items():
        cps = spec.get("unicode")
        if cps and len(cps) == 1:
            rev.setdefault(chr(int(cps[0][2:], 16)), tok)
    return [rev[c] for c in text if c in rev]


def vocabulary() -> list[str]:
    """Ordered list of all token symbols (stable; used as the recognizer's alphabet)."""
    return list(load_tokens().keys())


if __name__ == "__main__":  # quick self-check
    for w in ["kamooks", "alta", "kanawe", "chako", "kanamoxt", "khell", "kilapai", "wawa", "kloshe nanich"]:
        toks = latin_to_tokens(w)
        print(f"{w:<15} {' '.join(toks):<28} {tokens_to_unicode(toks)}")
