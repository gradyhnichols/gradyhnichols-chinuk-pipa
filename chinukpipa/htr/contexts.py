"""Recognizer errors broken down by spelling context.

    python -m chinukpipa.htr.contexts results/sweep5/base_v*.json [--min 12] [--token E --latin i]

For every rule token of every held-out word, finds the Roman letters it came from (as chinukpipa.translit
segments the word) and the class of the letters on either side: V = vowel letter (a e i o u y), C = consonant,
^ = word start, $ = word end. It then counts how often the recognizer's free reading dropped (`del`) or replaced
(`sub`) that token. With --token/--latin it prints only that token and adds a vowel-adjacent / not summary.
Words are counted once per run in which they were held out.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re

from chinukpipa import translit as T
from chinukpipa.htr.confusions import align

VOWELS = set("aeiouy")


def token_sources(word: str):
    """[(token, latin letters, previous letter, next letter)] following latin_to_tokens' segmentation."""
    out = []
    for wi, raw in enumerate(w for w in re.split(r"[\s\-]+", word) if w):
        if wi:
            out.append(("_", " ", "", ""))
        w = T.normalize_latin(raw).replace("x", "ks")
        i, toks = 0, []
        while i < len(w):
            for lat, tok in T._latin_table():
                if w.startswith(lat, i):
                    break
            else:
                lat, tok = w[i], "OO"          # lone w, as in latin_to_tokens
            toks.append((tok, lat, w[i - 1] if i else "^", w[i + len(lat)] if i + len(lat) < len(w) else "$"))
            i += len(lat)
        dedup = []
        for t in toks:
            if dedup and t[0] == dedup[-1][0] and t[0] not in T._VOWEL_TOKENS and t[0] not in T._W_DIPHTHONGS:
                continue
            dedup.append(t)
        out += dedup
    return out


def cls(ch: str) -> str:
    return ch if ch in ("^", "$", "") else ("V" if ch in VOWELS else "C")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("evals", nargs="+")
    ap.add_argument("--min", type=int, default=12)
    ap.add_argument("--token")
    ap.add_argument("--latin")
    a = ap.parse_args()
    stats = collections.defaultdict(lambda: [0, 0, 0, set()])
    skipped = 0
    for pattern in a.evals:
        for path in glob.glob(pattern) or [pattern]:
            for r in json.load(open(path, encoding="utf-8"))["rows"]:
                src = token_sources(r["latin"])
                ref = r["ref"].split()
                if [s[0] for s in src] != ref:
                    skipped += 1
                    continue
                k = 0
                for x, y in align(ref, r["greedy"].split()):
                    if x is None:
                        continue
                    tok, lat, pv, nx = src[k]
                    k += 1
                    s = stats[(tok, lat, cls(pv) + "_" + cls(nx))]
                    s[0] += 1
                    s[1] += y is None
                    s[2] += y is not None and y != x
                    s[3].add(r["latin"])
    keys = [k for k in stats if (not a.token or k[0] == a.token) and (not a.latin or k[1] == a.latin)]
    keys = [k for k in keys if stats[k][0] >= (1 if a.token else a.min)]
    keys.sort(key=lambda k: -(stats[k][1] + stats[k][2]) / stats[k][0])
    print("token latin context    n  del  sub  err%  words")
    for k in keys:
        n, d, s, ws = stats[k]
        print(f"{k[0]:>5} {k[1]:>5} {k[2]:7} {n:4d} {d:4d} {s:4d} {100 * (d + s) / n:5.0f}  {', '.join(sorted(ws)[:6])}")
    if a.token:
        groups = {"vowel on at least one side": [k for k in keys if "V" in k[2]],
                  "no vowel on either side": [k for k in keys if "V" not in k[2]]}
        for name, ks in groups.items():
            n = sum(stats[k][0] for k in ks)
            e = sum(stats[k][1] + stats[k][2] for k in ks)
            d = sum(stats[k][1] for k in ks)
            words = set().union(*[stats[k][3] for k in ks]) if ks else set()
            print(f"{name}: {e} of {n} misread ({d} dropped), {len(words)} distinct spellings")
    if skipped:
        print(f"({skipped} rows skipped: rule tokens differ from the current transliterator)")


if __name__ == "__main__":
    main()
