"""Where do the recognizer's readings of held-out words disagree with the spelling rules?

    python -m chinukpipa.htr.confusions runs/f0/eval.json runs/f1/eval.json ... [--min 3]

Each eval.json (from chinukpipa.htr.evaluate) lists, for every test word, the rule-derived tokens (`ref`) and the
recognizer's free reading (`greedy`). Aligning the two shows which sign substitutions, insertions and deletions
recur. A pattern that recurs across many different words and across models trained on different splits is a
lead worth checking by eye: either the recognizer has a systematic weakness, or the spelling-to-sign rule in
chinukpipa/translit.py is wrong for that case. It is not evidence by itself.
"""
from __future__ import annotations

import argparse
import collections
import json


def align(ref: list[str], hyp: list[str]):
    """Levenshtein alignment -> list of (ref_token or None, hyp_token or None)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            out.append((ref[i - 1], None))
            i -= 1
        else:
            out.append((None, hyp[j - 1]))
            j -= 1
    return out[::-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("evals", nargs="+")
    ap.add_argument("--min", type=int, default=3, help="report patterns seen in at least this many distinct words")
    a = ap.parse_args()
    pats = collections.defaultdict(set)
    totals = collections.Counter()
    for path in a.evals:
        for r in json.load(open(path, encoding="utf-8"))["rows"]:
            ref, hyp = r["ref"].split(), r["greedy"].split()
            for x, y in align(ref, hyp):
                if x:
                    totals[x] += 1
                if x != y:
                    pats[(x or "-", y or "-")].add(r["latin"])
    rows = sorted(((len(ws), k, ws) for k, ws in pats.items() if len(ws) >= a.min), reverse=True)
    print("rule -> read  words  (rule token occurrences)  examples")
    for n, (x, y), ws in rows:
        occ = f"({totals[x]} {x})" if x != "-" else ""
        print(f"{x:>4} -> {y:<4} {n:5d}  {occ:>12}  {', '.join(sorted(ws)[:8])}")


if __name__ == "__main__":
    main()
