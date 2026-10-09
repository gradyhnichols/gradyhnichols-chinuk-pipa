"""Score machine readings of a page against an alignment of its Roman text to its word boxes.

    python -m chinukpipa.text.score_alignment ALIGNMENT_TSV READ_OR_MERGED_JSON [READ_OR_MERGED_JSON ...]
        [--margins 0,2,4] [--merge-rule]

The alignment has one row per Roman word of the page (columns used: `unit_kind`, `main_box`, `box_ids`,
`roman_tokens`). A unit is the set of word boxes that carries one or more Roman words: `match` (one box, one
word), `split` (several boxes, one word), `merge` (one box, several words), `merge+split` (several boxes, several
words). The alignment is made elsewhere; this module only reads it. Percentages use the number of rows of the
alignment as denominator, so a Roman word without a reading counts as wrong.

Each JSON file is scored by its kind, with a strict rule:

* stage B, `<stem>_read.json` (rows keyed by `line` / `index`): a Roman word counts as read right only if it got
  exactly one word box of its own (unit_kind `match`, `main_box` = "line.index") and the reading's tokens equal the
  word's `roman_tokens`. A box that the segmenter did not class as a word has no reading and counts as wrong.
* stage C, `<stem>_merged.json` (rows with `boxes`): a Roman word counts only if its unit is `match` or `split`, one
  output word covers exactly the boxes of its unit (`box_ids`) and its tokens equal `roman_tokens`. Also reported: right words per unit kind,
  and single-box words (`match`) whose box was joined to another by stage C.

Readings compared: the word list's first choice (`top1`), any of its five (`top5`), the first model's free reading
(`free`, stage B only), `free_best` (of the models' free readings, the one with the lowest ensemble loss), and a
hybrid: `free_best` where its loss plus a margin is below the loss of the list's first choice, otherwise the list's
first choice. The hybrid is shown for each margin in --margins.

--merge-rule (experimental, v0.4) also scores the units that hold several Roman words (`merge`: one box, `merge+split`:
several boxes), which the strict rule never counts as right. The k Roman words of such a unit (its rows with the same
`box_ids`, in the file's order) are right if one output word covers exactly the unit's boxes and the reading's tokens
equal the k words' `roman_tokens` joined with " _ " (the word-space token), the way a two-word reading is written;
each of the k words then counts right in every measure (`top1`, `top5`, `free`, `free_best`, `hybrid`). For a stage-B
file only a one-box unit can be covered by an output word. `match` and `split` units are scored exactly as before, and
so is everything else when the option is not given.
"""
from __future__ import annotations

import argparse
import csv
import json

MARGINS = (-2, 0, 1, 2, 3, 4, 6, 8, 1e9)   # 1e9: never use the free reading when the list has an answer


def load_alignment(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def percent(v: int, n: int) -> str:
    return f"{v}/{n} = {100 * v / n:.1f}%"


MERGE_KINDS = ("merge", "merge+split")    # unit kinds that hold several Roman words


def scored_items(al: list[dict], merge_rule: bool = False) -> list[tuple[dict, int, str]]:
    """What to score: (alignment row, number of Roman words it stands for, tokens to match). Without `merge_rule`
    that is every row, one word, its own `roman_tokens`. With it, the rows of one multi-word unit (same `box_ids`,
    kind `merge` or `merge+split`) become one item: its first row, k words, the k words' tokens joined with " _ "."""
    if not merge_rule:
        return [(r, 1, r["roman_tokens"]) for r in al]
    items: list = []
    units: dict[str, list] = {}
    for r in al:
        if r["unit_kind"] not in MERGE_KINDS:
            items.append((r, 1, r["roman_tokens"]))
        elif r["box_ids"] not in units:
            units[r["box_ids"]] = item = [r, 1, r["roman_tokens"]]
            items.append(item)
        else:
            item = units[r["box_ids"]]
            item[1] += 1
            item[2] += " _ " + r["roman_tokens"]
    return [tuple(item) for item in items]


def hybrid_right(w: dict, ref: str, margin: float) -> bool:
    """Is the hybrid reading of word `w` right? Free reading (free_best) if it fits the image better than the
    list's first choice by more than `margin`, or if the list has no scored answer; else the list's first choice."""
    t1 = w["top5"][0] if w["top5"] else None
    fb = w.get("free_best")
    if fb and (not t1 or t1["score"] is None or fb["score"] + margin < t1["score"]):
        return fb["tokens"] == ref
    return bool(t1) and t1["tokens"] == ref


def score_read(al: list[dict], d: dict, margins=MARGINS, merge_rule: bool = False) -> dict:
    """Strict score of a stage-B file: the word must have a box of its own (unit_kind `match`); with `merge_rule`
    also a unit of kind `merge` (one box, several words) whose box reads as the words joined by " _ "."""
    n = len(al)
    words = {(w["line"], w["index"]): w for w in d["words"]}
    res = {"top1": 0, "top5": 0, "free": 0, "free_best": 0}
    hyb = {m: 0 for m in margins}
    kinds = ("match", "merge") if merge_rule else ("match",)
    for r, k, ref in scored_items(al, merge_rule):
        if r["unit_kind"] not in kinds or not r["main_box"].replace(".", "").isdigit():
            continue
        ln, ix = (int(x) for x in r["main_box"].split("."))
        w = words.get((ln, ix))
        if w is None:          # box not classed as a word by the segmenter: no reading
            continue
        t1 = w["top5"][0] if w["top5"] else None
        res["top1"] += k * (bool(t1) and t1["tokens"] == ref)
        res["top5"] += k * any(t["tokens"] == ref for t in w["top5"])
        res["free"] += k * (w["free_tokens"] == ref)
        fb = w.get("free_best")
        res["free_best"] += k * (bool(fb) and fb["tokens"] == ref)
        for m in hyb:
            hyb[m] += k * hybrid_right(w, ref, m)
    out = {k: percent(v, n) for k, v in res.items()}
    out["hybrid"] = {str(m): percent(v, n) for m, v in hyb.items()}
    return out


def score_merged(al: list[dict], d: dict, margins=MARGINS, merge_rule: bool = False) -> dict:
    """Strict score of a stage-C file: one output word must cover exactly the boxes of the Roman word's unit. A box
    that holds two words is never right, unless `merge_rule` says it is when it reads as the words joined by " _ "."""
    n = len(al)
    by_boxes = {frozenset(w["boxes"]): w for w in d["words"]}
    owner = {b: frozenset(w["boxes"]) for w in d["words"] for b in w["boxes"]}
    res = {"top1": 0, "top5": 0, "free_best": 0}
    hyb = {m: 0 for m in margins}
    kinds: dict[str, int] = {}
    wrong_join = 0
    scorable = ("match", "split") + (MERGE_KINDS if merge_rule else ())
    for r, k, ref in scored_items(al, merge_rule):
        w = by_boxes.get(frozenset(r["box_ids"].split("+")))
        if r["unit_kind"] == "match" and w is None and r["main_box"] in owner:
            wrong_join += 1
        if w is None or r["unit_kind"] not in scorable:   # a box that holds two words is never right
            continue
        t1 = w["top5"][0] if w["top5"] else None
        ok = bool(t1) and t1["tokens"] == ref
        res["top1"] += k * ok
        kinds[r["unit_kind"]] = kinds.get(r["unit_kind"], 0) + k * ok
        res["top5"] += k * any(t["tokens"] == ref for t in w["top5"])
        fb = w.get("free_best")
        res["free_best"] += k * (bool(fb) and fb["tokens"] == ref)
        for m in hyb:
            hyb[m] += k * hybrid_right(w, ref, m)
    out = {k: percent(v, n) for k, v in res.items()}
    out["right_by_unit_kind"] = kinds
    out["single_words_lost_to_wrong_joins"] = wrong_join
    out["joins"] = d["joins"]
    out["hybrid"] = {str(m): percent(v, n) for m, v in hyb.items()}
    return out


def is_merged(d: dict) -> bool:
    """Stage-C files have `boxes` in every word row (and a `joins` count); stage-B files do not."""
    return "joins" in d or any("boxes" in w for w in d["words"])


def score(al: list[dict], d: dict, margins=MARGINS, merge_rule: bool = False) -> dict:
    return (score_merged if is_merged(d) else score_read)(al, d, margins, merge_rule)


def parse_margins(spec: str) -> tuple:
    """'0,2,4' -> (0, 2, 4); a value with a decimal point or exponent stays a float."""
    return tuple(int(x) if x.lstrip("-").isdigit() else float(x) for x in spec.split(","))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("alignment", help="alignment TSV")
    ap.add_argument("readings", nargs="+", help="<stem>_read.json (stage B) or <stem>_merged.json (stage C) files")
    ap.add_argument("--margins", type=parse_margins, default=MARGINS,
                    help="comma list of margins for the hybrid reading (default: %s)" % ",".join(map(str, MARGINS)))
    ap.add_argument("--merge-rule", action="store_true",
                    help="also count the units that hold several Roman words (merge, merge+split) as right when one "
                         "output word covers exactly their boxes and reads as the words joined by ' _ ' "
                         "(experimental, v0.4)")
    a = ap.parse_args()
    al = load_alignment(a.alignment)
    if not al:
        ap.error(f"no rows in {a.alignment}")
    for path in a.readings:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        print(path, json.dumps(score(al, d, a.margins, a.merge_rule), indent=1))


if __name__ == "__main__":
    main()
