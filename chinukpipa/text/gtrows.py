"""Read ground-truth word crops with the stage-B reader, and score the readings against the rows' labels.

Ground-truth rows (`data/gt/*.jsonl`) give one shorthand crop per word (cut by `chinukpipa.gt.make_crops` as
`<id>_shorthand.png`). To read them with the same reader as running text, pack them as stage-A pages (one page per
`page_index`), read them with `readcrops`, and score. (These crops are cut from the page with a small margin and
border cleaning by make_crops; they are not the masked crops stage A cuts from a segmented page.)

    python -m chinukpipa.text.gtrows pack ROWS.jsonl CROPS_DIR PAGES_DIR --book NAME
    python -m chinukpipa.text.readcrops --pages PAGES_DIR --models M1.pt M2.pt ... --out-dir READ_DIR [--lexicon TSV]
    python -m chinukpipa.text.gtrows score ROWS.jsonl PAGES_DIR READ_DIR --book NAME [--lexicon TSV]

Within a page, words are numbered in the rows' reading order (`seq` when present, else file order); the numbers
are saved in `PAGES_DIR/<book>_rows.json`. Scored are the Chinook rows (`language` "chn") with a label
(`tokens_verified` if present, else `tokens_rule`): the word list's first choice and top five against the label, the
first model's free reading (exact, and token error rate),
the free_best reading, and the same first-choice and top-five scores for the words whose token string is a
candidate of the word list (pass the lexicon used for reading). Words are read from correct boxes, so this measures
reading, not segmentation.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

from . import DEFAULT_LEXICON


def load_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def pack(rows_path: str, crops: str, out: str, book: str) -> None:
    rows = load_rows(rows_path)
    by: dict[int, list[tuple[int, dict]]] = {}
    for k, r in enumerate(rows):
        by.setdefault(r["page_index"], []).append((k, r))
    os.makedirs(out, exist_ok=True)
    keys = {}
    for page, items in sorted(by.items()):
        items.sort(key=lambda kr: (kr[1].get("seq", kr[0]), kr[0]))
        arrs, boxes = [], []
        for n, (_, r) in enumerate(items, 1):
            arrs.append(np.asarray(Image.open(os.path.join(crops, f"{r['id']}_shorthand.png")).convert("L"),
                                   dtype=np.uint8))
            boxes.append(r["bbox_shorthand"])
            keys[r["id"]] = [page, n]
        np.savez_compressed(os.path.join(out, f"{book}_{page}_words.npz"),
                            flat=np.concatenate([a.ravel() for a in arrs]),
                            shapes=np.array([a.shape for a in arrs], dtype=np.int32).reshape(-1, 2),
                            boxes=np.array(boxes, dtype=np.int32).reshape(-1, 4),
                            line=np.arange(1, len(arrs) + 1, dtype=np.int32),
                            index=np.ones(len(arrs), dtype=np.int32))
    with open(os.path.join(out, f"{book}_rows.json"), "w", encoding="utf-8") as fh:
        json.dump(keys, fh)
    print(f"packed {len(keys)} words on {len(by)} pages")


def lexicon_keys(path: str) -> set[str]:
    """Token strings of the word list's candidates, selected as readcrops selects them (A/B rows; a `tokens`
    column, when present and filled, is used as given)."""
    from chinukpipa.htr.data import vocab
    from chinukpipa.translit import latin_to_tokens
    alphabet = set(vocab())
    keys = set()
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        hi, ci = header.index("headword"), header.index("best_conf")
        ti = header.index("tokens") if "tokens" in header else None
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c[ci] not in ("A", "B"):
                continue
            try:
                toks = c[ti].split() if ti is not None and ti < len(c) and c[ti].strip() else latin_to_tokens(c[hi])
            except ValueError:
                continue
            if toks and all(t in alphabet for t in toks):
                keys.add(" ".join(toks))
    return keys


def edit_distance(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def score(rows_path: str, pages: str, read_dir: str, book: str, lexicon: str) -> dict:
    keys = json.load(open(os.path.join(pages, f"{book}_rows.json"), encoding="utf-8"))
    readings = {}
    for p in glob.glob(os.path.join(read_dir, f"{book}_*_read.json")):
        d = json.load(open(p, encoding="utf-8"))
        for w in d["words"]:
            readings[(d["leaf"], w["line"])] = w
    cand = lexicon_keys(lexicon)
    n_read = {json.load(open(p, encoding="utf-8")).get("n_candidates")
              for p in glob.glob(os.path.join(read_dir, f"{book}_*_read.json"))} - {None}
    if n_read and n_read != {len(cand)}:
        raise SystemExit(f"the readings used {sorted(n_read)} candidates, the lexicon given has {len(cand)}: "
                         "pass the lexicon used for reading")
    c = dict.fromkeys(["words", "top1", "top5", "free", "free_best", "edits", "ref_tokens", "in_list",
                       "top1_in_list", "top5_in_list", "chn_without_tokens", "no_reading"], 0)
    for r in load_rows(rows_path):
        if r.get("language") != "chn":
            continue
        ref = r.get("tokens_verified") or r.get("tokens_rule")
        if not ref:
            c["chn_without_tokens"] += 1
            continue
        w = readings.get(tuple(keys[r["id"]])) if r["id"] in keys else None
        if w is None:
            c["no_reading"] += 1
            continue
        top = [t["tokens"] for t in w["top5"]]
        first = bool(top) and top[0] == ref
        c["words"] += 1
        c["top1"] += first
        c["top5"] += ref in top
        c["free"] += w["free_tokens"] == ref
        c["free_best"] += bool(w.get("free_best")) and w["free_best"]["tokens"] == ref
        c["edits"] += edit_distance(w["free_tokens"].split(), ref.split())
        c["ref_tokens"] += len(ref.split())
        if ref in cand:
            c["in_list"] += 1
            c["top1_in_list"] += first
            c["top5_in_list"] += ref in top
    if c["no_reading"]:
        raise SystemExit(f"{c['no_reading']} labelled rows have no reading: read every page packed for them")
    n, m = max(1, c["words"]), max(1, c["in_list"])
    out = {"words": c["words"], "list_first_choice": round(c["top1"] / n, 4), "list_top5": round(c["top5"] / n, 4),
           "free_exact": round(c["free"] / n, 4), "free_best_exact": round(c["free_best"] / n, 4),
           "free_token_error_rate": round(c["edits"] / max(1, c["ref_tokens"]), 4),
           "in_list": c["in_list"], "in_list_first_choice": round(c["top1_in_list"] / m, 4),
           "in_list_top5": round(c["top5_in_list"] / m, 4), "counts": c}
    print(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("rows")
    p.add_argument("crops")
    p.add_argument("pages")
    p.add_argument("--book", required=True)
    s = sub.add_parser("score")
    s.add_argument("rows")
    s.add_argument("pages")
    s.add_argument("read")
    s.add_argument("--book", required=True)
    s.add_argument("--lexicon", default=DEFAULT_LEXICON)
    s.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "pack":
        pack(a.rows, a.crops, a.pages, a.book)
    else:
        res = score(a.rows, a.pages, a.read, a.book, a.lexicon)
        if a.out:
            with open(a.out, "w", encoding="utf-8") as fh:
                json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
