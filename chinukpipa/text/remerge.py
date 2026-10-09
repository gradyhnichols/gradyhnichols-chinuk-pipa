"""Stage C of the text pipeline: recognition-guided merging of word pieces.

The segmenter cuts a line into words at gaps wider than a page-level threshold (the valley of the gap histogram);
a word whose own signs are spaced a little wider gets cut in two. Here, walking each text line left to right,
the current group of boxes and the next box are joined when the gap between them is below the page's typical
word gap (gap_stats.mode_large_px, a page statistic, not tuned) AND the joined image reads better than the
pieces: best word-list loss of the joined image < sum of the pieces' best losses (both are CTC negative
log-likelihoods of the same ink under the same models). Joined images are rebuilt from the stage-A crops.

    python -m chinukpipa.text.remerge --pages DIR --read DIR --out-dir DIR --models M0 M1 M2 [--lexicon TSV]
        [--books a,b] [--leaves 15] [--device cpu|cuda] [--extra-lexicon TSV ...] [--pair-penalty NATS]

Input: <stem>_seg.json + <stem>_words.npz (stage A), <stem>_read.json (stage B, read with the same lexicon and
models). Output: <stem>_merged.json: the page's words after merging, each with its box ids "line.index", bbox,
and the reading (same fields as stage B). Machine readings; no person has checked them.

--extra-lexicon and --pair-penalty are the experimental options of readcrops (v0.4) and must be given the same
values as in stage B; the joined images are read with them. With --pair-penalty the joins are decided as without
it, on the best list word's score (`best_word_score`, written by readcrops for the pieces and computed for the joined
image): two-word readings take no part in the decision, so they do not change which boxes are joined. They are part
of each unit's reading: the stage-B reading of a box that is not joined, the new reading of a joined image. This is
the path tested on the Creation page (joins on list-word scores, then two-word decoding of the units). A stage-B
file read with another penalty, or without `best_word_score`, is refused. Without the options nothing changes.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

from . import DEFAULT_LEXICON
from . import readcrops as rc


def compose(items, pad):
    """items: [(bbox, pixels)] of stage-A crops (each cut at bbox +- pad, clipped at 0). Darkest pixel wins."""
    ox = [max(0, b[0] - pad) for b, _ in items]
    oy = [max(0, b[1] - pad) for b, _ in items]
    x0, y0 = min(ox), min(oy)
    x1 = max(o + p.shape[1] for o, (_, p) in zip(ox, items))
    y1 = max(o + p.shape[0] for o, (_, p) in zip(oy, items))
    canvas = np.full((y1 - y0, x1 - x0), 255, np.uint8)
    for (b, p), a, c in zip(items, ox, oy):
        sub = canvas[c - y0:c - y0 + p.shape[0], a - x0:a - x0 + p.shape[1]]
        np.minimum(sub, p, out=sub)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", required=True)
    ap.add_argument("--read", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--lexicon", default=DEFAULT_LEXICON, help="word list TSV (default: %(default)s)")
    ap.add_argument("--extra-lexicon", action="append", metavar="TSV",
                    help="append the rows of this word list TSV to the main one (repeatable; as in stage B)")
    ap.add_argument("--pair-penalty", type=float, metavar="NATS",
                    help="propose two-word readings and add NATS to their loss (as in stage B; default: off)")
    ap.add_argument("--books", default="")
    ap.add_argument("--leaves", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--max-group", type=int, default=3)
    a = ap.parse_args()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.pair_penalty is not None and not math.isfinite(a.pair_penalty):
        sys.exit("--pair-penalty must be a finite number")
    ens = rc.Ensemble(a.models, a.lexicon, a.device, a.extra_lexicon or [])
    want = set(filter(None, a.books.split(",")))
    leaves = {int(x) for x in a.leaves.split(",") if x}
    os.makedirs(a.out_dir, exist_ok=True)
    tot = {"pages": 0, "boxes": 0, "joins": 0}
    for rp in sorted(glob.glob(os.path.join(a.read, "*_read.json"))):
        stem = os.path.basename(rp)[: -len("_read.json")]
        book, leaf = rc.book_of(stem + "_words.npz")
        if (want and book not in want) or (leaves and leaf not in leaves):
            continue
        rd = json.load(open(rp, encoding="utf-8"))
        if rd.get("n_candidates") != ens.N or len(rd.get("models", [])) != len(a.models):
            sys.exit(f"{rp}: read with another word list or model set (n_candidates {rd.get('n_candidates')} vs "
                     f"{ens.N}, {len(rd.get('models', []))} vs {len(a.models)} models); use the stage-B lexicon and models")
        if rd.get("pair_penalty") != a.pair_penalty:
            sys.exit(f"{rp}: read with pair penalty {rd.get('pair_penalty')}, this run has {a.pair_penalty}: the "
                     "scores of the pieces and of the joined images must be of the same kind; use the stage-B value")
        if a.pair_penalty is not None and any("best_word_score" not in w for w in rd["words"]):
            sys.exit(f"{rp}: a word has no best_word_score; read stage B again with this version")
        seg = json.load(open(os.path.join(a.pages, f"{stem}_seg.json"), encoding="utf-8"))
        arrs, boxes, line, index = rc.load_npz(os.path.join(a.pages, f"{stem}_words.npz"))
        crop = {(int(l), int(i)): (list(map(int, b)), p) for p, b, l, i in zip(arrs, boxes, line, index)}
        reading = {(w["line"], w["index"]): w for w in rd["words"]}
        pad = max(4, int(round(0.12 * seg["stats"]["h_med"])))
        gmax = (seg["stats"].get("gap_stats") or {}).get("mode_large_px") or 2.2 * seg["stats"]["word_gap_threshold_px"]
        scale = rd["scale"]

        def best(r):
            """The score a join is decided on: the best list word's score (never a two-word reading)."""
            if a.pair_penalty is not None:
                if "best_word_score" not in r:
                    sys.exit(f"{rp}: a word has no best_word_score; read stage B again with this version")
                s = r["best_word_score"]
            else:
                t = r["top5"][0] if r["top5"] else None
                s = t["score"] if t else None
            return s if s is not None else float("inf")

        def read_join(keys):
            x = rc.prepare_word(compose([crop[k] for k in keys], pad), [scale])[0]
            return ens.read_batch([x], pair_penalty=a.pair_penalty)[0]

        out_words = []
        joins = 0
        for ln in seg["lines"]:
            if ln["kind"] != "text":
                continue
            ws = [((ln["number"], wi + 1), wd) for wi, wd in enumerate(ln["words"])
                  if wd["kind"] == "word" and (ln["number"], wi + 1) in reading]
            ws.sort(key=lambda kw: kw[1]["bbox"][0])
            g_keys, g_read = None, None
            for key, wd in ws:
                r = reading[key]
                if g_keys is not None:
                    gap = wd["bbox"][0] - max(crop[k][0][2] for k in g_keys)
                    if gap < gmax and len(g_keys) < a.max_group:
                        jr = read_join(g_keys + [key])
                        if best(jr) < best(g_read) + best(r):
                            g_keys, g_read = g_keys + [key], jr
                            joins += 1
                            continue
                    out_words.append((g_keys, g_read))
                g_keys, g_read = [key], r
            if g_keys is not None:
                out_words.append((g_keys, g_read))
        words = []
        for keys, r in out_words:
            bb = [crop[k][0] for k in keys]
            w = {"boxes": [f"{k[0]}.{k[1]}" for k in keys], "line": keys[0][0], "index": keys[0][1],
                 "bbox": [min(b[0] for b in bb), min(b[1] for b in bb), max(b[2] for b in bb), max(b[3] for b in bb)]}
            w.update({k: v for k, v in r.items() if k not in ("line", "index", "bbox")})
            words.append(w)
        rule = "join if gap < gap_stats.mode_large_px and best loss(joined) < sum of best losses"
        used = {}           # what the options add to the file; with none of them the file is what it always was
        if a.extra_lexicon:
            used["extra_lexicons"] = [os.path.basename(p) for p in a.extra_lexicon]
        if a.pair_penalty is not None:
            used["pair_penalty"] = a.pair_penalty
            rule += (" (best loss = best list word's score, best_word_score: two-word readings take no part in the"
                     " decision; the units are read with them)")
        out = {"item": book, "leaf": leaf, "scale": scale, "models": a.models, "lexicon": os.path.basename(a.lexicon),
               **used, "merge_rule": rule, "gap_max_px": gmax, "joins": joins,
               "reading": "machine reading (CRNN ensemble), not checked by a person", "words": words}
        with open(os.path.join(a.out_dir, f"{stem}_merged.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
        tot["pages"] += 1
        tot["boxes"] += len(reading)
        tot["joins"] += joins
        print(json.dumps({"page": stem, "boxes": len(reading), "joins": joins, "words": len(words)}), flush=True)
    print(json.dumps(tot), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)   # all outputs are closed; skips library unloading, which can crash some CUDA installs at exit
