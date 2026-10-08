"""Stage A of the text pipeline (CPU): segment pages of running text and keep each page's word images in one .npz.

    python -m chinukpipa.text.segcrops ITEM_DIR --leaves 15-126,130 --out-dir OUT [--shard K/N]

For each leaf (every N-th one starting at K with --shard): segment.py's page analysis (writes <stem>_seg.json and
<stem>_overlay.png), then one masked crop per 'word' box (the same cut as read_page.word_crops), saved as
<stem>_words.npz: flat uint8 pixels + shapes + boxes + line/index (readcrops.load_npz reads it back). Pages whose
.npz exists are skipped, so a run can be restarted. Needs numpy, scipy, scikit-image, OpenCV, Pillow (no PyTorch).
One JSON line per page on stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy import ndimage as ndi

from . import segment as seglib


def word_crops(seg: dict, norm: np.ndarray, labels: np.ndarray, pad_rel: float = 0.12):
    """Copy of read_page.word_crops (kept here so that this stage needs no PyTorch)."""
    h_med = seg["stats"]["h_med"]
    pad = max(4, int(round(pad_rel * h_med)))
    grow = max(1, int(round(0.4 * seg["stats"]["stroke_width"])))
    H, W = labels.shape
    out = []
    for ln in seg["lines"]:
        if ln["kind"] != "text":
            continue
        for wi, wd in enumerate(ln["words"]):
            if wd["kind"] != "word":
                continue
            ids = wd["comps"] + wd["marks"]
            x0, y0, x1, y1 = wd["bbox"]
            X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
            keep = ndi.binary_dilation(np.isin(labels[Y0:Y1, X0:X1], ids), iterations=grow)
            g = (norm[Y0:Y1, X0:X1] * 255).astype(np.uint8)
            out.append({"line": ln["number"], "index": wi + 1, "bbox": [int(v) for v in wd["bbox"]],
                        "pixels": np.where(keep, g, 255).astype(np.uint8)})
    return out


def save_crops(npz: str, crops: list[dict]) -> None:
    """Write a page's crops (dicts from word_crops) as one compressed .npz. The file is written under a temporary
    name and renamed, so that an interrupted run never leaves a half-written page behind."""
    arrs = [c["pixels"] for c in crops]
    tmp = npz + ".tmp.npz"
    np.savez_compressed(tmp, flat=np.concatenate([x.ravel() for x in arrs]) if arrs else np.zeros(0, np.uint8),
                        shapes=np.array([x.shape for x in arrs], dtype=np.int32).reshape(-1, 2),
                        boxes=np.array([c["bbox"] for c in crops], dtype=np.int32).reshape(-1, 4),
                        line=np.array([c["line"] for c in crops], dtype=np.int32),
                        index=np.array([c["index"] for c in crops], dtype=np.int32))
    os.replace(tmp, npz)


def parse_leaves(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("item_dir")
    ap.add_argument("--leaves", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard", default="0/1", help="K/N: take every N-th leaf starting with the K-th")
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))
    os.makedirs(a.out_dir, exist_ok=True)
    for leaf in parse_leaves(a.leaves)[k::n]:
        stem = seglib.page_stem(a.item_dir, leaf)
        npz = os.path.join(a.out_dir, f"{stem}_words.npz")
        if os.path.exists(npz):
            continue
        t = time.time()
        try:
            seg, norm = seglib.run(a.item_dir, leaf, a.out_dir)
            crops = word_crops(seg, norm, seg["_labels"])
            save_crops(npz, crops)
            print(json.dumps({"page": stem, "words": len(crops), "sec": round(time.time() - t, 1)}), flush=True)
        except Exception as e:  # keep going: one bad page should not stop a shard
            print(json.dumps({"page": stem, "error": repr(e)[:300]}), flush=True)


if __name__ == "__main__":
    main()
