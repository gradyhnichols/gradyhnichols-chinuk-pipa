"""Regenerate ground-truth crops from Internet Archive scans.

First download the source items (original JP2 page archives), then cut the crops:

    python -m chinukpipa.corpus.internet_archive download --ids cihm_15465 --dest corpus/lejeune --kinds orig_jp2
    python -m chinukpipa.gt.make_crops data/gt/vocab_annotations.jsonl corpus/lejeune gt_crops/

Each annotation stores the page index inside <item>_orig_jp2.tar, the deskew angle that was applied, and
boxes in that frame's pixel coordinates (`frame`: {"deskew_deg"} or {"affine", "size"}).
"""
from __future__ import annotations

import argparse
import json
import os

from PIL import Image

from chinukpipa.gt.vocab_pages import load_ia_page


def apply_frame(im: Image.Image, frame: dict) -> Image.Image:
    """Reproduce the page frame the boxes were drawn in: a deskew rotation or a full affine map
    (PIL AFFINE coefficients mapping output pixels to source pixels, plus output size)."""
    if "affine" in frame:
        return im.transform(tuple(frame["size"]), Image.AFFINE, data=tuple(frame["affine"]),
                            resample=Image.BICUBIC, fillcolor=255)
    a = frame.get("deskew_deg", 0.0)
    if abs(a) >= 0.05:
        return im.rotate(a, resample=Image.BICUBIC, fillcolor=255)
    return im


def make_crops(jsonl: str, corpus_dir: str, out_dir: str, fields=("shorthand", "latin", "gloss"), pad: int = 6):
    os.makedirs(out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(jsonl, encoding="utf-8")]
    cache = {}
    n = 0
    for r in rows:
        key = (r["ia_item"], r["page_index"])
        if key not in cache:
            cache.clear()
            im = load_ia_page(os.path.join(corpus_dir, r["ia_item"]), r["page_index"])
            cache[key] = apply_frame(im, r.get("frame") or {"deskew_deg": r.get("deskew_deg", 0.0)})
        im = cache[key]
        for f in fields:
            bb = r.get(f"bbox_{f}")
            if not bb:
                continue
            x0, y0, x1, y1 = bb
            im.crop((max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad)).save(
                os.path.join(out_dir, f"{r['id']}_{f}.png"))
            n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl"); ap.add_argument("corpus_dir"); ap.add_argument("out_dir")
    a = ap.parse_args()
    print(make_crops(a.jsonl, a.corpus_dir, a.out_dir), "crops written")
