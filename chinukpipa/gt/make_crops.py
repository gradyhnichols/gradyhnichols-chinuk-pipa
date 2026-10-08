"""Regenerate ground-truth crops from Internet Archive scans.

First download the source items (original JP2 page archives), then cut the crops:

    python -m chinukpipa.corpus.internet_archive download --ids cihm_15465 --dest corpus/lejeune --kinds orig_jp2
    python -m chinukpipa.gt.make_crops data/gt/vocab_annotations.jsonl corpus/lejeune gt_crops/

Each annotation stores the page index inside <item>_orig_jp2.tar, the deskew angle that was applied, and
boxes in that frame's pixel coordinates (`frame`: {"deskew_deg"}, {"affine", "size"} or {"perspective", "size"}). Rows with
`"crop_clean": true` have ink touching the shorthand crop's border (parts of neighbouring rows) whitened.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from chinukpipa.gt.vocab_pages import binarize, load_ia_page


def clear_border_ink(im: Image.Image) -> Image.Image:
    """Whiten ink components that touch the crop border (pieces of neighbouring rows). Used for rows marked
    `"crop_clean": true`, whose boxes are the union of the row's own components plus a margin, so the row's own
    ink never touches the border."""
    g = np.asarray(im.convert("L"))
    if g.max() - g.min() < 10:
        return im
    lab, _ = ndi.label(binarize(g.astype(np.float32)), structure=np.ones((3, 3)))
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    if not edge:
        return im
    mask = ndi.binary_dilation(np.isin(lab, list(edge)), iterations=2)
    out = g.copy()
    out[mask] = 255
    return Image.fromarray(out)


def apply_frame(im: Image.Image, frame: dict) -> Image.Image:
    """Reproduce the page frame the boxes were drawn in: a deskew rotation, or a full affine or perspective map
    (PIL AFFINE / PERSPECTIVE coefficients mapping output pixels to source pixels, plus output size)."""
    if "affine" in frame:
        return im.transform(tuple(frame["size"]), Image.AFFINE, data=tuple(frame["affine"]),
                            resample=Image.BICUBIC, fillcolor=255)
    if "perspective" in frame:
        return im.transform(tuple(frame["size"]), Image.PERSPECTIVE, data=tuple(frame["perspective"]),
                            resample=Image.BICUBIC, fillcolor=255)
    a = frame.get("deskew_deg", 0.0)
    if abs(a) >= 0.05:
        return im.rotate(a, resample=Image.BICUBIC, fillcolor=255)
    return im


def make_crops(jsonl: str, corpus_dir: str, out_dir: str, fields=("shorthand", "latin", "gloss"), pad: int = 6):
    os.makedirs(out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(jsonl, encoding="utf-8")]
    pages, framed = {}, {}
    n = 0
    for r in rows:
        page = (r["ia_item"], r["page_index"])
        frame = r.get("frame") or {"deskew_deg": r.get("deskew_deg", 0.0)}
        key = (page, json.dumps(frame, sort_keys=True))     # one page can carry rows in different frames
        if key not in framed:
            if page not in pages:
                pages.clear()
                framed.clear()
                pages[page] = load_ia_page(os.path.join(corpus_dir, r["ia_item"]), r["page_index"])
            framed[key] = apply_frame(pages[page], frame)
        im = framed[key]
        for f in fields:
            bb = r.get(f"bbox_{f}")
            if not bb:
                continue
            x0, y0, x1, y1 = bb
            crop = im.crop((max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad))
            if f == "shorthand" and r.get("crop_clean"):
                crop = clear_border_ink(crop)
            crop.save(os.path.join(out_dir, f"{r['id']}_{f}.png"))
            n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl"); ap.add_argument("corpus_dir"); ap.add_argument("out_dir")
    a = ap.parse_args()
    print(make_crops(a.jsonl, a.corpus_dir, a.out_dir), "crops written")
