"""Row harvesting for vocabulary pages whose three columns (Roman word : shorthand : gloss) are separated by
dotted vertical rules, as in the 1892 mimeographed vocabulary.

The dotted rules are found from the x-positions of small solid dots (or given with --rules); their dots are removed, the remaining ink
components are split into the three columns by the rules, rows are formed from the Roman-word column, and each
row takes the shorthand and gloss components whose vertical centres fall inside it.

    python -m chinukpipa.gt.rule_columns ITEM_DIR PAGES OUT_JSONL --crops DIR

Output rows have the same fields as chinukpipa.gt.harvest_vocab proposals (boxes in the deskewed page frame),
without OCR. They are proposals only and must be read and checked against the images.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import ndimage as ndi

from chinukpipa.gt.make_crops import clear_border_ink
from chinukpipa.gt.vocab_pages import binarize, deskew, load_ia_page


def _components(b: np.ndarray):
    lab, n = ndi.label(b)
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        ys, xs = sl
        area = int((lab[sl] == i).sum())
        h, w = ys.stop - ys.start, xs.stop - xs.start
        out.append({"x0": xs.start, "x1": xs.stop, "y0": ys.start, "y1": ys.stop, "area": area,
                    "cx": (xs.start + xs.stop) / 2, "cy": (ys.start + ys.stop) / 2, "fill": area / max(1, h * w),
                    "h": h, "w": w})
    return out


def find_rules(comps, page_w: int, max_dot: int = 900, min_dots: int = 8, tol: int = 30):
    """x-positions of dotted vertical rules: columns of small, solid, roughly round blobs."""
    dots = [c for c in comps if 30 < c["area"] < max_dot and c["fill"] > 0.5 and 0.4 < c["h"] / max(1, c["w"]) < 2.5]
    xs = np.array([c["cx"] for c in dots])
    ys = np.array([c["cy"] for c in dots])
    cands = []
    for x in sorted(xs):
        sel = np.abs(xs - x) < tol
        if sel.sum() >= min_dots:
            m = float(np.median(xs[sel]))
            # a rule's dots are many and spread down the whole page
            score = float(sel.sum() * (np.percentile(ys[sel], 90) - np.percentile(ys[sel], 10)))
            cands.append((score, m))
    rules = []
    for score, m in sorted(cands, reverse=True):
        if 0.05 * page_w < m < 0.95 * page_w and not any(abs(m - r) < 4 * tol for _, r in rules):
            rules.append((score, m))
    return [m for _, m in rules], dots


def harvest_page(gray, *, row_gap: int = 30, rules: tuple[float, float] | None = None, straighten: bool = True,
                 flatten: bool = False):
    g, angle = deskew(gray) if straighten else (gray, 0.0)
    a = np.asarray(g, dtype=np.float32)
    if flatten:   # even out uneven lighting / pale ink before thresholding
        a = np.clip(a / np.maximum(ndi.gaussian_filter(a, 25), 1), 0, 1.2) * 200
        a = ndi.gaussian_filter(a, 2.0)   # join dotted (porous) ink into solid strokes
    b = binarize(a)
    b = ndi.binary_opening(b, iterations=1)
    comps = [c for c in _components(b) if c["area"] >= 30]
    found, dots = find_rules(comps, g.width)
    if rules is None:
        if len(found) < 2:
            return angle, found, []
        rules = found[:2]            # the two strongest rules
    r1, r2 = sorted(rules)
    dot_ids = {id(d) for d in dots if min(abs(d["cx"] - r1), abs(d["cx"] - r2)) < 40}
    ink = [c for c in comps if id(c) not in dot_ids and c["h"] < 300 and c["w"] < 900]
    left_edge = r1 - 0.38 * g.width
    lat = [c for c in ink if left_edge < c["cx"] < r1 - 15]
    sh = [c for c in ink if r1 + 15 < c["cx"] < r2 - 15]
    gl = [c for c in ink if r2 + 15 < c["cx"] < r2 + 0.4 * g.width]
    # rows from the Roman-word column: merge components whose vertical extents overlap or nearly touch
    lat.sort(key=lambda c: c["cy"])
    bands = []
    for c in lat:
        if c["area"] < 120:            # stray specks
            continue
        if bands and c["cy"] - bands[-1]["cy_max"] < row_gap:
            B = bands[-1]
            B["comps"].append(c)
            B["cy_max"] = max(B["cy_max"], c["cy"])
        else:
            bands.append({"comps": [c], "cy_max": c["cy"]})
    bands = [B for B in bands if sum(c["area"] for c in B["comps"]) >= 400 and
             max(c["x1"] for c in B["comps"]) - min(c["x0"] for c in B["comps"]) >= 40]
    rows = []
    for B in bands:
        y0 = min(c["y0"] for c in B["comps"])
        y1 = max(c["y1"] for c in B["comps"])
        rows.append({"mid": (y0 + y1) / 2, "half": (y1 - y0) / 2,
                     "bbox_latin": [min(c["x0"] for c in B["comps"]), y0, max(c["x1"] for c in B["comps"]), y1],
                     "sh": [], "gl": []})
    # each shorthand / gloss component goes to the single nearest row (if it is close enough)
    for key, cs in (("sh", sh), ("gl", gl)):
        for c in cs:
            if not rows:
                break
            r = min(rows, key=lambda r: abs(c["cy"] - r["mid"]))
            if abs(c["cy"] - r["mid"]) <= r["half"] + 30:
                r[key].append(c)
    for r in rows:
        for key, f in (("sh", "bbox_shorthand"), ("gl", "bbox_gloss")):
            got = r.pop(key)
            r[f] = [min(c["x0"] for c in got), min(c["y0"] for c in got), max(c["x1"] for c in got),
                    max(c["y1"] for c in got)] if got else None
    return angle, (r1, r2), rows, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("item_dir")
    ap.add_argument("pages")
    ap.add_argument("out_jsonl")
    ap.add_argument("--crops", required=True)
    ap.add_argument("--rules", default="", help="manual rule positions (deskewed x), e.g. 20:1180:1470,22:1250:1600")
    a = ap.parse_args()
    manual = {int(k): (float(x1), float(x2)) for k, x1, x2 in (r.split(":") for r in a.rules.split(",") if r)}
    item = os.path.basename(os.path.normpath(a.item_dir))
    os.makedirs(a.crops, exist_ok=True)
    with open(a.out_jsonl, "w", encoding="utf-8") as out:
        for p in [int(x) for x in a.pages.split(",")]:
            res = harvest_page(load_ia_page(a.item_dir, p), rules=manual.get(p))
            if len(res) == 3:
                print(item, p, "rules not found", res[1])
                continue
            angle, rules, rows, g = res
            print(item, p, f"deskew={angle:.2f} rules={[round(r) for r in rules]} rows={len(rows)}")
            for k, r in enumerate(rows):
                rid = f"{item}_p{p:03d}_R_r{k:02d}"
                rec = {"id": rid, "ia_item": item, "page_index": p, "frame": {"deskew_deg": angle},
                       "page_size": list(g.size), "crop_clean": True, "bbox_latin": r["bbox_latin"],
                       "bbox_shorthand": r["bbox_shorthand"], "bbox_gloss": r["bbox_gloss"],
                       "latin_ocr": "", "gloss_ocr": ""}
                for f in ("latin", "shorthand", "gloss"):
                    bb = rec[f"bbox_{f}"]
                    if bb:
                        x0, y0, x1, y1 = bb
                        crop = g.crop((max(0, x0 - 6), max(0, y0 - 6), x1 + 6, y1 + 6))
                        if f == "shorthand":
                            crop = clear_border_ink(crop)
                        crop.save(os.path.join(a.crops, f"{rid}_{f}.png"))
                out.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
