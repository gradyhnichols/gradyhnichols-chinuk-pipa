"""Carry annotations over to a second scanned copy of the same printed (or mimeographed) pages.

Two physical copies of one edition carry the same writing, but each sheet was inked, aged and photographed
differently, so the second copy gives new images of already-read words. For each annotated page, the source
page (in its annotation frame) is matched to the target scan with ORB keypoints and a RANSAC homography; each
row's boxes are then refined by template matching of its shorthand crop within a small window. The output rows
keep the readings, point to the target item, and record the transform as a `perspective` frame.

    python -m chinukpipa.gt.register_copy data/gt/vocab_annotations.jsonl corpus/lejeune \\
        --source cihm_15474 --target Ayer_PM848_L4_1892 --offset -4 --out rows_ayer.jsonl [--check-dir DIR]

`--offset` maps a source page index to the target page index (`--page-map` lists exceptions). Rows whose
refined match is weak are dropped.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

from chinukpipa.gt.make_crops import apply_frame
from chinukpipa.gt.vocab_pages import load_ia_page


def _norm(a: np.ndarray) -> np.ndarray:
    """Flatten uneven lighting and stretch contrast, so two scans of different tone can be matched."""
    a = a.astype(np.float32)
    bg = cv2.GaussianBlur(a, (0, 0), 25)
    r = np.clip(a / np.maximum(bg, 1), 0, 1.2)
    return cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def page_homography(src: Image.Image, dst: Image.Image, scale: float = 0.5):
    """3x3 H mapping source-frame pixels to target-scan pixels, plus the inlier count."""
    s = np.asarray(src.resize((int(src.width * scale), int(src.height * scale))))
    d = np.asarray(dst.resize((int(dst.width * scale), int(dst.height * scale))))
    s, d = _norm(s), _norm(d)
    orb = cv2.ORB_create(8000)
    ks, ds = orb.detectAndCompute(s, None)
    kd, dd = orb.detectAndCompute(d, None)
    if ds is None or dd is None:
        return None, 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ds, dd, k=2)
    good = [a for a, b in (p for p in m if len(p) == 2) if a.distance < 0.8 * b.distance]
    if len(good) < 20:
        return None, len(good)
    ps = np.float32([ks[g.queryIdx].pt for g in good]) / scale
    pd = np.float32([kd[g.trainIdx].pt for g in good]) / scale
    H, inl = cv2.findHomography(ps, pd, cv2.RANSAC, 6.0)
    return H, int(inl.sum()) if inl is not None else 0


def perspective_frame(H: np.ndarray, size) -> dict:
    """PIL PERSPECTIVE coefficients (output pixel -> input pixel) for the target scan, output in the source frame."""
    H = H / H[2, 2]
    return {"perspective": [float(x) for x in H.flatten()[:8]], "size": list(size)}


def refine_box(src_f: np.ndarray, dst_f: np.ndarray, bb, search: int = 40):
    """Shift `bb` (same coordinates in both framed pages) to where its source patch best matches the target."""
    x0, y0, x1, y1 = bb
    tpl = src_f[y0:y1, x0:x1]
    X0, Y0 = max(0, x0 - search), max(0, y0 - search)
    win = dst_f[Y0:y1 + search, X0:x1 + search]
    if tpl.size == 0 or win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1]:
        return bb, 0.0
    r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, (mx, my) = cv2.minMaxLoc(r)
    dx, dy = X0 + mx - x0, Y0 + my - y0
    return [x0 + dx, y0 + dy, x1 + dx, y1 + dy], float(score)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl")
    ap.add_argument("corpus_dir")
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--offset", type=int, required=True)
    ap.add_argument("--page-map", default="", help="exceptions to --offset, e.g. 7:5,8:6 (source:target)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--pages", help="comma-separated source page indices (default: all annotated)")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.jsonl, encoding="utf-8")]
    rows = [r for r in rows if r["ia_item"] == a.source]
    pages = sorted({r["page_index"] for r in rows})
    if a.pages:
        pages = [p for p in pages if p in {int(x) for x in a.pages.split(",")}]
    pmap = {int(k): int(v) for k, v in (x.split(":") for x in a.page_map.split(",") if x)}
    out, kept, dropped = [], 0, 0
    for p in pages:
        tp = pmap.get(p, p + a.offset)
        raw_src = load_ia_page(os.path.join(a.corpus_dir, a.source), p)
        dst = load_ia_page(os.path.join(a.corpus_dir, a.target), tp)
        by_frame = {}
        for r in rows:
            if r["page_index"] == p:
                by_frame.setdefault(json.dumps(r["frame"], sort_keys=True), []).append(r)
        for fkey, frs in by_frame.items():
            src = apply_frame(raw_src, frs[0]["frame"])
            H, ninl = page_homography(src, dst)
            if H is None or ninl < 40:
                print(f"page {p}: no reliable match ({ninl} inliers)")
                dropped += len(frs)
                continue
            frame = perspective_frame(H, src.size)
            frame["note"] = f"registered to {a.source} page {p} (ORB + RANSAC homography, {ninl} inliers)"
            dst_f = apply_frame(dst, frame)
            sf, df = _norm(np.asarray(src)), _norm(np.asarray(dst_f))
            for r in frs:
                bb, score = refine_box(sf, df, r["bbox_shorthand"])
                if score < a.min_score:
                    dropped += 1
                    continue
                dx, dy = bb[0] - r["bbox_shorthand"][0], bb[1] - r["bbox_shorthand"][1]
                n = dict(r)
                n["id"] = f"{a.target}_p{tp:03d}_" + r["id"].split("_p", 1)[1].split("_", 1)[1]
                n["ia_item"], n["page_index"], n["frame"], n["page_size"] = a.target, tp, frame, list(src.size)
                for f in ("shorthand", "latin", "gloss"):
                    b = r.get(f"bbox_{f}")
                    n[f"bbox_{f}"] = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy] if b else None
                n["copy_of"] = r["id"]
                n["reviewed_by"] = f"readings carried over from {r['id']} (same edition, another copy)"
                n["notes"] = f"template match {score:.2f}" + (f"; {r['notes']}" if r.get("notes") else "")
                out.append(n)
                kept += 1
            print(f"page {p} -> {tp}: {ninl} inliers, rows kept so far {kept}")
    with open(a.out, "w", encoding="utf-8") as f:
        for n in out:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    print(f"kept {kept}, dropped {dropped}")


if __name__ == "__main__":
    main()
