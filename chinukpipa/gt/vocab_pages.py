"""Segment Le Jeune's vocabulary pages (rows of: Roman word | shorthand | English gloss).

Works on pages like Chinook and Shorthand Rudiments (1898) pp. 7-10: one or two text columns separated by a
vertical (double) rule, left-aligned Roman words, a shorthand outline, then a left-aligned English gloss.
All boxes are returned in the page's full-resolution pixel coordinates as [x0, y0, x1, y1].
"""
from __future__ import annotations

import io
import os
import tarfile
from dataclasses import dataclass, field, asdict

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu


def load_ia_page(item_dir: str, page_index: int) -> Image.Image:
    item = os.path.basename(os.path.normpath(item_dir))
    tp = os.path.join(item_dir, f"{item}_orig_jp2.tar")
    with tarfile.open(tp) as t:
        names = sorted(m.name for m in t.getmembers() if m.name.lower().endswith(".jp2"))
        im = Image.open(io.BytesIO(t.extractfile(names[page_index]).read()))
        im.load()
    return im.convert("L")


def binarize(gray: np.ndarray) -> np.ndarray:
    return gray < threshold_otsu(gray)


def _runs(mask: np.ndarray, min_len: int = 1, max_gap: int = 0):
    """Contiguous True runs in a 1-D mask -> list of (start, end_exclusive)."""
    runs, start, gap = [], None, 0
    for i, v in enumerate(mask):
        if v:
            if start is None:
                start = i
            gap = 0
            end = i + 1
        elif start is not None:
            gap += 1
            if gap > max_gap:
                if end - start >= min_len:
                    runs.append((start, end))
                start, gap = None, 0
    if start is not None and end - start >= min_len:
        runs.append((start, end))
    return runs


def text_rows(sub: np.ndarray, *, min_row_h: int = 25, smooth: int = 15, min_pitch: int = 45):
    """Row bands by valley detection on the smoothed horizontal ink profile (robust to descenders)."""
    prof = ndi.uniform_filter1d(sub.sum(1).astype(float), smooth)
    if prof.max() <= 0:
        return []
    on = prof > 0.04 * np.percentile(prof[prof > 0], 90)
    bands = []
    for y0, y1 in _runs(on, min_len=min_row_h, max_gap=4):
        seg = prof[y0:y1]
        # split tall bands at deep valleys
        cuts = [y0]
        peak = np.percentile(seg, 90)
        i = min_pitch // 2
        while i < len(seg) - min_pitch // 2:
            win = seg[i - min_pitch // 2: i + min_pitch // 2]
            if seg[i] == win.min() and seg[i] < 0.35 * peak and i + y0 - cuts[-1] >= min_pitch * 0.7:
                cuts.append(y0 + i)
                i += min_pitch // 2
            i += 1
        cuts.append(y1)
        bands += [(a, b) for a, b in zip(cuts, cuts[1:]) if b - a >= min_row_h]
    return bands


@dataclass
class Row:
    column: int
    index: int
    bbox: list
    latin_bbox: list | None = None
    shorthand_bbox: list | None = None
    gloss_bbox: list | None = None
    blobs: list = field(default_factory=list)


def text_columns(b: np.ndarray, min_rule_frac: float = 0.25):
    """Split the page at long vertical rules / gutters / dark margins. Returns [(x0, x1)] column spans.

    A rule is anything that survives a vertical opening with a line min_rule_frac * page height long
    (after closing small gaps), i.e. long vertical strokes; ordinary text never does.
    """
    h, w = b.shape
    closed = ndi.binary_closing(b, structure=np.ones((25, 1)))
    closed = ndi.binary_dilation(closed, structure=np.ones((1, 7)))  # tolerate residual slant
    opened = ndi.binary_opening(closed, structure=np.ones((int(min_rule_frac * h), 1)))
    rules = _runs(opened.any(0), min_len=1, max_gap=25)
    # treat the far edges as rules too
    cuts = [(0, 0)] + rules + [(w, w)]
    cols = []
    for (a0, a1), (b0, b1) in zip(cuts, cuts[1:]):
        if b0 - a1 > 0.2 * w:
            cols.append((a1, b0))
    return cols


def estimate_skew(gray: Image.Image, max_deg: float = 3.0, step: float = 0.1) -> float:
    """Angle (degrees) that maximises the sharpness of the horizontal ink profile."""
    small = gray.copy()
    small.thumbnail((800, 800))
    b = binarize(np.asarray(small, dtype=np.float32)).astype(np.uint8) * 255
    im = Image.fromarray(b)
    best = (0.0, -1.0)
    for a in np.arange(-max_deg, max_deg + 1e-9, step):
        prof = np.asarray(im.rotate(a, resample=Image.BILINEAR, fillcolor=0), dtype=np.float32).sum(1)
        score = float(np.var(np.diff(prof)))
        if score > best[1]:
            best = (float(a), score)
    return best[0]


def deskew(gray: Image.Image) -> tuple[Image.Image, float]:
    a = estimate_skew(gray)
    if abs(a) < 0.05:
        return gray, 0.0
    return gray.rotate(a, resample=Image.BICUBIC, fillcolor=255, expand=False), a


def segment_page(gray: Image.Image, *, min_row_h: int = 25, merge_gap: int = 7):
    """Segment an (already deskewed) page; see deskew()."""
    g = np.asarray(gray, dtype=np.float32)
    b = binarize(g)
    h, w = b.shape
    rows: list[Row] = []
    for ci, (cx0, cx1) in enumerate(text_columns(b)):
        sub = b[:, cx0:cx1]
        # drop the facing-page strip / gutter: keep the widest low-density stretch
        bands = text_rows(sub, min_row_h=min_row_h)
        for ri, (y0, y1) in enumerate(bands):
            band = sub[y0:y1]
            # word blobs: dilate horizontally then label
            lab, n = ndi.label(ndi.binary_dilation(band, structure=np.ones((5, merge_gap))))
            objs = ndi.find_objects(lab)
            blobs = []
            for k, sl in enumerate(objs, 1):
                if sl is None:
                    continue
                ys, xs = sl
                ink = band[ys, xs][lab[ys, xs] == k].sum()
                if ink < 40:
                    continue
                blobs.append([xs.start + cx0, ys.start + y0, xs.stop + cx0, ys.stop + y0])
            blobs.sort(key=lambda bb: bb[0])
            if blobs:
                rows.append(Row(ci, ri, [cx0, y0, cx1, y1], blobs=blobs))
    _assign_fields(rows)
    return rows


def _assign_fields(rows: list[Row], tol: int = 18):
    """Gloss = blobs from the column's gloss-alignment x onward. Among the remaining (non-punctuation) blobs,
    split at the widest horizontal gap: left = Roman word, right = shorthand."""
    by_col: dict[int, list[Row]] = {}
    for r in rows:
        by_col.setdefault(r.column, []).append(r)
    for col_rows in by_col.values():
        lefts = [bb[0] for r in col_rows for bb in r.blobs[1:]]
        if not lefts:
            continue
        cx0, cx1 = col_rows[0].bbox[0], col_rows[0].bbox[2]
        mid = cx0 + 0.45 * (cx1 - cx0)
        cand = np.array([x for x in lefts if x > mid])
        if len(cand) == 0:
            continue
        hist = np.bincount(((cand - cand.min()) // tol).astype(int))
        gx = cand.min() + hist.argmax() * tol
        gloss_x = cand[(cand >= gx) & (cand < gx + tol)].min()
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for r in col_rows for b in r.blobs]
        med_area = float(np.median(areas)) if areas else 1.0
        for r in col_rows:
            rh = r.bbox[3] - r.bbox[1]
            # drop rule/gutter fragments (very thin, tall) and anything touching the column edges
            blobs = [bb for bb in r.blobs
                     if not ((bb[2] - bb[0]) < 10 and (bb[3] - bb[1]) > 0.5 * rh)
                     and bb[0] > cx0 + 3 and bb[2] < cx1 - 3]
            # punctuation (commas, apostrophes, stray dots) is kept out of the gap analysis
            body = sorted([bb for bb in blobs
                           if not ((bb[2] - bb[0]) * (bb[3] - bb[1]) < 0.12 * med_area
                                   and (bb[3] - bb[1]) < 0.45 * rh)], key=lambda b: b[0])
            if len(body) < 3:
                continue
            right, gaps = body[0][2], [0]
            for i in range(1, len(body)):
                gaps.append(body[i][0] - right)
                right = max(right, body[i][2])
            best = None
            for j in range(2, len(body)):               # j = first gloss blob
                if body[j][0] < gloss_x - 160:
                    continue
                for i in range(1, j):                   # i = first shorthand blob
                    score = gaps[i] + gaps[j] - 0.15 * abs(body[j][0] - gloss_x)
                    if gaps[i] <= 2 or gaps[j] <= 2:
                        continue
                    if best is None or score > best[0]:
                        best = (score, i, j)
            if best is None:
                continue
            _, i, j = best
            r.latin_bbox = _union(body[:i])
            r.shorthand_bbox = _union(body[i:j])
            # gloss keeps its punctuation (e.g. "what?") but not stray marks beyond the column
            r.gloss_bbox = _union([bb for bb in blobs if bb[0] >= body[j][0] - 2])


def _union(bbs):
    return [min(b[0] for b in bbs), min(b[1] for b in bbs), max(b[2] for b in bbs), max(b[3] for b in bbs)]


def rows_as_dicts(rows):
    return [asdict(r) for r in rows]
