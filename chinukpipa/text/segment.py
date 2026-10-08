"""Segment a page of running Chinuk Pipa text into text lines and word boxes.

    python -m chinukpipa.text.segment ITEM_DIR LEAF --out-dir DIR   # writes <id>_<leaf>_seg.json and _overlay.png

All thresholds are measured on the page itself, mostly as multiples of the median height `h_med` of word-sized
components, of the stroke width and of the line pitch.

1. **Page**: raw camera captures are rotated and cropped with IA's scandata (pagesrc.py); processed derivatives
   are cut to the sheet of paper (largest bright region).
2. **Binarize**: grey page / smooth background estimate, Otsu threshold on the ratio, grown by hysteresis
   through slightly lighter connected pixels (rejoins faint mimeograph strokes, ignores show-through).
3. **Deskew**: the angle (+-3 degrees) with the sharpest projection profile of the ink pixels, long rules left out.
4. **Furniture removal**: long straight rules (a vertical rule splits columns; a header rule cuts off the running
   head and page number); dotted/dashed page frames (>= 6 alike, evenly spaced dots or dashes on one row or column
   with no word reaching across) and everything outside them; border-touching, huge and thin-rule components;
   stanza separators ("~~~": isolated, flat, wide, centred); braces (thin, taller than 1.1 line pitches).
5. **Lines**: per column, horizontal ink profile of word-sized components, pitch from its autocorrelation, line
   centres at the profile peaks; components go to the nearest line, lines are refitted with the column's median
   slope, components are reassigned. Small marks (dots, vowel circles, punctuation) go to the nearest line.
6. **Words**: word-sized components of a line are chained left to right by nearest-ink distance; a distance above
   the page's word-gap threshold starts a new word. The threshold is the density valley between the two modes
   of the page's log-distances (within-word pen lifts vs. spaces between words). Marks never bridge two words:
   each joins its nearest word, inside the word image when it lies within the word's span or is a vowel circle,
   otherwise as an `outside_mark` (punctuation, the sign of the cross). Groups of marks without a closed loop,
   a lone 'x'/'+', and narrow one-stroke marks with a dot (';', '!') are labelled `punct`; runs of >= 6 small
   glyphs on one baseline (typeset Roman/italic text) are labelled `roman`. Only `word` boxes are read.
7. **Non-text lines**: page numbers (a narrow centred group as first or last line of a column) and, in the
   hymn-book layout, headings (a line directly above a separator, preceded by a separator, a page number or
   nothing) are labelled and left out of the numbered text lines.

Coordinates in the JSON are pixels of the deskewed page (`page_size`); `load_and_prepare(..., angle)` rebuilds
that page, so crops can be taken later without storing the page image.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from skimage.filters import apply_hysteresis_threshold, threshold_otsu
from skimage.morphology import skeletonize

from .pagesrc import is_raw_capture, load_page

# ----------------------------------------------------------------------------------------------- image prep


def normalize_background(gray: np.ndarray) -> np.ndarray:
    """Grey page / smooth background estimate, clipped to [0, 1] (paper ~1, ink well below)."""
    g = gray.astype(np.float32)
    h, w = g.shape
    f = 600.0 / max(h, w)
    small = cv2.resize(g, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    # paper level: a closing removes the (thin, dark) ink, a median smooths what is left
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    bg = cv2.medianBlur(np.clip(bg, 0, 255).astype(np.uint8), 21).astype(np.float32)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(g / np.maximum(bg, 1.0), 0.0, 1.0)


def binarize(norm: np.ndarray, t: float | None = None, weak: float = 0.3) -> tuple[np.ndarray, float]:
    """Ink mask by hysteresis: pixels darker than Otsu's threshold `t` seed the ink, which then grows through
    connected pixels darker than t + weak * (1 - t). This rejoins faint, broken pen strokes (mimeograph) without
    taking in isolated light marks such as show-through from the back of the leaf."""
    if t is None:
        t = float(threshold_otsu(norm[::2, ::2]))
    return apply_hysteresis_threshold(1.0 - norm, 1.0 - (t + weak * (1.0 - t)), 1.0 - t), t


def estimate_skew(binary: np.ndarray, max_deg: float = 3.0, step: float = 0.05, bin_px: float = 2.0,
                  max_points: int = 400_000, seed: int = 0) -> float:
    """Angle (degrees, PIL rotate() convention) that makes the text lines horizontal: the ink pixels are projected
    onto the rotated vertical axis and the angle with the sharpest projection profile (largest sum of squared
    differences between neighbouring bins) wins. Projecting coordinates avoids the bias toward 0 degrees that
    rotating (and so blurring) the image itself would introduce."""
    ys, xs = np.nonzero(binary)
    if len(ys) == 0:
        return 0.0
    if len(ys) > max_points:
        k = np.random.default_rng(seed).choice(len(ys), max_points, replace=False)
        ys, xs = ys[k], xs[k]
    ys = ys.astype(np.float64)
    xs = xs.astype(np.float64) - xs.mean()
    best = (0.0, -1.0)
    for a in np.arange(-max_deg, max_deg + 1e-9, step):
        t = np.deg2rad(a)
        yr = ys * np.cos(t) - xs * np.sin(t)
        hist = np.bincount(((yr - yr.min()) / bin_px).astype(np.int64)).astype(np.float64)
        hist = ndi.gaussian_filter1d(hist, 1.0)
        score = float(np.sum(np.diff(hist) ** 2))
        if score > best[1]:
            best = (round(float(a), 2), score)
    return best[0]


def stroke_width(binary: np.ndarray) -> float:
    """Median stroke width (pixels): 2 x distance-to-background sampled on the skeleton."""
    sk = skeletonize(binary)
    if not sk.any():
        return 1.0
    dist = ndi.distance_transform_edt(binary)
    return float(2 * np.median(dist[sk]))


def paper_region(gray: np.ndarray, inset: float = 0.01) -> list[int]:
    """Bounding box [x0, y0, x1, y1] of the sheet of paper: the largest bright region at low resolution (holes
    filled), shrunk by `inset` x its size. Falls back to the whole image if that region is implausibly small."""
    h, w = gray.shape
    f = 500.0 / max(h, w)
    small = cv2.resize(gray, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    bright = small > threshold_otsu(small)
    bright = ndi.binary_opening(bright, structure=np.ones((5, 5)))
    lab, n = ndi.label(bright)
    if n == 0:
        return [0, 0, w, h]
    sizes = ndi.sum(bright, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(ndi.binary_fill_holes(lab == k))
    x0, x1, y0, y1 = xs.min() / f, (xs.max() + 1) / f, ys.min() / f, (ys.max() + 1) / f
    if (x1 - x0) * (y1 - y0) < 0.3 * w * h:
        return [0, 0, w, h]
    dx, dy = inset * (x1 - x0), inset * (y1 - y0)
    return [int(max(0, x0 + dx)), int(max(0, y0 + dy)), int(min(w, x1 - dx)), int(min(h, y1 - dy))]


def load_and_prepare(item_dir: str, leaf: int, angle: float | None = None):
    """Load a page, cut it to the sheet of paper (paper_region), normalize its background and deskew it.
    Returns (norm [0..1] float32, binary, angle, threshold); norm is white outside the paper."""
    gray = np.asarray(load_page(item_dir, leaf), dtype=np.uint8)
    # raw captures are already cut to the page with IA's crop box; processed derivatives still show the
    # cradle / facing page, so find the sheet of paper
    x0, y0, x1, y1 = [0, 0, gray.shape[1], gray.shape[0]] if is_raw_capture(item_dir) else paper_region(gray)
    sheet = gray[y0:y1, x0:x1]
    norm = np.ones(gray.shape, np.float32)
    norm[y0:y1, x0:x1] = normalize_background(sheet)
    binary = np.zeros(gray.shape, bool)
    binary[y0:y1, x0:x1], thr = binarize(norm[y0:y1, x0:x1])
    if angle is None:
        hm, vm = find_rules(binary)          # long rules can have a slightly different angle from the text
        angle = estimate_skew(binary & ~hm & ~vm)
    if abs(angle) >= 0.05:
        im = Image.fromarray((norm * 255).astype(np.uint8)).rotate(angle, resample=Image.BICUBIC, fillcolor=255)
        norm = np.asarray(im, dtype=np.float32) / 255.0
        binary = binarize(norm, thr)[0]
    return norm, binary, angle, thr


# ------------------------------------------------------------------------------------------------ furniture


def find_rules(binary: np.ndarray, frac: float = 0.2):
    """Masks of long horizontal and vertical straight lines (length >= frac x page side)."""
    b = binary.astype(np.uint8)
    h, w = b.shape
    out = []
    for horizontal in (True, False):
        L = int(frac * (w if horizontal else h))
        thick = cv2.dilate(b, np.ones((5, 1) if horizontal else (1, 5), np.uint8))  # tolerate slight tilt
        k = np.ones((1, L) if horizontal else (L, 1), np.uint8)
        m = cv2.morphologyEx(thick, cv2.MORPH_OPEN, k)
        m = cv2.dilate(m, np.ones((3, 3), np.uint8)) & b
        out.append(m.astype(bool))
    return out[0], out[1]


def _boxes(mask: np.ndarray):
    """Bounding boxes [x0, y0, x1, y1] of the connected pieces of a mask."""
    lab, _ = ndi.label(mask)
    return [[sl[1].start, sl[0].start, sl[1].stop, sl[0].stop] for sl in ndi.find_objects(lab) if sl is not None]


@dataclass
class Comp:
    id: int
    bbox: list          # x0, y0, x1, y1 (exclusive)
    area: int
    cx: float
    cy: float
    role: str = "text"  # text | frame | outside | border | huge | rule | header | separator | brace | stray

    @property
    def w(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def h(self):
        return self.bbox[3] - self.bbox[1]


def components(binary: np.ndarray, min_area: int):
    """8-connected components with at least `min_area` pixels -> (label image, [Comp])."""
    lab, n = ndi.label(binary, structure=np.ones((3, 3)))
    objs = ndi.find_objects(lab)
    areas = np.bincount(lab.ravel())
    comps = []
    for k, sl in enumerate(objs, 1):
        if sl is None or areas[k] < min_area:
            continue
        ys, xs = sl
        yy, xx = np.nonzero(lab[sl] == k)
        comps.append(Comp(k, [xs.start, ys.start, xs.stop, ys.stop], int(areas[k]),
                          float(xx.mean() + xs.start), float(yy.mean() + ys.start)))
    return lab, comps


def mark_frame(comps: list[Comp], page_w: int, page_h: int, h_ref: float, sw: float, word_area: float):
    """Dotted/dashed page frame: >= 6 small components (dots/dashes, much smaller than a word) whose centres lie on
    one column (or one row) within a few stroke widths, evenly spaced and alike, spread over >= 30% of the page
    side, with (almost) no word-sized ink reaching across. Members and any small component near the fitted frame
    line become role='frame'; components outside the frame rectangle become role='outside'."""
    def small_of(c, f=1.0):
        """dots and short dashes (a dash may be as long as a word is tall, but it is thin)"""
        thin = min(c.w, c.h) <= 0.35 * max(c.w, c.h)
        return c.role == "text" and c.area < 0.25 * f * word_area and \
            max(c.w, c.h) < (1.6 if thin else 0.9) * f * h_ref

    small = [c for c in comps if small_of(c)]
    words = [c for c in comps if c.role == "text" and c.area >= 0.25 * word_area]
    found = []
    tol = max(2.5 * sw, 0.12 * h_ref)
    for axis in ("x", "y"):
        key = (lambda c: c.cx) if axis == "x" else (lambda c: c.cy)
        other = (lambda c: c.cy) if axis == "x" else (lambda c: c.cx)
        span_needed = 0.3 * (page_h if axis == "x" else page_w)
        cands = sorted(small, key=key)
        used: set[int] = set()
        for c in cands:
            if c.id in used:
                continue
            group = [d for d in cands if abs(key(d) - key(c)) <= tol and d.id not in used]
            if len(group) < 6:
                continue
            o = np.array(sorted(other(d) for d in group))
            if o[-1] - o[0] < span_needed:
                continue
            sp = np.diff(o)
            sizes = np.array([d.area for d in group], float)
            if np.std(sp) > 0.6 * np.mean(sp) or np.std(sizes) > 0.8 * np.mean(sizes):
                continue
            pos = float(np.median([key(d) for d in group]))
            lo, hi = float(o[0]), float(o[-1])
            near = [w for w in words if lo <= other(w) <= hi]
            if near:
                i0, i1 = (0, 2) if axis == "x" else (1, 3)
                before = np.mean([w.bbox[i0] < pos for w in near])
                after = np.mean([w.bbox[i1] > pos for w in near])
                if min(before, after) > 0.02:
                    continue
            # fit the frame line (it may be slightly tilted) and absorb every small component close to it
            if len(group) >= 3:
                b, a = np.polyfit([other(d) for d in group], [key(d) for d in group], 1)
            else:
                a, b = pos, 0.0
            ext = 0.1 * (hi - lo)
            for d in comps:
                if small_of(d, 1.5) and lo - ext <= other(d) <= hi + ext and abs(key(d) - (a + b * other(d))) <= 2 * tol:
                    d.role = "frame"
                    used.add(d.id)
            found.append({"axis": axis, "pos": round(pos, 1), "n": len(group), "extent": [round(lo), round(hi)]})
    # the frame rectangle: innermost frame line on each side of the page centre
    xs = [f["pos"] for f in found if f["axis"] == "x"]
    ys = [f["pos"] for f in found if f["axis"] == "y"]
    left = max([x for x in xs if x < 0.5 * page_w], default=None)
    right = min([x for x in xs if x > 0.5 * page_w], default=None)
    top = max([y for y in ys if y < 0.5 * page_h], default=None)
    bottom = min([y for y in ys if y > 0.5 * page_h], default=None)
    for c in comps:
        if c.role != "text":
            continue
        if (left is not None and c.cx < left) or (right is not None and c.cx > right) or \
                (top is not None and c.cy < top) or (bottom is not None and c.cy > bottom):
            c.role = "outside"
    return found, {"left": left, "right": right, "top": top, "bottom": bottom}


def mark_separators(comps: list[Comp], columns, h_med: float) -> list[list]:
    """Stanza separators such as '~~~': a cluster of flat pieces (one or a few components at the same height,
    together >= 1.2 h_med wide, <= 0.5 h_med tall), centred on the text of its column, with no other ink at the
    same height within 1.5 h_med. Marks the members role='separator'; returns their union boxes."""
    text = [c for c in comps if c.role == "text"]
    flat = [c for c in text if c.h <= 0.5 * h_med and c.w >= 0.3 * h_med and c.w >= 2 * c.h]
    seen: set[int] = set()
    seps = []
    for c in sorted(flat, key=lambda c: c.bbox[0]):
        if c.id in seen:
            continue
        cl = [c]
        grew = True
        while grew:
            grew = False
            x0, x1 = min(d.bbox[0] for d in cl), max(d.bbox[2] for d in cl)
            ycen = np.mean([d.cy for d in cl])
            for d in flat:
                if d not in cl and abs(d.cy - ycen) < 0.3 * h_med and d.bbox[0] < x1 + 0.5 * h_med \
                        and d.bbox[2] > x0 - 0.5 * h_med:
                    cl.append(d)
                    grew = True
        seen.update(d.id for d in cl)
        box = union([d.bbox for d in cl])
        w, h = box[2] - box[0], box[3] - box[1]
        if not (h <= 0.5 * h_med and w >= 1.2 * h_med and w >= 3 * h):
            continue
        cx, cy = 0.5 * (box[0] + box[2]), float(np.mean([d.cy for d in cl]))
        col = next(((x0, x1) for x0, x1 in columns if x0 <= cx < x1), None)
        if col is None:
            continue
        mid = [d.cx for d in text if col[0] <= d.cx < col[1]]
        tx0, tx1 = (min(mid), max(mid)) if mid else col
        if abs(cx - 0.5 * (tx0 + tx1)) > 0.15 * (tx1 - tx0):
            continue
        ids = {d.id for d in cl}
        neighbours = [d for d in text if d.id not in ids and abs(d.cy - cy) < 0.3 * h_med
                      and d.bbox[0] < box[2] + 1.5 * h_med and d.bbox[2] > box[0] - 1.5 * h_med]
        if neighbours:
            continue
        for d in cl:
            d.role = "separator"
        seps.append(box)
    return seps


# --------------------------------------------------------------------------------------------------- lines


@dataclass
class Word:
    bbox: list
    comps: list                                         # components forming the word (with `marks`)
    marks: list = field(default_factory=list)          # small components inside the word's span
    outside_marks: list = field(default_factory=list)  # small components before/after it (punctuation, cross)
    kind: str = "word"                                  # word | roman | punct
    gap_before: float = -1.0                            # nearest-ink distance to the previous word (-1: first)


@dataclass
class Line:
    number: int
    column: int
    kind: str           # text | separator | pagenum | heading
    bbox: list
    fit: list           # y = fit[0] + fit[1] * x  (line centre)
    words: list = field(default_factory=list)


def _profile_peaks(comps: list[Comp], lab: np.ndarray, h_med: float):
    """Line centres of one column from the horizontal ink profile of word-sized components."""
    H = lab.shape[0]
    prof = np.zeros(H, np.float64)
    for c in comps:
        bx0, by0, bx1, by1 = c.bbox
        prof[by0:by1] += (lab[by0:by1, bx0:bx1] == c.id).sum(1)
    if prof.max() <= 0:
        return [], 0.0
    sm = ndi.gaussian_filter1d(prof, max(2.0, 0.2 * h_med))
    # line pitch: the strongest local maximum of the profile's autocorrelation between 0.5 and 5 h_med
    z = sm - sm.mean()
    ac = np.correlate(z, z, mode="full")[len(z) - 1:]
    lo, hi = int(0.5 * h_med), int(min(len(ac) - 1, 5 * h_med))
    pk, _ = find_peaks(ac[lo:hi])
    pitch = float(lo + pk[np.argmax(ac[lo:hi][pk])]) if len(pk) else 1.5 * h_med
    peaks, _ = find_peaks(sm, distance=max(3, int(0.55 * pitch)), prominence=0.04 * sm.max())
    return [float(p) for p in peaks], pitch


def _fit_line(cs: list[Comp], max_slope: float = 0.02):
    """Weighted least squares y = a + b x through component centres (weights = ink area), slope clipped."""
    xs = np.array([c.cx for c in cs])
    ys = np.array([c.cy for c in cs])
    wts = np.array([c.area for c in cs], float)
    if len(cs) < 3 or xs.max() - xs.min() < 1:
        return [float(np.average(ys, weights=wts)), 0.0]
    A = np.stack([np.ones_like(xs), xs], 1) * np.sqrt(wts)[:, None]
    sol, *_ = np.linalg.lstsq(A, ys * np.sqrt(wts), rcond=None)
    b = float(np.clip(sol[1], -max_slope, max_slope))
    a = float(np.average(ys - b * xs, weights=wts))
    return [a, b]


def assign_lines(body: list[Comp], centres: list[float], iters: int = 3):
    """Assign components to lines (nearest centre), refit each line, reassign. Returns list of (fit, comps).

    After the free fits, every line of the column gets the median slope of the lines with >= 4 components (the
    page is already deskewed, so lines share one residual slope; single-line fits are noisy because of ascending
    and descending strokes) and its intercept is refitted."""
    fits = [[c, 0.0] for c in centres]

    def assign(fits):
        groups = [[] for _ in fits]
        for c in body:
            d = [abs(c.cy - (a + b * c.cx)) for a, b in fits]
            groups[int(np.argmin(d))].append(c)
        return groups

    for _ in range(iters):
        groups = assign(fits)
        fits = [_fit_line(g) for g in groups if g]
        if not fits:
            return []
        groups = [g for g in groups if g]
        slopes = [f[1] for f, g in zip(fits, groups) if len(g) >= 4]
        if slopes:
            b = float(np.median(slopes))
            fits = [[float(np.average([c.cy - b * c.cx for c in g], weights=[c.area for c in g])), b]
                    for g in groups]
    groups = assign(fits)
    return [(f, g) for f, g in zip(fits, groups) if g]


# --------------------------------------------------------------------------------------------------- words


def boundary_points(lab: np.ndarray, c: Comp) -> np.ndarray:
    """(x, y) coordinates of the outline pixels of one component."""
    x0, y0, x1, y1 = c.bbox
    m = lab[y0:y1, x0:x1] == c.id
    yy, xx = np.nonzero(m & ~ndi.binary_erosion(m))
    return np.stack([xx + x0, yy + y0], 1).astype(np.float32)


def ink_distances(cs: list[Comp], pts: dict) -> np.ndarray:
    """Matrix of nearest-ink distances (pixels) between the components of one line."""
    n = len(cs)
    D = np.zeros((n, n), np.float32)
    trees = [cKDTree(pts[c.id]) for c in cs]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = (i, j) if len(pts[cs[i].id]) <= len(pts[cs[j].id]) else (j, i)
            d = float(trees[b].query(pts[cs[a].id], k=1)[0].min())
            D[i, j] = D[j, i] = d
    return D


def chain_gaps(cs: list[Comp], pts: dict) -> list[float]:
    """For each component after the first (in x0 order): nearest-ink distance to any component left of it.
    Nearest-ink distance, unlike a bounding-box gap, is not fooled by slanted strokes that overhang a neighbour."""
    cs = sorted(cs, key=lambda c: c.bbox[0])
    D = ink_distances(cs, pts)
    return [float(D[j, :j].min()) for j in range(1, len(cs))]


def gap_threshold(all_gaps: list[float], h_med: float, bandwidth: float = 0.15):
    """Within-word / between-word gap threshold from the page's own gap distribution.

    Gaps are nearest-ink distances between neighbouring components of a line. Their log-density (Gaussian KDE,
    `bandwidth` in log units) is usually bimodal: pen lifts and marks inside words, and the spaces between words.
    The threshold is the density minimum between the two strongest modes that lie at least a factor 2 apart. Otsu's
    threshold on log(gap) is reported for comparison and used when no such pair of modes exists; 0.3 x h_med is
    the last resort (fewer than 8 gaps). Returns (threshold_px, stats)."""
    g = np.array([x for x in all_gaps if x >= 1], float)
    stats = {"n_gaps": int(len(g))}
    if len(g) < 8:
        stats["method"] = "fallback (few gaps)"
        return 0.3 * h_med, stats
    lg = np.log(g)
    otsu = float(np.exp(threshold_otsu(lg)))
    grid = np.linspace(lg.min() - 0.5, lg.max() + 0.5, 400)
    dens = np.exp(-0.5 * ((grid[:, None] - lg[None, :]) / bandwidth) ** 2).sum(1)
    pk, _ = find_peaks(dens)
    best = None
    for i in pk:
        for j in pk:
            if grid[j] - grid[i] < np.log(2.0):
                continue
            v = i + int(np.argmin(dens[i:j + 1]))
            depth = min(dens[i], dens[j]) - dens[v]
            if best is None or depth > best[0]:
                best = (depth, i, j, v)
    stats.update({"otsu_px": round(otsu, 1), "otsu_rel_h": round(otsu / h_med, 3)})
    if best is None or best[0] <= 0:
        stats["method"] = "otsu (no two modes)"
        return otsu, stats
    _, i, j, v = best
    thr = float(np.exp(grid[v]))
    stats.update({"method": "kde valley", "mode_small_px": round(float(np.exp(grid[i])), 1),
                  "mode_large_px": round(float(np.exp(grid[j])), 1), "valley_px": round(thr, 1),
                  "valley_rel_h": round(thr / h_med, 3), "valley_depth_rel": round(best[0] / dens.max(), 3),
                  "n_below": int((g <= thr).sum()), "n_above": int((g > thr).sum())})
    return thr, stats


def split_chain(cs: list[Comp], thr: float, pts: dict) -> list[tuple[list[Comp], float]]:
    """Walk the components in x0 order; a component farther than `thr` (nearest ink) from every component of the
    current word starts a new word. Returns (components, distance to the previous word) per word."""
    cs = sorted(cs, key=lambda c: c.bbox[0])
    D = ink_distances(cs, pts)
    groups, cur, gap_before = [], [0], [-1.0]
    for j in range(1, len(cs)):
        d = float(D[j, cur].min())
        if d > thr:
            groups.append(cur)
            cur = []
            gap_before.append(d)
        cur.append(j)
    groups.append(cur)
    return [([cs[i] for i in g], gb) for g, gb in zip(groups, gap_before)]


def has_hole(lab: np.ndarray, c: Comp, min_hole: int = 4) -> bool:
    """True if the component encloses background (a closed loop such as a vowel circle)."""
    x0, y0, x1, y1 = c.bbox
    m = lab[y0:y1, x0:x1] == c.id
    return int((ndi.binary_fill_holes(m) & ~m).sum()) >= min_hole


def is_ring(lab: np.ndarray, c: Comp, h_med: float, min_hole: int = 4) -> bool:
    """A small vowel circle, possibly not quite closed: encloses background, or is round, not tiny
    (>= 0.25 h_med) and hollow (ink covers < 55% of its box), unlike a dot or a comma."""
    if has_hole(lab, c, min_hole):
        return True
    return max(c.w, c.h) >= 0.25 * h_med and 0.75 < c.w / max(1, c.h) < 1.33 and c.area < 0.55 * c.w * c.h


def is_cross(lab: np.ndarray, c: Comp) -> bool:
    """A free-standing 'x' / '+' sign (sign of the cross in prayer books): skeleton with four end points and a
    branch point."""
    x0, y0, x1, y1 = c.bbox
    m = np.pad(lab[y0:y1, x0:x1] == c.id, 1)
    sk = skeletonize(m)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    ends = int(((nb == 1) & sk).sum())
    branch = int(((nb >= 3) & sk).sum())
    return ends == 4 and branch >= 1 and 0.5 < c.w / max(1, c.h) < 2.0


def roman_runs(cs: list[Comp], h_med: float) -> set[int]:
    """Ids of components in runs of >= 6 small, narrow, closely spaced glyph-like components standing on one
    baseline (typeset Roman or italic text: most letters end at the same height). Duployan words are mostly one
    to three joined components, and their pieces do not share a baseline."""
    cs = sorted(cs, key=lambda c: c.bbox[0])
    out, run = set(), []

    def flush():
        if len(run) >= 6:
            bottoms = np.array([c.bbox[3] for c in run], float)
            if np.mean(np.abs(bottoms - np.median(bottoms)) <= 0.12 * h_med) >= 0.7:
                out.update(c.id for c in run)

    for c in cs:
        glyph = c.h <= 1.2 * h_med and c.w <= 0.9 * h_med
        if glyph and run and c.bbox[0] - max(r.bbox[2] for r in run) <= 0.25 * h_med:
            run.append(c)
        elif glyph:
            flush()
            run = [c]
        else:
            flush()
            run = []
    flush()
    return out


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _nearest(pts_a: np.ndarray, tree) -> float:
    return float(tree.query(pts_a, k=1)[0].min())


def build_words(cs: list[Comp], is_mark, thr: float, h_med: float, pts: dict, lab: np.ndarray,
                min_hole: int = 4) -> list[Word]:
    """Words of one line. Word-sized components are chained by nearest-ink distance (marks never bridge two
    words); each mark then joins the nearest word if it is within `thr` of it -- inside the word image when it
    lies within the word's horizontal span or is a small closed loop (a vowel circle), otherwise as an outside
    mark (punctuation). Marks farther than `thr` from every word form their own groups: a short word when they
    contain a closed loop, else 'punct'."""
    body_all = [c for c in cs if not is_mark(c)]
    marks_all = [c for c in cs if is_mark(c)]
    roman = roman_runs(body_all, h_med)
    groups = split_chain(body_all, thr, pts) if body_all else []
    words: list[Word] = []
    for grp, gap in groups:
        words.append(Word([], [c.id for c in grp], [], [], "word", gap))
    cmap = {c.id: c for c in cs}
    trees = [cKDTree(np.concatenate([pts[i] for i in w.comps])) for w in words]
    loose = []
    for m in marks_all:
        if words:
            d = [_nearest(pts[m.id], t) for t in trees]
            k = int(np.argmin(d))
        if not words or d[k] > thr:
            loose.append(m)
            continue
        wd = words[k]
        bx0 = min(cmap[i].bbox[0] for i in wd.comps)
        bx1 = max(cmap[i].bbox[2] for i in wd.comps)
        # inside the word's span, or a small vowel circle just before/after it -> part of the word
        if bx0 - 0.1 * h_med <= m.cx <= bx1 + 0.1 * h_med or is_ring(lab, m, h_med, min_hole):
            wd.marks.append(m.id)
        else:
            wd.outside_marks.append(m.id)
    for grp, _ in (split_chain(loose, thr, pts) if loose else []):
        box = union([c.bbox for c in grp])
        gp = min([_nearest(np.concatenate([pts[c.id] for c in grp]), t) for t in trees], default=-1.0)
        # punctuation (. , : ; ! ? - =) has no closed loop; a lone small circle is a word (a vowel, e.g. vocative O)
        if any(is_ring(lab, c, h_med, min_hole) for c in grp):
            words.append(Word([], [c.id for c in grp], [], [], "word", gp))
        else:
            words.append(Word(box, [], [], [c.id for c in grp], "punct", gp))
    for wd in words:
        if wd.kind == "punct":
            continue
        body = [cmap[i] for i in wd.comps]
        # narrow punctuation made of one stroke: ';' '!' (stroke + dot) or a lone comma; the narrowest real words
        # (e.g. 'pe', an upright stroke with a hook) are wider and carry no dot
        narrow = len(body) == 1 and body[0].w <= 0.34 * h_med and not is_ring(lab, body[0], h_med, min_hole)
        dotted = any(max(cmap[i].w, cmap[i].h) <= 0.25 * h_med for i in wd.marks)
        if narrow and (dotted or (not wd.marks and body[0].h <= 0.5 * h_med)):
            wd.kind, wd.outside_marks, wd.marks, wd.comps = "punct", wd.comps + wd.marks + wd.outside_marks, [], []
            wd.bbox = union([cmap[i].bbox for i in wd.outside_marks])
            continue
        if len(body) == 1 and not wd.marks and max(body[0].w, body[0].h) < 0.8 * h_med and is_cross(lab, body[0]):
            wd.kind = "punct"
            wd.outside_marks = wd.comps + wd.outside_marks
            wd.comps = []
            wd.bbox = list(body[0].bbox)
            continue
        if roman and sum(c.id in roman for c in body) >= 0.5 * len(body):
            wd.kind = "roman"
        wd.bbox = union([cmap[i].bbox for i in wd.comps + wd.marks])
    words.sort(key=lambda w: w.bbox[0])
    return words


# ------------------------------------------------------------------------------------------------- the page


def segment_page(norm: np.ndarray, binary: np.ndarray, *, verbose: bool = False) -> dict:
    """Segment a prepared page (see load_and_prepare). Returns {stats, lines, components, _labels}."""
    H, W = binary.shape
    # --- rules
    hmask, vmask = find_rules(binary)
    hrules = [b for b in _boxes(hmask) if b[2] - b[0] > 0.2 * W]
    vrules = [b for b in _boxes(vmask) if b[3] - b[1] > 0.2 * H]
    work = binary & ~hmask & ~vmask
    # --- components and size statistics
    sw = stroke_width(work[H // 4: 3 * H // 4, W // 4: 3 * W // 4])
    min_area = max(4, int(round(0.5 * sw * sw)))
    lab, comps = components(work, min_area)
    for c in comps:
        x0, y0, x1, y1 = c.bbox
        if x0 <= 2 or y0 <= 2 or x1 >= W - 2 or y1 >= H - 2:
            c.role = "border"
    areas = np.array([c.area for c in comps if c.role == "text"])
    big_area = np.percentile(areas, 60) if len(areas) else 1
    sized = [c for c in comps if c.role == "text" and c.area >= big_area]
    h_med = float(np.median([c.h for c in sized])) if sized else 20.0
    word_area = float(np.median([c.area for c in sized])) if sized else 100.0
    for c in comps:
        if c.role == "text" and (c.h > 0.25 * H or c.w > 0.5 * W):
            c.role = "huge"
        elif c.role == "text" and c.h <= 2.5 * sw and c.w >= 3 * h_med:
            c.role = "rule"
    frames, frame_box = mark_frame(comps, W, H, h_med, sw, word_area)
    # --- columns (vertical rules away from the page edges split the page) and header / footer zones
    cuts = sorted(0.5 * (b[0] + b[2]) for b in vrules if 0.15 * W < 0.5 * (b[0] + b[2]) < 0.85 * W)
    col_edges = [0.0] + cuts + [float(W)]
    columns = [(col_edges[i], col_edges[i + 1]) for i in range(len(col_edges) - 1)]
    header_y = max([float(b[3]) for b in hrules if b[3] < 0.15 * H], default=0.0)
    footer_y = min([float(b[1]) for b in hrules if b[1] > 0.85 * H], default=float(H))
    for c in comps:
        if c.role == "text" and ((header_y > 0 and c.cy < header_y + 0.3 * h_med) or
                                 (footer_y < H and c.cy > footer_y - 0.3 * h_med)):
            c.role = "header"
    mark_area = 0.08 * word_area

    def is_mark(c: Comp) -> bool:
        return c.area < mark_area or max(c.w, c.h) <= 0.35 * h_med

    seps = mark_separators(comps, columns, h_med)
    text = [c for c in comps if c.role == "text"]
    # --- lines per column
    lines_raw = []      # (column, fit, comps, pitch)
    pitch_by_col = []
    for ci, (cx0, cx1) in enumerate(columns):
        col = [c for c in text if cx0 <= c.cx < cx1]
        body = [c for c in col if not is_mark(c)]
        if not body:
            pitch_by_col.append(0.0)
            continue
        centres, pitch = _profile_peaks(body, lab, h_med)
        pitch_by_col.append(pitch)
        for c in body:
            if c.h > 1.1 * pitch and c.w < 0.5 * c.h:
                c.role = "brace"
        body = [c for c in body if c.role == "text"]
        if not centres or not body:
            continue
        for fit, g in assign_lines(body, centres):
            lines_raw.append([ci, fit, g, pitch])
    # --- marks to lines
    for m in [c for c in text if is_mark(c) and c.role == "text"]:
        best, bd = None, None
        for k, (ci, fit, g, pitch) in enumerate(lines_raw):
            cx0, cx1 = columns[ci]
            if not (cx0 <= m.cx < cx1):
                continue
            d = abs(m.cy - (fit[0] + fit[1] * m.cx))
            if bd is None or d < bd:
                best, bd = k, d
        if best is not None and bd < 0.75 * lines_raw[best][3]:
            lines_raw[best][2].append(m)
        else:
            m.role = "stray"
    # --- word-gap threshold from all text lines of the page
    pts = {c.id: boundary_points(lab, c) for ln in lines_raw for c in ln[2]}
    all_gaps = []
    for ci, fit, g, pitch in lines_raw:
        body = [c for c in g if not is_mark(c)]
        if len(body) >= 2:
            all_gaps += chain_gaps(body, pts)
    thr, gap_stats = gap_threshold(all_gaps, h_med)
    # --- build lines and words; separators become their own lines
    cmap = {c.id: c for c in comps}
    lines: list[Line] = []
    for ci, fit, g, pitch in lines_raw:
        words = build_words(g, is_mark, thr, h_med, pts, lab, max(4, int(0.5 * sw * sw)))
        allb = [w.bbox for w in words] + [cmap[i].bbox for w in words for i in w.outside_marks]
        lines.append(Line(0, ci, "text", union(allb), [round(fit[0], 2), round(fit[1], 5)], words))
    for box in seps:
        ci = next(i for i, (x0, x1) in enumerate(columns) if x0 <= 0.5 * (box[0] + box[2]) < x1)
        lines.append(Line(0, ci, "separator", list(box), [round(0.5 * (box[1] + box[3]), 2), 0.0], []))
    centre_x = {ci: 0.5 * (x0 + x1) for ci, (x0, x1) in enumerate(columns)}
    lines.sort(key=lambda l: (l.column, l.fit[0] + l.fit[1] * centre_x[l.column]))
    _classify_lines(lines, columns)
    k = 0
    for ln in lines:
        if ln.kind == "text":
            k += 1
            ln.number = k
    roles: dict[str, int] = {}
    for c in comps:
        roles[c.role] = roles.get(c.role, 0) + 1
    stats = {"page_size": [W, H], "stroke_width": round(sw, 2), "h_med": round(h_med, 1),
             "word_area": round(word_area, 1), "min_area": min_area, "mark_area": round(mark_area, 1),
             "pitch_by_column": [round(p, 1) for p in pitch_by_col], "word_gap_threshold_px": round(thr, 1),
             "gap_stats": gap_stats, "columns": columns, "header_y": header_y, "footer_y": footer_y,
             "hrules": hrules, "vrules": vrules, "frames": frames, "frame_box": frame_box,
             "component_roles": roles}
    if verbose:
        print(json.dumps({k: stats[k] for k in ("h_med", "stroke_width", "pitch_by_column", "word_gap_threshold_px",
                                                 "gap_stats", "frames", "component_roles")}))
    comp_info = {c.id: {"bbox": c.bbox, "area": c.area, "role": c.role} for c in comps}
    return {"stats": stats, "lines": [asdict(l) for l in lines], "components": comp_info, "_labels": lab}


def _classify_lines(lines: list[Line], columns):
    """Label page-number and heading lines (see module docstring). `lines` is in reading order."""
    for ci in sorted({l.column for l in lines}):
        col = [l for l in lines if l.column == ci]
        texts = [l for l in col if l.kind == "text"]
        if texts:
            tx0 = min(l.bbox[0] for l in texts)
            tx1 = max(l.bbox[2] for l in texts)
            tw = max(1.0, tx1 - tx0)
            for l in {id(texts[0]): texts[0], id(texts[-1]): texts[-1]}.values():
                x0, y0, x1, y1 = l.bbox
                centred = abs(0.5 * (x0 + x1) - 0.5 * (tx0 + tx1)) < 0.12 * tw
                if centred and (x1 - x0) < 0.2 * tw and len(l.words) <= 3:
                    l.kind = "pagenum"
        for i, l in enumerate(col):
            if l.kind != "text":
                continue
            nxt = col[i + 1] if i + 1 < len(col) else None
            prv = col[i - 1] if i > 0 else None
            if nxt is not None and nxt.kind == "separator" and (prv is None or prv.kind in ("separator", "pagenum")):
                l.kind = "heading"


# ------------------------------------------------------------------------------------------------- drawing


def _font(size: int):
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "FreeSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def draw_overlay(norm: np.ndarray, seg: dict, path: str, max_side: int = 2400):
    """Page with word boxes (red: words, orange: Roman-text runs, blue: punctuation and outside marks), text-line
    numbers (green) left of each line, grey boxes around dropped furniture and non-text lines."""
    im = Image.fromarray((norm * 255).astype(np.uint8)).convert("RGB")
    f = min(1.0, max_side / max(im.size))
    if f < 1:
        im = im.resize((int(im.width * f), int(im.height * f)), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    font = _font(max(14, int(30 * f)))
    small = _font(max(10, int(20 * f)))
    comps = seg["components"]

    def S(b):
        return [int(v * f) for v in b]

    for c in comps.values():
        if c["role"] not in ("text",):
            d.rectangle(S(c["bbox"]), outline=(160, 160, 160), width=1)
    for ln in seg["lines"]:
        if ln["kind"] != "text":
            d.rectangle(S(ln["bbox"]), outline=(110, 110, 110), width=2)
            d.text((int(ln["bbox"][2] * f) + 4, int(ln["bbox"][1] * f)), ln["kind"], fill=(90, 90, 90), font=small)
            continue
        for wd in ln["words"]:
            col = {"word": (220, 0, 0), "roman": (240, 140, 0), "punct": (0, 90, 255)}[wd["kind"]]
            d.rectangle(S(wd["bbox"]), outline=col, width=2)
            for mid in wd["outside_marks"]:
                d.rectangle(S(comps[mid]["bbox"]), outline=(0, 90, 255), width=1)
        x0, y0, x1, y1 = ln["bbox"]
        yc = ln["fit"][0] + ln["fit"][1] * x0
        d.text((max(0, int(x0 * f) - int(44 * f) - 10), int(yc * f) - 12), str(ln["number"]), fill=(0, 140, 0),
               font=font)
    im.save(path)


def page_stem(item_dir: str, leaf: int) -> str:
    return f"{os.path.basename(os.path.normpath(item_dir))}_{leaf}"


def run(item_dir: str, leaf: int, out_dir: str, verbose: bool = False) -> tuple[dict, np.ndarray]:
    """Segment one page, write <stem>_seg.json and <stem>_overlay.png. Returns (seg, normalized page)."""
    norm, binary, angle, thr = load_and_prepare(item_dir, leaf)
    seg = segment_page(norm, binary, verbose=verbose)
    seg["stats"].update({"item": os.path.basename(os.path.normpath(item_dir)), "leaf": leaf,
                         "deskew_deg": round(angle, 2), "binarize_threshold": round(thr, 3)})
    os.makedirs(out_dir, exist_ok=True)
    stem = page_stem(item_dir, leaf)
    draw_overlay(norm, seg, os.path.join(out_dir, f"{stem}_overlay.png"))
    js = {k: v for k, v in seg.items() if not k.startswith("_")}
    with open(os.path.join(out_dir, f"{stem}_seg.json"), "w", encoding="utf-8") as fh:
        json.dump(js, fh, indent=1)
    return seg, norm


def main():
    ap = argparse.ArgumentParser(description="Segment a page of running Chinuk Pipa text into lines and words.")
    ap.add_argument("item_dir", help="local Internet Archive item folder (contains <id>_orig_jp2.tar or _jp2.zip)")
    ap.add_argument("leaf", type=int, help="leaf index (0-based)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    seg, _ = run(a.item_dir, a.leaf, a.out_dir, a.verbose)
    text = [l for l in seg["lines"] if l["kind"] == "text"]
    kinds: dict[str, int] = {}
    for l in text:
        for w in l["words"]:
            kinds[w["kind"]] = kinds.get(w["kind"], 0) + 1
    other = {}
    for l in seg["lines"]:
        if l["kind"] != "text":
            other[l["kind"]] = other.get(l["kind"], 0) + 1
    print(f"{page_stem(a.item_dir, a.leaf)}: {len(text)} text lines, boxes {kinds}, other lines {other}, "
          f"deskew {seg['stats']['deskew_deg']} deg")


if __name__ == "__main__":
    main()
