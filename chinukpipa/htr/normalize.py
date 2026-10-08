"""Scale-normalize word images by stroke width, so that synthetic and scanned words look alike to the model.

A word image is binarized (Otsu), its median stroke width is estimated from the distance transform sampled on
the skeleton, and it is rescaled so that strokes are `target_sw` pixels wide. It is then placed, vertically
centred, on a white canvas of height `height` (shrunk further only if it would not fit).
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize


def stroke_width(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    if g.max() - g.min() < 10:
        return 0.0
    ink = g < threshold_otsu(g)
    if ink.sum() < 10:
        return 0.0
    dist = ndi.distance_transform_edt(ink)
    sk = skeletonize(ink)
    vals = dist[sk]
    return float(2 * np.median(vals)) if len(vals) else 0.0


def stretch_contrast(im: Image.Image) -> Image.Image:
    """Map the ink level to black and the paper level to white (scans differ a lot in paper tone and ink density;
    synthetic words are black on white)."""
    g = np.asarray(im.convert("L"), dtype=np.float32)
    if g.max() - g.min() < 10:
        return im.convert("L")
    t = threshold_otsu(g)
    ink, paper = np.median(g[g <= t]), np.median(g[g > t])
    if paper - ink < 10:
        return im.convert("L")
    out = np.clip((g - ink) / (paper - ink) * 255.0, 0, 255)
    return Image.fromarray(out.astype(np.uint8))


def normalize(im: Image.Image, *, height: int = 64, target_sw: float = 3.0, max_w: int = 640,
              sw: float | None = None, scale: float = 1.0) -> Image.Image:
    """`scale` multiplies the stroke-width scaling; use it to calibrate a source whose ratio of stroke length to
    pen width differs from the synthetic fonts (see htr/calibrate.py)."""
    g = np.asarray(im.convert("L"))
    sw = sw or stroke_width(g)
    scale = scale * (target_sw / sw if sw > 0.5 else height * 0.6 / max(1, g.shape[0]))
    w, h = max(1, round(g.shape[1] * scale)), max(1, round(g.shape[0] * scale))
    if h > height - 4:  # too tall: shrink to fit
        f = (height - 4) / h
        w, h = max(1, round(w * f)), height - 4
    w = min(w, max_w)
    small = im.convert("L").resize((w, h), Image.LANCZOS)
    canvas = Image.new("L", (max(w + 8, 16), height), 255)
    canvas.paste(small, (4, (height - h) // 2))
    return canvas


def rescale(a: np.ndarray, mult: float, height: int = 64, max_w: int = 640) -> np.ndarray:
    """Rescale an already-normalized word (uint8, white background) by `mult`, re-centred on a `height` canvas.
    Used for scale augmentation; words that would no longer fit are shrunk to fit."""
    ink = a < 128
    if not ink.any() or abs(mult - 1) < 1e-3:
        return a
    ys, xs = np.nonzero(ink)
    y0, y1, x0, x1 = max(0, ys.min() - 1), ys.max() + 2, max(0, xs.min() - 1), xs.max() + 2
    crop = Image.fromarray(a[y0:y1, x0:x1])
    w, h = max(1, round(crop.width * mult)), max(1, round(crop.height * mult))
    if h > height - 4:
        f = (height - 4) / h
        w, h = max(1, round(w * f)), height - 4
    w = min(w, max_w)
    small = crop.resize((w, h), Image.BILINEAR)
    canvas = Image.new("L", (max(w + 8, 16), height), 255)
    canvas.paste(small, (4, (height - h) // 2))
    return np.asarray(canvas)
