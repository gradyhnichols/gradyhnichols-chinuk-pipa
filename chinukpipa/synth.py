"""Synthetic Chinuk Pipa word images for training a recognizer.

Renders token sequences with Duployan fonts (Pillow+Raqm when available, otherwise HarfBuzz+fontTools via
chinukpipa.render), then degrades them as a rough stand-in for scanned handwriting: uneven stroke width, slant,
small rotation, elastic wobble, blur, speckle and thresholding.

Words come from the A/B-confidence headwords of the merged word list, minus the held-out test words (see
chinukpipa/split.py), plus random syllable strings for coverage.

    python -m chinukpipa.synth OUT_DIR --n 20000 --fonts fonts/*.otf fonts/*.ttf

Writes OUT_DIR/images/NNNNNN.png (stroke-width normalized, height 64, white background; see htr/normalize.py)
and OUT_DIR/labels.tsv (file<TAB>space-separated tokens<TAB>source word, or ~random).
"""
from __future__ import annotations

import argparse
import glob
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features
from scipy import ndimage as ndi

from chinukpipa.htr.normalize import normalize
from chinukpipa.render import Renderer
from chinukpipa.split import split_of
from chinukpipa.translit import latin_to_tokens, load_tokens, tokens_to_unicode

RAQM = features.check("raqm")
_CONS = ["P", "B", "T", "D", "K", "G", "L", "R", "M", "N", "SH", "S", "CH", "TS", "TH", "KH", "HL", "LH", "H", "F", "V", "NG"]
_VOW = ["A", "O", "OO", "E", "U", "OW", "WA", "WE", "WO"]


def encodable(tokens: list[str]) -> bool:
    spec = load_tokens()
    return all(spec[t].get("unicode") for t in tokens)


def random_tokens(rng: random.Random) -> list[str]:
    """Random CV(C) syllable strings, for coverage of rare letter combinations."""
    out = []
    for _ in range(rng.randint(1, 4)):
        if rng.random() < 0.8:
            out.append(rng.choice(_CONS))
        out.append(rng.choice(_VOW))
        if rng.random() < 0.35:
            out.append(rng.choice(_CONS))
    return out


def lexicon_words(path: str) -> list[str]:
    words = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        hi, ci = header.index("headword"), header.index("best_conf")
        for line in f:
            cols = line.rstrip("\n").split("\t")
            w = cols[hi].strip()
            if w and " " not in w and w.isalpha() and cols[ci] in ("A", "B") and split_of(w) != "test":
                words.append(w)
    return words


_RENDERERS: dict[str, Renderer] = {}


def shapes_cleanly(text: str, font_path: str) -> bool:
    """False if the font has to insert a dotted circle (an invalid sign sequence) when shaping `text`."""
    try:
        if font_path not in _RENDERERS:
            _RENDERERS[font_path] = Renderer(font_path)
    except ImportError:  # uharfbuzz/fontTools missing: shaping check unavailable, accept
        return True
    return not _RENDERERS[font_path].has_dotted_circle(text)


def render(text: str, font: ImageFont.FreeTypeFont, stroke: int = 0) -> Image.Image:
    canvas = Image.new("L", (1600, 400), 255)
    ImageDraw.Draw(canvas).text((150, 140), text, font=font, fill=0, stroke_width=stroke, stroke_fill=0)
    bb = Image.eval(canvas, lambda p: 255 - p).getbbox()
    if not bb:
        return canvas.crop((0, 0, 10, 10))
    return canvas.crop((bb[0] - 2, bb[1] - 2, bb[2] + 2, bb[3] + 2))


def degrade(im: Image.Image, rng: random.Random) -> Image.Image:
    w, h = im.size
    pad = 20
    big = Image.new("L", (w + 2 * pad, h + 2 * pad), 255)
    big.paste(im, (pad, pad))
    # slant + small rotation
    shear = rng.uniform(-0.25, 0.25)
    big = big.transform(big.size, Image.AFFINE, (1, shear, -shear * big.size[1] / 2, 0, 1, 0),
                        resample=Image.BILINEAR, fillcolor=255)
    big = big.rotate(rng.uniform(-3, 3), resample=Image.BILINEAR, fillcolor=255, expand=True)
    a = np.asarray(big, dtype=np.float32) / 255.0
    # elastic wobble (hand-drawn strokes are not geometric)
    if rng.random() < 0.8:
        sigma = rng.uniform(4, 8)
        alpha = rng.uniform(1.0, 3.5)
        dx = ndi.gaussian_filter(np.random.uniform(-1, 1, a.shape), sigma) * alpha * sigma
        dy = ndi.gaussian_filter(np.random.uniform(-1, 1, a.shape), sigma) * alpha * sigma
        yy, xx = np.meshgrid(np.arange(a.shape[0]), np.arange(a.shape[1]), indexing="ij")
        a = ndi.map_coordinates(a, [yy + dy, xx + dx], order=1, mode="constant", cval=1.0)
    # stroke weight: occasionally thicken the ink (erosion is not used: in our renders it broke thin strokes apart)
    ink = a < 0.5
    if rng.random() < 0.2:
        ink = ndi.binary_dilation(ink, iterations=1)
    a = np.where(ink, 0.0, 1.0)
    # blur, speckle, contrast, threshold
    a = ndi.gaussian_filter(a, rng.uniform(0.3, 1.6))
    a = a + np.random.normal(0, rng.uniform(0.0, 0.12), a.shape)
    if rng.random() < 0.4:
        a = np.where(a < rng.uniform(0.35, 0.65), 0.0, 1.0)
        a = ndi.gaussian_filter(a, rng.uniform(0.2, 0.8))
    lo, hi = rng.uniform(0.0, 0.25), rng.uniform(0.75, 1.0)
    a = lo + (hi - lo) * np.clip(a, 0, 1)
    out = Image.fromarray((a * 255).astype(np.uint8))
    bb = Image.eval(out, lambda p: 255 if p < 128 else 0).getbbox()
    if bb:
        out = out.crop((max(0, bb[0] - 3), max(0, bb[1] - 3), bb[2] + 3, bb[3] + 3))
    return out


def generate(out_dir: str, n: int, fonts: list[str], lexicon: str, seed: int = 0, p_random: float = 0.25,
             engine: str = "auto", jitter: float = 0.0):
    rng = random.Random(seed)
    np.random.seed(seed)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    words = lexicon_words(lexicon) if lexicon and os.path.exists(lexicon) else []
    loaded = {}
    rows = []
    i = 0
    while i < n:
        if words and rng.random() > p_random:
            src = rng.choice(words)
            try:
                toks = latin_to_tokens(src)
            except ValueError:
                continue
        else:
            toks = random_tokens(rng)
            src = "~random"
        if not toks or not encodable(toks):
            continue
        fp = rng.choice(fonts)
        if not shapes_cleanly(tokens_to_unicode(toks), fp):
            continue
        size = rng.choice([56, 64, 72, 80])
        text = tokens_to_unicode(toks)
        if engine == "raqm" or (engine == "auto" and RAQM and jitter == 0):
            key = (fp, size)
            if key not in loaded:
                loaded[key] = ImageFont.truetype(fp, size, layout_engine=ImageFont.Layout.RAQM)
            im = render(text, loaded[key], stroke=rng.choice([0, 0, 1, 2]))
        else:
            if fp not in loaded:
                loaded[fp] = Renderer(fp)
            im = loaded[fp].render(text, size=size, pad=2, jitter=rng.uniform(0.3, 1.0) * jitter, rng=rng)
            if rng.random() < 0.5:   # stand-in for stroke_width: thicken a little
                im = im.filter(ImageFilter.MinFilter(3))
        if im.width < 4 or im.height < 4:
            continue
        im = normalize(degrade(im, rng))
        fn = f"{i:06d}.png"
        im.save(os.path.join(out_dir, "images", fn))
        rows.append((fn, " ".join(toks), src))
        i += 1
    with open(os.path.join(out_dir, "labels.tsv"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--fonts", nargs="+", default=sorted(glob.glob("fonts/*.otf") + glob.glob("fonts/*.ttf")))
    ap.add_argument("--lexicon", default="data/lexicon/lexicon_merged.tsv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", choices=["auto", "raqm", "hb"], default="auto")
    ap.add_argument("--jitter", type=float, default=0.0, help="outline-space handwriting jitter (hb engine), e.g. 1.0")
    a = ap.parse_args()
    if not a.fonts:
        ap.error("no fonts given and none found in fonts/: add Duployan .otf/.ttf files (e.g. Noto Sans Duployan)")
    print(generate(a.out_dir, a.n, a.fonts, a.lexicon, a.seed, engine=a.engine, jitter=a.jitter), "images")
