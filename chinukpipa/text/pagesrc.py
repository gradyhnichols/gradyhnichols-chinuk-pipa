"""Load one page image of an Internet Archive book item, upright and cropped to the page.

Two kinds of local IA downloads are handled:

* ``<id>_orig_jp2.tar`` -- raw camera captures. Each leaf is rotated and cropped as recorded in the item's
  ``<id>_scandata.xml`` (``rotateDegree`` = clockwise rotation, ``cropBox`` = page box in the rotated frame),
  which is what IA itself does when it makes its derivatives.
* ``<id>_jp2.zip`` -- processed derivatives, already upright and cropped; used as they are.

Page indices are leaf numbers (0-based position in the sorted image list), the same numbering that
``chinukpipa.gt.vocab_pages.load_ia_page`` uses.

    python -m chinukpipa.text.pagesrc ITEM_DIR LEAF OUT.png [--max-side 1600]
"""
from __future__ import annotations

import io
import os
import re
import tarfile
import zipfile

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _scandata_leaf(item_dir: str, leaf: int) -> dict:
    """rotateDegree / cropBox / orig size for one leaf from <id>_scandata.xml ({} if not available)."""
    item = os.path.basename(os.path.normpath(item_dir))
    path = os.path.join(item_dir, f"{item}_scandata.xml")
    if not os.path.exists(path):
        return {}
    text = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(rf'<page leafNum="{leaf}">(.*?)</page>', text, re.S)
    if not m:
        return {}
    body = m.group(1)
    out: dict = {}
    r = re.search(r"<rotateDegree>(-?\d+)</rotateDegree>", body)
    if r:
        out["rotate"] = int(r.group(1))
    c = re.search(r"<cropBox>\s*<x>(-?\d+)</x>\s*<y>(-?\d+)</y>\s*<w>(\d+)</w>\s*<h>(\d+)</h>", body)
    if c:
        out["crop"] = tuple(int(v) for v in c.groups())
    for tag in ("origWidth", "origHeight"):
        t = re.search(rf"<{tag}>(\d+)</{tag}>", body)
        if t:
            out[tag] = int(t.group(1))
    return out


def is_raw_capture(item_dir: str) -> bool:
    """True when the item is read from raw captures (<id>_orig_jp2.tar), which load_page crops with scandata."""
    item = os.path.basename(os.path.normpath(item_dir))
    return os.path.exists(os.path.join(item_dir, f"{item}_orig_jp2.tar"))


def _read_member(item_dir: str, leaf: int) -> tuple[Image.Image, bool]:
    """Return (image, is_raw_capture)."""
    item = os.path.basename(os.path.normpath(item_dir))
    tar_path = os.path.join(item_dir, f"{item}_orig_jp2.tar")
    if os.path.exists(tar_path):
        with tarfile.open(tar_path) as t:
            names = sorted(m.name for m in t.getmembers() if m.name.lower().endswith(".jp2"))
            im = Image.open(io.BytesIO(t.extractfile(names[leaf]).read()))
            im.load()
        return im, True
    zip_path = os.path.join(item_dir, f"{item}_jp2.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith(".jp2"))
            im = Image.open(io.BytesIO(z.read(names[leaf])))
            im.load()
        return im, False
    raise FileNotFoundError(f"no {item}_orig_jp2.tar or {item}_jp2.zip in {item_dir}")


def load_page(item_dir: str, leaf: int, *, margin: int = 40) -> Image.Image:
    """Grayscale page image, rotated upright and cropped to the scandata page box (+ `margin` px)."""
    im, raw = _read_member(item_dir, leaf)
    im = im.convert("L")
    if not raw:
        return im
    sd = _scandata_leaf(item_dir, leaf)
    rot = sd.get("rotate", 0) % 360
    if rot:
        # IA's rotateDegree is clockwise; PIL rotates counter-clockwise
        im = im.rotate(-rot, expand=True)
    if "crop" in sd:
        x, y, w, h = sd["crop"]
        im = im.crop((max(0, x - margin), max(0, y - margin),
                      min(im.width, x + w + margin), min(im.height, y + h + margin)))
    return im


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="save one upright, cropped IA page as PNG")
    ap.add_argument("item_dir")
    ap.add_argument("leaf", type=int)
    ap.add_argument("out")
    ap.add_argument("--max-side", type=int, default=0, help="downscale so the longer side is at most this")
    a = ap.parse_args()
    page = load_page(a.item_dir, a.leaf)
    if a.max_side:
        page.thumbnail((a.max_side, a.max_side), Image.LANCZOS)
    page.save(a.out)
    print(a.out, page.size)
