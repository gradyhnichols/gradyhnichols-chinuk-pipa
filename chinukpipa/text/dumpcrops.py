"""Write the word images of one stage-A page (<stem>_words.npz) as PNGs named L<line>_W<index>.png (the same
line/index as the rows of <stem>_read.json), plus a contact sheet per text line.

    python -m chinukpipa.text.dumpcrops PAGE_words.npz OUT_DIR [--lines 1-20]
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("out_dir")
    ap.add_argument("--lines", default="")
    a = ap.parse_args()
    z = np.load(a.npz)
    flat, shapes, line, index = z["flat"], z["shapes"], z["line"], z["index"]
    keep = None
    if a.lines:
        lo, _, hi = a.lines.partition("-")
        keep = range(int(lo), int(hi or lo) + 1)
    os.makedirs(a.out_dir, exist_ok=True)
    o = 0
    by_line = {}
    for (h, w), ln, ix in zip(shapes, line, index):
        n = int(h) * int(w)
        arr = flat[o:o + n].reshape(int(h), int(w))
        o += n
        if keep is not None and int(ln) not in keep:
            continue
        im = Image.fromarray(arr)
        im.save(os.path.join(a.out_dir, f"L{int(ln):02d}_W{int(ix):02d}.png"))
        by_line.setdefault(int(ln), []).append((int(ix), im))
    for ln, items in by_line.items():   # one strip per line: crops left to right with their index above
        items.sort()
        H = max(im.height for _, im in items) + 22
        W = sum(im.width + 12 for _, im in items) + 12
        sheet = Image.new("L", (W, H), 255)
        d = ImageDraw.Draw(sheet)
        x = 6
        for ix, im in items:
            d.text((x, 2), str(ix), fill=0)
            sheet.paste(im, (x, 20))
            x += im.width + 12
        sheet.save(os.path.join(a.out_dir, f"line_L{ln:02d}.png"))
    print(f"{sum(len(v) for v in by_line.values())} crops, {len(by_line)} lines -> {a.out_dir}")


if __name__ == "__main__":
    main()
