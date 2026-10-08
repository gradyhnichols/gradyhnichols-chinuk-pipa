#!/usr/bin/env python3
"""Render every sign in signs.yaml that has a code point, as a labelled grid (sanity test).

Usage:
  python3 -I render_sign_table.py --font /path/to/RawndMusmusDuployan-Regular.otf \
      [--signs signs.yaml] [--out sign_table.png] [--size 96]

Requires Pillow built with Raqm (HarfBuzz shaping), PyYAML. Rawnd Musmus Duployan font files are SIL OFL 1.1 (build code Apache-2.0)
(https://github.com/dscorbett/duployan-font). All glyphs share one scale; each is centred on its ink box.
Compare the output with Le Jeune's printed table (Chinook Rudiments 1924, p.5) by eye.
"""
import argparse
import os
import re
import sys

import yaml
from PIL import Image, ImageDraw, ImageFont, features

CGJ = "͏"
# Signs whose only code point is a combining/format character need a base to be visible.
OVERRIDE = {
    "NUM_ENCLOSE_1000": "\U0001BC02⃝",
    "NUM_ENCLOSE_1000000": "\U0001BC02⃝⃝",
    "DIAC_ORIENT_CGJ3": "\U0001BC03\U0001BC44" + CGJ * 3,
    "DIAC_YEE_DIAERESIS": "\U0001BC46\U0001BC46̈",
    "PUNCT_SYLLABLE_BREAK": "\U0001BC05\U0001BC41‌\U0001BC03\U0001BC41",
    "PUNCT_WORD_SPACE": "\U0001BC05\U0001BC41 \U0001BC03\U0001BC41",
    "ABBR_OVERLAP": "\U0001BC1C\U0001BCA1\U0001BC03",
}
ORDER = ["consonant", "vowel", "diphthong", "numeral", "logogram", "punctuation", "abbreviation", "diacritic"]


def to_str(cps):
    out = []
    for item in cps:
        for tok in re.findall(r"U\+([0-9A-Fa-f]{4,6})", item):
            out.append(chr(int(tok, 16)))
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", required=True)
    ap.add_argument("--signs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "signs.yaml"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "research", "img", "sign_table.png"))
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--joining-tests", default=None, help="also write the Rawnd joining/orientation test sheet to this path")
    a = ap.parse_args()
    if not features.check("raqm"):
        sys.exit("Pillow was built without Raqm; shaped Duployan cannot be rendered")
    data = yaml.safe_load(open(a.signs, encoding="utf-8"))
    gf = ImageFont.truetype(a.font, a.size, layout_engine=ImageFont.Layout.RAQM)
    lab = ImageFont.load_default(size=13)
    cap = ImageFont.load_default(size=12)
    hdr = ImageFont.load_default(size=22)

    cells = {c: [] for c in ORDER}
    skipped = []
    for s in data["signs"]:
        base = OVERRIDE.get(s["id"]) or (to_str(s["unicode"]) if s.get("unicode") else "")
        if base:
            cells[s["category"]].append((s["id"], base, "+".join(re.findall(r"U\+[0-9A-Fa-f]{4,6}", " ".join(s["unicode"])))[:30] if s.get("unicode") else ""))
        else:
            skipped.append(s["id"])
        for k, v in enumerate(s.get("unicode_variants") or [], 1):
            cells[s["category"]].append((s["id"] + "/v%d" % k, to_str([v]), v.replace("U+", "")[:30]))

    W, H = 190, 200
    cols = a.cols
    pad = 12
    rows_total = sum((len(v) + cols - 1) // cols for v in cells.values() if v)
    nsec = sum(1 for v in cells.values() if v)
    img = Image.new("RGB", (cols * W + 2 * pad, rows_total * H + nsec * 40 + 70), "white")
    d = ImageDraw.Draw(img)
    d.text((pad, 10), "chinuk-pipa sign inventory: Rawnd Musmus Duployan via Pillow+Raqm (common scale %dpx; id above, code points below)" % a.size, fill="black", font=lab)
    d.text((pad, 34), "%d signs, %d with a code point (variants shown as id/vN); %d without a code point are not drawn: %s"
           % (len(data["signs"]), len(data["signs"]) - len(skipped), len(skipped), ", ".join(skipped)[:140] + ("..." if len(", ".join(skipped)) > 140 else "")), fill="#444", font=cap)
    y = 62
    for cat in ORDER:
        items = cells[cat]
        if not items:
            continue
        d.rectangle((pad, y, img.width - pad, y + 28), fill="#e8eef7")
        d.text((pad + 6, y + 3), "%s (%d)" % (cat.upper(), len(items)), fill="#123", font=hdr)
        y += 36
        for n, (i, s, cp) in enumerate(items):
            r, c = divmod(n, cols)
            x0 = pad + c * W
            y0 = y + r * H
            d.rectangle((x0 + 2, y0 + 2, x0 + W - 3, y0 + H - 3), outline="#cccccc")
            d.text((x0 + 8, y0 + 6), i, fill="#b00000", font=lab)
            # glyph on a large scratch canvas, anchored on the baseline, then centred on the ink box
            scratch = Image.new("L", (900, 900), 0)
            ImageDraw.Draw(scratch).text((200, 450), s, fill=255, font=gf, anchor="ls")
            bb = scratch.getbbox()
            if bb:
                gw, gh = bb[2] - bb[0], bb[3] - bb[1]
                crop = scratch.crop(bb)
                px = x0 + (W - gw) // 2
                py = y0 + 26 + (H - 26 - 26 - gh) // 2
                img.paste((0, 0, 0), (px, py), crop)
            d.text((x0 + 8, y0 + H - 22), cp, fill="#555", font=cap)
        y += ((len(items) + cols - 1) // cols) * H + 4
    img = img.crop((0, 0, img.width, y + 8))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    img.save(a.out)
    if a.joining_tests:
        joining_tests(a.font, a.joining_tests)
    print("wrote", a.out, img.size, "cells drawn:", sum(len(v) for v in cells.values()), "skipped (no code point):", len(skipped))


def c(*cps):
    return "".join(chr(x) for x in cps)


def joining_tests(font_path, out, size=90):
    """Sheet of short sequences that exercise joining, orientation, overlap and numerals."""
    OV, ZW = 0x1BCA1, 0x200C
    tests = [
        ("K L  (kl)", c(0x1BC05, 0x1BC06)), ("K T  (kt)", c(0x1BC05, 0x1BC03)), ("R K  (rk)", c(0x1BC0B, 0x1BC05)),
        ("R P  (rp)", c(0x1BC0B, 0x1BC02)), ("P A P  (pap = 11)", c(0x1BC02, 0x1BC41, 0x1BC02)), ("P P  (identical)", c(0x1BC02, 0x1BC02)),
        ("K A  (ka)", c(0x1BC05, 0x1BC41)), ("A K  (ak)", c(0x1BC41, 0x1BC05)), ("L A  (la)", c(0x1BC06, 0x1BC41)),
        ("A L  (al)", c(0x1BC41, 0x1BC06)), ("T O", c(0x1BC03, 0x1BC44)), ("T O + CGJ x3", c(0x1BC03, 0x1BC44, 0x34F, 0x34F, 0x34F)),
        ("T A2 (U+1BC42)", c(0x1BC03, 0x1BC42)), ("S + overlap + T  (ST)", c(0x1BC1C, OV, 0x1BC03)), ("Sh + overlap + K  (ShK)", c(0x1BC1B, OV, 0x1BC05)),
        ("S ov Sh ov B  (SShB)", c(0x1BC1C, OV, 0x1BC1B, OV, 0x1BC07)), ("I ov T ov S  (ItS)", c(0x1BC46, OV, 0x1BC03, OV, 0x1BC1C)),
        ("M N Sh S  (numerals 6 7 8 9)", c(0x1BC19, 0x1BC1A, 0x1BC1B, 0x1BC1C)), ("ka ZWNJ ta", c(0x1BC05, 0x1BC41, ZW, 0x1BC03, 0x1BC41)),
        ("P WA  (100?)", c(0x1BC02, 0x1BC5C)), ("P + U+20DD  (1000)", c(0x1BC02, 0x20DD)), ("P + U+20DD x2  (1,000,000?)", c(0x1BC02, 0x20DD, 0x20DD)),
        ("H ZWNJ L  (h-l)", c(0x1BC00, ZW, 0x1BC06)), ("HL (U+1BC16)", c(0x1BC16)), ("LH (U+1BC17)", c(0x1BC17)),
        ("T U+2E3C T", c(0x1BC03, 0x2E3C, 0x1BC03)), ("I I U+0308  (Yee?)", c(0x1BC46, 0x1BC46, 0x308)), ("WE / WEYIE", c(0x1BC5E, 0x1BC5F)),
    ]
    gf = ImageFont.truetype(font_path, size, layout_engine=ImageFont.Layout.RAQM)
    lab = ImageFont.load_default(size=14)
    cols, W, H = 3, 470, 250
    rows = (len(tests) + cols - 1) // cols
    img = Image.new("RGB", (cols * W, rows * H + 30), "white")
    d = ImageDraw.Draw(img)
    d.text((8, 6), "Rawnd Musmus Duployan, Pillow+Raqm: joining, orientation, overlap and numeral tests (blue tick = baseline)", fill="black", font=lab)
    for n, (label, s) in enumerate(tests):
        r, cc = divmod(n, cols)
        x0, y0 = cc * W, 30 + r * H
        d.rectangle((x0 + 2, y0 + 2, x0 + W - 3, y0 + H - 3), outline="#cccccc")
        d.text((x0 + 8, y0 + 6), label, fill="#b00000", font=lab)
        d.text((x0 + 60, y0 + 150), s, fill="black", font=gf, anchor="ls")
        d.line((x0 + 30, y0 + 150, x0 + 52, y0 + 150), fill="#2060d0", width=2)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    img.save(out)
    print("wrote", out, img.size)


if __name__ == "__main__":
    main()
