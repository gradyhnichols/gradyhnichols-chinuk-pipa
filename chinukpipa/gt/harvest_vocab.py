"""Harvest (Roman word, shorthand image, gloss) rows from Le Jeune vocabulary pages.

python -m chinukpipa.gt.harvest_vocab ITEM_DIR PAGES OUT_JSONL --crops DIR
Writes one proposal per 3-field row: boxes in full-resolution page coordinates (after the recorded
deskew angle), OCR/lexicon candidates for the Roman word and gloss. Proposals must then be reviewed
against the images before they become ground truth (in v0.1 this review was done by AI models; see
data/gt/README.md).
"""
from __future__ import annotations

import argparse
import difflib
import json
import os

import pytesseract

from chinukpipa.gt.vocab_pages import load_ia_page, deskew, segment_page
from chinukpipa.translit import normalize_latin


def _ocr(img, psm=7):
    try:
        return pytesseract.image_to_string(img, config=f"--psm {psm}").strip()
    except Exception:
        return ""


def load_headwords(path):
    words = {}
    if not os.path.exists(path):
        return words
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        hi, gi, si = header.index("headword"), header.index("gloss_en"), header.index("source_key")
        for line in f:
            c = line.rstrip("\n").split("\t")
            words.setdefault(c[hi], []).append((c[si], c[gi]))
    return words


def best_match(cand: str, words: dict, n=3):
    key = normalize_latin(cand).replace(" ", "")
    return difflib.get_close_matches(key, list(words), n=n, cutoff=0.5)


def harvest(item_dir, pages, out_jsonl, crops_dir, lexicon_tsv):
    words = load_headwords(lexicon_tsv)
    item = os.path.basename(os.path.normpath(item_dir))
    os.makedirs(crops_dir, exist_ok=True)
    n = 0
    with open(out_jsonl, "a", encoding="utf-8") as out:
        for pg in pages:
            im = load_ia_page(item_dir, pg)
            im, angle = deskew(im)
            for r in segment_page(im):
                if not r.shorthand_bbox:
                    continue
                rid = f"{item}_p{pg:03d}_c{r.column}_r{r.index:02d}"
                pad = 6
                crops = {}
                for k in ("latin", "shorthand", "gloss"):
                    x0, y0, x1, y1 = getattr(r, f"{k}_bbox")
                    c = im.crop((max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad))
                    c.save(os.path.join(crops_dir, f"{rid}_{k}.png"))
                    crops[k] = c
                lat_ocr, glo_ocr = _ocr(crops["latin"]), _ocr(crops["gloss"])
                rec = dict(id=rid, ia_item=item, page_index=pg, deskew_deg=angle, page_size=list(im.size),
                           bbox_latin=r.latin_bbox, bbox_shorthand=r.shorthand_bbox, bbox_gloss=r.gloss_bbox,
                           latin_ocr=lat_ocr, gloss_ocr=glo_ocr,
                           lexicon_candidates=best_match(lat_ocr, words) if lat_ocr else [])
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("item_dir"); ap.add_argument("pages"); ap.add_argument("out_jsonl")
    ap.add_argument("--crops", required=True)
    ap.add_argument("--lexicon", default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "lexicon", "lexicon.tsv"))
    a = ap.parse_args()
    print(harvest(a.item_dir, [int(x) for x in a.pages.split(",")], a.out_jsonl, a.crops, a.lexicon), "rows")
