"""Read the words of one or more pages of running Chinuk Pipa text with the CRNN word recognizer.

    python -m chinukpipa.text.read_page ITEM_DIR LEAF [LEAF ...] --models M1.pt M2.pt M3.pt --out-dir DIR \
        [--lexicon TSV] [--scale auto|S] [--calib-leaves L ...]

For every page: segment it (segment.py), cut one image per word box (only the word's own components; ink of
neighbouring words and lines is whitened), and read it twice:

* **free reading**: greedy CTC decoding of the first model's output, as sign tokens and as Roman letters (the
  first Latin spelling listed for each token in data/signs/tokens.yaml);
* **word-list reading**: every candidate word of the merged word list (A/B-confidence headwords, converted to
  tokens with the project's spelling rules; about 1,600 distinct token strings) is scored by its CTC loss
  (negative log-likelihood) under each model, the losses are averaged over the models, and the five lowest are
  kept. Impossible candidates (too long for the image) get an infinite loss and rank last. `p_rel` is the
  softmax of the negative scores over all candidates: how much the best candidate stands out among the
  candidates, not a probability that the reading is right.

**Scale calibration** (`--scale auto`): the recognizer was trained with a per-book scale factor applied after
stroke-width normalization. For a new book every S in {0.4, 0.5, 0.63, 0.8, 1.0, 1.25} is tried on all words of
the given pages (plus --calib-leaves) and the S with the highest mean max-probability (per-frame maximum class
probability averaged over frames, words and models) is kept.

Outputs in --out-dir: <id>_<leaf>_read.json, <id>_<leaf>_crops/*.png, <id>_<leaf>_review_NN.png (crop | free
reading | top-3 candidates), plus the segmentation files of segment.py and <id>_calibration.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from chinukpipa.htr.data import to_tensor, vocab
from chinukpipa.htr.model import greedy_decode, load_model
from chinukpipa.htr.normalize import normalize, stretch_contrast
from chinukpipa.translit import latin_to_tokens, load_tokens

from . import DEFAULT_LEXICON
from . import segment as seglib

SCALE_GRID = [0.4, 0.5, 0.63, 0.8, 1.0, 1.25]


# ------------------------------------------------------------------------------------------------ crops


def word_crops(seg: dict, norm: np.ndarray, labels: np.ndarray, pad_rel: float = 0.12):
    """One masked grey crop per word box of the page's text lines (kind 'word' only). Returns dicts with line,
    index, bbox and the PIL image; the padding is pad_rel x h_med."""
    h_med = seg["stats"]["h_med"]
    pad = max(4, int(round(pad_rel * h_med)))
    grow = max(1, int(round(0.4 * seg["stats"]["stroke_width"])))
    H, W = labels.shape
    out = []
    for ln in seg["lines"]:
        if ln["kind"] != "text":
            continue
        for wi, wd in enumerate(ln["words"]):
            if wd["kind"] != "word":
                continue
            ids = wd["comps"] + wd["marks"]
            x0, y0, x1, y1 = wd["bbox"]
            X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
            lab = labels[Y0:Y1, X0:X1]
            keep = ndi.binary_dilation(np.isin(lab, ids), iterations=grow)
            g = (norm[Y0:Y1, X0:X1] * 255).astype(np.uint8)
            g = np.where(keep, g, 255).astype(np.uint8)
            out.append({"line": ln["number"], "index": wi + 1, "bbox": [int(v) for v in wd["bbox"]],
                        "image": Image.fromarray(g)})
    return out


# ------------------------------------------------------------------------------------------------ recognizer


class Reader:
    def __init__(self, model_paths: list[str], lexicon: str):
        self.v = vocab()
        self.models = [load_model(p, len(self.v), "cpu") for p in model_paths]
        self.idx = {t: i + 1 for i, t in enumerate(self.v)}
        self.keys, self.headwords = self._candidates(lexicon)
        self.targets = [torch.tensor([self.idx[t] for t in k.split()]) for k in self.keys]
        self.ctc = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)
        spec = load_tokens()
        self.roman = {t: ((spec[t].get("latin") or [t.lower()])[0]) for t in spec}
        self.roman["_"] = " "

    def _candidates(self, lexicon: str):
        """Same selection as chinukpipa.htr.evaluate.candidate_words (A/B headwords whose tokens are all in the
        recognizer's alphabet), but keeping the Roman headwords for each token string (first one = display)."""
        v = set(self.v)
        heads: dict[str, list[str]] = {}
        with open(lexicon, encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            hi, ci = header.index("headword"), header.index("best_conf")
            for line in f:
                c = line.rstrip("\n").split("\t")
                if c[ci] not in ("A", "B"):
                    continue
                try:
                    toks = latin_to_tokens(c[hi])
                except ValueError:
                    continue
                if toks and all(t in v for t in toks):
                    heads.setdefault(" ".join(toks), []).append(c[hi])
        keys = list(heads)
        return keys, heads

    def to_roman(self, toks: list[str]) -> str:
        return "".join(self.roman.get(t, t.lower()) for t in toks)

    @torch.no_grad()
    def logps(self, im: Image.Image, scale: float):
        x = to_tensor(normalize(stretch_contrast(im), height=64, scale=scale))
        return [m(x[None]) for m in self.models]          # each (T, 1, C)

    @staticmethod
    def confidence(lps) -> tuple[float, float]:
        """(mean per-frame max probability, same over frames whose best class is not the blank), averaged over
        models."""
        a, b = [], []
        for lp in lps:
            p = lp[:, 0].exp()
            mx, arg = p.max(-1)
            a.append(float(mx.mean()))
            nb = mx[arg != 0]
            b.append(float(nb.mean()) if len(nb) else 0.0)
        return float(np.mean(a)), float(np.mean(b))

    @torch.no_grad()
    def nll(self, lp) -> torch.Tensor:
        T = lp.shape[0]
        out = []
        for s in range(0, len(self.keys), 512):
            tg = self.targets[s:s + 512]
            n = len(tg)
            out.append(self.ctc(lp.expand(T, n, lp.shape[-1]), torch.cat(tg), torch.full((n,), T, dtype=torch.long),
                                torch.tensor([len(t) for t in tg])))
        out = torch.cat(out)
        return torch.where(torch.isfinite(out), out, torch.full_like(out, float("inf")))

    def read(self, im: Image.Image, scale: float, k: int = 5) -> dict:
        lps = self.logps(im, scale)
        frees = [[self.v[i - 1] for i in greedy_decode(lp)[0]] for lp in lps]
        scores = sum(self.nll(lp) for lp in lps) / len(lps)
        finite = torch.isfinite(scores)
        order = torch.argsort(scores)[:k].tolist()
        if finite.any():
            s = -scores[finite]
            logz = torch.logsumexp(s, 0)
        top = []
        for j in order:
            sc = float(scores[j])
            key = self.keys[j]
            top.append({"headword": self.headwords[key][0], "also": self.headwords[key][1:4], "tokens": key,
                        "score": round(sc, 3) if math.isfinite(sc) else None,
                        "p_rel": round(float(torch.exp(-scores[j] - logz)), 4) if math.isfinite(sc) else 0.0})
        conf, conf_nb = self.confidence(lps)
        return {"free_tokens": " ".join(frees[0]), "free_roman": self.to_roman(frees[0]),
                "free_all_models": [" ".join(f) for f in frees],
                "free_models_agree": all(f == frees[0] for f in frees),
                "confidence": round(conf, 4), "confidence_nonblank": round(conf_nb, 4), "top5": top,
                "frames": int(lps[0].shape[0])}


def calibrate(reader: Reader, images: list[Image.Image], grid=SCALE_GRID) -> tuple[float, list]:
    table = []
    for s in grid:
        cs = [reader.confidence(reader.logps(im, s)) for im in images]
        table.append({"scale": s, "mean_max_prob": round(float(np.mean([c[0] for c in cs])), 4),
                      "mean_max_prob_nonblank": round(float(np.mean([c[1] for c in cs])), 4), "n_words": len(cs)})
    best = max(table, key=lambda r: r["mean_max_prob"])["scale"]
    return best, table


# ------------------------------------------------------------------------------------------------ review sheet


def review_sheets(rows: list[dict], crops: list[Image.Image], out_prefix: str, per_sheet: int = 30,
                  crop_h: int = 72, crop_w: int = 260) -> list[str]:
    """Sheets of rows: word id | crop | free reading (tokens + Roman) | top-3 candidates with scores."""
    font = seglib._font(18)
    small = seglib._font(15)
    paths = []
    col_x = [0, 90, 90 + crop_w + 10, 90 + crop_w + 330]
    width = col_x[3] + 520
    for s in range(0, len(rows), per_sheet):
        chunk = list(zip(rows, crops))[s:s + per_sheet]
        H = 34 + len(chunk) * (crop_h + 8)
        sheet = Image.new("RGB", (width, H), "white")
        d = ImageDraw.Draw(sheet)
        d.text((col_x[0] + 4, 6), "word", fill="black", font=font)
        d.text((col_x[1] + 4, 6), "crop", fill="black", font=font)
        d.text((col_x[2] + 4, 6), "free reading (tokens / Roman)", fill="black", font=font)
        d.text((col_x[3] + 4, 6), "word list: top 3 (CTC loss, lower = better)", fill="black", font=font)
        for k, (r, im) in enumerate(chunk):
            y = 34 + k * (crop_h + 8)
            d.line([(0, y - 4), (width, y - 4)], fill=(200, 200, 200))
            d.text((col_x[0] + 4, y + 4), f"L{r['line']}.{r['index']}", fill="black", font=font)
            c = im.copy()
            c.thumbnail((crop_w, crop_h), Image.LANCZOS)
            sheet.paste(c.convert("RGB"), (col_x[1], y))
            ft = r["free_tokens"] or "(nothing)"
            d.text((col_x[2] + 4, y + 2), ft[:34], fill=(0, 0, 160), font=small)
            d.text((col_x[2] + 4, y + 24), r["free_roman"][:30], fill=(0, 0, 160), font=font)
            d.text((col_x[2] + 4, y + 48), f"conf {r['confidence']:.2f}" + ("" if r["free_models_agree"] else
                                                                            "  (models differ)"),
                   fill=(110, 110, 110), font=small)
            for j, t in enumerate(r["top5"][:3]):
                sc = "inf" if t["score"] is None else f"{t['score']:.1f}"
                d.text((col_x[3] + 4, y + 2 + 22 * j), f"{j + 1}. {t['headword']}  ({sc})   {t['tokens']}"[:60],
                       fill=(140, 0, 0) if j == 0 else (60, 60, 60), font=font if j == 0 else small)
        p = f"{out_prefix}_{s // per_sheet + 1:02d}.png"
        sheet.save(p)
        paths.append(p)
    return paths


# ------------------------------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser(description="Segment and read pages of running Chinuk Pipa text.")
    ap.add_argument("item_dir")
    ap.add_argument("leaves", nargs="+", type=int)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--lexicon", default=DEFAULT_LEXICON, help="word list TSV (default: %(default)s)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scale", default="auto", help="'auto' (calibrate on this book's words) or a number")
    ap.add_argument("--calib-leaves", nargs="*", type=int, default=[],
                    help="extra leaves of the same book used only for scale calibration")
    a = ap.parse_args()
    reader = Reader(a.models, a.lexicon)
    item = os.path.basename(os.path.normpath(a.item_dir))
    os.makedirs(a.out_dir, exist_ok=True)
    pages = {}
    for leaf in list(dict.fromkeys(a.leaves + a.calib_leaves)):
        seg, norm = seglib.run(a.item_dir, leaf, a.out_dir)
        pages[leaf] = (seg, word_crops(seg, norm, seg["_labels"]))
    if a.scale == "auto":
        imgs = [c["image"] for leaf in pages for c in pages[leaf][1]]
        scale, table = calibrate(reader, imgs)
        with open(os.path.join(a.out_dir, f"{item}_calibration.json"), "w") as fh:
            json.dump({"item": item, "leaves": list(pages), "chosen_scale": scale, "table": table}, fh, indent=1)
        print("scale table:", " ".join(f"{r['scale']}:{r['mean_max_prob']:.4f}/{r['mean_max_prob_nonblank']:.4f}"
                                       for r in table), "->", scale)
    else:
        scale = float(a.scale)
    for leaf in a.leaves:
        seg, crops = pages[leaf]
        stem = seglib.page_stem(a.item_dir, leaf)
        cdir = os.path.join(a.out_dir, f"{stem}_crops")
        os.makedirs(cdir, exist_ok=True)
        rows = []
        for c in crops:
            name = f"L{c['line']:02d}_W{c['index']:02d}.png"
            c["image"].save(os.path.join(cdir, name))
            r = {"line": c["line"], "index": c["index"], "bbox": c["bbox"], "crop": f"{stem}_crops/{name}"}
            r.update(reader.read(c["image"], scale))
            rows.append(r)
        out = {"item": item, "leaf": leaf, "scale": scale, "models": [os.path.basename(m) for m in a.models],
               "n_candidates": len(reader.keys), "segmentation": f"{stem}_seg.json", "words": rows}
        with open(os.path.join(a.out_dir, f"{stem}_read.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
        sheets = review_sheets(rows, [c["image"] for c in crops], os.path.join(a.out_dir, f"{stem}_review"))
        print(f"{stem}: {len(rows)} words read at scale {scale}; {len(sheets)} review sheets")


if __name__ == "__main__":
    main()
