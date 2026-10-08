"""Stage B of the text pipeline: read every word image of the stage-A pages with the recognizer ensemble (GPU or CPU).

    python -m chinukpipa.text.readcrops --pages DIR --models M0 M1 M2 --out-dir DIR [--lexicon TSV] [--books a,b]
        [--leaves 15,16] [--device cuda] [--scale auto|S] [--calib-words 600] [--workers 4] [--selftest]

Input: <book>_<leaf>_words.npz from segcrops.py (masked grey word crops). For each book:
1. scale calibration, the rule of read_page.calibrate: the scale in SCALE_GRID with the highest mean per-frame
   maximum class probability (averaged over frames, words and models) on a random sample of the book's words
   (seed 0, up to --calib-words words);
2. every word read at that scale, twice: free reading (greedy CTC decoding of the first model; whether all models
   agree) and word-list reading (CTC loss of every A/B headword's token string under each model, averaged over
   models; top 5; p_rel = softmax of the negative losses over all candidates, i.e. how much the best candidate
   stands out, not a probability of being right);
   words go through the CNN in width-sorted batches and through the LSTM as a packed sequence, so each word is
   read as if alone (models trained on width-bucketed batches read badly through long padding);
3. outputs: <book>_<leaf>_read.json per page, <book>_book.json per book. Restartable: pages with a _read.json
   are skipped unless --force. These are machine readings; no person has checked them.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

from chinukpipa.htr.normalize import normalize, stretch_contrast, stroke_width

from . import DEFAULT_LEXICON

# PyTorch and the rest of chinukpipa.htr are imported inside Ensemble, not here: this module is also imported by
# the CPU worker processes and by tools that only read the .npz files, which should not have to load PyTorch.

SCALE_GRID = [0.4, 0.5, 0.63, 0.8, 1.0, 1.25]
HEIGHT = 64


# ------------------------------------------------------------------------------------------ CPU workers


def load_npz(path: str):
    z = np.load(path)
    flat, shapes = z["flat"], z["shapes"]
    out, o = [], 0
    for h, w in shapes:
        n = int(h) * int(w)
        out.append(flat[o:o + n].reshape(int(h), int(w)))
        o += n
    return out, z["boxes"], z["line"], z["index"]


def prepare_word(arr: np.ndarray, scales: list[float]) -> list[np.ndarray]:
    """One crop -> one uint8 HEIGHT x W array per scale. Same preprocessing as read_page: stretch_contrast, then
    normalize (the stroke width is measured once)."""
    im = stretch_contrast(Image.fromarray(arr))
    sw = stroke_width(np.asarray(im.convert("L")))
    return [np.asarray(normalize(im, height=HEIGHT, sw=sw, scale=s)) for s in scales]


def prep_task(task):
    """(npz path, word indices or None, scales) -> (path, idxs, [[uint8 HEIGHT x W per scale] per word], boxes,
    line, index)."""
    path, idxs, scales = task
    arrs, boxes, line, index = load_npz(path)
    idxs = list(range(len(arrs))) if idxs is None else idxs
    res = [prepare_word(arrs[i], scales) for i in idxs]
    return path, idxs, res, boxes, line, index


# ------------------------------------------------------------------------------------------ recognizer


class Ensemble:
    """Several recognizers read as one: batched log-probabilities, free reading and word-list scoring."""

    def __init__(self, model_paths: list[str], lexicon: str, device: str):
        import torch
        from chinukpipa.htr.data import vocab
        from chinukpipa.htr.model import load_model
        self.v = vocab()
        dev = torch.device(device)
        self._set_models([load_model(p, len(self.v), dev) for p in model_paths], dev)
        self._set_candidates(lexicon)

    @classmethod
    def from_models(cls, models, device: str = "cpu") -> Ensemble:
        """An ensemble of models that are already built, with no word list: logps() and confidence() work, the
        word-list methods do not. For tests and for code that has its own models."""
        import torch
        dev = torch.device(device)
        ens = cls.__new__(cls)
        ens._set_models([m.to(dev).eval() for m in models], dev)
        return ens

    def _set_models(self, models, dev) -> None:
        import torch
        from chinukpipa.htr.model import greedy_decode
        self.torch, self.greedy = torch, greedy_decode
        self.dev = dev
        self.models = models
        self.C = models[0].out.out_features      # number of classes, the CTC blank included

    def _set_candidates(self, lexicon: str) -> None:
        """Word list: the A/B headwords of the lexicon as token strings (same selection as read_page.Reader). A row of
        a lexicon with a `tokens` column may give its tokens directly; otherwise the spelling rules compute them."""
        torch = self.torch
        from chinukpipa.translit import latin_to_tokens, load_tokens
        idx = {t: i + 1 for i, t in enumerate(self.v)}
        heads: dict[str, list[str]] = {}
        vs = set(self.v)
        with open(lexicon, encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            hi, ci = header.index("headword"), header.index("best_conf")
            ti = header.index("tokens") if "tokens" in header else None   # optional: tokens given directly
            for line in f:
                c = line.rstrip("\n").split("\t")
                if c[ci] not in ("A", "B"):
                    continue
                try:
                    toks = c[ti].split() if ti is not None and ti < len(c) and c[ti].strip() \
                        else latin_to_tokens(c[hi])
                except ValueError:
                    continue
                if toks and all(t in vs for t in toks):
                    heads.setdefault(" ".join(toks), []).append(c[hi])
        self.keys, self.heads = list(heads), heads
        tg = [[idx[t] for t in k.split()] for k in self.keys]
        self.N = len(tg)
        self.tflat = torch.tensor([x for t in tg for x in t], dtype=torch.long, device=self.dev)
        self.tlen = torch.tensor([len(t) for t in tg], dtype=torch.long, device=self.dev)
        spec = load_tokens()
        self.roman = {t: ((spec[t].get("latin") or [t.lower()])[0]) for t in spec}

    def to_roman(self, toks):
        return "".join(self.roman.get(t, t.lower()) for t in toks)

    def logps(self, arrs: list[np.ndarray]):
        """Width-padded batch through the CNN, packed sequence through the LSTM. Returns ([(t, B, C) per model],
        T lengths)."""
        torch = self.torch
        W = max(a.shape[1] for a in arrs)
        W = (W + 3) // 4 * 4
        X = torch.zeros(len(arrs), 1, HEIGHT, W, dtype=torch.uint8)
        for i, a in enumerate(arrs):
            X[i, 0, :, : a.shape[1]] = torch.from_numpy(255 - a.astype(np.uint8))
        X = X.to(self.dev).float().div_(255.0)
        W0 = torch.tensor([a.shape[1] for a in arrs], device=self.dev)
        outs = []
        for m in self.models:
            # emulate reading each word alone: after every ReLU / pooling, zero the columns beyond the word's own
            # width at that resolution (what the next convolution's zero padding would show it if read alone)
            f, w = X, W0.clone()
            for layer in m.cnn:
                f = layer(f)
                if isinstance(layer, torch.nn.MaxPool2d):
                    ks = layer.kernel_size
                    w = w // (ks if isinstance(ks, int) else ks[1])
                if isinstance(layer, (torch.nn.ReLU, torch.nn.MaxPool2d)):
                    keep = torch.arange(f.shape[-1], device=self.dev)[None, :] < w[:, None]
                    f = f * keep[:, None, None, :].to(f.dtype)
            T = w.clamp(min=1).cpu()
            b, c, h, t = f.shape
            f = f.permute(3, 0, 1, 2).reshape(t, b, c * h)
            pk = torch.nn.utils.rnn.pack_padded_sequence(f, T.clamp(max=t), enforce_sorted=False)
            o, _ = m.rnn(pk)
            o, _ = torch.nn.utils.rnn.pad_packed_sequence(o, total_length=t)
            outs.append(m.out(o).log_softmax(-1))
        return outs, T.clamp(max=outs[0].shape[0])

    def confidence(self, outs, T):
        """Per word: mean per-frame max probability (and over non-blank frames), averaged over models."""
        torch = self.torch
        B = len(T)
        a = torch.zeros(B, device=self.dev)
        b = torch.zeros(B, device=self.dev)
        mask = (torch.arange(outs[0].shape[0], device=self.dev)[:, None] < T.to(self.dev)[None]).float()
        for lp in outs:
            mx, arg = lp.exp().max(-1)                       # t, B
            a += (mx * mask).sum(0) / mask.sum(0)
            nb = mask * (arg != 0).float()
            b += torch.where(nb.sum(0) > 0, (mx * nb).sum(0) / nb.sum(0).clamp(min=1), torch.zeros_like(a))
        return (a / len(outs)).tolist(), (b / len(outs)).tolist()

    def nll(self, outs, T, budget: float = 3e8):
        """Mean CTC loss of every candidate for every word: (B, N)."""
        torch = self.torch
        t = outs[0].shape[0]
        B = len(T)
        per = max(1, int(budget // (t * self.N * self.C * 4)))
        tot = torch.zeros(B, self.N, device=self.dev)
        Td = T.to(self.dev)
        for lp in outs:
            for s in range(0, B, per):
                ids = torch.arange(s, min(B, s + per), device=self.dev)
                n = len(ids)
                x = lp[:, ids.repeat_interleave(self.N), :]
                loss = torch.nn.functional.ctc_loss(
                    x, self.tflat.repeat(n), Td[ids].repeat_interleave(self.N), self.tlen.repeat(n),
                    blank=0, reduction="none", zero_infinity=False)
                tot[s:s + n] += loss.view(n, self.N)
        tot /= len(outs)
        return torch.where(torch.isfinite(tot), tot, torch.full_like(tot, float("inf")))

    def read_batch(self, arrs, k: int = 5):
        torch = self.torch
        with torch.no_grad():
            outs, T = self.logps(arrs)
            conf, conf_nb = self.confidence(outs, T)
            sc = self.nll(outs, T)
            fin = torch.isfinite(sc)
            neg = torch.where(fin, -sc, torch.full_like(sc, -float("inf")))
            logz = torch.logsumexp(neg, 1)
            vals, order = torch.topk(sc, min(k, self.N), dim=1, largest=False)
            frees = [[self.greedy(lp[: int(T[i]), i:i + 1])[0] for lp in outs] for i in range(len(arrs))]
            # free_best: of the models' own free readings, the one with the lowest mean CTC loss under the ensemble
            # (comparable with the word-list scores; used to decide when a word is probably not in the list)
            fbest = []
            for i in range(len(arrs)):
                seqs = []
                for fr in frees[i]:
                    if fr and fr not in seqs:
                        seqs.append(fr)
                if not seqs:
                    fbest.append(None)
                    continue
                Ti, n = int(T[i]), len(seqs)
                tg = torch.tensor([x for s_ in seqs for x in s_], dtype=torch.long, device=self.dev)
                tl = torch.tensor([len(s_) for s_ in seqs], dtype=torch.long, device=self.dev)
                il = torch.full((n,), Ti, dtype=torch.long, device=self.dev)
                loss = sum(torch.nn.functional.ctc_loss(lp[:Ti, i:i + 1].expand(Ti, n, lp.shape[-1]), tg, il, tl,
                                                        blank=0, reduction="none") for lp in outs) / len(outs)
                j = int(loss.argmin())
                fbest.append((seqs[j], float(loss[j])))
            vals, order, logz = vals.cpu().tolist(), order.cpu().tolist(), logz.cpu().tolist()
        rows = []
        for i in range(len(arrs)):
            f0 = [self.v[j - 1] for j in frees[i][0]]
            top = []
            for s, j in zip(vals[i], order[i]):
                key = self.keys[j]
                ok = math.isfinite(s)
                top.append({"headword": self.heads[key][0], "also": self.heads[key][1:4], "tokens": key,
                            "score": round(s, 3) if ok else None,
                            "p_rel": round(math.exp(-s - logz[i]), 4) if ok and math.isfinite(logz[i]) else 0.0})
            rows.append({"free_tokens": " ".join(f0), "free_roman": self.to_roman(f0),
                         "free_models_agree": all(fr == frees[i][0] for fr in frees[i]),
                         "confidence": round(conf[i], 4), "confidence_nonblank": round(conf_nb[i], 4),
                         "frames": int(T[i]), "top5": top,
                         "free_best": None if fbest[i] is None else {
                             "tokens": " ".join(self.v[j - 1] for j in fbest[i][0]),
                             "roman": self.to_roman([self.v[j - 1] for j in fbest[i][0]]),
                             "score": round(fbest[i][1], 3)}})
        return rows


def batches_by_width(arrs, max_batch: int, max_pixels: int):
    order = sorted(range(len(arrs)), key=lambda i: arrs[i].shape[1])
    cur, w = [], 0
    for i in order:
        wi = arrs[i].shape[1]
        if cur and (len(cur) >= max_batch or (len(cur) + 1) * max(w, wi) > max_pixels):
            yield cur
            cur, w = [], 0
        cur.append(i)
        w = max(w, wi)
    if cur:
        yield cur


# ------------------------------------------------------------------------------------------ main


def book_of(path: str) -> tuple[str, int]:
    stem = os.path.basename(path)[: -len("_words.npz")]
    book, _, leaf = stem.rpartition("_")
    return book, int(leaf)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--lexicon", default=DEFAULT_LEXICON, help="word list TSV (default: %(default)s)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--books", default="")
    ap.add_argument("--leaves", default="", help="only these leaves (comma list) of the selected books")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--scale", default="auto")
    ap.add_argument("--calib-words", type=int, default=600)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="check batched reading against one-word reading")
    a = ap.parse_args()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available")
    ens = Ensemble(a.models, a.lexicon, a.device)
    os.makedirs(a.out_dir, exist_ok=True)
    books: dict[str, list[tuple[int, str]]] = {}
    for p in glob.glob(os.path.join(a.pages, "*_words.npz")):
        b, leaf = book_of(p)
        books.setdefault(b, []).append((leaf, p))
    want = set(filter(None, a.books.split(",")))
    leaves = {int(x) for x in a.leaves.split(",") if x}
    print(json.dumps({"candidates": ens.N, "books": {b: len(v) for b, v in sorted(books.items())}}), flush=True)
    pool = Pool(a.workers)
    for book in sorted(books):
        if want and book not in want:
            continue
        pages = sorted(books[book])
        todo = [(leaf, p) for leaf, p in pages if (not leaves or leaf in leaves)]
        outp = {leaf: os.path.join(a.out_dir, f"{book}_{leaf}_read.json") for leaf, _ in todo}
        if not a.force:
            todo = [(leaf, p) for leaf, p in todo if not os.path.exists(outp[leaf])]
        if not todo:
            continue
        t0 = time.time()
        calib_path = os.path.join(a.out_dir, f"{book}_calibration.json")
        if a.scale != "auto":
            scale, table = float(a.scale), None
        elif os.path.exists(calib_path) and not a.force:
            cal = json.load(open(calib_path, encoding="utf-8"))
            scale, table = cal["chosen_scale"], cal["table"]
        else:
            counts = [(p, int(np.load(p)["shapes"].shape[0])) for _, p in pages]
            allw = [(p, i) for p, n in counts for i in range(n)]
            sample = random.Random(0).sample(allw, min(a.calib_words, len(allw)))
            byp: dict[str, list[int]] = {}
            for p, i in sample:
                byp.setdefault(p, []).append(i)
            imgs = []
            for _, _, res, *_ in pool.imap_unordered(prep_task, [(p, sorted(ix), SCALE_GRID) for p, ix in byp.items()]):
                imgs += res
            table = []
            for si, s in enumerate(SCALE_GRID):
                arrs = [r[si] for r in imgs]
                ca, cb = [], []
                for ix in batches_by_width(arrs, 64, 64 * 640):
                    with torch.no_grad():
                        outs, T = ens.logps([arrs[i] for i in ix])
                        x, y = ens.confidence(outs, T)
                    ca += x
                    cb += y
                table.append({"scale": s, "mean_max_prob": round(float(np.mean(ca)), 4),
                              "mean_max_prob_nonblank": round(float(np.mean(cb)), 4), "n_words": len(ca)})
            scale = max(table, key=lambda r: r["mean_max_prob"])["scale"]
            with open(calib_path, "w", encoding="utf-8") as fh:
                json.dump({"book": book, "chosen_scale": scale, "table": table,
                           "rule": "highest mean per-frame max probability, sample seed 0"}, fh, indent=1)
        print(json.dumps({"book": book, "scale": scale, "pages": len(todo),
                          "calib": [(r["scale"], r["mean_max_prob"]) for r in table] if table else None}), flush=True)
        nw = 0
        tasks = [(p, None, [scale]) for _, p in todo]
        for path, _, res, boxes, line, index in pool.imap(prep_task, tasks):
            leaf = book_of(path)[1]
            arrs = [r[0] for r in res]
            rows: list = [None] * len(arrs)
            if a.selftest and arrs:
                k = min(24, len(arrs))
                with torch.no_grad():
                    outs, T = ens.logps(arrs[:k])
                    dmax = 0.0
                    for i, x in enumerate(arrs[:k]):     # reference: the model's own forward on the word alone
                        xt = torch.from_numpy(255 - x.astype(np.float32)).div(255.0)[None, None].to(ens.dev)
                        for m, lp in zip(ens.models, outs):
                            ref = m(xt)[:, 0]
                            if ref.shape[0] != int(T[i]):
                                dmax = float("inf")
                                break
                            dmax = max(dmax, float((ref - lp[: int(T[i]), i]).abs().max()))
                print(json.dumps({"selftest_words": k, "max_logp_diff_vs_alone": dmax}), flush=True)
                a.selftest = False
            for ix in batches_by_width(arrs, a.batch, a.batch * 320):
                for i, r in zip(ix, ens.read_batch([arrs[i] for i in ix])):
                    rows[i] = r
            words = []
            for i, r in enumerate(rows):
                w = {"line": int(line[i]), "index": int(index[i]), "bbox": [int(v) for v in boxes[i]]}
                w.update(r)
                words.append(w)
            out = {"item": book, "leaf": leaf, "scale": scale, "models": a.models, "n_candidates": ens.N,
                   "reading": "machine reading (CRNN ensemble), not checked by a person", "words": words}
            tmp = outp[leaf] + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp, outp[leaf])
            nw += len(words)
            print(json.dumps({"page": f"{book}_{leaf}", "words": len(words),
                              "sec": round(time.time() - t0, 1)}), flush=True)
        summarize(book, a.out_dir, scale, table, time.time() - t0)
        print(json.dumps({"book_done": book, "words": nw, "sec": round(time.time() - t0, 1)}), flush=True)
    pool.close()
    pool.join()
    if a.device == "cuda":
        torch.cuda.synchronize()


def summarize(book: str, out_dir: str, scale, table, sec: float) -> None:
    rows = []
    for p in glob.glob(os.path.join(out_dir, f"{book}_*_read.json")):
        if book_of(p.replace("_read.json", "_words.npz"))[0] != book:
            continue
        rows += json.load(open(p, encoding="utf-8"))["words"]
    if not rows:
        return
    agree = np.mean([r["free_models_agree"] for r in rows])
    same = np.mean([bool(r["top5"]) and r["free_tokens"] == r["top5"][0]["tokens"] for r in rows])
    prel = np.array([r["top5"][0]["p_rel"] if r["top5"] else 0.0 for r in rows])
    conf = np.array([r["confidence"] for r in rows])
    heads: dict[str, int] = {}
    for r in rows:
        if r["top5"]:
            heads[r["top5"][0]["headword"]] = heads.get(r["top5"][0]["headword"], 0) + 1
    top = sorted(heads.items(), key=lambda x: -x[1])[:100]
    s = {"book": book, "words": len(rows), "scale": scale, "calibration": table,
         "free_models_agree": round(float(agree), 4), "free_equals_top1": round(float(same), 4),
         "top1_p_rel_quantiles": {q: round(float(np.quantile(prel, q)), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
         "confidence_quantiles": {q: round(float(np.quantile(conf, q)), 4) for q in (0.1, 0.5, 0.9)},
         "top1_headwords": top, "seconds_last_run": round(sec, 1),
         "note": "machine readings; no person has checked them"}
    with open(os.path.join(out_dir, f"{book}_book.json"), "w", encoding="utf-8") as fh:
        json.dump(s, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    if os.name == "nt":
        # Some Windows CUDA installs crash while unloading libraries at exit, after the work is done. Every output
        # is written and closed by now, so end the process at once (same as htr/train.py's exit_now).
        import ctypes
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p          # a HANDLE is pointer-sized: without these two
        k.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]   # lines ctypes passes a 32-bit integer
        k.TerminateProcess(k.GetCurrentProcess(), 0)
