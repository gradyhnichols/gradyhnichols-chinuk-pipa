"""Evaluate a trained recognizer on the held-out real words, with and without a word list.

    python -m chinukpipa.htr.evaluate runs/x/model.pt --crops gt_crops [--split test] [--out eval.json]

Reports:
- free decoding: token error rate (TER) and exact-word rate, greedy CTC;
- closed-vocabulary reading: every candidate word (the A/B-confidence headwords of the merged word list in
  data/lexicon, plus the test words themselves) is scored by its CTC likelihood and the best is taken
  (top-1 / top-5 accuracy). Because the test words are among the candidates, this measures "pick the right word
  from a known vocabulary", not open reading.

Several models can be given: their CTC losses are averaged for the closed-vocabulary reading (an ensemble), and
--tta adds rescaled copies of each word image (test-time augmentation). Free decoding uses the first model only.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from PIL import Image

from chinukpipa.htr.data import RealDataset, encode, parse_scales, to_tensor, vocab
from chinukpipa.htr.model import CRNN, greedy_decode
from chinukpipa.htr.normalize import rescale
from chinukpipa.htr.train import edit_distance
from chinukpipa.translit import latin_to_tokens


def candidate_words(lexicon_tsv: str, extra: list[str]) -> dict[str, list[str]]:
    extra = list(extra)
    cands = {}
    with open(lexicon_tsv, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        hi, ci = header.index("headword"), header.index("best_conf")
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c[ci] in ("A", "B"):
                extra.append(c[hi])
    v = set(vocab())
    for w in extra:
        try:
            toks = latin_to_tokens(w)
        except ValueError:
            continue
        if toks and all(t in v for t in toks):
            cands.setdefault(" ".join(toks), w)
    return {k: k.split() for k in cands}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="+")
    ap.add_argument("--tta", default="1", help="rescale factors for the closed-vocabulary reading, e.g. 0.85,1,1.2")
    ap.add_argument("--crops", required=True)
    ap.add_argument("--real", nargs="+", default=["data/gt/vocab_annotations.jsonl", "data/gt/copy_annotations.jsonl"])
    ap.add_argument("--lexicon", default="data/lexicon/lexicon_merged.tsv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--real-scales", help="per-source calibration, e.g. LJ1892=0.5 (see htr/calibrate.py)")
    ap.add_argument("--split-seed", default="v0")
    ap.add_argument("--out")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v = vocab()
    models = []
    for path in a.model:
        m = CRNN(len(v)).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)
    tta = [float(f) for f in a.tta.split(",")]
    ds = RealDataset(a.real, a.crops, v, split=a.split, scales=parse_scales(a.real_scales), split_seed=a.split_seed)
    test_words = [it[3] for it in ds.items]
    cands = candidate_words(a.lexicon, list(test_words))
    keys = list(cands)
    cand_targets = [torch.tensor(encode(cands[k], v)) for k in keys]
    # zero_infinity must stay False here: an impossible candidate (e.g. longer than the image allows) has infinite
    # loss and must rank last, not first
    ctc = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)

    def nll(logp):
        T = logp.shape[0]
        out = []
        for s in range(0, len(keys), 512):
            tg = cand_targets[s:s + 512]
            n = len(tg)
            lp = logp.expand(T, n, logp.shape[-1])
            out.append(ctc(lp, torch.cat(tg).to(device), torch.full((n,), T, dtype=torch.long),
                           torch.tensor([len(t) for t in tg])).cpu())
        out = torch.cat(out)
        return torch.where(torch.isfinite(out), out, torch.full_like(out, float("inf")))

    errs = refs = exact = top1 = top5 = 0
    rows = []
    for i in range(len(ds)):
        x, y, ref = ds[i]
        logp = models[0](x[None].to(device))              # T,1,C
        hyp = [v[k - 1] for k in greedy_decode(logp)[0]]
        r = ref.split()
        e = edit_distance(hyp, r)
        errs, refs, exact = errs + e, refs + len(r), exact + (hyp == r)
        im = np.asarray(ds.items[i][0].convert("L"))
        variants = [x if f == 1 else to_tensor(Image.fromarray(rescale(im, f))) for f in tta]
        scores = sum(nll(m(xv[None].to(device))) for m in models for xv in variants) / (len(models) * len(variants))
        order = scores.argsort()[:5].tolist()
        best = [keys[k] for k in order]
        top1 += best[0] == ref
        top5 += ref in best
        rows.append({"id": ds.items[i][2], "source": ds.items[i][4], "latin": ds.items[i][3], "ref": ref,
                     "greedy": " ".join(hyp), "top5": best, "edits": e, "len": len(r)})
    res = {"n": len(ds), "models": len(models), "tta": tta, "candidates": len(keys), "ter": errs / max(1, refs), "exact": exact / max(1, len(ds)),
           "lexicon_top1": top1 / max(1, len(ds)), "lexicon_top5": top5 / max(1, len(ds)), "by_source": {}}
    for src in sorted({r["source"] for r in rows}):
        rs = [r for r in rows if r["source"] == src]
        res["by_source"][src] = {"n": len(rs), "ter": sum(r["edits"] for r in rs) / max(1, sum(r["len"] for r in rs)),
                                 "exact": sum(r["greedy"] == r["ref"] for r in rs) / len(rs),
                                 "lexicon_top1": sum(r["top5"][0] == r["ref"] for r in rs) / len(rs),
                                 "lexicon_top5": sum(r["ref"] in r["top5"] for r in rs) / len(rs)}
    print(json.dumps(res))
    if a.out:
        json.dump({"summary": res, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
