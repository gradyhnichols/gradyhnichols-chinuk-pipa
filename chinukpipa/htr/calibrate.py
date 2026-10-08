"""Pick a per-source scale multiplier for real word images (label-using, on the *training* split only).

Normalizing every word to the same stroke width does not make all sources the same size: in this project's
measurements the words from the 1892 vocabulary came out roughly twice as large as the synthetic words, those from
the 1898 book about the same size. For each source this tries a
grid of multipliers and keeps the one with the lowest token error rate on the training words; test words are never
looked at.

    python -m chinukpipa.htr.calibrate MODEL --crops DIR [--out scales.json]
"""
from __future__ import annotations

import argparse
import json

import torch

from chinukpipa.htr.data import RealDataset, to_tensor, vocab
from chinukpipa.htr.model import greedy_decode, load_model
from chinukpipa.htr.train import edit_distance

GRID = [0.4, 0.5, 0.63, 0.8, 1.0, 1.25, 1.6]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--crops", required=True)
    ap.add_argument("--real", nargs="+", default=["data/gt/vocab_annotations.jsonl", "data/gt/copy_annotations.jsonl"])
    ap.add_argument("--split-seed", default="v0", help="use the training words of this split")
    ap.add_argument("--out")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v = vocab()
    model = load_model(a.model, len(v), device)
    sources = {json.loads(line)["source"] for path in a.real for line in open(path, encoding="utf-8")}
    table = {}
    for mult in GRID:
        ds = RealDataset(a.real, a.crops, v, split="train", scales={src: mult for src in sources},
                         split_seed=a.split_seed)
        for im, toks, _, _, src in ds.items:
            hyp = [v[k - 1] for k in greedy_decode(model(to_tensor(im)[None].to(device)))[0]]
            e = table.setdefault(src, {}).setdefault(mult, [0, 0])
            e[0] += edit_distance(hyp, toks)
            e[1] += len(toks)
    best = {}
    for src, row in sorted(table.items()):
        ters = {m: e / max(1, n) for m, (e, n) in row.items()}
        best[src] = min(ters, key=ters.get)
        print(src, " ".join(f"x{m}:{t:.3f}" for m, t in ters.items()), "->", best[src])
    print(json.dumps(best))
    if a.out:
        json.dump(best, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
