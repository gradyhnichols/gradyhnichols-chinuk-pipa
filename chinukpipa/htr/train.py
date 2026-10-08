"""Train / evaluate the Chinuk Pipa word recognizer.

    python -m chinukpipa.htr.train --synth SYNTH_DIR [SYNTH_DIR ...] --real data/gt/vocab_annotations.jsonl \\
        --crops gt_crops/ --out runs/exp1 --epochs 10 [--real-train-weight 0] [--init runs/x/model.pt]

Metrics (token error rate = edit distance / reference length, and exact-word accuracy) are reported for a
synthetic validation slice and for the real test words. Test words are held out *by word* (see
chinukpipa/split.py): synthetic renderings of a test word (same Latin key or same token sequence) are dropped
from the training data as well. Real labels are rule-derived from Le Jeune's Roman spellings unless verified,
so real-word scores measure model and rule errors together.
With --all-real every real word is used for training (for a final model); the "real_test" numbers it logs are
then not held out.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

from chinukpipa.htr.data import RealDataset, SynthDataset, collate, parse_scales, split_of, vocab
from chinukpipa.htr.gpu_aug import augment_batch
from chinukpipa.htr.model import CRNN, greedy_decode


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


@torch.no_grad()
def evaluate(model, loader, v, device, keep=0):
    model.eval()
    errs = refs = exact = n = 0
    samples = []
    for X, targets, lengths, widths, txt in loader:
        hyp = greedy_decode(model(X.to(device)))
        for h, ref in zip(hyp, txt):
            r = ref.split()
            hs = [v[k - 1] for k in h]
            errs += edit_distance(hs, r)
            refs += len(r)
            exact += hs == r
            n += 1
            if len(samples) < keep:
                samples.append((" ".join(r), " ".join(hs)))
    model.train()
    return {"ter": errs / max(1, refs), "exact": exact / max(1, n), "n": n, "samples": samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", nargs="*", default=[])
    ap.add_argument("--real", nargs="+", default=["data/gt/vocab_annotations.jsonl", "data/gt/copy_annotations.jsonl"])
    ap.add_argument("--crops", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--real-train-weight", type=float, default=0.0,
                    help="share of each training batch drawn from real training words (0 = synthetic only)")
    ap.add_argument("--steps-per-epoch", type=int, default=1000)
    ap.add_argument("--init")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--real-scales", help="per-source calibration, e.g. LJ1892=0.5,LJ1898=1.0, or a JSON file")
    ap.add_argument("--scale-aug", type=float, default=0.0, help="random rescale in [1/(1+s), 1+s]")
    ap.add_argument("--gpu-aug", type=float, default=0.0, help="probability of warping each training sample")
    ap.add_argument("--gpu-aug-strength", type=float, default=1.0)
    ap.add_argument("--split-seed", default="v0", help="train/test split by word (other values give other folds)")
    ap.add_argument("--hidden", type=int, default=128, help="LSTM width")
    ap.add_argument("--all-real", action="store_true",
                    help="train on every real word (for a final model; test metrics are then not held out)")
    a = ap.parse_args()

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    v = vocab()

    scales = parse_scales(a.real_scales)
    ss = a.split_seed
    tr_split = None if a.all_real else "train"
    real_train = RealDataset(a.real, a.crops, v, split=tr_split, augment_p=0.7, seed=a.seed, scales=scales,
                             scale_aug=a.scale_aug, split_seed=ss)
    real_train_eval = RealDataset(a.real, a.crops, v, split="train", scales=scales, split_seed=ss)  # for metrics
    real_test = RealDataset(a.real, a.crops, v, split="test", scales=scales, split_seed=ss)
    test_seqs = {" ".join(it[1]) for it in real_test.items}

    def is_test_word(toks, latin):
        if a.all_real:
            return False
        return " ".join(toks) in test_seqs or (not latin.startswith("~") and split_of(latin, seed=ss) == "test")
    parts, weights = [], []
    synth_val = None
    if a.synth:
        synth = SynthDataset(a.synth, v, scale_aug=a.scale_aug, seed=a.seed, exclude=is_test_word)
        print(f"synthetic samples of held-out test words dropped: {synth.excluded}", flush=True)
        idx = list(range(len(synth)))
        random.Random(a.seed).shuffle(idx)
        nval = min(1000, len(idx) // 50)
        synth_val = Subset(synth, idx[:nval])
        synth_tr = Subset(synth, idx[nval:])
        parts.append(synth_tr)
        weights += [(1 - a.real_train_weight) / len(synth_tr)] * len(synth_tr)
    if a.real_train_weight > 0:
        parts.append(real_train)
        weights += [a.real_train_weight / len(real_train)] * len(real_train)
    if not parts:
        ap.error("nothing to train on: give --synth and/or --real-train-weight > 0")
    train = ConcatDataset(parts)
    sampler = WeightedRandomSampler(weights, num_samples=a.steps_per_epoch * a.bs, replacement=True)
    tl = DataLoader(train, batch_size=a.bs, sampler=sampler, collate_fn=collate, num_workers=0)
    loaders = {"real_test": DataLoader(real_test, batch_size=64, collate_fn=collate),
               "real_train": DataLoader(real_train_eval, batch_size=64, collate_fn=collate)}
    for src in sorted({it[4] for it in real_test.items}):
        idx = [i for i, it in enumerate(real_test.items) if it[4] == src]
        loaders[f"real_test_{src}"] = DataLoader(Subset(real_test, idx), batch_size=64, collate_fn=collate)
    if synth_val is not None:
        loaders["synth_val"] = DataLoader(synth_val, batch_size=64, collate_fn=collate)

    model = CRNN(len(v), hidden=a.hidden).to(device)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=device))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * a.steps_per_epoch)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    log = []
    print(f"device={device} vocab={len(v)} real_train={len(real_train)} real_test={len(real_test)} "
          f"synth={sum(len(p) for p in parts[:1]) if a.synth else 0}", flush=True)
    for ep in range(1, a.epochs + 1):
        t0, tot = time.time(), 0.0
        for X, targets, lengths, widths, _ in tl:
            X = X.to(device)
            if a.gpu_aug > 0:
                X = augment_batch(X, widths.to(device) * 4, p=a.gpu_aug, strength=a.gpu_aug_strength)
            logp = model(X)
            loss = ctc(logp, targets.to(device), widths.to(device), lengths.to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item()
        rec = {"epoch": ep, "loss": tot / a.steps_per_epoch, "sec": round(time.time() - t0)}
        for k, L in loaders.items():
            m = evaluate(model, L, v, device, keep=8 if k == "real_test" else 0)
            rec[k] = {kk: vv for kk, vv in m.items() if kk != "samples"}
            if m["samples"]:
                rec["real_test_samples"] = m["samples"]
        log.append(rec)
        print(json.dumps({k: rec[k] for k in rec if k != "real_test_samples"}), flush=True)
        torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
        json.dump({"args": vars(a), "vocab": v, "log": log}, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
