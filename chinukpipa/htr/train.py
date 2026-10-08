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

Speed options (--workers, --bucket, --pad-to, --precision, --channels-last, --cudnn-benchmark) change how fast
a run goes, not what it learns from; some of them (batch grouping, precision) can still shift results within
run-to-run noise, so check accuracy when changing them (chinukpipa/htr/bench.py measures the speed).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from functools import partial

import torch
from torch.utils.data import BatchSampler, DataLoader, Subset, WeightedRandomSampler

from chinukpipa.htr.data import (BucketBatchSampler, MixDataset, RealDataset, SynthDataset, collate,
                                 dataset_widths, parse_scales, seed_worker, split_of, vocab)
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
        hyp = greedy_decode(model(as_float(X.to(device))))
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


def expand_dirs(specs: list[str]) -> list[str]:
    """Expand wildcards ourselves (the Windows shell does not)."""
    out = []
    for s in specs:
        out += sorted(glob.glob(s)) if any(c in s for c in "*?[") else [s]
    return out


def as_float(X: torch.Tensor) -> torch.Tensor:
    """uint8 ink batches (see data.ink_u8) -> the model's float input in [0, 1]."""
    return X.float().div_(255.0) if X.dtype == torch.uint8 else X


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", nargs="*", default=[], help="synthetic directories (wildcards allowed: synth*/part*)")
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
    ap.add_argument("--eval-every", type=int, default=1, help="evaluate every N epochs (and always after the last)")
    g = ap.add_argument_group("speed")
    g.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (0 = prepare batches in the "
                   "training process)")
    g.add_argument("--bucket", type=int, default=0, help="group each N batches' worth of samples by width (0 = off)")
    g.add_argument("--pad-to", type=int, default=4, help="pad each batch's width to a multiple of this (of 4)")
    g.add_argument("--precision", choices=["fp32", "fp16", "bf16"], help="CUDA autocast precision (default fp32)")
    g.add_argument("--amp", action="store_true", help="same as --precision fp16")
    g.add_argument("--channels-last", action="store_true")
    g.add_argument("--cudnn-benchmark", action="store_true", help="let cuDNN time its algorithms per input shape")
    return ap


def build_data(a, v, device):
    """Datasets and loaders for a run: (training loader, {name: evaluation loader}, info)."""
    scales = parse_scales(a.real_scales)
    ss = a.split_seed
    tr_split = None if a.all_real else "train"
    real_train = RealDataset(a.real, a.crops, v, split=tr_split, augment_p=0.7, seed=a.seed, scales=scales,
                             scale_aug=a.scale_aug, split_seed=ss, u8=True)
    real_train_eval = RealDataset(a.real, a.crops, v, split="train", scales=scales, split_seed=ss)  # for metrics
    real_test = RealDataset(a.real, a.crops, v, split="test", scales=scales, split_seed=ss)
    test_seqs = {" ".join(it[1]) for it in real_test.items}

    def is_test_word(toks, latin):
        if a.all_real:
            return False
        return " ".join(toks) in test_seqs or (not latin.startswith("~") and split_of(latin, seed=ss) == "test")
    parts, weights = [], []
    synth_val = None
    info = {"real_train": len(real_train), "real_test": len(real_test), "synth": 0, "synth_excluded": 0}
    a.synth = expand_dirs(a.synth)
    if a.synth:
        synth = SynthDataset(a.synth, v, scale_aug=a.scale_aug, seed=a.seed, exclude=is_test_word, u8=True)
        info["synth_excluded"] = synth.excluded
        idx = list(range(len(synth)))
        random.Random(a.seed).shuffle(idx)
        nval = min(1000, len(idx) // 50)
        synth_val = Subset(synth, idx[:nval])     # evaluate() turns uint8 batches into floats too
        tr_idx = idx[nval:]
        parts.append((synth, tr_idx))
        weights += [(1 - a.real_train_weight) / len(tr_idx)] * len(tr_idx)
        info["synth"] = len(tr_idx)
    if a.real_train_weight > 0:
        parts.append((real_train, None))
        weights += [a.real_train_weight / len(real_train)] * len(real_train)
    if not parts:
        raise SystemExit("nothing to train on: give --synth and/or --real-train-weight > 0")
    if a.pad_to % 4:
        raise SystemExit("--pad-to must be a multiple of 4")
    train = MixDataset(parts)
    widths = dataset_widths(train) if a.bucket else None
    if a.bucket and widths is not None:
        bsampler = BucketBatchSampler(weights, widths, a.bs, a.steps_per_epoch, pool=a.bucket, seed=a.seed,
                                      scale_aug=a.scale_aug, width_model=train.width_model() if a.scale_aug else None)
    else:
        bsampler = BatchSampler(WeightedRandomSampler(weights, num_samples=a.steps_per_epoch * a.bs,
                                                      replacement=True), a.bs, drop_last=False)
    kw = {"num_workers": a.workers, "pin_memory": device == "cuda"}
    if a.workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4, worker_init_fn=seed_worker)
    tl = DataLoader(train, batch_sampler=bsampler, collate_fn=partial(collate, pad_to=a.pad_to), **kw)
    # evaluation batches are sorted by width: a model trained on width-bucketed batches rarely sees long runs of
    # padding, and reading such padding (in the backward LSTM) degrades its output; sorted batches have little padding
    def by_width(ds, idx=None):
        idx = list(range(len(ds))) if idx is None else list(idx)
        w = dataset_widths(ds)
        if w is not None:
            idx = sorted(idx, key=lambda i: w[i])
        return DataLoader(Subset(ds, idx), batch_size=16, collate_fn=collate)
    loaders = {"real_test": by_width(real_test), "real_train": by_width(real_train_eval)}
    for src in sorted({it[4] for it in real_test.items}):
        loaders[f"real_test_{src}"] = by_width(real_test, [i for i, it in enumerate(real_test.items) if it[4] == src])
    if synth_val is not None:
        loaders["synth_val"] = by_width(synth_val)
    return tl, loaders, info


def resolve_precision(a, device) -> str:
    p = a.precision or ("fp16" if a.amp else "fp32")
    if device != "cuda":
        return "fp32"
    if p == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("this GPU does not support bf16; use --precision fp16")
    return p


class Trainer:
    """One optimisation step per call. Returns the loss as a GPU tensor, so the CPU never waits for the GPU
    inside the loop (reading a value back with .item() each step would stall the pipeline)."""

    def __init__(self, a, v, device, total_steps):
        self.device, self.a = device, a
        self.precision = resolve_precision(a, device)
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.precision)
        self.model = CRNN(len(v), hidden=a.hidden).to(device)
        if a.init:
            self.model.load_state_dict(torch.load(a.init, map_location=device))
        if a.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=a.lr, weight_decay=1e-4, fused=device == "cuda")
        self.sched = torch.optim.lr_scheduler.OneCycleLR(self.opt, max_lr=a.lr, total_steps=total_steps)
        self.ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        # loss scaling is needed for fp16 only; with a fused optimizer it does not force a sync either
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.precision == "fp16")

    def __call__(self, X, targets, lengths, widths):
        dev, a = self.device, self.a
        X = as_float(X.to(dev, non_blocking=True))
        if a.channels_last:
            X = X.contiguous(memory_format=torch.channels_last)
        widths_d = widths.to(dev, non_blocking=True)
        if a.gpu_aug > 0:
            X = augment_batch(X, widths_d * 4, p=a.gpu_aug, strength=a.gpu_aug_strength)
        with torch.autocast(device_type="cuda", dtype=self.dtype or torch.float32, enabled=self.dtype is not None):
            logp = self.model(X)
        loss = self.ctc(logp.float(), targets.to(dev, non_blocking=True), widths_d, lengths.to(dev, non_blocking=True))
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        return loss.detach()


def main():
    a = build_parser().parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = a.cudnn_benchmark
    os.makedirs(a.out, exist_ok=True)
    v = vocab()
    tl, loaders, info = build_data(a, v, device)
    if a.synth:
        print(f"synthetic samples of held-out test words dropped: {info['synth_excluded']}", flush=True)
    tr = Trainer(a, v, device, a.epochs * a.steps_per_epoch)
    log = []
    print(f"device={device} precision={tr.precision} vocab={len(v)} real_train={info['real_train']} "
          f"real_test={info['real_test']} synth={info['synth']} workers={a.workers} bucket={a.bucket}", flush=True)
    for ep in range(1, a.epochs + 1):
        t0, n = time.time(), 0
        tot = torch.zeros((), device=device)
        for X, targets, lengths, widths, _ in tl:
            tot += tr(X, targets, lengths, widths)
            n += 1
        rec = {"epoch": ep, "loss": float(tot) / max(1, n)}   # float() waits for the GPU to finish the epoch
        rec["sec"] = round(time.time() - t0)
        if ep % max(1, a.eval_every) == 0 or ep == a.epochs:
            t1 = time.time()
            for k, L in loaders.items():
                m = evaluate(tr.model, L, v, device, keep=8 if k == "real_test" else 0)
                rec[k] = {kk: vv for kk, vv in m.items() if kk != "samples"}
                if m["samples"]:
                    rec["real_test_samples"] = m["samples"]
            rec["eval_sec"] = round(time.time() - t1)
        log.append(rec)
        print(json.dumps({k: rec[k] for k in rec if k != "real_test_samples"}), flush=True)
        torch.save(tr.model.state_dict(), os.path.join(a.out, "model.pt"))
        with open(os.path.join(a.out, "metrics.json"), "w") as f:
            json.dump({"args": vars(a), "vocab": v, "log": log}, f, indent=1)


def exit_now(code: int = 0) -> None:
    """End the process at once, skipping interpreter and library teardown. On one Windows machine (torch 2.14 + CUDA
    12.6), processes that ran fp16 autocast crash while unloading libraries (exit code 0xC0000409) after all
    their work is done, even with os._exit; outputs are written and closed before this is called."""
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p          # a HANDLE is pointer-sized: without these
        k.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]   # ctypes passes a 32-bit int
        k.TerminateProcess(k.GetCurrentProcess(), code)
    os._exit(code)


if __name__ == "__main__":
    main()
    if os.name == "nt":
        exit_now(0)
