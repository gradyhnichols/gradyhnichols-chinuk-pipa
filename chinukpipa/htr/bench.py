"""Measure how fast this machine trains the recognizer, and where the time goes.

    python -m chinukpipa.htr.bench <train.py arguments> [--bench-steps 300] [--bench-warmup 40]

Three numbers, in samples per second:
  data  - how fast the DataLoader alone delivers batches (no GPU work);
  gpu   - how fast the GPU trains on batches that are already prepared (cycling over a fixed set of real batches);
  e2e   - both together, as in a real run.
If e2e is close to data, the input pipeline is the bottleneck; if close to gpu, the GPU is. While e2e runs, GPU
utilization, power and SM clock are sampled with nvidia-smi (when present), and the CPU used by the training process itself is
reported, in cores (data preparation in DataLoader workers is not included).
Prints one JSON line. The model trained here is thrown away; --out is not needed.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time

import torch

from chinukpipa.htr.data import vocab
from chinukpipa.htr.train import Trainer, build_data, build_parser, exit_now


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


class GpuSampler:
    def __init__(self):
        self.p = None
        exe = shutil.which("nvidia-smi")
        if exe:
            self.p = subprocess.Popen([exe, "--query-gpu=utilization.gpu,power.draw,clocks.sm,memory.used",
                                       "--format=csv,noheader,nounits", "-lms", "200"],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def stop(self) -> dict:
        if not self.p:
            return {}
        self.p.terminate()
        out = self.p.communicate(timeout=10)[0]
        rows = []
        for line in out.splitlines():
            try:
                rows.append([float(x) for x in line.split(",")])
            except ValueError:
                pass
        rows = rows[2:] or rows      # skip the first samples (start-up)
        if not rows:
            return {}
        mean = [sum(c) / len(rows) for c in zip(*rows)]
        return {"gpu_util": round(mean[0], 1), "gpu_watts": round(mean[1], 1), "sm_mhz": round(mean[2]),
                "gpu_mem_mb": round(max(r[3] for r in rows)), "gpu_samples": len(rows)}


def main():
    ap = build_parser()
    ap.add_argument("--bench-steps", type=int, default=300)
    ap.add_argument("--bench-warmup", type=int, default=40)
    ap.add_argument("--bench-modes", default="data,gpu,e2e")
    ap.add_argument("--label", default="")
    argv = sys.argv[1:]
    if "--out" not in argv:
        argv += ["--out", "."]
    a = ap.parse_args(argv)
    modes = a.bench_modes.split(",")
    N, W = a.bench_steps, a.bench_warmup
    a.epochs = 1
    a.steps_per_epoch = W + 3 * N + 80
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = a.cudnn_benchmark
    v = vocab()
    t = time.time()
    tl, _, info = build_data(a, v, device)
    res = {"label": a.label, "bs": a.bs, "workers": a.workers, "bucket": a.bucket, "pad_to": a.pad_to,
           "channels_last": a.channels_last, "cudnn_benchmark": a.cudnn_benchmark, "setup_s": round(time.time() - t, 1)}
    tr = Trainer(a, v, device, a.steps_per_epoch)
    res["precision"] = tr.precision
    it = iter(tl)
    t = time.time()
    first = next(it)
    res["first_batch_s"] = round(time.time() - t, 1)
    pad = []
    for _ in range(W):                    # warm up workers, cuDNN and the allocator
        X, targets, lengths, widths, _ = next(it)
        tr(X, targets, lengths, widths)
        pad.append(1 - float(widths.sum()) * 4 / (X.shape[0] * X.shape[-1]))
    _sync(device)
    res["padding_share"] = round(sum(pad) / len(pad), 3)
    if "data" in modes:
        t = time.time()
        for _ in range(N):
            next(it)
        res["data_sps"] = round(N * a.bs / (time.time() - t))
    if "gpu" in modes:
        held = [next(it) for _ in range(min(N, 40))]
        if device == "cuda":
            held = [tuple(x.pin_memory() if torch.is_tensor(x) else x for x in b) for b in held]
        _sync(device)
        t = time.time()
        for k in range(N):
            X, targets, lengths, widths, _ = held[k % len(held)]
            tr(X, targets, lengths, widths)
        _sync(device)
        res["gpu_sps"] = round(N * a.bs / (time.time() - t))
    if "e2e" in modes:
        smi = GpuSampler() if device == "cuda" else None
        t, c = time.time(), time.process_time()
        for _ in range(N):
            X, targets, lengths, widths, _ = next(it)
            tr(X, targets, lengths, widths)
        _sync(device)
        dt = time.time() - t
        res["e2e_sps"] = round(N * a.bs / dt)
        res["main_cpu_cores"] = round((time.process_time() - c) / dt, 2)
        if smi:
            res.update(smi.stop())
    if device == "cuda":
        res["peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    del first
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
    if os.name == "nt":
        exit_now(0)
