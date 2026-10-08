"""Datasets for the Chinuk Pipa recognizer: synthetic renders and the real ground-truth word crops."""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from chinukpipa.htr.normalize import normalize, rescale, stretch_contrast, stroke_width
from chinukpipa.split import split_of, word_key  # noqa: F401  (re-exported)
from chinukpipa.translit import load_tokens

HEIGHT = 64


def vocab() -> list[str]:
    """Recognizer alphabet: every token in tokens.yaml except line/syllable markup. The CTC blank is not in this
    list; encode() shifts token indices by one so that class 0 is the blank."""
    return [t for t in load_tokens() if t not in ("|", "=")]


_IDX: dict[tuple, dict] = {}


def encode(tokens: list[str], v: list[str]) -> list[int]:
    key = tuple(v)
    if key not in _IDX:
        _IDX[key] = {t: i + 1 for i, t in enumerate(v)}
    idx = _IDX[key]
    return [idx[t] for t in tokens]


def to_tensor(im: Image.Image) -> torch.Tensor:
    a = 1.0 - np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(a)[None]


def augment(im: Image.Image, rng: random.Random) -> Image.Image:
    """Gentle geometric augmentation only (small shear/rotation/scale)."""
    w, h = im.size
    sh = rng.uniform(-0.15, 0.15)
    im = im.transform(im.size, Image.AFFINE, (1, sh, -sh * h / 2, 0, 1, 0), resample=Image.BILINEAR, fillcolor=255)
    im = im.rotate(rng.uniform(-2, 2), resample=Image.BILINEAR, fillcolor=255)
    s = rng.uniform(0.9, 1.1)
    return im.resize((max(8, int(w * s)), h), Image.BILINEAR)


def _load_dir_cached(d: str, paths: list[str]) -> list[np.ndarray]:
    """Decode every image of a synthetic directory once and keep a packed copy (<dir>/cache_u8.npz) so later runs
    load in seconds instead of minutes."""
    cp = os.path.join(d, "cache_u8.npz")
    lab = os.path.join(d, "labels.tsv")
    if os.path.exists(cp) and os.path.getmtime(cp) >= os.path.getmtime(lab):
        try:
            z = np.load(cp)
            flat, shapes = z["flat"], z["shapes"]
            if len(shapes) == len(paths) and int((shapes[:, 0] * shapes[:, 1]).sum()) == len(flat):
                out, o = [], 0
                for h, w in shapes:
                    out.append(flat[o:o + h * w].reshape(h, w))
                    o += h * w
                return out
        except Exception:  # damaged or partial cache: rebuild it
            pass
    arrs = [np.asarray(Image.open(p).convert("L"), dtype=np.uint8) for p in paths]
    np.savez(cp, flat=np.concatenate([a.ravel() for a in arrs]), shapes=np.array([a.shape for a in arrs]))
    return arrs


def scale_jitter(rng: random.Random, spread: float) -> float:
    """Log-uniform multiplier in [1/(1+spread), 1+spread]."""
    import math
    return math.exp(rng.uniform(-math.log1p(spread), math.log1p(spread)))


class SynthDataset(Dataset):
    """Synthetic renders listed in <dir>/labels.tsv. With cache=True all images are decoded once into memory
    (uint8), which keeps a GPU busy without DataLoader workers. scale_aug > 0 rescales each sample by a random
    log-uniform factor in [1/(1+scale_aug), 1+scale_aug]. `exclude(tokens, latin)` drops samples, e.g. renderings
    of held-out test words."""

    def __init__(self, dirs: list[str], v: list[str], limit: int | None = None, cache: bool = True,
                 scale_aug: float = 0.0, seed: int = 0, exclude=None):
        self.items, self.arrays = [], [] if cache else None
        self.excluded = 0
        for d in dirs:
            items = []
            with open(os.path.join(d, "labels.tsv"), encoding="utf-8") as f:
                for line in f:
                    fn, toks, src = line.rstrip("\n").split("\t")
                    items.append((os.path.join(d, "images", fn), toks.split(), src))
            arrays = _load_dir_cached(d, [it[0] for it in items]) if cache else None
            keep = [k for k, it in enumerate(items) if not (exclude and exclude(it[1], it[2]))]
            self.excluded += len(items) - len(keep)
            self.items += [items[k] for k in keep]
            if cache:
                self.arrays += [arrays[k] for k in keep]
        if limit:
            self.items = self.items[:limit]
            if cache:
                self.arrays = self.arrays[:limit]
        self.v, self.scale_aug, self.rng = v, scale_aug, random.Random(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, toks = self.items[i][:2]
        a = self.arrays[i] if self.arrays is not None else np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
        if self.scale_aug:
            a = rescale(a, scale_jitter(self.rng, self.scale_aug))
        x = torch.from_numpy(1.0 - a.astype(np.float32) / 255.0)[None]
        return x, encode(toks, self.v), " ".join(toks)


def parse_scales(spec: str | None) -> dict[str, float]:
    """'LJ1892=0.5,LJ1898=1.0' or a JSON file path -> {source: multiplier}."""
    if not spec:
        return {}
    if os.path.exists(spec):
        return {k: float(x) for k, x in json.load(open(spec, encoding="utf-8")).items()}
    return {k: float(x) for k, x in (kv.split("=") for kv in spec.split(","))}


class RealDataset(Dataset):
    """Real crops from ground-truth JSONL files (crops regenerated with chinukpipa.gt.make_crops).

    Labels are `tokens_verified` when present, else `tokens_rule` (rule-derived: a hypothesis).
    `scales` maps a source id (e.g. LJ1892) to a calibration multiplier passed to normalize(). Rows with
    `copy_of` (the same writing in another copy of the edition, see chinukpipa/gt/register_copy.py) are scaled
    exactly like their original, and are left out of the test split unless copies_in_test=True."""

    def __init__(self, jsonl, crops_dir: str, v: list[str], split: str | None = None,
                 augment_p: float = 0.0, seed: int = 0, scales: dict[str, float] | None = None,
                 scale_aug: float = 0.0, copies_in_test: bool = False, split_seed: str = "v0"):
        self.v, self.augment_p, self.rng, self.scale_aug = v, augment_p, random.Random(seed), scale_aug
        scales = scales or {}
        paths = [jsonl] if isinstance(jsonl, str) else list(jsonl)
        self.items = []
        for path in paths:
            if not os.path.exists(path):
                continue
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                toks = r.get("tokens_verified") or r.get("tokens_rule")
                if r.get("language") != "chn" or not toks:
                    continue
                if split and split_of(r["latin"], seed=split_seed) != split:
                    continue
                if split == "test" and r.get("copy_of") and not copies_in_test:
                    continue
                if any(t not in v for t in toks.split()):
                    continue
                im = Image.open(os.path.join(crops_dir, f"{r['id']}_shorthand.png"))
                sw = None
                if r.get("copy_of"):
                    orig = os.path.join(crops_dir, f"{r['copy_of']}_shorthand.png")
                    if os.path.exists(orig):
                        sw = stroke_width(np.asarray(Image.open(orig).convert("L"))) or None
                im = normalize(stretch_contrast(im), height=HEIGHT, sw=sw, scale=scales.get(r["source"], 1.0))
                self.items.append((im, toks.split(), r["id"], r["latin"], r["source"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        im, toks = self.items[i][:2]
        if self.augment_p and self.rng.random() < self.augment_p:
            im = augment(im, self.rng)
            if self.scale_aug:
                im = Image.fromarray(rescale(np.asarray(im.convert("L")), scale_jitter(self.rng, self.scale_aug)))
        return to_tensor(im), encode(toks, self.v), " ".join(toks)


def collate(batch):
    xs, ys, txt = zip(*batch)
    W = max(x.shape[-1] for x in xs)
    W = (W + 3) // 4 * 4
    X = torch.zeros(len(xs), 1, HEIGHT, W)
    for i, x in enumerate(xs):
        X[i, :, :, : x.shape[-1]] = x
    widths = torch.tensor([max(1, x.shape[-1] // 4) for x in xs])
    targets = torch.tensor([k for y in ys for k in y], dtype=torch.long)
    lengths = torch.tensor([len(y) for y in ys])
    return X, targets, lengths, widths, list(txt)
