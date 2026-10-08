"""Datasets for the Chinuk Pipa recognizer: synthetic renders and the real ground-truth word crops."""
from __future__ import annotations

import json
import math
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, Sampler, Subset, get_worker_info

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


def ink_u8(a: np.ndarray) -> torch.Tensor:
    """Ink as uint8 (255 - gray), shape (1, H, W): a quarter of the bytes of the float input, for DataLoader
    workers to hand over; the trainer divides by 255 on the GPU, giving exactly to_tensor()'s values."""
    return torch.from_numpy(255 - np.ascontiguousarray(a, dtype=np.uint8))[None]


def augment(im: Image.Image, rng: random.Random) -> Image.Image:
    """Gentle geometric augmentation only (small shear/rotation/scale)."""
    w, h = im.size
    sh = rng.uniform(-0.15, 0.15)
    im = im.transform(im.size, Image.AFFINE, (1, sh, -sh * h / 2, 0, 1, 0), resample=Image.BILINEAR, fillcolor=255)
    im = im.rotate(rng.uniform(-2, 2), resample=Image.BILINEAR, fillcolor=255)
    s = rng.uniform(0.9, 1.1)
    return im.resize((max(8, int(w * s)), h), Image.BILINEAR)


def _cache_paths(d: str) -> tuple[str, str]:
    return os.path.join(d, "cache_u8.flat.npy"), os.path.join(d, "cache_u8.shapes.npy")


def _atomic_save(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def _build_dir_cache(d: str, paths: list[str]) -> None:
    """Decode every image of a synthetic directory once into two .npy files: all pixels as one flat uint8 array,
    and each image's (height, width). An older single-file cache (cache_u8.npz) is converted instead of
    re-decoding the images."""
    flat = shapes = None
    old, lab = os.path.join(d, "cache_u8.npz"), os.path.join(d, "labels.tsv")
    if os.path.exists(old) and os.path.getmtime(old) >= os.path.getmtime(lab):
        try:
            z = np.load(old)
            flat, shapes = z["flat"], z["shapes"]
        except Exception:  # damaged or partial: decode again
            flat = None
    if flat is None or len(shapes) != len(paths) or int((shapes[:, 0] * shapes[:, 1]).sum()) != len(flat):
        arrs = [np.asarray(Image.open(p).convert("L"), dtype=np.uint8) for p in paths]
        flat = np.concatenate([a.ravel() for a in arrs])
        shapes = np.array([a.shape for a in arrs])
    fp, sp = _cache_paths(d)
    _atomic_save(fp, np.ascontiguousarray(flat, dtype=np.uint8))
    _atomic_save(sp, np.asarray(shapes, dtype=np.int64))


def _open_dir_cache(d: str, paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(flat pixels as a read-only memory map, shapes) for a synthetic directory, building the cache if needed.
    A memory map is shared by all processes that open it (through the OS file cache), so DataLoader worker
    processes do not each need their own copy of the images."""
    fp, sp = _cache_paths(d)
    lab = os.path.join(d, "labels.tsv")
    for attempt in (0, 1):
        if all(os.path.exists(p) and os.path.getmtime(p) >= os.path.getmtime(lab) for p in (fp, sp)):
            try:
                shapes = np.load(sp)
                flat = np.load(fp, mmap_mode="r")
                if len(shapes) == len(paths) and int((shapes[:, 0] * shapes[:, 1]).sum()) == len(flat):
                    return flat, shapes
            except Exception:
                pass
        if attempt == 0:
            _build_dir_cache(d, paths)
    raise RuntimeError(f"could not build the image cache of {d}")


def ink_box(a: np.ndarray) -> tuple[int, int]:
    """(width, height) of the region rescale() crops to: the ink bounding box plus its 1-pixel margins."""
    ink = a < 128
    cols, rows = np.flatnonzero(ink.any(axis=0)), np.flatnonzero(ink.any(axis=1))
    if not len(cols):
        return a.shape[1], a.shape[0]
    H, W = a.shape
    return int(min(cols[-1] + 2, W) - max(0, cols[0] - 1)), int(min(rows[-1] + 2, H) - max(0, rows[0] - 1))


def rescaled_width(box_w, box_h, mult, height: int = 64, max_w: int = 640):
    """Width of rescale(a, mult) for an image whose ink box is (box_w, box_h); vectorized over numpy arrays."""
    w, h = np.round(box_w * mult), np.round(box_h * mult)
    w = np.where(h > height - 4, np.round(w * (height - 4) / np.maximum(h, 1)), w)
    return np.maximum(np.minimum(w, max_w) + 8, 16)


def scale_jitter(rng: random.Random, spread: float) -> float:
    """Log-uniform multiplier in [1/(1+spread), 1+spread]."""
    import math
    return math.exp(rng.uniform(-math.log1p(spread), math.log1p(spread)))


class SynthDataset(Dataset):
    """Synthetic renders listed in <dir>/labels.tsv. With cache=True every image is decoded once into a packed
    uint8 cache next to the images and read through a memory map (see _open_dir_cache). scale_aug > 0 rescales
    each sample by a random log-uniform factor in [1/(1+scale_aug), 1+scale_aug]. `exclude(tokens, latin)` drops
    samples, e.g. renderings of held-out test words."""

    def __init__(self, dirs: list[str], v: list[str], limit: int | None = None, cache: bool = True,
                 scale_aug: float = 0.0, seed: int = 0, exclude=None, u8: bool = False):
        self.dirs, self.cache, self.u8 = list(dirs), cache, u8
        self.items, self.excluded = [], 0
        loc, self._shapes, self._offsets = [], [], []
        for k, d in enumerate(self.dirs):
            items = []
            with open(os.path.join(d, "labels.tsv"), encoding="utf-8") as f:
                for line in f:
                    fn, toks, src = line.rstrip("\n").split("\t")
                    items.append((os.path.join(d, "images", fn), toks.split(), src))
            if cache:
                _, shapes = _open_dir_cache(d, [it[0] for it in items])
                self._shapes.append(shapes)
                self._offsets.append(np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])]))
            keep = [j for j, it in enumerate(items) if not (exclude and exclude(it[1], it[2]))]
            self.excluded += len(items) - len(keep)
            self.items += [items[j] for j in keep]
            loc += [(k, j) for j in keep]
        if limit:
            self.items, loc = self.items[:limit], loc[:limit]
        self._loc = np.array(loc, dtype=np.int64).reshape(-1, 2)
        self._flat = [None] * len(self.dirs)   # memory maps, opened on first use in each process
        self.v, self.scale_aug, self.rng = v, scale_aug, random.Random(seed)

    def __getstate__(self):   # DataLoader workers reopen the memory maps instead of receiving a copy of them
        st = dict(self.__dict__)
        st["_flat"] = [None] * len(self.dirs)
        return st

    def __len__(self):
        return len(self.items)

    def widths(self) -> np.ndarray | None:
        """Stored image widths (before any scale augmentation), for grouping samples of similar width."""
        if not self.cache:
            return None
        return np.array([self._shapes[k][j][1] for k, j in self._loc], dtype=np.int64)

    def width_model(self):
        """(stored width, ink box width, ink box height, probability that a sample is rescaled) per sample, so a
        batch sampler can predict each sample's width for a given scale factor."""
        if not self.cache:
            return None
        box = np.array([ink_box(self._array(i)) for i in range(len(self))], dtype=np.float64).reshape(-1, 2)
        p = np.full(len(self), 1.0 if self.scale_aug else 0.0)
        return self.widths().astype(np.float64), box[:, 0], box[:, 1], p

    def _array(self, i: int) -> np.ndarray:
        if not self.cache:
            return np.asarray(Image.open(self.items[i][0]).convert("L"), dtype=np.uint8)
        k, j = self._loc[i]
        if self._flat[k] is None:
            self._flat[k] = np.load(_cache_paths(self.dirs[k])[0], mmap_mode="r")
        h, w = self._shapes[k][j]
        o = self._offsets[k][j]
        return np.asarray(self._flat[k][o:o + h * w]).reshape(h, w)

    def get(self, i: int, scale: float | None = None, aug: bool | None = None):
        """Sample i; with scale_aug, rescaled by `scale` (drawn here when not given, e.g. by the batch sampler).
        `aug` is accepted for symmetry with RealDataset.get and ignored."""
        toks = self.items[i][1]
        a = self._array(i)
        if self.scale_aug:
            a = rescale(a, scale if scale is not None else scale_jitter(self.rng, self.scale_aug))
        x = ink_u8(a) if self.u8 else torch.from_numpy(1.0 - a.astype(np.float32) / 255.0)[None]
        return x, encode(toks, self.v), " ".join(toks)

    def __getitem__(self, i):
        return self.get(i)


def parse_scales(spec: str | None) -> dict[str, float]:
    """'LJ1892=0.5,LJ1898=1.0' or a JSON file path -> {source: multiplier}."""
    if not spec:
        return {}
    if os.path.exists(spec):
        return {k: float(x) for k, x in json.load(open(spec, encoding="utf-8")).items()}
    return {k: float(x) for k, x in (kv.split("=") for kv in spec.split(","))}


_CROP_CACHE: dict[tuple, Image.Image] = {}


def _normalized_crop(crops_dir: str, rid: str, copy_of: str | None, scale: float) -> Image.Image:
    """A ground-truth crop, contrast-stretched and scale-normalized (memoized: a run builds several datasets
    from the same crops). A copy (`copy_of`) is scaled with its original's stroke width."""
    key = (os.path.abspath(crops_dir), rid, copy_of, scale)
    if key not in _CROP_CACHE:
        im = Image.open(os.path.join(crops_dir, f"{rid}_shorthand.png"))
        sw = None
        if copy_of:
            orig = os.path.join(crops_dir, f"{copy_of}_shorthand.png")
            if os.path.exists(orig):
                sw = stroke_width(np.asarray(Image.open(orig).convert("L"))) or None
        _CROP_CACHE[key] = normalize(stretch_contrast(im), height=HEIGHT, sw=sw, scale=scale)
    return _CROP_CACHE[key]


class RealDataset(Dataset):
    """Real crops from ground-truth JSONL files (crops regenerated with chinukpipa.gt.make_crops).

    Labels are `tokens_verified` when present, else `tokens_rule` (rule-derived: a hypothesis).
    `scales` maps a source id (e.g. LJ1892) to a calibration multiplier passed to normalize(). Rows with
    `copy_of` (the same writing in another copy of the edition, see chinukpipa/gt/register_copy.py) are scaled
    exactly like their original, and are left out of the test split unless copies_in_test=True."""

    def __init__(self, jsonl, crops_dir: str, v: list[str], split: str | None = None,
                 augment_p: float = 0.0, seed: int = 0, scales: dict[str, float] | None = None,
                 scale_aug: float = 0.0, copies_in_test: bool = False, split_seed: str = "v0", u8: bool = False):
        self.v, self.augment_p, self.rng, self.scale_aug = v, augment_p, random.Random(seed), scale_aug
        self.u8 = u8
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
                im = _normalized_crop(crops_dir, r["id"], r.get("copy_of"), scales.get(r["source"], 1.0))
                self.items.append((im, toks.split(), r["id"], r["latin"], r["source"]))

    def __len__(self):
        return len(self.items)

    def widths(self) -> np.ndarray:
        return np.array([it[0].width for it in self.items], dtype=np.int64)

    def width_model(self):
        """See SynthDataset.width_model; a real word is rescaled only when it is augmented (augment() also
        resizes it by up to 10%, which the model ignores)."""
        box = np.array([ink_box(np.asarray(it[0].convert("L"))) for it in self.items], dtype=np.float64).reshape(-1, 2)
        p = np.full(len(self), self.augment_p if self.scale_aug else 0.0)
        return self.widths().astype(np.float64), box[:, 0], box[:, 1], p

    def get(self, i: int, scale: float | None = None, aug: bool | None = None):
        """Sample i. When it is augmented (probability augment_p, or as `aug` says) and scale_aug is set, it is
        also rescaled, by `scale` if given or a fresh random factor. A batch sampler can choose `aug` and `scale`
        in advance (BucketBatchSampler does, to predict widths)."""
        im, toks = self.items[i][:2]
        if aug is None:
            aug = bool(self.augment_p) and self.rng.random() < self.augment_p
        if aug:
            im = augment(im, self.rng)
            if self.scale_aug:
                m = scale if scale is not None else scale_jitter(self.rng, self.scale_aug)
                im = Image.fromarray(rescale(np.asarray(im.convert("L")), m))
        x = ink_u8(np.asarray(im.convert("L"))) if self.u8 else to_tensor(im)
        return x, encode(toks, self.v), " ".join(toks)

    def __getitem__(self, i):
        return self.get(i)


def collate(batch, pad_to: int = 4):
    """Pad a batch to a common width (a multiple of `pad_to`, itself a multiple of 4: the model reads one column
    per 4 pixels). A coarser `pad_to` gives the GPU fewer distinct shapes, at the cost of some extra padding."""
    xs, ys, txt = zip(*batch)
    W = max(x.shape[-1] for x in xs)
    W = (W + pad_to - 1) // pad_to * pad_to
    X = torch.zeros(len(xs), 1, HEIGHT, W, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        X[i, :, :, : x.shape[-1]] = x
    widths = torch.tensor([max(1, x.shape[-1] // 4) for x in xs])
    targets = torch.tensor([k for y in ys for k in y], dtype=torch.long)
    lengths = torch.tensor([len(y) for y in ys])
    return X, targets, lengths, widths, list(txt)


class MixDataset(Dataset):
    """Concatenation of datasets (each optionally restricted to some indices) whose items can also be requested
    as (index, scale) pairs, so that a batch sampler can choose each sample's scale augmentation in advance."""

    def __init__(self, parts):
        self.datasets = [d for d, _ in parts]
        self.indices = [None if ix is None else np.asarray(ix, dtype=np.int64) for _, ix in parts]
        sizes = [len(d) if ix is None else len(ix) for d, ix in zip(self.datasets, self.indices)]
        self.cum = np.cumsum([0] + sizes)

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, key):
        i, scale, aug = (tuple(key) + (None, None))[:3] if isinstance(key, tuple) else (key, None, None)
        p = int(np.searchsorted(self.cum, i, side="right")) - 1
        j = i - int(self.cum[p])
        if self.indices[p] is not None:
            j = int(self.indices[p][j])
        d = self.datasets[p]
        return d.get(j, scale, aug) if hasattr(d, "get") else d[j]

    def widths(self) -> np.ndarray | None:
        out = []
        for d, ix in zip(self.datasets, self.indices):
            w = dataset_widths(d)
            if w is None:
                return None
            out.append(w if ix is None else w[ix])
        return np.concatenate(out)

    def width_model(self):
        cols = []
        for d, ix in zip(self.datasets, self.indices):
            m = d.width_model() if hasattr(d, "width_model") else None
            if m is None:
                return None
            cols.append([c if ix is None else c[ix] for c in m])
        return tuple(np.concatenate(c) for c in zip(*cols))


def dataset_widths(ds) -> np.ndarray | None:
    """Per-sample widths of a dataset built from SynthDataset / RealDataset with Subset and ConcatDataset."""
    if isinstance(ds, Subset):
        w = dataset_widths(ds.dataset)
        return None if w is None else w[np.asarray(ds.indices, dtype=np.int64)]
    if isinstance(ds, ConcatDataset):
        parts = [dataset_widths(d) for d in ds.datasets]
        return None if any(p is None for p in parts) else np.concatenate(parts)
    return ds.widths() if hasattr(ds, "widths") else None


class BucketBatchSampler(Sampler):
    """Weighted random batches of similar width. Draws `pool` batches' worth of indices at a time, with
    replacement and with the given weights (the same distribution as WeightedRandomSampler), sorts them by width,
    cuts them into batches and shuffles the batch order. Less padding means less wasted computation; the samples
    drawn are unchanged, only how they are grouped into batches."""

    def __init__(self, weights, widths, batch_size: int, num_batches: int, pool: int = 64, seed: int = 0,
                 scale_aug: float = 0.0, width_model=None):
        """`widths`: each sample's width. With scale_aug > 0 the sampler also draws each sample's scale factor and
        passes it to the dataset as (index, factor); `width_model` (stored width, ink box width, ink box height,
        probability of being rescaled; see SynthDataset.width_model) then predicts the rescaled widths."""
        self.w = torch.as_tensor(weights, dtype=torch.double)
        self.widths = np.asarray(widths, dtype=np.float64)
        self.model = width_model
        self.bs, self.n, self.pool = batch_size, num_batches, max(1, pool)
        self.seed, self.epoch, self.scale_aug = seed, 0, scale_aug

    def predicted_widths(self, idx: np.ndarray, m: np.ndarray, aug: np.ndarray) -> np.ndarray:
        if self.model is None:
            return self.widths[idx] * m
        base, bw, bh, _ = (c[idx] for c in self.model)
        return np.where(aug, rescaled_width(bw, bh, m), base)

    def __len__(self):
        return self.n

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 100003 + self.epoch)
        self.epoch += 1
        left = self.n
        while left > 0:
            nb = min(self.pool, left)
            idx = torch.multinomial(self.w, nb * self.bs, replacement=True, generator=g).numpy()
            if self.scale_aug:   # draw each sample's rescaling here, so that it counts in the grouping
                r = math.log1p(self.scale_aug)
                m = np.exp((torch.rand(len(idx), generator=g, dtype=torch.float64).numpy() * 2 - 1) * r)
                u = torch.rand(len(idx), generator=g, dtype=torch.float64).numpy()
                aug = u < (self.model[3][idx] if self.model is not None else 1.0)
                order = np.argsort(self.predicted_widths(idx, m, aug), kind="stable").reshape(nb, self.bs)
                for b in torch.randperm(nb, generator=g).tolist():
                    yield [(int(idx[k]), float(m[k]), bool(aug[k])) for k in order[b]]
            else:
                idx = idx[np.argsort(self.widths[idx], kind="stable")].reshape(nb, self.bs)
                for b in torch.randperm(nb, generator=g).tolist():
                    yield idx[b].tolist()
            left -= nb


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: give each worker's copy of the datasets its own random stream (otherwise every
    worker would repeat the same augmentation sequence)."""
    info = get_worker_info()
    s = torch.initial_seed() % 2 ** 32       # differs per worker
    random.seed(s)
    np.random.seed(s)
    stack, seen = [info.dataset], set()
    while stack:
        d = stack.pop()
        if id(d) in seen:
            continue
        seen.add(id(d))
        if isinstance(getattr(d, "rng", None), random.Random):
            d.rng.seed(s * 1000 + len(seen))
        stack += list(getattr(d, "datasets", [])) + ([d.dataset] if isinstance(d, Subset) else [])
