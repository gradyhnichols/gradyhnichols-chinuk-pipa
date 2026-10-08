"""Checks of the recognizer's data pipeline (skipped where PyTorch is not installed)."""
import os
import pickle
import random

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from chinukpipa.htr.data import BucketBatchSampler, SynthDataset, vocab  # noqa: E402


def _make_synth(d, n=30):
    os.makedirs(os.path.join(d, "images"))
    v = vocab()
    rng = random.Random(1)
    with open(os.path.join(d, "labels.tsv"), "w", encoding="utf-8") as f:
        for k in range(n):
            w = rng.randint(20, 120)
            a = np.full((64, w), 255, np.uint8)
            a[20:40, 4:w - 4] = k                 # each image is identifiable by its ink value
            Image.fromarray(a).save(os.path.join(d, "images", f"{k}.png"))
            f.write(f"{k}.png\t{v[k % len(v)]}\tw{k}\n")


def test_memory_mapped_cache_round_trip(tmp_path):
    d = str(tmp_path / "s")
    _make_synth(d)
    v = vocab()
    ds = SynthDataset([d], v, exclude=lambda toks, latin: latin == "w3")
    assert len(ds) == 29 and ds.excluded == 1
    clone = pickle.loads(pickle.dumps(ds))            # what a DataLoader worker process receives
    assert all(m is None for m in clone._flat)
    for i in (0, 3, 28):
        x, y, _ = clone[i]
        k = int(ds.items[i][2][1:])
        assert x.shape[-1] == ds.widths()[i]
        assert abs(float(x.max()) - (1 - k / 255)) < 1e-6
    ds2 = SynthDataset([d], v)                         # second load uses the cache files
    assert len(ds2) == 30 and os.path.exists(os.path.join(d, "cache_u8.flat.npy"))


def test_bucket_sampler_keeps_distribution_and_groups_widths():
    w = np.array([0.7, 0.1, 0.1, 0.1])
    widths = np.array([10, 20, 30, 40])
    s = BucketBatchSampler(w, widths, batch_size=8, num_batches=500, pool=16, seed=0)
    batches = list(s)
    assert len(batches) == 500 and all(len(b) == 8 for b in batches)
    counts = np.bincount(np.concatenate(batches), minlength=4) / (500 * 8)
    assert abs(counts[0] - 0.7) < 0.03
    spread = np.mean([np.ptp(widths[b]) for b in batches])
    assert spread < 15          # unsorted batches of this mix usually span 20-30


def test_sampler_drawn_scales_reach_the_samples(tmp_path):
    from chinukpipa.htr.data import MixDataset
    d = str(tmp_path / "s")
    _make_synth(d, n=12)
    ds = SynthDataset([d], vocab(), scale_aug=0.3)
    mix = MixDataset([(ds, list(range(2, 12))), (ds, None)])
    assert len(mix) == 22
    s = BucketBatchSampler(np.ones(len(mix)), mix.widths(), batch_size=4, num_batches=20, pool=5, seed=1,
                           scale_aug=0.3)
    for batch in s:
        for i, m, aug in batch:
            assert 1 / 1.3 - 1e-9 <= m <= 1.3 + 1e-9
            x, _, _ = mix[(i, m, aug)]
            assert x.shape[1] == 64
    x1, y1, _ = mix[3]                       # plain integer keys still work (index 3 of part 0 = item 5)
    assert y1 == SynthDataset([d], vocab())[5][1]


def test_width_model_predicts_rescaled_widths(tmp_path):
    from chinukpipa.htr.data import ink_box, rescale, rescaled_width
    d = str(tmp_path / "s")
    _make_synth(d, n=10)
    ds = SynthDataset([d], vocab(), scale_aug=0.3)
    for i in range(len(ds)):
        a = ds._array(i)
        bw, bh = ink_box(a)
        for m in (0.77, 0.9, 1.1, 1.3):
            assert rescale(a, m).shape[1] == rescaled_width(np.array(bw), np.array(bh), m)


def test_uint8_batches_equal_float_batches(tmp_path):
    from chinukpipa.htr.data import collate
    from chinukpipa.htr.train import as_float
    d = str(tmp_path / "s")
    _make_synth(d, n=6)
    f = SynthDataset([d], vocab())
    q = SynthDataset([d], vocab(), u8=True)
    Xf = collate([f[i] for i in range(6)])[0]
    Xq = collate([q[i] for i in range(6)])[0]
    assert Xq.dtype == torch.uint8
    assert torch.allclose(as_float(Xq), Xf, atol=1e-6)
