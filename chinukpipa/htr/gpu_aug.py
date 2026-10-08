"""Batch augmentation on the GPU (or CPU): smooth elastic warping, small rotation/shear, pen-width change, blur
and noise, applied per sample to a padded batch of word images (ink = 1, background = 0).

Synthetic words come from a fixed set of rendered images; warping them freshly at every step stops the network
from memorising those images and imitates the irregularity of handwriting.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _gauss_kernel(sigma: float, device) -> torch.Tensor:
    r = max(1, int(math.ceil(2 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-x * x / (2 * sigma * sigma))
    return k / k.sum()


def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = _gauss_kernel(sigma, x.device)
    r = (len(k) - 1) // 2
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))


def augment_batch(X: torch.Tensor, widths_px: torch.Tensor, p: float = 0.8, strength: float = 1.0,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """X: (B,1,H,W) float in [0,1]; widths_px: (B,) content width of each sample in pixels.
    Each sample is augmented with probability p; strength scales every perturbation."""
    B, _, H, W = X.shape
    dev = X.device
    g = generator

    def U(*shape, lo=-1.0, hi=1.0):
        return torch.rand(*shape, device=dev, generator=g) * (hi - lo) + lo

    on = (torch.rand(B, device=dev, generator=g) < p).float()
    s = strength * on                                                      # per-sample strength, 0 = untouched

    # --- geometry: rotation + shear about each word's centre, plus smooth elastic displacement (pixels)
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    cx = (widths_px.to(dev).float() / 2).view(B, 1, 1)
    cy = H / 2
    th = (U(B) * 0.05 * s).view(B, 1, 1)                                    # about +-3 degrees
    sh = (U(B) * 0.20 * s).view(B, 1, 1)                                    # slant
    dx, dy = xs[None] - cx, ys[None] - cy
    sx = torch.cos(th) * dx - torch.sin(th) * dy + sh * dy + cx
    sy = torch.sin(th) * dx + torch.cos(th) * dy + cy
    gh, gw = max(2, H // 12), max(2, W // 12)
    disp = torch.randn(B, 2, gh, gw, device=dev, generator=g) * (1.6 * s).view(B, 1, 1, 1)
    disp = F.interpolate(disp, size=(H, W), mode="bicubic", align_corners=True)
    sx = sx + disp[:, 0]
    sy = sy + disp[:, 1]
    grid = torch.stack([2 * sx / max(1, W - 1) - 1, 2 * sy / max(1, H - 1) - 1], dim=-1)
    Y = F.grid_sample(X, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    # --- pen width: blur then a soft threshold; a lower threshold gives a thicker pen, a higher one a thinner pen
    Yb = _blur(Y, 0.8)
    t = (0.47 + U(B) * 0.15 * s).view(B, 1, 1, 1)
    Y = torch.sigmoid((Yb - t) * 14)

    # --- print/scan texture: slight blur and noise on a random subset
    blur_on = (torch.rand(B, device=dev, generator=g) < 0.3 * on).float().view(B, 1, 1, 1)
    Y = blur_on * _blur(Y, 0.7) + (1 - blur_on) * Y
    Y = Y + torch.randn(Y.shape, device=dev, generator=g) * (0.06 * s).view(B, 1, 1, 1)
    Y = Y.clamp(0, 1)

    keep = on.view(B, 1, 1, 1)
    return keep * Y + (1 - keep) * X
