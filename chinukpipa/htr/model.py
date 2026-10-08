"""A small CRNN (convolutional + bidirectional LSTM) trained with CTC to read Chinuk Pipa word images.

Input: batch of grayscale images, ink = 1, background = 0, shape (B, 1, 64, W).
Output: per-column log-probabilities over the token vocabulary plus a CTC blank (index 0), shape (T, B, C+1),
with T = W // 4.
"""
from __future__ import annotations

import torch
from torch import nn


def _block(cin, cout, pool):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(pool))
    return layers


class CRNN(nn.Module):
    def __init__(self, n_classes: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            *_block(1, 32, (2, 2)),      # 32 x W/2
            *_block(32, 64, (2, 2)),     # 16 x W/4
            *_block(64, 128, None),
            *_block(128, 128, (2, 1)),   # 8 x W/4
            *_block(128, 256, None),
            *_block(256, 256, (2, 1)),   # 4 x W/4
        )
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(256 * 4, hidden, num_layers=2, bidirectional=True, batch_first=False, dropout=dropout)
        self.out = nn.Linear(2 * hidden, n_classes + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x)                       # B, 256, 4, T
        b, c, h, t = f.shape
        f = f.permute(3, 0, 1, 2).reshape(t, b, c * h)
        f, _ = self.rnn(self.drop(f))
        return self.out(self.drop(f)).log_softmax(-1)   # T, B, C+1


def greedy_decode(logp: torch.Tensor) -> list[list[int]]:
    """Best-path CTC decoding: argmax per column, collapse repeats, drop blanks (0)."""
    best = logp.argmax(-1).transpose(0, 1).tolist()   # B, T
    out = []
    for seq in best:
        res, prev = [], 0
        for k in seq:
            if k != prev and k != 0:
                res.append(k)
            prev = k
        out.append(res)
    return out
