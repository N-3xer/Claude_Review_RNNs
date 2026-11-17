"""Tiny helpers shared by every stage. Read once, then forget it."""
import random

import numpy as np
import torch


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def header(title):
    print(f"\n{'=' * 6} {title} {'=' * 6}")


def check_shape(t, expected, name=""):
    assert tuple(t.shape) == tuple(expected), (
        f"{name}: got shape {tuple(t.shape)}, expected {tuple(expected)}"
    )
    print(f"[ok]   shape {name:<30} {tuple(t.shape)}")


def check_close(a, b, name="", atol=1e-5, rtol=1e-4):
    diff = (a - b).abs().max().item()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    print(f"[{'ok' if ok else 'FAIL'}]   {name:<44} max|diff| = {diff:.2e}")
    assert ok, f"{name}: tensors differ"
