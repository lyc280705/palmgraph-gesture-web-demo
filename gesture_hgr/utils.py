from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(z)
    return exp / np.clip(exp.sum(axis=axis, keepdims=True), 1e-8, None)


def configure_torch_threads(num_threads: int | None = None) -> int:
    if num_threads is None:
        num_threads = max(1, min(8, os.cpu_count() or 1))
    return max(1, int(num_threads))
