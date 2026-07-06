"""Deterministic image / array IO. PNG saves carry no metadata; loads ignore
EXIF/ICC (bytes-in decides pixels-out for a given pinned Pillow)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 512_000_000


def load_rgb(path: Path | str) -> np.ndarray:
    im = Image.open(path)
    im = im.convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def save_png(path: Path | str, arr: np.ndarray) -> None:
    if arr.dtype != np.uint8:
        raise TypeError(f"save_png expects uint8, got {arr.dtype}")
    im = Image.fromarray(arr)
    im.save(str(path), format="PNG", optimize=False)


def save_mask_png(path: Path | str, mask: np.ndarray) -> None:
    if mask.dtype != np.bool_:
        raise TypeError("mask must be bool")
    save_png(path, (mask.astype(np.uint8)) * 255)


def load_mask_png(path: Path | str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return arr >= 128


def save_npy(path: Path | str, arr: np.ndarray) -> None:
    with open(path, "wb") as f:
        np.save(f, arr)


def load_npy(path: Path | str) -> np.ndarray:
    return np.load(path)
