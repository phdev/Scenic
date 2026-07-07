"""Deterministic image + coverage metrics shared by gates, S8 and tools.

Everything here is pure numpy/scipy (no torch, no RNG, no wall-clock) so it is
bit-reproducible across runs. SSIM is the ENFORCED fidelity metric. LPIPS is
ADVISORY ONLY: its VGG/ImageNet weight provenance is an OPEN license question
(see weights/LICENSES.md), so the weights are never provisioned into the
enforced weights/ tree and are never downloaded at runtime — lpips_advisory()
returns unavailable unless a human has placed local weights outside the
enforced tree and set SCENIC_LPIPS_DIR. This keeps the license guard and the
no-network guard green with no new enforced weight.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy import ndimage

# SSIM (Wang et al. 2004) constants for dynamic range L = 1.0.
_C1 = (0.01) ** 2
_C2 = (0.03) ** 2
_SSIM_SIGMA = 1.5
_SSIM_TRUNCATE = 3.0  # -> radius 5 -> 11-tap window, classic 11x11 Gaussian


def to_gray01(img: np.ndarray) -> np.ndarray:
    """uint8 or float image -> float64 grayscale in [0,1]. uint8 is scaled by
    1/255; float is assumed already in [0,1] and clipped."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        a = a.astype(np.float64) / 255.0
    else:
        a = np.clip(a.astype(np.float64), 0.0, 1.0)
    if a.ndim == 3:
        a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return a


def _gauss(x: np.ndarray) -> np.ndarray:
    return ndimage.gaussian_filter(
        x, sigma=_SSIM_SIGMA, truncate=_SSIM_TRUNCATE, mode="reflect"
    )


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel SSIM map between two same-shape images (any dtype accepted)."""
    x = to_gray01(a)
    y = to_gray01(b)
    if x.shape != y.shape:
        raise ValueError(f"ssim shape mismatch {x.shape} vs {y.shape}")
    mx, my = _gauss(x), _gauss(y)
    mxx, myy, mxy = _gauss(x * x), _gauss(y * y), _gauss(x * y)
    vx = mxx - mx * mx
    vy = myy - my * my
    vxy = mxy - mx * my
    num = (2 * mx * my + _C1) * (2 * vxy + _C2)
    den = (mx * mx + my * my + _C1) * (vx + vy + _C2)
    return num / den


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM in [-1, 1] (deterministic)."""
    return float(np.mean(ssim_map(a, b)))


def solid_angle_fraction(mask: np.ndarray) -> float:
    """Fraction of the full sphere's solid angle covered by a boolean equirect
    mask. Equirect pixel solid angle is proportional to cos(latitude)."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    lat = np.pi / 2 - (np.arange(h, dtype=np.float64) + 0.5) / h * np.pi
    wrow = np.cos(lat)
    num = float((mask.astype(np.float64) * wrow[:, None]).sum())
    den = float(wrow.sum()) * w
    return num / max(den, 1e-12)


def equirect_tile_views(
    tiles_lon: int, tiles_lat: int, fov_pad_deg: float = 6.0
) -> list[tuple[str, float, float, float]]:
    """Deterministic perspective view grid tiling the sphere for the
    fidelity_at_origin gate. Returns (name, yaw_deg, pitch_deg, fov_deg) at
    tile centers; fov = tile angular size + fov_pad_deg (slight overlap).
    Latitude tiles are centered inside (-90, 90) bands; longitude wraps."""
    views: list[tuple[str, float, float, float]] = []
    dlon = 360.0 / tiles_lon
    dlat = 180.0 / tiles_lat
    fov = float(max(dlon, dlat) + fov_pad_deg)
    for j in range(tiles_lat):
        pitch = 90.0 - (j + 0.5) * dlat
        for i in range(tiles_lon):
            yaw = -180.0 + (i + 0.5) * dlon
            views.append((f"tile_{j}_{i}", float(yaw), float(pitch), fov))
    return views


def lpips_advisory_available() -> tuple[bool, str]:
    """Whether advisory LPIPS can run. False by default: VGG/ImageNet weight
    provenance is an OPEN license question, so weights are never in the
    enforced tree. A human may set SCENIC_LPIPS_DIR to a local weight dir
    (outside weights/) AND install the optional `lpips` package to enable it;
    absent that, the fidelity gate reports lpips as advisory_unavailable."""
    d = os.environ.get("SCENIC_LPIPS_DIR", "")
    if not d or not Path(d).is_dir():
        return False, "SCENIC_LPIPS_DIR unset (LPIPS weight provenance OPEN)"
    try:
        import lpips  # noqa: F401
    except Exception:
        return False, "optional lpips package not installed"
    return True, "advisory LPIPS available"
