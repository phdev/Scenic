"""Shared spherical / camera geometry. Conventions (see docs/CONTRACTS.md):
right-handed, +Y up, theta=0 -> +Z. Equirect: lon = (u+0.5)/W*2pi - pi,
lat = pi/2 - (v+0.5)/H*pi. dir = [cos(lat)sin(lon), sin(lat), cos(lat)cos(lon)].
All internal math float64; callers cast artifacts to float32."""
from __future__ import annotations

import numpy as np

CUBE_FACES: tuple[tuple[str, float, float], ...] = (
    ("front", 0.0, 0.0),
    ("right", np.pi / 2, 0.0),
    ("back", np.pi, 0.0),
    ("left", -np.pi / 2, 0.0),
    ("up", 0.0, np.pi / 2),
    ("down", 0.0, -np.pi / 2),
)


def equirect_lonlat(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    u = (np.arange(w, dtype=np.float64) + 0.5) / w * 2 * np.pi - np.pi
    v = np.pi / 2 - (np.arange(h, dtype=np.float64) + 0.5) / h * np.pi
    return np.meshgrid(u, v)


def lonlat_to_dirs(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    cl = np.cos(lat)
    return np.stack([cl * np.sin(lon), np.sin(lat), cl * np.cos(lon)], axis=-1)


def equirect_dirs(w: int, h: int) -> np.ndarray:
    lon, lat = equirect_lonlat(w, h)
    return lonlat_to_dirs(lon, lat)


def dirs_to_uv(dirs: np.ndarray, w: int, h: int) -> np.ndarray:
    """Float pixel coords (u right, v down); u in [-0.5, w-0.5) wrap domain."""
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    lon = np.arctan2(x, z)
    lat = np.arcsin(np.clip(y / np.maximum(np.linalg.norm(dirs, axis=-1), 1e-12), -1, 1))
    u = (lon + np.pi) / (2 * np.pi) * w - 0.5
    v = (np.pi / 2 - lat) / np.pi * h - 0.5
    return np.stack([u, v], axis=-1)


def pitch_of_dirs(dirs: np.ndarray) -> np.ndarray:
    n = np.maximum(np.linalg.norm(dirs, axis=-1), 1e-12)
    return np.arcsin(np.clip(dirs[..., 1] / n, -1, 1))


def rotation_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    """world_dir = R @ cam_dir; cam looks +Z, x right, y up. Positive yaw turns
    toward +X (east); positive pitch looks up."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rp = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]], dtype=np.float64)
    return ry @ rp


def camera_grid(fov_deg: float, w: int, h: int) -> np.ndarray:
    """Unnormalized cam-space dirs for each pixel (z=1 plane), x right y up."""
    f = (w / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    xs = (np.arange(w, dtype=np.float64) + 0.5 - w / 2.0) / f
    ys = (h / 2.0 - (np.arange(h, dtype=np.float64) + 0.5)) / f
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx, gy, np.ones_like(gx)], axis=-1)


def perspective_dirs(
    fov_deg: float, w: int, h: int, yaw: float, pitch: float
) -> np.ndarray:
    cam = camera_grid(fov_deg, w, h)
    cam = cam / np.linalg.norm(cam, axis=-1, keepdims=True)
    r = rotation_yaw_pitch(yaw, pitch)
    return cam @ r.T


def sample_equirect(img: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Bilinear sample; lon wraps, lat clamps. img HxW or HxWxC float."""
    single = img.ndim == 2
    if single:
        img = img[..., None]
    h, w, c = img.shape
    uv = dirs_to_uv(dirs, w, h)
    u, v = uv[..., 0], uv[..., 1]
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    fu = (u - u0)[..., None]
    fv = (v - v0)[..., None]
    u0w, u1w = u0 % w, (u0 + 1) % w
    v0c = np.clip(v0, 0, h - 1)
    v1c = np.clip(v0 + 1, 0, h - 1)
    p00 = img[v0c, u0w]
    p01 = img[v0c, u1w]
    p10 = img[v1c, u0w]
    p11 = img[v1c, u1w]
    out = (
        p00 * (1 - fu) * (1 - fv)
        + p01 * fu * (1 - fv)
        + p10 * (1 - fu) * fv
        + p11 * fu * fv
    )
    return out[..., 0] if single else out


def render_perspective(
    img: np.ndarray, fov_deg: float, w: int, h: int, yaw: float, pitch: float
) -> np.ndarray:
    return sample_equirect(img, perspective_dirs(fov_deg, w, h, yaw, pitch))


def face_project(
    dirs: np.ndarray, yaw: float, pitch: float, fov_deg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world dirs into a perspective face.
    Returns (uv01 (...,2) in [0,1], in_frustum bool, center_cos)."""
    r = rotation_yaw_pitch(yaw, pitch)
    cam = dirs @ r  # R^T @ dir per row
    z = cam[..., 2]
    t = np.tan(np.deg2rad(fov_deg) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = cam[..., 0] / np.where(z > 1e-9, z, np.nan)
        y = cam[..., 1] / np.where(z > 1e-9, z, np.nan)
    u01 = (x / t + 1) / 2
    v01 = (1 - y / t) / 2
    inside = (z > 1e-9) & (u01 >= 0) & (u01 <= 1) & (v01 >= 0) & (v01 <= 1)
    center_cos = np.where(z > 0, z / np.linalg.norm(cam, axis=-1), 0.0)
    uv = np.stack([np.nan_to_num(u01), np.nan_to_num(v01)], axis=-1)
    return uv, inside, center_cos


def angular_pixel_size(h: int) -> float:
    return np.pi / h


def normals_from_depth(depth: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Per-pixel unit normal from 3D point grid (central differences),
    oriented to face the origin. The x axis is equirect longitude and WRAPS
    (columns 0 and w-1 difference across the seam); the y axis clamps (rows
    0 and h-1 fall back). Invalid depths propagate NaN-free: falls back to
    -dir (facing camera)."""
    finite = np.isfinite(depth)
    pts = dirs * np.where(finite, depth, 0.0)[..., None]
    dx = np.roll(pts, -1, axis=1) - np.roll(pts, 1, axis=1)
    dy = np.zeros_like(pts)
    dy[1:-1, :] = pts[2:, :] - pts[:-2, :]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    finite_nbrs = np.roll(finite, -1, axis=1) & np.roll(finite, 1, axis=1)
    finite_nbrs[1:-1, :] &= finite[2:, :] & finite[:-2, :]
    good = (norm[..., 0] > 1e-12) & finite & finite_nbrs
    n = np.where(good[..., None], n / np.maximum(norm, 1e-12), -dirs)
    flip = np.sum(n * dirs, axis=-1) > 0
    n = np.where(flip[..., None], -n, n)
    return n
