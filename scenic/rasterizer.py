"""Deterministic CPU 3D-Gaussian-splat rasterizer (EWA splatting).

Used by the gates (s7) and review (s8) stages to render `SplatData` scenes.
Pure numpy, sequential compositing, float32 accumulation: identical inputs
produce bit-identical outputs.

Conventions (docs/CONTRACTS.md, scenic/geometry.py): right-handed, +Y up;
camera looks +Z, x right, y up; world_dir = rotation_yaw_pitch(yaw, pitch)
@ cam_dir. Pixel mapping matches geometry.camera_grid's +0.5-center
convention: f = (px_w/2)/tan(fov/2), u = f*x/z + px_w/2 - 0.5,
v = px_h/2 - 0.5 - f*y/z (u right, v down).

Algorithm (3DGS / EWA):
- cam-space mean p = R.T @ (xyz - pos); cull z <= 0.05.
- Sigma3d = R_q diag(exp(log_scales)^2) R_q^T; Sigma2d = J W Sigma3d W^T J^T
  + 0.3*I (dilation), J the perspective Jacobian at the mean. As in the
  reference 3DGS rasterizer, x/z and y/z are clamped to 1.3*tan(fov/2) when
  building J so grazing near-plane splats keep bounded footprints (without
  this, splats beside the camera in a 360 bubble smear across the screen and
  defeat the bbox cull).
- Cull splats whose 3-sigma axis-aligned bbox (radii 3*sqrt(diag(Sigma2d)),
  the exact AABB of the 3-sigma ellipse) misses the image.
- Stable front-to-back sort by (z, index); per-splat gaussian evaluated on
  the 3-sigma bbox patch; alpha_px = min(0.99, opacity * exp(-0.5 q)),
  contributions below 1/255 dropped; front-to-back "over" compositing with
  per-pixel transmittance T; splats whose patch max T < 1e-3 are skipped.
- depth = expected depth: sum(T*alpha*z) / (1-T_final), +inf where the final
  alpha < 1e-3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenic.geometry import rotation_yaw_pitch
from scenic.plyio import SplatData, dc_to_rgb01, logit_to_opacity

_Z_NEAR = 0.05           # cull splats at or behind this cam-space depth
_DILATION = 0.3          # 2D covariance dilation (3DGS anti-alias floor)
_ALPHA_MAX = 0.99        # per-pixel alpha cap
_ALPHA_MIN = 1.0 / 255.0  # per-pixel contribution cutoff
_T_SKIP = 1e-3           # transmittance early-out threshold
_ALPHA_BG = 1e-3         # below this final alpha, depth output is +inf
_DET_MIN = 1e-12         # degenerate 2D covariance cutoff
_J_CLAMP = 1.3           # Jacobian x/z, y/z clamp factor (3DGS reference)
_GLOBAL_CHECK_EVERY = 4096  # fixed cadence for the whole-image early-out


@dataclass
class Camera:
    """Pinhole camera: world position + yaw/pitch (radians, geometry conv)."""

    pos: np.ndarray  # (3,) world position
    yaw: float
    pitch: float


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(n,4) quats wxyz -> (n,3,3) rotation matrices (normalizes first)."""
    q = q.astype(np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def _finish(rgb_acc: np.ndarray, t: np.ndarray, depth_acc: np.ndarray) -> dict:
    alpha = (np.float32(1.0) - t).astype(np.float32)
    depth = (depth_acc / np.maximum(alpha, np.float32(1e-6))).astype(np.float32)
    depth[alpha < _ALPHA_BG] = np.inf
    rgb = np.clip(np.round(rgb_acc * np.float32(255.0)), 0, 255).astype(np.uint8)
    return {"rgb": rgb, "alpha": alpha, "depth": depth}


def render(
    splats: SplatData,
    cam: Camera,
    px_w: int,
    px_h: int,
    fov_deg: float,
    override_rgb: np.ndarray | None = None,
) -> dict:
    """Render splats through cam. Returns {"rgb": uint8 (h,w,3),
    "alpha": float32 (h,w), "depth": float32 (h,w) expected cam-space depth
    (+inf background)}. override_rgb: optional (n,3) float01 per-splat color
    replacing the f_dc colors (e.g. the magenta-shell hole gate)."""
    if px_w < 1 or px_h < 1:
        raise ValueError(f"invalid image size {px_w}x{px_h}")
    if not (0.0 < fov_deg < 180.0):
        raise ValueError(f"fov_deg must be in (0, 180), got {fov_deg}")
    n = len(splats)
    if override_rgb is not None:
        override_rgb = np.asarray(override_rgb, dtype=np.float64)
        if override_rgb.shape != (n, 3):
            raise ValueError(
                f"override_rgb shape {override_rgb.shape} != ({n}, 3)"
            )

    rgb_acc = np.zeros((px_h, px_w, 3), dtype=np.float32)
    depth_acc = np.zeros((px_h, px_w), dtype=np.float32)
    t = np.ones((px_h, px_w), dtype=np.float32)
    if n == 0:
        return _finish(rgb_acc, t, depth_acc)

    # --- project means to cam space (float64 math) -------------------------
    r_cam = rotation_yaw_pitch(float(cam.yaw), float(cam.pitch))  # world=R@cam
    pos = np.asarray(cam.pos, dtype=np.float64).reshape(3)
    p_cam = (splats.xyz.astype(np.float64) - pos[None, :]) @ r_cam  # R.T@(p-pos)
    idx = np.nonzero(p_cam[:, 2] > _Z_NEAR)[0]  # near-plane cull, keeps order
    if idx.size == 0:
        return _finish(rgb_acc, t, depth_acc)
    xc, yc, zc = p_cam[idx, 0], p_cam[idx, 1], p_cam[idx, 2]

    tan_x = np.tan(np.deg2rad(fov_deg) / 2.0)
    f = (px_w / 2.0) / tan_x
    u = f * xc / zc + px_w / 2.0 - 0.5
    v = px_h / 2.0 - 0.5 - f * yc / zc
    # Jacobian-only clamp of x/z, y/z (3DGS reference rasterizer): keeps
    # grazing near-plane splats from smearing across the whole image.
    lim_x = _J_CLAMP * tan_x
    lim_y = _J_CLAMP * tan_x * (px_h / px_w)
    xj = np.clip(xc / zc, -lim_x, lim_x) * zc
    yj = np.clip(yc / zc, -lim_y, lim_y) * zc

    # --- EWA: 2D covariance -------------------------------------------------
    s2 = np.exp(2.0 * splats.log_scales[idx].astype(np.float64))  # (m,3)
    r_q = _quat_to_rot(splats.quat_wxyz[idx])                     # (m,3,3)
    # A = W @ R_q with W = R.T; Sigma_cam = A diag(s2) A^T (= W Sigma3d W^T)
    a_rot = np.einsum("ji,njk->nik", r_cam, r_q)
    sig = np.einsum("nij,nj,nkj->nik", a_rot, s2, a_rot)
    s00, s01, s02 = sig[:, 0, 0], sig[:, 0, 1], sig[:, 0, 2]
    s11, s12, s22 = sig[:, 1, 1], sig[:, 1, 2], sig[:, 2, 2]
    # Perspective Jacobian rows: (f/z, 0, -f x/z^2), (0, -f/z, f y/z^2);
    # the y sign flip maps cam-up to v-down.
    j00 = f / zc
    j02 = -f * xj / (zc * zc)
    j11 = -f / zc
    j12 = f * yj / (zc * zc)
    cov_a = j00 * j00 * s00 + 2.0 * j00 * j02 * s02 + j02 * j02 * s22 + _DILATION
    cov_c = j11 * j11 * s11 + 2.0 * j11 * j12 * s12 + j12 * j12 * s22 + _DILATION
    cov_b = j00 * j11 * s01 + j00 * j12 * s02 + j02 * j11 * s12 + j02 * j12 * s22
    det = cov_a * cov_c - cov_b * cov_b

    # --- frustum cull: 3-sigma AABB vs image --------------------------------
    rx = 3.0 * np.sqrt(cov_a)
    ry = 3.0 * np.sqrt(cov_c)
    ok = (
        (det > _DET_MIN)
        & (u + rx >= 0.0)
        & (u - rx <= px_w - 1.0)
        & (v + ry >= 0.0)
        & (v - ry <= px_h - 1.0)
    )
    if not np.any(ok):
        return _finish(rgb_acc, t, depth_acc)

    # --- per-splat color -----------------------------------------------------
    if override_rgb is not None:
        col_all = np.clip(override_rgb[idx], 0.0, 1.0)
    else:
        col_all = dc_to_rgb01(splats.f_dc[idx].astype(np.float64))

    # --- stable front-to-back sort by (z, original index) --------------------
    # idx is ascending, so stable argsort on z ties out to original index.
    order = np.argsort(zc[ok], kind="stable")

    def _sel(arr: np.ndarray) -> np.ndarray:
        return arr[ok][order]

    u_s = _sel(u).astype(np.float32)
    v_s = _sel(v).astype(np.float32)
    z_s = _sel(zc).astype(np.float32)
    det_s = _sel(det)
    con_a = (_sel(cov_c) / det_s).astype(np.float32)   # dx^2 coeff
    con_b = (-_sel(cov_b) / det_s).astype(np.float32)  # dx*dy coeff (x2 below)
    con_c = (_sel(cov_a) / det_s).astype(np.float32)   # dy^2 coeff
    opac = logit_to_opacity(
        splats.opacity_logit[idx].astype(np.float64)
    )
    op_s = _sel(opac).astype(np.float32)
    col_s = _sel(col_all).astype(np.float32)
    x0 = np.clip(np.floor(_sel(u) - _sel(rx)), 0, px_w - 1).astype(np.int64)
    x1 = np.clip(np.ceil(_sel(u) + _sel(rx)), 0, px_w - 1).astype(np.int64)
    y0 = np.clip(np.floor(_sel(v) - _sel(ry)), 0, px_h - 1).astype(np.int64)
    y1 = np.clip(np.ceil(_sel(v) + _sel(ry)), 0, px_h - 1).astype(np.int64)
    x0l, x1l = x0.tolist(), x1.tolist()
    y0l, y1l = y0.tolist(), y1.tolist()

    # --- composite (front-to-back "over") ------------------------------------
    xs_pix = np.arange(px_w, dtype=np.float32)
    ys_pix = np.arange(px_h, dtype=np.float32)
    alpha_min = np.float32(_ALPHA_MIN)
    alpha_max = np.float32(_ALPHA_MAX)
    one = np.float32(1.0)
    m = int(z_s.shape[0])
    for k in range(m):
        if k and k % _GLOBAL_CHECK_EVERY == 0 and float(t.max()) < _T_SKIP:
            break  # image saturated everywhere; fixed cadence, deterministic
        xa, xb, ya, yb = x0l[k], x1l[k], y0l[k], y1l[k]
        t_patch = t[ya : yb + 1, xa : xb + 1]
        if float(t_patch.max()) < _T_SKIP:
            continue
        dx = xs_pix[xa : xb + 1] - u_s[k]
        dy = ys_pix[ya : yb + 1] - v_s[k]
        q = (
            (con_a[k] * dx * dx)[None, :]
            + (con_c[k] * dy * dy)[:, None]
            + (np.float32(2.0) * con_b[k]) * (dy[:, None] * dx[None, :])
        )
        alpha = op_s[k] * np.exp(np.float32(-0.5) * q)
        np.minimum(alpha, alpha_max, out=alpha)
        alpha = np.where(alpha >= alpha_min, alpha, np.float32(0.0))
        contrib = t_patch * alpha
        rgb_acc[ya : yb + 1, xa : xb + 1] += contrib[:, :, None] * col_s[k]
        depth_acc[ya : yb + 1, xa : xb + 1] += contrib * z_s[k]
        t[ya : yb + 1, xa : xb + 1] = t_patch * (one - alpha)

    return _finish(rgb_acc, t, depth_acc)
