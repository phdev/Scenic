"""s4_place: shell-inward, density/scale-aware splat placement -> 3DGS PLY.

Reads the s3 layer outputs (fg/bg rgb + depth, bg_mask band, layers.json)
and the s2 sky mask, then places Gaussians on deterministic strided pixel
grids with v2 "shell-inward" routing by METRIC depth d (fg_depth is metric):

  fg layer  — three importance classes (priority edge > ground > base):
              'edge'   = within EDGE_BAND_PX of the s3 band (bg_mask; if the
                         band is unavailable it is recomputed from the
                         fg_depth log-gradient),
              'ground' = pitch below params.s4.ground_band_pitch_deg & finite,
              'base'   = everything else.
              Class K uses stride sK with fixed offsets (sK//2, sK//2).
              fg/bg splats are placed ONLY where a direction is finite, ~sky,
              and min_content_distance_m <= d <= shell_distance_m.
  bg layer  — bg_mask pixels whose bg_depth is inside the same [min_content,
              shell_distance] window; scale CLAMPED to the fg-equivalent radius
              at that depth (never inflated by the coarser bg stride).
  shell     — every direction that is sky OR non-finite OR d > shell_distance_m
              OR d < min_content_distance_m is pushed to a textured sphere at
              shell_radius_m (normals facing the camera). The shell ALSO covers
              every direction with d > (shell_distance_m - feather_m) so a
              backdrop always sits behind a fading fg splat (no hard seam).

Feather: fg splats with d in [shell_distance_m - feather_m, shell_distance_m]
have opacity multiplied by clip((shell_distance_m - d)/feather_m, 0, 1) BEFORE
the logit transform, ramping to 0 at the frontier.

Color-variance importance sampling: local color variance v (max over channels
of the variance in a color_var_window_px half-window box on fg_rgb01) yields a
per-pixel boost b = 1 + (color_var_boost - 1) * clip(v/color_var_ref, 0, 1).
A class-selected pixel is kept iff u(r,c) < b/color_var_boost, where u(r,c) =
(((r*73856093) ^ (c*19349663)) & 0xffffff)/0x1000000 is a closed-form
deterministic uniform in [0,1) (never RNG). Flat areas are decimated by up to
1/color_var_boost; textured areas are kept.

Splat math per selected pixel: pos = dir*depth, color -> DC coeffs, normal
from the depth grid, orientation quat from the frame R=[t1,t2,n] (columns),
radius = depth * angular_pixel_size(H) * class_stride * scale_multiplier,
log_scales = [ln r, ln r, ln(r*flatten_ratio)] (disc flattened along n).

run() accepts stride_multiplier (default 1.0); s6 re-invokes with larger
values to enforce the splat cap. This stage never retries on cap overflow —
it just records the count (and exceeds_cap) in splats_meta.json / the
receipt notes. Pure numpy: no torch, no RNG, no wall-clock.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from scenic import geometry, imageio, plyio, receipts, schema
from scenic.plyio import LAYER_BG, LAYER_FG, LAYER_SHELL, SplatData
from scenic.stage import Ctx

STAGE = "s4_place"
ORIGIN_STAGE = 4

EDGE_BAND_PX = 2                  # 'within 2px of the s3 band/edges'
BG_BASE_STRIDE = 2.0              # bg grid stride = max(1, round(2*mult))
SHELL_OPACITY = 0.995
DEFAULT_EDGE_LOG_GRAD_MIN = 0.30  # fallback if params.s3 is absent


# ------------------------------------------------------------- rotations


def rotmat_from_quat(q: np.ndarray) -> np.ndarray:
    """(n,4) wxyz -> (n,3,3) rotation matrices (verification helper)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
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


def quat_from_rotmats(m: np.ndarray) -> np.ndarray:
    """(n,3,3) rotation matrices -> (n,4) unit quats wxyz, canonical w>=0.

    Vectorized Shepperd: per row pick the largest of {4w^2,4x^2,4y^2,4z^2}
    as the pivot (their sum is exactly 4, so the pivot is >= 1 and the
    divisor S = 2*sqrt(pivot) >= 2 — no division hazard)."""
    m = np.asarray(m, dtype=np.float64)
    n = m.shape[0]
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    four = np.stack(
        [
            np.maximum(0.0, 1.0 + m00 + m11 + m22),  # 4w^2
            np.maximum(0.0, 1.0 + m00 - m11 - m22),  # 4x^2
            np.maximum(0.0, 1.0 - m00 + m11 - m22),  # 4y^2
            np.maximum(0.0, 1.0 - m00 - m11 + m22),  # 4z^2
        ],
        axis=0,
    )
    k = np.argmax(four, axis=0)
    q = np.zeros((n, 4), dtype=np.float64)
    for case in range(4):
        msk = k == case
        if not np.any(msk):
            continue
        s = 2.0 * np.sqrt(four[case][msk])
        if case == 0:
            q[msk, 0] = 0.25 * s
            q[msk, 1] = (m21[msk] - m12[msk]) / s
            q[msk, 2] = (m02[msk] - m20[msk]) / s
            q[msk, 3] = (m10[msk] - m01[msk]) / s
        elif case == 1:
            q[msk, 0] = (m21[msk] - m12[msk]) / s
            q[msk, 1] = 0.25 * s
            q[msk, 2] = (m01[msk] + m10[msk]) / s
            q[msk, 3] = (m02[msk] + m20[msk]) / s
        elif case == 2:
            q[msk, 0] = (m02[msk] - m20[msk]) / s
            q[msk, 1] = (m01[msk] + m10[msk]) / s
            q[msk, 2] = 0.25 * s
            q[msk, 3] = (m12[msk] + m21[msk]) / s
        else:
            q[msk, 0] = (m10[msk] - m01[msk]) / s
            q[msk, 1] = (m02[msk] + m20[msk]) / s
            q[msk, 2] = (m12[msk] + m21[msk]) / s
            q[msk, 3] = 0.25 * s
    return plyio.canonical_quat(q)


def frames_from_normals(n_vec: np.ndarray) -> np.ndarray:
    """(n,3) normals -> (n,3,3) rotations with columns [t1, t2, n].
    t1 = normalize(cross(a, n)), a = +Y unless |n.y| > 0.9 then +X;
    t2 = cross(n, t1). Proper orthonormal (det = +1) by construction."""
    n_vec = np.asarray(n_vec, dtype=np.float64)
    n_unit = n_vec / np.maximum(np.linalg.norm(n_vec, axis=-1, keepdims=True), 1e-12)
    use_x = np.abs(n_unit[:, 1]) > 0.9
    a = np.where(
        use_x[:, None],
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )
    t1 = np.cross(a, n_unit)
    t1 = t1 / np.maximum(np.linalg.norm(t1, axis=-1, keepdims=True), 1e-12)
    t2 = np.cross(n_unit, t1)
    return np.stack([t1, t2, n_unit], axis=-1)  # columns


# ------------------------------------------------------------- selection


def _stride(x: float) -> int:
    return max(1, int(round(x)))


def _grid_mask(h: int, w: int, stride: int) -> np.ndarray:
    """(r,c) selected iff r%s == s//2 and c%s == s//2 (fixed offsets)."""
    rows = (np.arange(h) % stride) == (stride // 2)
    cols = (np.arange(w) % stride) == (stride // 2)
    return rows[:, None] & cols[None, :]


def _dilate_chebyshev(mask: np.ndarray, radius: int) -> np.ndarray:
    """'within radius px' (Chebyshev). x wraps (equirect lon), y clamps."""
    h = mask.shape[0]
    out = np.zeros_like(mask)
    for du in range(-radius, radius + 1):
        m = np.roll(mask, du, axis=1)
        for dv in range(-radius, radius + 1):
            if dv == 0:
                out |= m
            elif dv > 0:
                out[dv:] |= m[: h - dv]
            else:
                out[: h + dv] |= m[-dv:]
    return out


def _log_grad_edges(depth: np.ndarray, thr: float) -> np.ndarray:
    """Fallback edge detection: |delta log depth| > thr against any 4-nbr
    (x wraps, y edge-pads). Non-finite neighbors never trigger (nan-safe)."""
    finite = np.isfinite(depth) & (depth > 0)
    with np.errstate(all="ignore"):
        lg = np.where(finite, np.log(np.where(finite, depth, 1.0)), np.nan)
        dxa = np.abs(lg - np.roll(lg, 1, axis=1))
        dxb = np.abs(lg - np.roll(lg, -1, axis=1))
        dya = np.zeros_like(lg)
        dyb = np.zeros_like(lg)
        dya[1:] = np.abs(lg[1:] - lg[:-1])
        dyb[:-1] = np.abs(lg[:-1] - lg[1:])
        edge = (dxa > thr) | (dxb > thr) | (dya > thr) | (dyb > thr)
    return edge & finite


# ------------------------------------------------------------- color variance


def _box_sum(x: np.ndarray, k: int) -> np.ndarray:
    """Sum over a (2k+1)x(2k+1) box per pixel; lon (axis 1) wraps, lat (axis 0)
    edge-clamps. Deterministic float64 integral image (constant box size)."""
    h, w = x.shape
    xp = np.pad(x, ((0, 0), (k, k)), mode="wrap")
    xp = np.pad(xp, ((k, k), (0, 0)), mode="edge")
    ii = np.zeros((xp.shape[0] + 1, xp.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(xp, axis=0), axis=1)
    win = 2 * k + 1
    return (
        ii[win : win + h, win : win + w]
        - ii[0:h, win : win + w]
        - ii[win : win + h, 0:w]
        + ii[0:h, 0:w]
    )


def _local_color_variance(rgb01: np.ndarray, k: int) -> np.ndarray:
    """Per-pixel local color variance = max over channels of the variance in a
    (2k+1)^2 box (constant sample count -> exact E[x^2]-E[x]^2)."""
    h, w, c = rgb01.shape
    if k <= 0:
        return np.zeros((h, w), dtype=np.float64)
    n = float((2 * k + 1) * (2 * k + 1))
    var = np.zeros((h, w), dtype=np.float64)
    for ch in range(c):
        x = rgb01[..., ch].astype(np.float64)
        mean = _box_sum(x, k) / n
        meansq = _box_sum(x * x, k) / n
        var = np.maximum(var, np.maximum(meansq - mean * mean, 0.0))
    return var


def _hash_uniform(h: int, w: int) -> np.ndarray:
    """Closed-form deterministic uniform in [0,1) per integer (r,c):
    u = (((r*73856093) ^ (c*19349663)) & 0xffffff) / 0x1000000."""
    r = np.arange(h, dtype=np.int64)[:, None]
    c = np.arange(w, dtype=np.int64)[None, :]
    hv = ((r * 73856093) ^ (c * 19349663)) & 0xFFFFFF
    return hv.astype(np.float64) / float(0x1000000)


# ------------------------------------------------------------- assembly


def _empty_part() -> SplatData:
    return SplatData(
        xyz=np.zeros((0, 3), np.float32),
        normals=np.zeros((0, 3), np.float32),
        f_dc=np.zeros((0, 3), np.float32),
        opacity_logit=np.zeros((0,), np.float32),
        log_scales=np.zeros((0, 3), np.float32),
        quat_wxyz=np.zeros((0, 4), np.float32),
        layer=np.zeros((0,), np.uint8),
        origin_stage=np.zeros((0,), np.uint8),
    )


def _make_part(
    pos: np.ndarray,
    normals: np.ndarray,
    rgb01: np.ndarray,
    radii: np.ndarray,
    opacity,
    flatten_ratio: float,
    layer: int,
) -> SplatData:
    n = int(pos.shape[0])
    if n == 0:
        return _empty_part()
    lr = np.log(radii)
    log_scales = np.stack([lr, lr, np.log(radii * flatten_ratio)], axis=1)
    quats = quat_from_rotmats(frames_from_normals(normals))
    opac_arr = np.broadcast_to(np.asarray(opacity, dtype=np.float64), (n,))
    opac = plyio.opacity_to_logit(opac_arr)
    return SplatData(
        xyz=pos.astype(np.float32),
        normals=normals.astype(np.float32),
        f_dc=plyio.rgb01_to_dc(rgb01).astype(np.float32),
        opacity_logit=opac.astype(np.float32),
        log_scales=log_scales.astype(np.float32),
        quat_wxyz=quats.astype(np.float32),
        layer=np.full(n, layer, np.uint8),
        origin_stage=np.full(n, ORIGIN_STAGE, np.uint8),
    )


# ------------------------------------------------------------- stage


def run(run_dir: Path, params: dict, ctx: Ctx, stride_multiplier: float = 1.0) -> None:
    run_dir = Path(run_dir)
    out = ctx.out(run_dir, STAGE)
    s3_out = run_dir / "s3_layers" / "out"
    s2_out = run_dir / "s2_depth" / "out"

    p4 = params["s4"]
    splat_cap = int(params["splat_cap"])
    min_content = float(params["min_content_distance_m"])
    shell_dist = float(p4["shell_distance_m"])
    feather_m = float(p4["feather_m"])
    shell_r = float(p4["shell_radius_m"])
    cv_boost = float(p4["color_var_boost"])
    cv_window = int(p4["color_var_window_px"])
    cv_ref = float(p4["color_var_ref"])
    edge_thr = float(
        params.get("s3", {}).get("edge_log_grad_min", DEFAULT_EDGE_LOG_GRAD_MIN)
    )
    mult = float(stride_multiplier)
    if not (math.isfinite(mult) and mult > 0):
        raise ValueError(f"stride_multiplier must be finite and > 0, got {mult}")
    if not (feather_m > 0):
        raise ValueError(f"feather_m must be > 0, got {feather_m}")
    if not (cv_boost >= 1.0):
        raise ValueError(f"color_var_boost must be >= 1, got {cv_boost}")

    # -- inputs (hard errors on missing files / shape mismatches)
    fg_rgb_path = s3_out / "fg_rgb.png"
    fg_depth_path = s3_out / "fg_depth.npy"
    fg_mask_path = s3_out / "fg_mask.png"
    bg_rgb_path = s3_out / "bg_rgb.png"
    bg_depth_path = s3_out / "bg_depth.npy"
    bg_mask_path = s3_out / "bg_mask.png"
    layers_path = s3_out / "layers.json"
    sky_path = s2_out / "sky_mask.png"

    fg_rgb = imageio.load_rgb(fg_rgb_path)
    fg_depth = imageio.load_npy(fg_depth_path).astype(np.float64)
    bg_rgb = imageio.load_rgb(bg_rgb_path)
    bg_depth = imageio.load_npy(bg_depth_path).astype(np.float64)
    sky = imageio.load_mask_png(sky_path)
    schema.read_validated(layers_path, "layers")  # provenance + hard validation
    if fg_depth.ndim != 2:
        raise ValueError(f"fg_depth must be HxW, got {fg_depth.shape}")
    h, w = fg_depth.shape
    for name, shp in [
        ("fg_rgb", fg_rgb.shape[:2]),
        ("bg_rgb", bg_rgb.shape[:2]),
        ("bg_depth", bg_depth.shape),
        ("sky_mask", sky.shape),
    ]:
        if tuple(shp) != (h, w):
            raise ValueError(f"{name} shape {shp} != depth shape {(h, w)}")

    have_bg_mask = bg_mask_path.exists()
    if have_bg_mask:
        bg_mask = imageio.load_mask_png(bg_mask_path)
        if bg_mask.shape != (h, w):
            raise ValueError(f"bg_mask shape {bg_mask.shape} != {(h, w)}")
    else:
        bg_mask = np.isfinite(bg_depth)  # adapt: contract lists no bg_mask.png

    # -- shared geometry (float64; artifacts cast to float32 at assembly)
    dirs = geometry.equirect_dirs(w, h)
    angpix = geometry.angular_pixel_size(h)
    pitch_deg = np.degrees(geometry.pitch_of_dirs(dirs))
    scale_mult = float(p4["scale_multiplier"])
    flatten = float(p4["flatten_ratio"])
    fg_rgb01 = fg_rgb.astype(np.float64) / 255.0
    with np.errstate(all="ignore"):
        normals_fg = geometry.normals_from_depth(fg_depth, dirs)
        normals_bg = geometry.normals_from_depth(bg_depth, dirs)

    # -- class strides
    base = float(p4["base_stride"])
    se = _stride(base * mult / math.sqrt(float(p4["edge_boost"])))
    sg = _stride(base * mult / math.sqrt(float(p4["ground_boost"])))
    sb = _stride(base * mult)
    s_bg = _stride(BG_BASE_STRIDE * mult)
    s_sh = _stride(float(p4["shell_stride"]) * mult)

    # -- shell-inward routing by METRIC depth d (fg_depth is metric)
    # Near content (d < min_content) STAYS as fg splats at true depth: it is
    # real geometry the viewer must see, so shelling it would read as a hole
    # below the skyline (and kill parallax). Its discomfort is flagged honestly
    # by the min_content_distance and stereo gates off the depth map, not
    # hidden. Only sky + FAR content (d > shell_distance) + non-finite route to
    # the textured shell.
    finite_fg = np.isfinite(fg_depth) & (fg_depth > 0)
    with np.errstate(invalid="ignore"):
        content_ok = finite_fg & ~sky & (fg_depth <= shell_dist)
        near_fg = content_ok & (fg_depth < min_content)
        too_far = finite_fg & ~sky & (fg_depth > shell_dist)
        # backdrop: shell also covers everything past (shell_dist - feather) so a
        # fading fg splat always has a solid shell behind it (no hard seam).
        backdrop = finite_fg & ~sky & (fg_depth > (shell_dist - feather_m))
    shell_route = sky | ~finite_fg | backdrop
    near_shell_px = 0  # near content is no longer shelled (kept as fg splats)
    near_fg_px = int(near_fg.sum())
    far_shell_px = int(too_far.sum())

    # -- fg importance classes (priority edge > ground > base; exclusive)
    if have_bg_mask and bool(bg_mask.any()):
        band = bg_mask
    else:
        band = _log_grad_edges(fg_depth, edge_thr)
    edge_c = _dilate_chebyshev(band, EDGE_BAND_PX)
    ground_c = ~edge_c & (pitch_deg < float(p4["ground_band_pitch_deg"])) & finite_fg
    base_c = ~edge_c & ~ground_c
    stride_map = np.where(edge_c, se, np.where(ground_c, sg, sb))
    grid_sel = (
        (edge_c & _grid_mask(h, w, se))
        | (ground_c & _grid_mask(h, w, sg))
        | (base_c & _grid_mask(h, w, sb))
    )

    # -- color-variance importance sampling (closed-form deterministic uniform)
    cvar = _local_color_variance(fg_rgb01, cv_window)
    boost = 1.0 + (cv_boost - 1.0) * np.clip(cvar / cv_ref, 0.0, 1.0)
    keep = _hash_uniform(h, w) < (boost / cv_boost)

    feather_zone = content_ok & (fg_depth >= (shell_dist - feather_m))
    feather_px = int(feather_zone.sum())

    # -- fg layer (routed content, color-var decimated, frontier feathered)
    sel_fg_pre = content_ok & grid_sel
    n_fg_pre = int(sel_fg_pre.sum())
    sel_fg = sel_fg_pre & keep
    rows, cols = np.nonzero(sel_fg)  # row-major deterministic order
    d = fg_depth[rows, cols]
    with np.errstate(invalid="ignore"):
        feather = np.clip((shell_dist - d) / feather_m, 0.0, 1.0)
    fg_opacity = np.full(rows.shape[0], float(p4["fg_opacity"]), dtype=np.float64)
    fg_opacity = fg_opacity * feather
    fg_part = _make_part(
        pos=dirs[rows, cols] * d[:, None],
        normals=normals_fg[rows, cols],
        rgb01=fg_rgb01[rows, cols],
        radii=d * angpix * stride_map[rows, cols] * scale_mult,
        opacity=fg_opacity,
        flatten_ratio=flatten,
        layer=LAYER_FG,
    )

    # -- bg layer (routed content; scale clamped to fg-equivalent at depth)
    finite_bg = np.isfinite(bg_depth) & (bg_depth > 0)
    with np.errstate(invalid="ignore"):
        bg_content = finite_bg & ~sky & (bg_depth <= shell_dist)
    sel_bg_pre = bg_mask & bg_content & _grid_mask(h, w, s_bg)
    n_bg_pre = int(sel_bg_pre.sum())
    sel_bg = sel_bg_pre & keep
    rows, cols = np.nonzero(sel_bg)
    d = bg_depth[rows, cols]
    # fg-equivalent radius clamp: never inflate past the fg stride at that pixel.
    bg_stride_eff = np.minimum(float(s_bg), stride_map[rows, cols].astype(np.float64))
    bg_part = _make_part(
        pos=dirs[rows, cols] * d[:, None],
        normals=normals_bg[rows, cols],
        rgb01=bg_rgb[rows, cols].astype(np.float64) / 255.0,
        radii=d * angpix * bg_stride_eff * scale_mult,
        opacity=float(p4["bg_opacity"]),
        flatten_ratio=flatten,
        layer=LAYER_BG,
    )

    # -- textured shell (sky / non-finite / near / far pushed to shell_radius_m)
    sel_sh = shell_route & _grid_mask(h, w, s_sh)
    rows, cols = np.nonzero(sel_sh)
    n_sh = rows.shape[0]
    shell_dirs = dirs[rows, cols]
    sh_part = _make_part(
        pos=shell_dirs * shell_r,
        normals=-shell_dirs,
        rgb01=fg_rgb01[rows, cols],
        radii=np.full(n_sh, shell_r * angpix * s_sh * scale_mult, dtype=np.float64),
        opacity=SHELL_OPACITY,
        flatten_ratio=flatten,
        layer=LAYER_SHELL,
    )

    # -- assemble (fg row-major, then bg, then shell) + artifacts
    splats = SplatData.concat([fg_part, bg_part, sh_part])
    count = len(splats)
    ply_path = out / "splats.ply"
    plyio.write_splats(ply_path, splats)

    n_pre = n_fg_pre + n_bg_pre
    n_post = len(fg_part) + len(bg_part)
    retained = float(n_post / n_pre) if n_pre > 0 else 1.0

    strides = {"edge": se, "ground": sg, "base": sb, "bg": s_bg, "shell": s_sh}
    meta = {
        "count": count,
        "counts_by_layer": {
            "fg": len(fg_part),
            "bg": len(bg_part),
            "shell": len(sh_part),
        },
        "stride_multiplier": mult,
        "strides": strides,
        "near_shell_px": near_shell_px,
        "near_fg_px": near_fg_px,
        "far_shell_px": far_shell_px,
        "feather_px": feather_px,
        "color_var_retained_frac": retained,
        "exceeds_cap": bool(count > splat_cap),  # s6 owns the retry, not s4
    }
    meta_path = out / "splats_meta.json"
    schema.write_validated(meta_path, meta, "splats_meta")

    inputs = {
        "fg_rgb": fg_rgb_path,
        "fg_depth": fg_depth_path,
        "fg_mask": fg_mask_path,
        "bg_rgb": bg_rgb_path,
        "bg_depth": bg_depth_path,
        "layers": layers_path,
        "sky_mask": sky_path,
    }
    if have_bg_mask:
        inputs["bg_mask"] = bg_mask_path
    receipts.write_receipt(
        run_dir,
        STAGE,
        inputs=inputs,
        outputs={"splats": ply_path, "splats_meta": meta_path},
        params_used={
            "s4": p4,
            "splat_cap": splat_cap,
            "min_content_distance_m": min_content,
            "s3": {"edge_log_grad_min": edge_thr},
        },
        weights_used=[],
        gates=[],
        notes={
            "strides": strides,
            "stride_multiplier": mult,
            "count": count,
            "near_shell_px": near_shell_px,
            "far_shell_px": far_shell_px,
            "feather_px": feather_px,
            "color_var_retained_frac": retained,
            "exceeds_cap": bool(count > splat_cap),
        },
    )
