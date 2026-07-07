"""s3_layers: split the scene into foreground/background layers.

Occlusion edges from log-depth forward differences (x wraps, y edge-padded)
now require BOTH a log-gradient AND a depth ratio (a real occlusion, not a
gentle slope). An analytic inpaint band width (head-box translation vs the
near/far depths at edges) is hard-capped. A head-box visibility test keeps
only edges that actually disocclude for some reachable pose, and a
deterministic push-pull pyramid fill synthesizes background rgb + log-depth
inside the surviving band.

Reads (all at depth resolution; pano is resampled bilinearly to it):
  s2b_scale/out/depth_m.npy     float32 HxW radial metric depth, inf invalid
  s2_depth/out/sky_mask.png     bool mask
  s1_cleanplate/out/pano_clean.png if present else s0_ingest/out/pano.png

Writes: fg_rgb.png fg_depth.npy fg_mask.png bg_rgb.png bg_depth.npy
bg_mask.png layers.json (+ receipt with the bg_solid_angle gate). Pure numpy,
no torch, no RNG.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from scenic import geometry, imageio, metrics, receipts, schema
from scenic.stage import Ctx

STAGE = "s3_layers"


# ---------------------------------------------------------------- helpers


def _resample_mask_nearest(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbor mask resample (defensive; inputs should match)."""
    sh, sw = mask.shape
    ys = (np.arange(h, dtype=np.int64) * sh) // h
    xs = (np.arange(w, dtype=np.int64) * sw) // w
    return mask[ys][:, xs]


def _dilate4(mask: np.ndarray, iters: int) -> np.ndarray:
    """Binary dilation with a 4-neighborhood; x wraps (equirect lon),
    y clamps at the poles."""
    out = mask.copy()
    for _ in range(int(iters)):
        up = np.concatenate([out[:1], out[:-1]], axis=0)
        down = np.concatenate([out[1:], out[-1:]], axis=0)
        left = np.roll(out, 1, axis=1)
        right = np.roll(out, -1, axis=1)
        out = out | up | down | left | right
    return out


def _push_pull(values: np.ndarray, weight: np.ndarray, min_size: int) -> np.ndarray:
    """Deterministic push-pull fill. values HxWxC float64, weight HxW in
    {0,1}. Returns values with weight==0 pixels filled from coarser levels
    (weighted-average downsample, nearest upsample into invalid px only)."""
    h, w = weight.shape
    c = values.shape[-1]
    if h <= min_size or w <= min_size:
        tot = weight.sum()
        if tot > 0:
            avg = (values * weight[..., None]).sum(axis=(0, 1)) / tot
        else:
            avg = np.zeros(c, dtype=np.float64)
        return np.where(weight[..., None] > 0, values, avg)
    ph, pw = ((h + 1) // 2) * 2, ((w + 1) // 2) * 2
    vpad = np.zeros((ph, pw, c), dtype=np.float64)
    wpad = np.zeros((ph, pw), dtype=np.float64)
    vpad[:h, :w] = values * weight[..., None]
    wpad[:h, :w] = weight
    vsum = vpad.reshape(ph // 2, 2, pw // 2, 2, c).sum(axis=(1, 3))
    wsum = wpad.reshape(ph // 2, 2, pw // 2, 2).sum(axis=(1, 3))
    cvals = np.where(wsum[..., None] > 0, vsum / np.maximum(wsum, 1e-12)[..., None], 0.0)
    cweight = (wsum > 0).astype(np.float64)
    cfilled = _push_pull(cvals, cweight, min_size)
    upv = np.repeat(np.repeat(cfilled, 2, axis=0), 2, axis=1)[:h, :w]
    return np.where(weight[..., None] > 0, values, upv)


def _head_box_poses(head_box: dict, squat_y_m: float) -> list[tuple[str, np.ndarray]]:
    """Reimplemented locally (single-owner): the head-box extreme translation
    offsets used by the visibility test — center plus the four lateral extremes
    at head_box.lateral_m, the up extreme at head_box.up_m, and the squat
    extreme at s7.squat_y_m. Offsets are camera translations in metres
    (right-handed, +Y up)."""
    lat = float(head_box["lateral_m"])
    up = float(head_box["up_m"])
    sq = float(squat_y_m)
    return [
        ("center", np.array([0.0, 0.0, 0.0], dtype=np.float64)),
        ("xpos", np.array([lat, 0.0, 0.0], dtype=np.float64)),
        ("xneg", np.array([-lat, 0.0, 0.0], dtype=np.float64)),
        ("zpos", np.array([0.0, 0.0, lat], dtype=np.float64)),
        ("zneg", np.array([0.0, 0.0, -lat], dtype=np.float64)),
        ("up", np.array([0.0, up, 0.0], dtype=np.float64)),
        ("squat", np.array([0.0, sq, 0.0], dtype=np.float64)),
    ]


# ---------------------------------------------------------------- stage


def run(run_dir: Path, params: dict, ctx: Ctx) -> None:
    run_dir = Path(run_dir)
    out = ctx.out(run_dir, STAGE)
    p3 = params["s3"]
    head_box = params["head_box"]
    squat_y_m = float(params["s7"]["squat_y_m"])

    depth_path = run_dir / "s2b_scale" / "out" / "depth_m.npy"
    sky_path = run_dir / "s2_depth" / "out" / "sky_mask.png"
    clean_path = run_dir / "s1_cleanplate" / "out" / "pano_clean.png"
    if clean_path.exists():
        pano_path, pano_source = clean_path, "s1_cleanplate"
    else:
        pano_path, pano_source = run_dir / "s0_ingest" / "out" / "pano.png", "s0_ingest"

    depth = imageio.load_npy(depth_path).astype(np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth_m.npy must be HxW, got {depth.shape}")
    h, w = depth.shape

    sky = imageio.load_mask_png(sky_path)
    if sky.shape != depth.shape:
        sky = _resample_mask_nearest(sky, h, w)

    rgb01 = imageio.load_rgb(pano_path).astype(np.float64) / 255.0
    if rgb01.shape[:2] != (h, w):
        # bilinear resample onto the depth grid via the shared equirect
        # conventions (lon wraps, lat clamps)
        rgb01 = geometry.sample_equirect(rgb01, geometry.equirect_dirs(w, h))
    rgb01 = np.clip(rgb01, 0.0, 1.0)

    # -- 1. occlusion edges on log depth (forward diffs; x wraps, y pads edge).
    #        An edge now requires BOTH the log-gradient AND a depth ratio
    #        d_far/d_near > s3.edge_depth_ratio_min, with per-edge-pixel
    #        near/far taken from the dominant-diff neighbor.
    finite = np.isfinite(depth)
    logd = np.zeros_like(depth)
    logd[finite] = np.log(depth[finite])

    logd_x = np.roll(logd, -1, axis=1)
    finite_x = np.roll(finite, -1, axis=1)
    depth_x = np.roll(depth, -1, axis=1)
    logd_y = np.concatenate([logd[1:], logd[-1:]], axis=0)  # edge pad: last diff 0
    finite_y = np.concatenate([finite[1:], finite[-1:]], axis=0)
    depth_y = np.concatenate([depth[1:], depth[-1:]], axis=0)

    ex = np.where(finite & finite_x, np.abs(logd - logd_x), 0.0)
    ey = np.where(finite & finite_y, np.abs(logd - logd_y), 0.0)
    grad = np.maximum(ex, ey)
    grad_ok = grad > float(p3["edge_log_grad_min"])

    # per-edge-pixel near/far from the dominant-diff neighbor
    use_x = ex >= ey
    d_nbr = np.where(use_x, depth_x, depth_y)
    near_pp = np.minimum(depth, d_nbr)
    far_pp = np.maximum(depth, d_nbr)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = far_pp / near_pp
    ratio = np.where(np.isfinite(ratio), ratio, 0.0)
    ratio_ok = ratio > float(p3["edge_depth_ratio_min"])

    edge = grad_ok & ratio_ok
    edge_px_count = int(edge.sum())

    # -- 2. head-box visibility test: keep an edge only if the near surface,
    #        viewed from some reachable head-box pose, shifts by >= 1 px
    #        against the far surface. Parallax(pose) = |t_perp| / d_near (rad),
    #        with t_perp the pose offset perpendicular to the ray.
    dirs = geometry.equirect_dirs(w, h)  # (h,w,3) unit
    angpix = float(geometry.angular_pixel_size(h))
    safe_near = np.where(near_pp > 0.0, near_pp, np.inf)
    max_parallax = np.zeros((h, w), dtype=np.float64)
    for _, t in _head_box_poses(head_box, squat_y_m):
        t_dot = dirs @ t                       # (h,w) component along the ray
        t_norm2 = float(t @ t)
        perp = np.sqrt(np.maximum(t_norm2 - t_dot * t_dot, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            par = perp / safe_near
        max_parallax = np.maximum(max_parallax, par)
    visible = max_parallax > angpix
    visible_edge = edge & visible
    visible_edge_px_count = int(visible_edge.sum())

    # -- 3. analytic band width from head-box translation vs the VISIBLE edge
    #        depths, hard-capped to [2, s3.band_px_max].
    t_max = float(max(head_box["lateral_m"], head_box["up_m"], head_box["down_m"]))
    band_px_max = int(p3["band_px_max"])
    band_extra = int(p3["band_extra_px"])
    if visible_edge_px_count > 0:
        near_all = near_pp[visible_edge]
        far_all = far_pp[visible_edge]
        d_near = float(np.percentile(near_all, 10.0))
        d_far = float(np.percentile(far_all, 90.0))
        band_angle = t_max * abs(1.0 / d_near - 1.0 / d_far)
        raw_band = int(math.ceil(band_angle / angpix)) + band_extra
        band_px = int(np.clip(raw_band, 2, band_px_max))
        clamped = bool(raw_band > band_px_max)
        derivation = {
            "t_max": t_max,
            "d_near": d_near,
            "d_far": d_far,
            "band_angle_rad": float(band_angle),
            "raw_band_px": raw_band,
            "band_px": band_px,
            "band_px_max": band_px_max,
            "clamped": clamped,
        }
    else:
        raw_band = band_extra
        band_px = int(np.clip(raw_band, 2, band_px_max))
        derivation = {
            "t_max": t_max,
            "d_near": None,
            "d_far": None,
            "band_angle_rad": None,
            "raw_band_px": raw_band,
            "band_px": band_px,
            "band_px_max": band_px_max,
            "clamped": False,
        }

    # -- 4. background layer: band around VISIBLE edges only, push-pull fill of
    #        rgb + log-depth. bg_depth stays the metric far-side depth.
    bg_region = _dilate4(visible_edge, band_px) & ~sky
    valid_src = finite & ~bg_region

    fill_in = np.concatenate([rgb01, logd[..., None]], axis=-1)
    filled = _push_pull(
        fill_in, valid_src.astype(np.float64), int(p3["pyramid_min_size"])
    )
    bg_rgb01 = np.where(valid_src[..., None], rgb01, filled[..., 0:3])
    bg_rgb = np.round(np.clip(bg_rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
    bg_depth = np.where(valid_src, depth, np.exp(filled[..., 3])).astype(np.float32)

    # -- 5. foreground layer: original content everywhere finite (minus sky)
    fg_mask = finite & ~sky
    fg_rgb = np.round(rgb01 * 255.0).astype(np.uint8)
    fg_depth = np.where(fg_mask, depth, np.inf).astype(np.float32)

    bg_solid_angle_frac = float(metrics.solid_angle_fraction(bg_region))

    imageio.save_png(out / "fg_rgb.png", fg_rgb)
    imageio.save_npy(out / "fg_depth.npy", fg_depth)
    imageio.save_mask_png(out / "fg_mask.png", fg_mask)
    imageio.save_png(out / "bg_rgb.png", bg_rgb)
    imageio.save_npy(out / "bg_depth.npy", bg_depth)
    imageio.save_mask_png(out / "bg_mask.png", bg_region)

    layers = {
        "band_px": band_px,
        "band_derivation": derivation,
        "edge_px_count": edge_px_count,
        "visible_edge_px_count": visible_edge_px_count,
        "bg_filled_px": int(bg_region.sum()),
        "bg_scale_clamp": bool(p3["bg_scale_clamp"]),
        "bg_solid_angle_frac": bg_solid_angle_frac,
    }
    schema.write_validated(out / "layers.json", layers, "layers")

    max_frac = float(p3["bg_solid_angle_max_frac"])
    gate = {
        "gate": "bg_solid_angle",
        "pass": bool(bg_solid_angle_frac <= max_frac),
        "metrics": {
            "bg_solid_angle_frac": bg_solid_angle_frac,
            "edge_px_count": edge_px_count,
            "band_px": band_px,
        },
        "thresholds": {"max_frac": max_frac},
    }

    receipts.write_receipt(
        run_dir,
        STAGE,
        inputs={"depth_m": depth_path, "sky_mask": sky_path, "pano": pano_path},
        outputs={
            "fg_rgb": out / "fg_rgb.png",
            "fg_depth": out / "fg_depth.npy",
            "fg_mask": out / "fg_mask.png",
            "bg_rgb": out / "bg_rgb.png",
            "bg_depth": out / "bg_depth.npy",
            "bg_mask": out / "bg_mask.png",
            "layers": out / "layers.json",
        },
        params_used={
            "head_box": head_box,
            "s3": p3,
            "s7": {"squat_y_m": squat_y_m},
        },
        weights_used=[],
        gates=[gate],
        notes={"pano_source": pano_source},
    )
