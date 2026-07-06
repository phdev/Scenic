"""s2b_scale: metric scale from a robust ground-plane fit in the nadir cone.

Reads s2_depth out/depth_rel.npy (+ sky_mask.png) and s0_ingest
out/pano_meta.json (camera_height_m). Fits the ground as the graph
y = a*x + b*z + c over nadir-cone points via deterministic IRLS (Tukey
bisquare, NO RANSAC), derives the relative camera height h_rel as the
origin-to-plane distance, and scales relative depth to meters:

    scale_factor = camera_height_m / h_rel        (scale_source ground_plane)
    UNLESS params.s2b.explicit_scale is non-null  (scale_source explicit)

Outputs out/depth_m.npy (float32, inf stays inf) and out/scale.json
(schema "scale"). Records two gate verdicts (never aborts on gate fail):
ground_plane (residual_rel + tilt_deg) and min_content_distance (near
percentile of horizon-band metric depth).

Plane math (verified in tests/test_s2b.py on synthetic data):
y = a*x + b*z + c  <=>  a*x - y + b*z + c = 0, gradient (a, -1, b).
Unit normal sign-fixed so n.y > 0:  n = (-a, 1, -b) / N,  N = sqrt(a^2+1+b^2).
Plane as n.p + d = 0  =>  d = -c / N;  origin distance h_rel = |c| / N.
tilt_deg = angle(n, +Y) = acos(1 / N). Orthogonal residual = |y-residual| / N.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from scenic import geometry, imageio, receipts, schema
from scenic.stage import Ctx

STAGE = "s2b_scale"

_TUKEY = 4.685
_MAD_TO_SIGMA = 1.4826
_MAD_GUARD = 1e-9
_H_REL_GUARD = 1e-9


def _num(x) -> float | str:
    """JSON-safe number for gate metrics (canonical JSON forbids NaN/Inf)."""
    x = float(x)
    return x if math.isfinite(x) else repr(x)


def fit_ground_plane(
    points: np.ndarray, irls_iters: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic IRLS Tukey-bisquare fit of y = a*x + b*z + c.

    points: (N,3) xyz. Returns (coef [a,b,c] float64, inlier bool mask,
    y-residuals float64). Init = unweighted least squares on all points;
    then `irls_iters` reweighted solves with Tukey c = 4.685 * 1.4826 *
    median|res| (median guarded >= 1e-9). No randomness.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        raise ValueError(f"fit_ground_plane needs (N>=3,3) points, got {pts.shape}")
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    a_mat = np.stack([x, z, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(a_mat, y, rcond=None)
    for _ in range(int(irls_iters)):
        res = y - a_mat @ coef
        mad = max(float(np.median(np.abs(res))), _MAD_GUARD)
        c_t = _TUKEY * _MAD_TO_SIGMA * mad
        u = res / c_t
        w = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
        if float(w.sum()) < 1e-12:
            break  # degenerate; keep previous solution
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(a_mat * sw[:, None], y * sw, rcond=None)
    res = y - a_mat @ coef
    mad = max(float(np.median(np.abs(res))), _MAD_GUARD)
    inliers = np.abs(res) < _TUKEY * _MAD_TO_SIGMA * mad
    return coef, inliers, res


def plane_from_coef(coef: np.ndarray) -> dict:
    """Derive unit normal (n.y>0), d (plane n.p + d = 0), origin distance
    h_rel = |c|/||(a,-1,b)||, tilt_deg vs +Y, and the gradient norm."""
    a, b, c = (float(v) for v in coef)
    n_norm = math.sqrt(a * a + 1.0 + b * b)
    normal = [-a / n_norm, 1.0 / n_norm, -b / n_norm]
    return {
        "normal": normal,
        "d": -c / n_norm,
        "h_rel": abs(c) / n_norm,
        "tilt_deg": math.degrees(math.acos(min(1.0, 1.0 / n_norm))),
        "norm": n_norm,
    }


def run(run_dir: Path, params: dict, ctx: Ctx) -> None:
    run_dir = Path(run_dir)
    out_dir = ctx.out(run_dir, STAGE)

    depth_rel_path = run_dir / "s2_depth" / "out" / "depth_rel.npy"
    sky_mask_path = run_dir / "s2_depth" / "out" / "sky_mask.png"
    pano_meta_path = run_dir / "s0_ingest" / "out" / "pano_meta.json"

    depth_rel = imageio.load_npy(depth_rel_path)
    if depth_rel.ndim != 2:
        raise ValueError(f"depth_rel must be HxW, got shape {depth_rel.shape}")
    sky = imageio.load_mask_png(sky_mask_path)
    if sky.shape != depth_rel.shape:
        raise ValueError(
            f"sky_mask shape {sky.shape} != depth_rel shape {depth_rel.shape}"
        )
    meta = schema.read_validated(pano_meta_path, "pano_meta")
    camera_height_m = float(meta["camera_height_m"])

    s2b = params["s2b"]
    min_content_m = float(params["min_content_distance_m"])
    explicit = s2b.get("explicit_scale")

    h, w = depth_rel.shape
    lon, lat = geometry.equirect_lonlat(w, h)
    dirs = geometry.lonlat_to_dirs(lon, lat)
    pitch_deg = np.degrees(lat)

    finite = np.isfinite(depth_rel)
    cone = (pitch_deg < float(s2b["nadir_cone_pitch_deg"])) & finite & ~sky
    n_cone = int(cone.sum())

    if n_cone >= 3:
        pts = dirs[cone] * depth_rel[cone].astype(np.float64)[:, None]
        coef, inliers, res = fit_ground_plane(pts, int(s2b["irls_iters"]))
        plane = plane_from_coef(coef)
        h_rel = plane["h_rel"]
        tilt_deg = plane["tilt_deg"]
        orth = np.abs(res) / plane["norm"]
        med_orth = float(np.median(orth[inliers])) if inliers.any() else float(
            np.median(orth)
        )
        residual_rel = med_orth / max(h_rel, _H_REL_GUARD)
        n_inliers = int(inliers.sum())
        plane_fit = True
    elif explicit is not None:
        # No usable ground content, but an explicit scale was supplied:
        # record a degenerate (identity-up) plane instead of aborting.
        plane = {"normal": [0.0, 1.0, 0.0], "d": 0.0, "h_rel": 0.0,
                 "tilt_deg": 0.0, "norm": 1.0}
        h_rel = 0.0
        tilt_deg = 0.0
        residual_rel = 0.0
        n_inliers = 0
        plane_fit = False
    else:
        raise ValueError(
            f"s2b_scale: only {n_cone} finite non-sky nadir-cone pixels "
            "(need >= 3) and no explicit_scale override"
        )

    if explicit is not None:
        scale_factor = float(explicit)
        scale_source = "explicit"
    else:
        if h_rel < _H_REL_GUARD:
            raise ValueError(
                f"s2b_scale: degenerate ground plane (h_rel={h_rel!r})"
            )
        scale_factor = camera_height_m / h_rel
        scale_source = "ground_plane"

    depth_m = (depth_rel.astype(np.float64) * scale_factor).astype(np.float32)
    depth_m_path = out_dir / "depth_m.npy"
    imageio.save_npy(depth_m_path, depth_m)

    # --- gate: ground_plane (recorded; never aborts) ---
    if scale_source == "explicit":
        gp_pass = True
        gp_details = (
            "explicit_scale override in effect; plane-fit quality not enforced"
        )
    else:
        gp_pass = (
            residual_rel <= float(s2b["residual_max_rel"])
            and tilt_deg <= float(s2b["plane_tilt_max_deg"])
        )
        gp_details = "ground plane IRLS fit over nadir cone"
    ground_gate = {
        "gate": "ground_plane",
        "pass": bool(gp_pass),
        "metrics": {
            "residual_rel": _num(residual_rel),
            "tilt_deg": _num(tilt_deg),
            "h_rel": _num(h_rel),
            "cone_px": n_cone,
            "inlier_px": n_inliers,
            "scale_factor": _num(scale_factor),
            "scale_source": scale_source,
        },
        "thresholds": {
            "residual_max_rel": float(s2b["residual_max_rel"]),
            "plane_tilt_max_deg": float(s2b["plane_tilt_max_deg"]),
            "nadir_cone_pitch_deg": float(s2b["nadir_cone_pitch_deg"]),
        },
        "details": gp_details,
    }

    # --- gate: min_content_distance (recorded; never aborts) ---
    # Asymmetric band: the pano's own ground legitimately approaches the
    # camera below the horizon (flat ground at -20 deg with a 1.6 m camera is
    # 4.7 m away), so the down side stops at horizon_band_down_deg where flat
    # ground is still >= camera_height/sin(down_deg) away.
    band = (pitch_deg <= float(s2b["horizon_band_deg"])) & (
        pitch_deg >= -float(s2b["horizon_band_down_deg"])
    )
    band_vals = depth_m[band].astype(np.float64)
    if band_vals.size:
        near_m = float(np.percentile(band_vals, float(s2b["near_percentile"])))
        mc_pass = near_m >= min_content_m
        mc_details = "near-percentile metric depth over horizon band"
    else:
        near_m = float("nan")
        mc_pass = False
        mc_details = "horizon band contains no pixels"
    mc_gate = {
        "gate": "min_content_distance",
        "pass": bool(mc_pass),
        "metrics": {
            "near_distance_m": _num(near_m),
            "band_px": int(band.sum()),
            "band_finite_px": int((band & np.isfinite(depth_m)).sum()),
        },
        "thresholds": {
            "min_content_distance_m": min_content_m,
            "near_percentile": float(s2b["near_percentile"]),
            "horizon_band_deg": float(s2b["horizon_band_deg"]),
            "horizon_band_down_deg": float(s2b["horizon_band_down_deg"]),
        },
        "details": mc_details,
    }

    gates = [ground_gate, mc_gate]

    scale_obj = {
        "scale_factor": float(scale_factor),
        "camera_height_m": camera_height_m,
        "scale_source": scale_source,
        "plane": {
            "normal": [float(v) for v in plane["normal"]],
            "d": float(plane["d"]),
        },
        "h_rel": float(h_rel),
        "residual_rel": float(min(residual_rel, 1e12)),  # keep JSON finite
        "tilt_deg": float(tilt_deg),
        "gates": gates,
    }
    scale_path = out_dir / "scale.json"
    schema.write_validated(scale_path, scale_obj, "scale")

    receipts.write_receipt(
        run_dir,
        STAGE,
        inputs={
            "depth_rel": depth_rel_path,
            "pano_meta": pano_meta_path,
            "sky_mask": sky_mask_path,
        },
        outputs={"depth_m": depth_m_path, "scale": scale_path},
        params_used={
            "s2b": s2b,
            "min_content_distance_m": min_content_m,
        },
        weights_used=[],
        gates=gates,
        notes={"plane_fit": bool(plane_fit), "cone_px": n_cone},
    )
