"""Stereo gate: render an eye pair (+-ipd/2 along the camera-right axis) at
the center pose for each of the 4 yaws and check three failure modes:

1. vertical disparity — a global vertical shift between L and R estimated on
   gradient-magnitude images: cost(dy) = mean |GL - shift(GR, dy)| for
   integer dy in [-2..2] over a fixed central row band (rows 2..h-2 of L, so
   every shift compares the SAME number of rows — a varying overlap region
   biases the argmin toward whichever crop drops the strongest gradients),
   argmin + parabolic subpixel refine. |dy*| must be <=
   s7.stereo_vdisp_max_px in every yaw. (Flat-cost guard: if the costs are
   indistinguishable there is no signal, dy* = 0.)
2. near limit — min finite depth over the FULL frame minus a fixed 2-px
   border margin of BOTH eyes' depth maps must be >=
   s7.stereo_near_depth_min_m. Views with no finite depth contribute the
   sentinel DEPTH_SENTINEL_M (receipts forbid Inf). Beyond the 4 yaws, two
   near-limit-ONLY eye pairs at (yaw 0, pitch -90) and (yaw 0, pitch +90)
   close the polar blind wedges; the eye offset stays horizontal (IPD is a
   yaw-frame offset, rotation_yaw_pitch(yaw, 0) @ [1,0,0]). Pitched pairs
   contribute ONLY to min_depth_m — vdisp and order stay on the pitch-0
   yaws, where the horizontal-disparity geometry they assume holds.
3. depth order — per-pixel horizontal disparity from geometry d = f*ipd/zL
   (f in px); warp the R depth map by d (uR = uL - d, nearest pixel) and
   count the fraction of finite L/R pairs with |zL - zR_warped| / zL <
   REL_DEPTH_TOL. Must be >= s7.stereo_order_min_frac in every yaw (views
   with no finite pairs pass vacuously, n recorded in details).

PASS iff all three hold. details.per_yaw carries the 4 yaw entries plus the
2 pitched near-only entries (flagged near_only, with a pitch_deg field).
Diagnostics: yaw-0 L/R renders in outdir/renders/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scenic.geometry import rotation_yaw_pitch
from scenic.plyio import SplatData

from gates import YAWS_DEG, render_view, save_render

REL_DEPTH_TOL = 0.2       # depth-order relative consistency tolerance
DEPTH_SENTINEL_M = 1.0e9  # stands in for "no finite depth" (no Inf in JSON)
_FLAT_COST_EPS = 1e-12    # below this cost spread, there is no vdisp signal
_BORDER_MARGIN_PX = 2     # near-limit frame margin (edge-clipped footprints)

# Near-limit-only eye pairs (yaw_deg, pitch_deg): straight down + straight
# up, closing the |pitch| blind wedges of the pitch-0 yaw ring.
NEAR_ONLY_VIEWS = ((0.0, -90.0), (0.0, 90.0))


def _grad_mag(rgb: np.ndarray) -> np.ndarray:
    """Gradient-magnitude image of the float01 grayscale render."""
    gray = rgb.astype(np.float64).mean(axis=-1) / 255.0
    gy, gx = np.gradient(gray)
    return np.hypot(gx, gy)


def _vertical_disparity_px(gl: np.ndarray, gr: np.ndarray) -> float:
    """Global vertical shift (px) of R vs L: argmin over integer dy in
    [-2..2] of mean |GL - shift(GR, dy)| on a fixed central row band (same
    row count for every dy), with parabolic subpixel refine."""
    h = gl.shape[0]
    m = 2  # band margin = max tested shift, keeps every cost comparable
    a = gl[m : h - m]
    costs = []
    for dy in range(-2, 3):
        b = gr[m + dy : h - m + dy]
        costs.append(float(np.mean(np.abs(a - b))))
    costs_arr = np.asarray(costs)
    if float(costs_arr.max() - costs_arr.min()) < _FLAT_COST_EPS:
        return 0.0
    i = int(np.argmin(costs_arr))
    dy_star = float(i - 2)
    if 0 < i < 4:
        c0, c1, c2 = costs_arr[i - 1], costs_arr[i], costs_arr[i + 1]
        den = float(c0 - 2.0 * c1 + c2)
        if den > _FLAT_COST_EPS:
            dy_star += float(0.5 * (c0 - c2) / den)
    return dy_star


def _frame_min_depth(depth: np.ndarray) -> float:
    """Min finite depth over the full frame minus a fixed 2-px border margin
    (partial splat footprints clipped at the frame edge carry unreliable
    expected depth); sentinel if no finite depth survives."""
    m = _BORDER_MARGIN_PX
    win = depth[m:-m, m:-m]
    finite = win[np.isfinite(win)]
    if finite.size == 0:
        return DEPTH_SENTINEL_M
    return float(finite.min())


def _order_consistency(
    zl: np.ndarray, zr: np.ndarray, f_px: float, ipd_m: float
) -> tuple[float, int]:
    """Warp R depth by the L-depth-derived disparity and count agreement.
    Returns (consistent fraction, n pixels checked); vacuous pass on n=0."""
    h, w = zl.shape
    finite_l = np.isfinite(zl)
    disp = np.zeros_like(zl, dtype=np.float64)
    disp[finite_l] = f_px * ipd_m / zl[finite_l].astype(np.float64)
    u = np.arange(w, dtype=np.float64)[None, :]
    ur = np.rint(u - disp)
    valid = finite_l & (ur >= 0.0) & (ur <= w - 1.0)
    if not valid.any():
        return 1.0, 0
    vv, uu = np.nonzero(valid)
    zr_w = zr[vv, ur[valid].astype(np.int64)]
    finite_pair = np.isfinite(zr_w)
    if not finite_pair.any():
        return 1.0, 0
    zl_v = zl[vv, uu][finite_pair].astype(np.float64)
    zr_v = zr_w[finite_pair].astype(np.float64)
    ok = np.abs(zl_v - zr_v) / zl_v < REL_DEPTH_TOL
    return float(np.mean(ok)), int(finite_pair.sum())


def run_gate(splats: SplatData, params: dict, outdir: Path | str) -> dict:
    s7 = params["s7"]
    px = int(s7["render_px"])
    fov = float(s7["render_fov_deg"])
    ipd = float(s7["stereo_ipd_m"])
    vdisp_max = float(s7["stereo_vdisp_max_px"])
    near_min = float(s7["stereo_near_depth_min_m"])
    order_min = float(s7["stereo_order_min_frac"])
    f_px = (px / 2.0) / np.tan(np.deg2rad(fov) / 2.0)

    per_yaw: list[dict] = []
    for yaw in YAWS_DEG:
        right = rotation_yaw_pitch(float(np.deg2rad(yaw)), 0.0) @ np.array(
            [1.0, 0.0, 0.0]
        )
        left_eye = render_view(splats, params, -0.5 * ipd * right, yaw)
        right_eye = render_view(splats, params, 0.5 * ipd * right, yaw)
        if yaw == 0.0:
            save_render(outdir, "stereo_yaw000_left.png", left_eye["rgb"])
            save_render(outdir, "stereo_yaw000_right.png", right_eye["rgb"])

        vdisp = _vertical_disparity_px(
            _grad_mag(left_eye["rgb"]), _grad_mag(right_eye["rgb"])
        )
        min_depth = min(
            _frame_min_depth(left_eye["depth"]),
            _frame_min_depth(right_eye["depth"]),
        )
        order_frac, n_order = _order_consistency(
            left_eye["depth"], right_eye["depth"], f_px, ipd
        )
        per_yaw.append(
            {
                "yaw_deg": float(yaw),
                "pitch_deg": 0.0,
                "vdisp_px": float(vdisp),
                "min_depth_m": float(min_depth),
                "order_frac": float(order_frac),
                "n_order_px": int(n_order),
                "near_only": False,
            }
        )

    # Polar near-limit-only pairs: the eye offset stays horizontal (IPD is a
    # yaw-frame offset), only min_depth_m is measured — vdisp and the
    # horizontal-disparity order model do not apply to pitched cameras.
    for yaw, pitch in NEAR_ONLY_VIEWS:
        right = rotation_yaw_pitch(float(np.deg2rad(yaw)), 0.0) @ np.array(
            [1.0, 0.0, 0.0]
        )
        left_eye = render_view(splats, params, -0.5 * ipd * right, yaw, pitch)
        right_eye = render_view(splats, params, 0.5 * ipd * right, yaw, pitch)
        min_depth = min(
            _frame_min_depth(left_eye["depth"]),
            _frame_min_depth(right_eye["depth"]),
        )
        per_yaw.append(
            {
                "yaw_deg": float(yaw),
                "pitch_deg": float(pitch),
                "min_depth_m": float(min_depth),
                "near_only": True,
            }
        )

    vdisp_worst = max(abs(v["vdisp_px"]) for v in per_yaw if not v["near_only"])
    min_depth_all = min(v["min_depth_m"] for v in per_yaw)
    order_worst = min(v["order_frac"] for v in per_yaw if not v["near_only"])
    passed = (
        vdisp_worst <= vdisp_max
        and min_depth_all >= near_min
        and order_worst >= order_min
    )
    return {
        "gate": "stereo",
        "pass": bool(passed),
        "metrics": {
            "vdisp_max_px": float(vdisp_worst),
            "min_depth_m": float(min_depth_all),
            "order_min_frac": float(order_worst),
        },
        "thresholds": {
            "stereo_vdisp_max_px": vdisp_max,
            "stereo_near_depth_min_m": near_min,
            "stereo_order_min_frac": order_min,
            "stereo_ipd_m": ipd,
            "rel_depth_tol": REL_DEPTH_TOL,
        },
        "details": {"per_yaw": per_yaw, "pose": "center"},
    }
