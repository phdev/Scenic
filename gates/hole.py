"""Hole gate: render the whole head-box view matrix with the background
shell dyed magenta; any magenta (or zero-alpha) pixel below the skyline is a
coverage hole in the placed content.

Per view (7 poses x 5 views: the 4-yaw pitch-0 ring + the straight-down
nadir view — the pitch-0 fov-90 ring only reaches ray pitch ~-45 deg, and
the nadir is where real defects live):
- magenta pixel:  r > 0.6, b > 0.6, g < 0.35 (float01, after compositing) —
  generous thresholds so blended shell still counts.
- alpha hole:     composite alpha < 0.05 (a true hole: the ray hit nothing) —
  tracked as a separate metric but failing under the same frac threshold.
- below skyline:  pixel ray pitch < -2 deg (translation does not change ray
  directions, so the mask is shared across poses; it is yaw-invariant per
  camera pitch and computed once per distinct pitch). For the straight-down
  view every ray qualifies (mask all-True), so the central-blob check
  covers the nadir directly.

FAIL if, in ANY view, magenta_below_skyline_frac > s7.hole_max_frac, or
alpha_below_skyline_frac > s7.hole_max_frac, or any connected blob of hole
pixels (magenta | alpha-hole, restricted below the skyline) inside the
central window (side s7.hole_center_frac of the image) exceeds
s7.hole_blob_max_px (scipy.ndimage.label, default 4-connectivity).

Contract deviations (documented): (1) the central-blob check labels the
combined hole mask (magenta OR alpha-hole), a superset of the spec's
magenta-only mask — a dead-center hole with no shell behind it must still
trip the blob rule. (2) The blob mask is restricted to below-skyline pixels:
above the horizon the magenta shell IS the sky by construction, so a central
window straddling the horizon would otherwise flag every legitimate sky
view.

Diagnostics: center-pose magenta renders (all 4 yaws, reused by s8, plus the
down view) and the worst view, under outdir/renders/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from scenic import geometry
from scenic.plyio import SplatData

from gates import (
    head_box_poses,
    pose_views,
    render_view,
    save_render,
    shell_magenta_override,
    view_name,
)

SKYLINE_PITCH_DEG = -2.0
MAGENTA_R_MIN = 0.6
MAGENTA_B_MIN = 0.6
MAGENTA_G_MAX = 0.35
ALPHA_HOLE_MAX = 0.05


def _below_skyline_mask(px: int, fov_deg: float, pitch_deg: float) -> np.ndarray:
    """Bool (px,px): pixel ray pitch < SKYLINE_PITCH_DEG for a camera at the
    given pitch. Yaw-invariant per pitch (yaw rotates about +Y, preserving
    each ray's pitch); for the straight-down view it is all-True."""
    dirs = geometry.perspective_dirs(
        fov_deg, px, px, 0.0, float(np.deg2rad(pitch_deg))
    )
    return geometry.pitch_of_dirs(dirs) < np.deg2rad(SKYLINE_PITCH_DEG)


def _center_window(px: int, center_frac: float) -> tuple[int, int]:
    lo = int(round(px * (1.0 - center_frac) / 2.0))
    hi = int(round(px * (1.0 + center_frac) / 2.0))
    return lo, hi


def _max_blob_px(mask: np.ndarray) -> int:
    """Largest connected component (scipy.ndimage.label, 4-connectivity)."""
    if not mask.any():
        return 0
    lbl, n = ndimage.label(mask)
    return int(np.bincount(lbl.ravel())[1:].max())


def run_gate(splats: SplatData, params: dict, outdir: Path | str) -> dict:
    s7 = params["s7"]
    px = int(s7["render_px"])
    fov = float(s7["render_fov_deg"])
    max_frac = float(s7["hole_max_frac"])
    blob_max = int(s7["hole_blob_max_px"])
    center_frac = float(s7["hole_center_frac"])

    override = shell_magenta_override(splats)
    views = pose_views()
    # One skyline mask per distinct camera pitch (yaw-invariant); the view
    # list is the deterministic iteration order.
    below_by_pitch = {
        pitch: _below_skyline_mask(px, fov, pitch)
        for pitch in dict.fromkeys(p for _, p in views)
    }
    n_below_by_pitch = {p: int(m.sum()) for p, m in below_by_pitch.items()}
    lo, hi = _center_window(px, center_frac)

    per_view: list[dict] = []
    worst_magenta = 0.0
    worst_alpha = 0.0
    worst_blob = 0
    worst_view = ""
    worst_score = -1.0
    worst_rgb: np.ndarray | None = None

    for pose_name, pos in head_box_poses(params):
        for yaw, pitch in views:
            name = view_name(pose_name, yaw, pitch)
            below = below_by_pitch[pitch]
            n_below = n_below_by_pitch[pitch]
            out = render_view(
                splats, params, pos, yaw, pitch, override_rgb=override
            )
            rgb01 = out["rgb"].astype(np.float64) / 255.0
            magenta = (
                (rgb01[..., 0] > MAGENTA_R_MIN)
                & (rgb01[..., 2] > MAGENTA_B_MIN)
                & (rgb01[..., 1] < MAGENTA_G_MAX)
            )
            alpha_hole = out["alpha"] < ALPHA_HOLE_MAX
            m_frac = float((magenta & below).sum()) / max(n_below, 1)
            a_frac = float((alpha_hole & below).sum()) / max(n_below, 1)
            # Blob check only below the skyline: magenta shell sky above the
            # horizon is the intended shell, not a hole, even dead-center.
            hole = (magenta | alpha_hole) & below
            blob = _max_blob_px(hole[lo:hi, lo:hi])
            per_view.append(
                {
                    "view": name,
                    "magenta_frac": m_frac,
                    "alpha_frac": a_frac,
                    "blob_px": int(blob),
                }
            )
            worst_magenta = max(worst_magenta, m_frac)
            worst_alpha = max(worst_alpha, a_frac)
            worst_blob = max(worst_blob, blob)
            score = max(m_frac, a_frac, blob / max(px * px, 1))
            if score > worst_score:
                worst_score = score
                worst_view = name
                worst_rgb = out["rgb"]
            if pose_name == "center":
                save_render(
                    outdir,
                    ("center_down_magenta.png" if pitch != 0.0
                     else f"center_yaw{int(yaw):03d}_magenta.png"),
                    out["rgb"],
                )

    if worst_rgb is not None:
        save_render(outdir, "hole_worst_magenta.png", worst_rgb)

    passed = (
        worst_magenta <= max_frac
        and worst_alpha <= max_frac
        and worst_blob <= blob_max
    )
    return {
        "gate": "hole",
        "pass": bool(passed),
        "metrics": {
            "worst_magenta_below_skyline_frac": float(worst_magenta),
            "worst_alpha_below_skyline_frac": float(worst_alpha),
            "worst_blob_px": int(worst_blob),
            "n_views": len(per_view),
        },
        "thresholds": {
            "hole_max_frac": max_frac,
            "hole_blob_max_px": blob_max,
            "hole_center_frac": center_frac,
            "alpha_hole_max": ALPHA_HOLE_MAX,
            "skyline_pitch_deg": SKYLINE_PITCH_DEG,
        },
        "details": {"per_view": per_view, "worst_view": worst_view},
    }
