"""Gate implementations for the s7_gates stage (the falsifiability layer).

Each gate module exposes `run_gate(splats, params, outdir) -> verdict dict`
(schema gate_verdict) and saves diagnostic renders under `outdir/renders/`.
Gates never abort the pipeline: failures become `pass: false` verdicts; hard
errors (missing files, schema violations) raise.

This package also hosts the helpers shared by all gates: the 7 head-box
poses, the fixed per-pose view set (4-yaw pitch-0 ring + one straight-down
nadir view), the render wrapper, and the magenta-shell color override.
Geometry follows docs/CONTRACTS.md (+Y up, camera looks +Z, yaw toward +X).

Owner: this module owns gates/*.py, pipeline/s7_gates.py and
tests/test_s7.py only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scenic import imageio
from scenic.plyio import (
    LAYER_BG,
    LAYER_FG,
    LAYER_SHELL,
    SplatData,
    dc_to_rgb01,
)
from scenic.rasterizer import Camera, render

# The six gates, in the order s7 runs them and embeds them in its receipt.
# `fidelity_at_origin` is the v2 quality-pass fidelity floor (gates/fidelity.py).
GATE_ORDER = (
    "hole", "jitter", "stereo", "people", "budgets", "fidelity_at_origin",
)

# Views per pose: yaw ring at pitch 0 (fov/res come from params s7). The
# pitch-0 fov-90 ring only covers ray pitches down to about -45 deg, so the
# standard view set (pose_views) adds one straight-down view per pose — the
# nadir is exactly where real defects live (tripod/watermark cleanplate
# region; the head box includes looking down).
YAWS_DEG = (0.0, 90.0, 180.0, 270.0)
DOWN_VIEW = (0.0, -90.0)  # (yaw_deg, pitch_deg) straight-down nadir view

# Layer forensics: (layer value, short name), in the fixed fg -> bg -> shell
# order used for the origin layer renders and the receipt notes.
LAYER_ITEMS: tuple[tuple[int, str], ...] = (
    (LAYER_FG, "fg"),
    (LAYER_BG, "bg"),
    (LAYER_SHELL, "shell"),
)

MAGENTA = (1.0, 0.0, 1.0)


def head_box_poses(params: dict) -> list[tuple[str, np.ndarray]]:
    """The 7 head-box poses (params head_box + s7): center, 4 lateral
    extremes (+-x, +-z), up, squat. Fixed, deterministic order."""
    lat = float(params["head_box"]["lateral_m"])
    up = float(params["head_box"]["up_m"])
    squat_y = float(params["s7"]["squat_y_m"])
    return [
        ("center", np.array([0.0, 0.0, 0.0])),
        ("xpos", np.array([lat, 0.0, 0.0])),
        ("xneg", np.array([-lat, 0.0, 0.0])),
        ("zpos", np.array([0.0, 0.0, lat])),
        ("zneg", np.array([0.0, 0.0, -lat])),
        ("up", np.array([0.0, up, 0.0])),
        ("squat", np.array([0.0, squat_y, 0.0])),
    ]


def pose_views() -> list[tuple[float, float]]:
    """The standard (yaw_deg, pitch_deg) view set per head-box pose: the 4
    pitch-0 yaws plus the straight-down nadir view. Fixed, deterministic
    order."""
    return [(yaw, 0.0) for yaw in YAWS_DEG] + [DOWN_VIEW]


def view_name(pose_name: str, yaw_deg: float, pitch_deg: float = 0.0) -> str:
    """Stable view id: `{pose}_yawNNN` on the pitch-0 ring, `{pose}_down`
    for the straight-down view."""
    if pitch_deg == 0.0:
        return f"{pose_name}_yaw{int(round(yaw_deg)):03d}"
    return f"{pose_name}_down"


def render_view(
    splats: SplatData,
    params: dict,
    pos: np.ndarray,
    yaw_deg: float,
    pitch_deg: float = 0.0,
    override_rgb: np.ndarray | None = None,
) -> dict:
    """Render one square head-box view at params s7 fov/res."""
    s7 = params["s7"]
    px = int(s7["render_px"])
    cam = Camera(
        pos=np.asarray(pos, dtype=np.float64).reshape(3),
        yaw=float(np.deg2rad(yaw_deg)),
        pitch=float(np.deg2rad(pitch_deg)),
    )
    return render(
        splats, cam, px, px, float(s7["render_fov_deg"]), override_rgb=override_rgb
    )


def shell_magenta_override(splats: SplatData) -> np.ndarray:
    """(n,3) float01 colors: normal DC colors, but LAYER_SHELL splats get
    pure magenta — the hole gate's tracer dye."""
    col = dc_to_rgb01(splats.f_dc.astype(np.float64))
    col[splats.layer == LAYER_SHELL] = MAGENTA
    return col


def renders_dir(outdir: Path | str) -> Path:
    d = Path(outdir) / "renders"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_render(outdir: Path | str, name: str, rgb: np.ndarray) -> None:
    imageio.save_png(renders_dir(outdir) / name, rgb)


def layer_subset(splats: SplatData, layer_value: int) -> SplatData:
    """Filter to just the splats of one layer (fg/bg/shell)."""
    return splats.take(splats.layer == int(layer_value))


def layer_direction_mask(
    xyz: np.ndarray, h: int = 256, w: int = 512
) -> np.ndarray:
    """Boolean equirect (h,w) mask of the direction bins a layer's splats
    occupy, seen from the ORIGIN. Deterministic, closed-form: project each
    splat's world xyz to (lon, lat) using the docs/CONTRACTS convention and
    mark its bin. Feeds metrics.solid_angle_fraction for the receipt's
    per-layer coverage note (an approximation, not a render)."""
    mask = np.zeros((int(h), int(w)), dtype=bool)
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] == 0:
        return mask
    r = np.maximum(np.linalg.norm(xyz, axis=1), 1e-12)
    lon = np.arctan2(xyz[:, 0], xyz[:, 2])                 # [-pi, pi]
    lat = np.arcsin(np.clip(xyz[:, 1] / r, -1.0, 1.0))     # [-pi/2, pi/2]
    col = np.floor((lon + np.pi) / (2.0 * np.pi) * w).astype(np.int64) % w
    row = np.clip(
        np.floor((np.pi / 2.0 - lat) / np.pi * h).astype(np.int64), 0, h - 1
    )
    mask[row, col] = True
    return mask


def render_layer_view_and_save(
    splats: SplatData,
    params: dict,
    outdir: Path | str,
    layer_value: int,
    layer_name: str,
    yaw_deg: float,
) -> dict:
    """Render ONE origin (center pose) layer-filtered view and save it as
    center_yaw{NNN}_layer_{layer_name}.png. Returns the render dict."""
    sub = layer_subset(splats, layer_value)
    out = render_view(sub, params, np.zeros(3), yaw_deg)
    save_render(
        outdir,
        f"center_yaw{int(round(yaw_deg)):03d}_layer_{layer_name}.png",
        out["rgb"],
    )
    return out
