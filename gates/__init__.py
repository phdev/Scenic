"""Gate implementations for the s7_gates stage (the falsifiability layer).

Each gate module exposes `run_gate(splats, params, outdir) -> verdict dict`
(schema gate_verdict) and saves diagnostic renders under `outdir/renders/`.
Gates never abort the pipeline: failures become `pass: false` verdicts; hard
errors (missing files, schema violations) raise.

This package also hosts the helpers shared by all gates: the 7 head-box
poses, the fixed 4-yaw view ring, the render wrapper, and the magenta-shell
color override. Geometry follows docs/CONTRACTS.md (+Y up, camera looks +Z,
yaw toward +X).

Owner: this module owns gates/*.py, pipeline/s7_gates.py and
tests/test_s7.py only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scenic import imageio
from scenic.plyio import LAYER_SHELL, SplatData, dc_to_rgb01
from scenic.rasterizer import Camera, render

# The five gates, in the order s7 runs them and embeds them in its receipt.
GATE_ORDER = ("hole", "jitter", "stereo", "people", "budgets")

# Views per pose: yaw ring at pitch 0 (fov/res come from params s7).
YAWS_DEG = (0.0, 90.0, 180.0, 270.0)

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


def view_name(pose_name: str, yaw_deg: float) -> str:
    return f"{pose_name}_yaw{int(round(yaw_deg)):03d}"


def render_view(
    splats: SplatData,
    params: dict,
    pos: np.ndarray,
    yaw_deg: float,
    override_rgb: np.ndarray | None = None,
) -> dict:
    """Render one square head-box view (pitch 0) at params s7 fov/res."""
    s7 = params["s7"]
    px = int(s7["render_px"])
    cam = Camera(
        pos=np.asarray(pos, dtype=np.float64).reshape(3),
        yaw=float(np.deg2rad(yaw_deg)),
        pitch=0.0,
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
