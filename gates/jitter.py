"""Jitter gate: a sub-millimeter head translation must not visibly change
the image (splat popping / z-fighting detector).

Renders the center pose (yaw 0) at P = origin and P + (s7.jitter_offset_m,
0, 0); energy = mean |rgb_a - rgb_b| / 255 over all pixels and channels.
FAIL if energy > s7.jitter_energy_max.

Diagnostics: both renders under outdir/renders/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scenic.plyio import SplatData

from gates import render_view, save_render


def run_gate(splats: SplatData, params: dict, outdir: Path | str) -> dict:
    s7 = params["s7"]
    offset_m = float(s7["jitter_offset_m"])
    energy_max = float(s7["jitter_energy_max"])

    base = render_view(splats, params, np.zeros(3), 0.0)
    moved = render_view(splats, params, np.array([offset_m, 0.0, 0.0]), 0.0)
    save_render(outdir, "jitter_base.png", base["rgb"])
    save_render(outdir, "jitter_offset.png", moved["rgb"])

    energy = float(
        np.mean(
            np.abs(
                base["rgb"].astype(np.float64) - moved["rgb"].astype(np.float64)
            )
        )
        / 255.0
    )
    return {
        "gate": "jitter",
        "pass": bool(energy <= energy_max),
        "metrics": {"energy": energy},
        "thresholds": {
            "jitter_energy_max": energy_max,
            "jitter_offset_m": offset_m,
        },
        "details": {"pose": "center", "yaw_deg": 0.0},
    }
