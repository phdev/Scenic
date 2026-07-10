"""Jitter gate: a sub-millimeter head translation must not visibly change
the image (splat popping / z-fighting detector).

For each center-pose view in the standard set (4 pitch-0 yaws + the
straight-down nadir view), renders P = origin and P + (s7.jitter_offset_m,
0, 0); per-view energy = mean |rgb_a - rgb_b| / 255 over all pixels and
channels. The verdict gates on the WORST per-view energy (metrics key
"energy"); per-view energies are recorded in details. FAIL if the worst
energy > s7.jitter_energy_max.

Diagnostics: the yaw-0 render pair under outdir/renders/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scenic.plyio import SplatData

from gates import pose_views, render_view, save_render, view_name


def run_gate(splats: SplatData, params: dict, outdir: Path | str) -> dict:
    s7 = params["s7"]
    offset_m = float(s7["jitter_offset_m"])
    energy_max = float(s7["jitter_energy_max"])

    per_view: list[dict] = []
    for yaw, pitch in pose_views():
        name = view_name("center", yaw, pitch)
        base = render_view(splats, params, np.zeros(3), yaw, pitch)
        moved = render_view(
            splats, params, np.array([offset_m, 0.0, 0.0]), yaw, pitch
        )
        if yaw == 0.0 and pitch == 0.0:
            save_render(outdir, "jitter_base.png", base["rgb"])
            save_render(outdir, "jitter_offset.png", moved["rgb"])
        energy = float(
            np.mean(
                np.abs(
                    base["rgb"].astype(np.float64)
                    - moved["rgb"].astype(np.float64)
                )
            )
            / 255.0
        )
        per_view.append({"view": name, "energy": energy})

    # Verdict gates on the worst view; ties keep the first (fixed view order).
    worst_entry = max(per_view, key=lambda pv: pv["energy"])
    worst = float(worst_entry["energy"])
    worst_view = str(worst_entry["view"])

    return {
        "gate": "jitter",
        "pass": bool(worst <= energy_max),
        "metrics": {"energy": worst},
        "thresholds": {
            "jitter_energy_max": energy_max,
            "jitter_offset_m": offset_m,
        },
        "details": {
            "pose": "center",
            "per_view": per_view,
            "worst_view": worst_view,
        },
    }
