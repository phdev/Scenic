"""People gate: no recognizable person may survive into the shipped scene.

Renders all 7x4 head-box views with normal colors (uint8) and runs the
pinned RT-DETR person detector (weights key rtdetr_r18; CPU, deterministic,
scenic.determinism.enforce() first, torch.no_grad()). Detections are
post-processed at threshold 0 so the max person score is observable even
when no detection crosses the gate threshold.

FAIL if any person detection scores >= s7.people_score_min in any view.
Metrics: max_score (0.0 if none), n_detections (count at/above threshold).

Diagnostics: center-pose normal-color renders (all 4 yaws, reused by s8)
under outdir/renders/.
"""
from __future__ import annotations

from pathlib import Path

from scenic import determinism, weights
from scenic.plyio import SplatData

from gates import YAWS_DEG, head_box_poses, render_view, save_render, view_name


def run_gate(splats: SplatData, params: dict, outdir: Path | str) -> dict:
    determinism.enforce()
    import torch
    from PIL import Image

    s7 = params["s7"]
    score_min = float(s7["people_score_min"])
    px = int(s7["render_px"])

    model, proc = weights.load_person_detector()
    person_id = weights.person_label_id(model)

    per_view: list[dict] = []
    max_score = 0.0
    n_detections = 0
    for pose_name, pos in head_box_poses(params):
        for yaw in YAWS_DEG:
            name = view_name(pose_name, yaw)
            out = render_view(splats, params, pos, yaw)
            if pose_name == "center":
                save_render(outdir, f"center_yaw{int(yaw):03d}_normal.png",
                            out["rgb"])
            inputs = proc(images=Image.fromarray(out["rgb"]),
                          return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            result = proc.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor([[px, px]]),
                threshold=0.0,
            )[0]
            person = result["labels"] == person_id
            scores = result["scores"][person]
            view_max = float(scores.max()) if scores.numel() else 0.0
            view_hits = int((scores >= score_min).sum())
            max_score = max(max_score, view_max)
            n_detections += view_hits
            per_view.append(
                {"view": name, "max_person_score": view_max,
                 "n_detections": view_hits}
            )

    return {
        "gate": "people",
        "pass": bool(n_detections == 0),
        "metrics": {
            "max_score": float(max_score),
            "n_detections": int(n_detections),
        },
        "thresholds": {"people_score_min": score_min},
        "details": {"per_view": per_view},
    }
