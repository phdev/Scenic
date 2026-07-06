"""S1 cleanplate: semi-automatic human-edit workflow for person removal.

Reads s0_ingest outputs (pano.png, person_mask.png, person_boxes.json).

- No persons detected by s0 -> passthrough: pano_clean.png is a byte-copy of
  the s0 master pano; both cleanplate gates pass trivially.
- Persons detected and no human edit present -> emit out/package/ (pano copy +
  red person-mask overlay for the editor) and raise SystemExit. No receipt is
  written: the pipeline halts unshippable until a human saves the edited plate
  next to the source pano as <pano>.cleanplate.png and the run is re-run.
- Persons detected and the edit exists -> re-entry gates:
    cleanplate_detector    RT-DETR re-run on the EDITED pano with the exact
                           s0 view recipe (8 horizon views + 2 caps);
                           pass iff 0 person hits.
    cleanplate_containment every pixel that differs from the s0 master by
                           more than DIFF_THRESH (any channel, uint8) must
                           fall inside binary_dilation(person_mask,
                           s0.mask_dilate_px); pass iff none escape.
  Failing verdicts are RECORDED (run becomes unshippable) but do not raise.

Artifacts (out/): pano_clean.png, cleanplate.json (schema cleanplate);
plus package/{pano.png, overlay.png} on the halt path only.

Notes on in-module choices (see docs/CONTRACTS.md):
- The edited plate is re-encoded via imageio.save_png (strips any editor
  metadata); the containment diff is computed on decoded pixels.
- The overlay tints mask pixels 50% red and draws a solid-red dilated
  boundary ring (OVERLAY_BOUNDARY_PX) so editors can see the allowed region.

Owner: this module owns pipeline/s1_cleanplate.py and tests/test_s1.py only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from scenic import determinism, geometry, imageio, receipts, schema, weights
from scenic.stage import Ctx

STAGE = "s1_cleanplate"

# A pixel counts as "edited" when any channel differs from the s0 master by
# more than this (uint8 levels) — tolerates lossy editor round-trips.
DIFF_THRESH = 2
# Solid-red ring width (px) drawn around the person mask in the overlay.
OVERLAY_BOUNDARY_PX = 3
# 50% red tint inside the mask.
OVERLAY_TINT = 0.5


# ---------------------------------------------------------------- helpers


def _views(s0: dict) -> list[dict]:
    """The exact s0 detection view recipe: `horizon_views` horizon views at
    yaw k*(360/n) fov horizon_fov_deg + up/down caps at fov cap_fov_deg.
    Reimplemented here (single-owner modules do not import each other)."""
    n = int(s0["horizon_views"])
    fov_h = float(s0["horizon_fov_deg"])
    fov_cap = float(s0["cap_fov_deg"])
    views = []
    for k in range(n):
        yaw = k * (360.0 / n)
        views.append(
            {
                "name": f"yaw{int(round(yaw)):03d}",
                "yaw_deg": yaw,
                "pitch_deg": 0.0,
                "fov_deg": fov_h,
            }
        )
    views.append({"name": "up", "yaw_deg": 0.0, "pitch_deg": 90.0, "fov_deg": fov_cap})
    views.append(
        {"name": "down", "yaw_deg": 0.0, "pitch_deg": -90.0, "fov_deg": fov_cap}
    )
    return views


def _detect_hits(
    pano_f32: np.ndarray, views: list[dict], view_px: int, score_min: float
) -> int:
    """Render each view from the pano and count pinned RT-DETR person hits
    (CPU, deterministic). Same recipe as s0's detection pass."""
    determinism.enforce()
    import torch
    from PIL import Image

    model, proc = weights.load_person_detector()
    person_id = weights.person_label_id(model)

    total = 0
    for v in views:
        img = geometry.render_perspective(
            pano_f32,
            v["fov_deg"],
            view_px,
            view_px,
            float(np.deg2rad(v["yaw_deg"])),
            float(np.deg2rad(v["pitch_deg"])),
        )
        u8 = np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
        pil = Image.fromarray(u8)
        inputs = proc(images=pil, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        result = proc.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([[view_px, view_px]]),
            threshold=score_min,
        )[0]
        total += sum(1 for label in result["labels"] if int(label) == person_id)
    return total


def _overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Editor overlay: person mask tinted red at 50% + a solid-red dilated
    boundary ring so the allowed edit region is unambiguous."""
    out = rgb.astype(np.float64)
    red = np.array([255.0, 0.0, 0.0])
    if mask.any():
        out[mask] = (1.0 - OVERLAY_TINT) * out[mask] + OVERLAY_TINT * red
        ring = ndimage.binary_dilation(mask, iterations=OVERLAY_BOUNDARY_PX) & ~mask
        out[ring] = red
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _containment(
    orig: np.ndarray, edited: np.ndarray, mask: np.ndarray, dilate_px: int
) -> tuple[bool, int, int]:
    """(ok, n_diff_px, n_diff_px_outside_allowed): every pixel differing by
    more than DIFF_THRESH in any channel must sit inside the dilated mask."""
    delta = np.abs(edited.astype(np.int16) - orig.astype(np.int16))
    diff = np.any(delta > DIFF_THRESH, axis=-1)
    allowed = mask
    if dilate_px > 0 and mask.any():
        allowed = ndimage.binary_dilation(mask, iterations=int(dilate_px))
    outside = diff & ~allowed
    return bool(not outside.any()), int(diff.sum()), int(outside.sum())


# -------------------------------------------------------------------- run


def run(run_dir: Path, params: dict, ctx: Ctx) -> None:
    run_dir = Path(run_dir)
    out = ctx.out(run_dir, STAGE)

    # --- s0 inputs (missing artifacts = broken chain = hard error)
    s0_out = run_dir / "s0_ingest" / "out"
    pano_path = s0_out / "pano.png"
    mask_path = s0_out / "person_mask.png"
    boxes_path = s0_out / "person_boxes.json"
    for p in (pano_path, mask_path, boxes_path):
        if not p.exists():
            raise FileNotFoundError(f"s1_cleanplate: missing s0 artifact {p}")
    boxes = schema.read_validated(boxes_path, "person_boxes")
    total_hits = int(boxes["total_hits"])

    # ------------------------------------------------ passthrough (0 hits)
    if total_hits == 0:
        (out / "pano_clean.png").write_bytes(pano_path.read_bytes())
        result = {
            "mode": "passthrough",
            "detector_hits_after": 0,
            "containment_ok": True,
        }
        schema.write_validated(out / "cleanplate.json", result, "cleanplate")
        gates = [
            {
                "gate": "cleanplate_detector",
                "pass": True,
                "metrics": {"hits": 0, "note": "carried from s0"},
                "thresholds": {"max_hits": 0},
            },
            {
                "gate": "cleanplate_containment",
                "pass": True,
                "metrics": {"diff_px": 0, "diff_px_outside_mask": 0},
                "thresholds": {"pixel_delta_max": DIFF_THRESH},
                "details": "trivial pass: no persons detected, pano untouched",
            },
        ]
        receipts.write_receipt(
            run_dir,
            STAGE,
            inputs={"pano": pano_path, "person_boxes": boxes_path},
            outputs={
                "pano_clean": out / "pano_clean.png",
                "cleanplate": out / "cleanplate.json",
            },
            params_used={},
            weights_used=[],
            gates=gates,
            notes={"mode": "passthrough", "detector_hits_before": 0},
        )
        return

    # ------------------------------------------- persons found: human loop
    s0p = params["s0"]
    edit_path = Path(str(ctx.pano_path) + ".cleanplate.png")
    orig = imageio.load_rgb(pano_path)
    mask = imageio.load_mask_png(mask_path)
    if mask.shape != orig.shape[:2]:
        raise ValueError(
            f"s1_cleanplate: person_mask {mask.shape} does not match pano "
            f"{orig.shape[:2]}"
        )

    if not edit_path.exists():
        # Emit the human-edit package and halt WITHOUT a receipt: the run is
        # unshippable until the edited plate exists.
        pkg = out / "package"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "pano.png").write_bytes(pano_path.read_bytes())
        imageio.save_png(pkg / "overlay.png", _overlay(orig, mask))
        raise SystemExit(
            f"cleanplate required: edit {pkg / 'pano.png'}, "
            f"save as {edit_path}, re-run"
        )

    # ------------------------------------------------- re-entry: gate 1+2
    edited = imageio.load_rgb(edit_path)
    if edited.shape != orig.shape:
        raise ValueError(
            f"s1_cleanplate: edited plate {edit_path.name} shape "
            f"{edited.shape} does not match s0 pano {orig.shape}"
        )

    hits_after = _detect_hits(
        edited.astype(np.float32) / 255.0,
        _views(s0p),
        int(s0p["view_px"]),
        float(s0p["person_score_min"]),
    )
    detector_gate = {
        "gate": "cleanplate_detector",
        "pass": hits_after == 0,
        "metrics": {"hits": int(hits_after)},
        "thresholds": {"max_hits": 0, "score_min": float(s0p["person_score_min"])},
        "details": "RT-DETR re-run on the edited pano (s0 view recipe)",
    }

    containment_ok, n_diff, n_outside = _containment(
        orig, edited, mask, int(s0p["mask_dilate_px"])
    )
    containment_gate = {
        "gate": "cleanplate_containment",
        "pass": containment_ok,
        "metrics": {"diff_px": n_diff, "diff_px_outside_mask": n_outside},
        "thresholds": {
            "pixel_delta_max": DIFF_THRESH,
            "mask_dilate_px": int(s0p["mask_dilate_px"]),
        },
    }

    # Re-encode (strips editor metadata; PNG is lossless so pixels are exact).
    imageio.save_png(out / "pano_clean.png", edited)
    result = {
        "mode": "edited",
        "detector_hits_after": int(hits_after),
        "containment_ok": containment_ok,
    }
    schema.write_validated(out / "cleanplate.json", result, "cleanplate")

    # Failing verdicts are recorded (unshippable) but never raise.
    receipts.write_receipt(
        run_dir,
        STAGE,
        inputs={
            "pano": pano_path,
            "person_boxes": boxes_path,
            "person_mask": mask_path,
            "cleanplate_edit": edit_path,
        },
        outputs={
            "pano_clean": out / "pano_clean.png",
            "cleanplate": out / "cleanplate.json",
        },
        params_used={"s0": s0p},
        weights_used=["rtdetr_r18"],
        gates=[detector_gate, containment_gate],
        notes={"mode": "edited", "detector_hits_before": total_hits},
    )
