"""S0 ingest: normalize the source pano, enforce the license sidecar, run the
nadir watermark/text heuristic, detect people across perspective views, and
reproject detections into an equirect person mask.

Artifacts (out/): pano.png, pano_meta.json, watermark.json, person_boxes.json,
person_mask.png. See docs/CONTRACTS.md ("Stage IO summary").

Owner: this module owns pipeline/s0_ingest.py and tests/test_s0.py only.
"""
from __future__ import annotations

from pathlib import Path

import jsonschema
import numpy as np
from scipy import ndimage

from scenic import determinism, geometry, hashing, imageio, receipts, schema, weights
from scenic.stage import Ctx

STAGE = "s0_ingest"

# A band pixel counts as an "edge" when its normalized Sobel gradient
# magnitude exceeds this. Normalization: Sobel responses on [0,1] grayscale
# are divided by 4 so a full-contrast unit step edge measures ~1.0.
EDGE_GRAD_THRESH = 0.15


# ---------------------------------------------------------------- helpers


def _load_sidecar(sidecar_path: Path) -> dict:
    """License sidecar is mandatory: missing or schema-invalid => SystemExit."""
    if not sidecar_path.exists():
        raise SystemExit(
            f"s0_ingest: license sidecar missing: {sidecar_path.name} — every "
            "pano needs a <pano>.license.json validating schema license_sidecar"
        )
    obj = hashing.read_json(sidecar_path)
    try:
        schema.validate(obj, "license_sidecar")
    except jsonschema.ValidationError as e:
        raise SystemExit(
            f"s0_ingest: license sidecar {sidecar_path.name} failed schema "
            f"license_sidecar: {e.message}"
        ) from e
    return obj


def _gray01(rgb: np.ndarray) -> np.ndarray:
    """Rec.601 luma on [0,1], float64 (deterministic fixed weights)."""
    f = rgb.astype(np.float64) / 255.0
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def _nadir_edge_density(rgb: np.ndarray, band_pitch_deg: float) -> float:
    """Fraction of nadir-band pixels whose normalized Sobel gradient
    magnitude exceeds EDGE_GRAD_THRESH. Band = equirect rows whose pixel
    pitch (lat) is below band_pitch_deg."""
    gray = _gray01(rgb)
    # 'reflect' mode: no fake wrap edges at the poles; the lon seam merely
    # loses cross-seam gradients, which is acceptable for this heuristic.
    gx = ndimage.sobel(gray, axis=1, mode="reflect") / 4.0
    gy = ndimage.sobel(gray, axis=0, mode="reflect") / 4.0
    mag = np.hypot(gx, gy)
    h = gray.shape[0]
    row_pitch_deg = 90.0 - (np.arange(h, dtype=np.float64) + 0.5) / h * 180.0
    band = row_pitch_deg < band_pitch_deg
    if not band.any():
        return 0.0
    return float(np.mean(mag[band] > EDGE_GRAD_THRESH))


def _views(s0: dict) -> list[dict]:
    """8 horizon views (yaw k*45deg, pitch 0) + up/down caps. Fixed order."""
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


def _detect_persons(
    pano_f32: np.ndarray, views: list[dict], view_px: int, score_min: float
) -> tuple[list[dict], int]:
    """Render each view from the pano and run the pinned RT-DETR person
    detector (CPU, deterministic). Returns (views-with-boxes, total_hits)."""
    determinism.enforce()
    import torch
    from PIL import Image

    model, proc = weights.load_person_detector()
    person_id = weights.person_label_id(model)

    out_views: list[dict] = []
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
        boxes = []
        for score, label, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            if int(label) != person_id:
                continue
            boxes.append(
                {"xyxy": [float(c) for c in box.tolist()], "score": float(score)}
            )
        total += len(boxes)
        out_views.append({**v, "boxes": boxes})
    return out_views, total


def _person_mask(
    w: int, h: int, det_views: list[dict], view_px: int, dilate_px: int
) -> np.ndarray:
    """Union of all detection boxes reprojected onto the equirect grid, then
    binary-dilated (4-connectivity cross, fixed iteration count)."""
    mask = np.zeros((h, w), dtype=bool)
    hit_views = [v for v in det_views if v["boxes"]]
    if hit_views:
        dirs = geometry.equirect_dirs(w, h)
        for v in hit_views:
            uv01, inside, _ = geometry.face_project(
                dirs,
                float(np.deg2rad(v["yaw_deg"])),
                float(np.deg2rad(v["pitch_deg"])),
                v["fov_deg"],
            )
            u, vv = uv01[..., 0], uv01[..., 1]
            for b in v["boxes"]:
                x0, y0, x1, y1 = [
                    float(np.clip(c / view_px, 0.0, 1.0)) for c in b["xyxy"]
                ]
                mask |= inside & (u >= x0) & (u <= x1) & (vv >= y0) & (vv <= y1)
    if dilate_px > 0 and mask.any():
        # Note: dilation does not wrap across the longitude seam; the
        # per-view frustums overlap enough that this is acceptable for a
        # conservative person mask.
        mask = ndimage.binary_dilation(mask, iterations=int(dilate_px))
    return mask


# -------------------------------------------------------------------- run


def run(run_dir: Path, params: dict, ctx: Ctx) -> None:
    out = ctx.out(run_dir, STAGE)
    s0 = params["s0"]

    # --- source pano: hash bytes, decode, enforce 2:1, write normalized master
    pano_path = Path(ctx.pano_path)
    source_sha256 = hashing.sha256_bytes(pano_path.read_bytes())
    rgb = imageio.load_rgb(pano_path)
    h, w = rgb.shape[:2]
    if w != 2 * h:
        raise ValueError(
            f"s0_ingest: pano must be 2:1 equirect, got {w}x{h} ({pano_path.name})"
        )
    imageio.save_png(out / "pano.png", rgb)

    # --- license sidecar (mandatory) + pano meta
    sidecar = _load_sidecar(Path(ctx.sidecar_path))
    camera_height_m = float(
        sidecar.get("camera_height_m", params["camera_height_m_default"])
    )
    meta = {
        "width": int(w),
        "height": int(h),
        "source_sha256": source_sha256,
        "license": sidecar,
        "camera_height_m": camera_height_m,
    }
    schema.write_validated(out / "pano_meta.json", meta, "pano_meta")

    # --- nadir watermark/text heuristic + gate verdict
    band_pitch_deg = float(s0["nadir_band_pitch_deg"])
    edge_density = _nadir_edge_density(rgb, band_pitch_deg)
    density_max = float(s0["watermark_edge_density_max"])
    suspicious = bool(edge_density > density_max)
    watermark = {
        "edge_density": edge_density,
        "suspicious": suspicious,
        "band_pitch_deg": band_pitch_deg,
        "grad_threshold": EDGE_GRAD_THRESH,
    }
    schema.write_validated(out / "watermark.json", watermark, "watermark")
    watermark_gate = {
        "gate": "watermark",
        "pass": not suspicious,
        "metrics": {"edge_density": edge_density},
        "thresholds": {"max": density_max},
    }

    # --- person detection across perspective views
    pano_f32 = rgb.astype(np.float32) / 255.0
    view_px = int(s0["view_px"])
    det_views, total_hits = _detect_persons(
        pano_f32, _views(s0), view_px, float(s0["person_score_min"])
    )
    schema.write_validated(
        out / "person_boxes.json",
        {"views": det_views, "total_hits": int(total_hits)},
        "person_boxes",
    )

    # --- reproject boxes to the equirect person mask
    mask = _person_mask(w, h, det_views, view_px, int(s0["mask_dilate_px"]))
    imageio.save_mask_png(out / "person_mask.png", mask)

    # --- receipt (exactly one per stage run)
    receipts.write_receipt(
        run_dir,
        STAGE,
        inputs={"pano": pano_path, "sidecar": Path(ctx.sidecar_path)},
        outputs={
            "pano": out / "pano.png",
            "pano_meta": out / "pano_meta.json",
            "person_boxes": out / "person_boxes.json",
            "person_mask": out / "person_mask.png",
            "watermark": out / "watermark.json",
        },
        params_used={
            "camera_height_m_default": params["camera_height_m_default"],
            "s0": s0,
        },
        weights_used=["rtdetr_r18"],
        gates=[watermark_gate],
        notes={"n_person_hits": int(total_hits)},
    )
