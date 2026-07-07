"""fidelity_at_origin gate: does the shipped scene, rendered from the origin,
still look like the source pano?

The v2 quality pass replaces "trust the pipeline" with a measured floor. We
tile the sphere into a deterministic perspective grid (metrics.equirect_tile_
views), and for every tile:

- render the PRIMARY splats (normal colors, no override) from the origin
  (Camera pos 0) at the tile's yaw/pitch/fov via scenic.rasterizer.render;
- sample the SOURCE pano into the SAME perspective via
  geometry.render_perspective — the s1 clean plate if present, else the s0
  ingest master;
- score SSIM (the enforced fidelity metric) between the two tiles.

PASS iff the WORST tile SSIM >= s7.fidelity.ssim_worst_tile_min AND the MEAN
tile SSIM >= s7.fidelity.ssim_mean_min. Both are float64 SSIM in [-1, 1].

LPIPS is ADVISORY ONLY and NEVER affects pass: if metrics.lpips_advisory_
available() reports a usable local install we record `lpips_mean`; otherwise
metrics carry `lpips: "advisory_unavailable"` and details carry the reason
(the VGG/ImageNet weight provenance is an OPEN license question, so those
weights are never in the enforced tree — see scenic/metrics.py).

Diagnostics: the worst tile's render + source-sample land under
outdir/renders/ as fidelity_worst_{render,source}.png.

Owner: this module owns gates/*.py, pipeline/s7_gates.py and
tests/test_s7.py only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scenic import geometry, imageio, metrics
from scenic.plyio import SplatData
from scenic.rasterizer import Camera, render

from gates import save_render


def _source_pano_path(run_dir: Path) -> Path:
    """The s1 clean plate if it exists, else the s0 ingest master. Raises if
    neither is present (a hard error: there is nothing to compare against)."""
    run_dir = Path(run_dir)
    clean = run_dir / "s1_cleanplate" / "out" / "pano_clean.png"
    if clean.exists():
        return clean
    ingest = run_dir / "s0_ingest" / "out" / "pano.png"
    if ingest.exists():
        return ingest
    raise FileNotFoundError(
        f"fidelity gate: no source pano at {clean} or {ingest}"
    )


def _to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)


def run_gate(
    splats: SplatData,
    params: dict,
    outdir: Path | str,
    run_dir: Path | str | None = None,
) -> dict:
    outdir = Path(outdir)
    if run_dir is None:
        # standard <run>/s7_gates/out layout
        run_dir = outdir.resolve().parent.parent
    fid = params["s7"]["fidelity"]
    tiles_lon = int(fid["tiles_lon"])
    tiles_lat = int(fid["tiles_lat"])
    px = int(fid["render_px"])
    worst_min = float(fid["ssim_worst_tile_min"])
    mean_min = float(fid["ssim_mean_min"])

    pano = imageio.load_rgb(_source_pano_path(run_dir)).astype(np.float64) / 255.0

    views = metrics.equirect_tile_views(tiles_lon, tiles_lat)

    per_tile: list[dict] = []
    ssims: list[float] = []
    worst_ssim = float("inf")
    worst_name = ""
    worst_render: np.ndarray | None = None
    worst_source: np.ndarray | None = None
    for name, yaw_deg, pitch_deg, fov_deg in views:
        yaw = float(np.deg2rad(yaw_deg))
        pitch = float(np.deg2rad(pitch_deg))
        cam = Camera(pos=np.zeros(3, dtype=np.float64), yaw=yaw, pitch=pitch)
        rendered = render(splats, cam, px, px, float(fov_deg))["rgb"]  # uint8
        source01 = geometry.render_perspective(
            pano, float(fov_deg), px, px, yaw, pitch
        )  # float01
        s = metrics.ssim(rendered, source01)
        ssims.append(s)
        per_tile.append(
            {
                "tile": name,
                "ssim": float(s),
                "yaw_deg": float(yaw_deg),
                "pitch_deg": float(pitch_deg),
                "fov_deg": float(fov_deg),
            }
        )
        if s < worst_ssim:
            worst_ssim = s
            worst_name = name
            worst_render = rendered
            worst_source = _to_uint8(source01)

    n_tiles = len(ssims)
    mean_ssim = float(np.mean(ssims)) if ssims else 0.0
    worst_ssim = float(worst_ssim) if ssims else 0.0

    if worst_render is not None:
        save_render(outdir, "fidelity_worst_render.png", worst_render)
    if worst_source is not None:
        save_render(outdir, "fidelity_worst_source.png", worst_source)

    passed = worst_ssim >= worst_min and mean_ssim >= mean_min

    metrics_out: dict = {
        "ssim_worst_tile": float(worst_ssim),
        "ssim_mean": float(mean_ssim),
        "worst_tile": worst_name,
        "n_tiles": int(n_tiles),
    }
    details: dict = {"per_tile": per_tile, "worst_tile": worst_name}

    # LPIPS: advisory only, never affects pass.
    available, reason = metrics.lpips_advisory_available()
    if available:
        try:
            lpips_mean = _advisory_lpips_mean(splats, pano, params, run_dir)
            metrics_out["lpips_mean"] = float(lpips_mean)
        except Exception as exc:  # never fail the gate on an advisory metric
            metrics_out["lpips"] = "advisory_unavailable"
            details["lpips_reason"] = f"advisory compute failed: {exc}"
    else:
        metrics_out["lpips"] = "advisory_unavailable"
        details["lpips_reason"] = reason

    return {
        "gate": "fidelity_at_origin",
        "pass": bool(passed),
        "metrics": metrics_out,
        "thresholds": {
            "ssim_worst_tile_min": worst_min,
            "ssim_mean_min": mean_min,
        },
        "details": details,
    }


def _advisory_lpips_mean(
    splats: SplatData, pano: np.ndarray, params: dict, run_dir: Path
) -> float:
    """Mean advisory LPIPS over the tile grid (only reached when
    metrics.lpips_advisory_available() is True — a human-provisioned local
    install outside the enforced weights tree). Deterministic CPU torch."""
    from scenic import determinism

    determinism.enforce()
    import lpips as lpips_pkg
    import torch

    fid = params["s7"]["fidelity"]
    px = int(fid["render_px"])
    net = lpips_pkg.LPIPS(net="vgg", verbose=False).eval()
    views = metrics.equirect_tile_views(
        int(fid["tiles_lon"]), int(fid["tiles_lat"])
    )
    vals: list[float] = []
    with torch.no_grad():
        for _name, yaw_deg, pitch_deg, fov_deg in views:
            yaw = float(np.deg2rad(yaw_deg))
            pitch = float(np.deg2rad(pitch_deg))
            cam = Camera(pos=np.zeros(3, dtype=np.float64), yaw=yaw, pitch=pitch)
            rendered = (
                render(splats, cam, px, px, float(fov_deg))["rgb"].astype(
                    np.float64
                )
                / 255.0
            )
            source01 = geometry.render_perspective(
                pano, float(fov_deg), px, px, yaw, pitch
            )

            def _t(img: np.ndarray) -> "torch.Tensor":
                a = np.ascontiguousarray(img.transpose(2, 0, 1))[None]
                return torch.from_numpy((a * 2.0 - 1.0).astype(np.float32))

            vals.append(float(net(_t(rendered), _t(source01)).item()))
    return float(np.mean(vals)) if vals else 0.0
