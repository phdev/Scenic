"""Tests for pipeline/s0_ingest.py (stage-local, no full pipeline).

The shared `s0_run` fixture executes the real stage once on a tiny synthetic
pano; that run loads the ACTUAL pinned RT-DETR person detector from ./weights
(~80MB, CPU-only) — that is intentional and fine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline import s0_ingest
from scenic import determinism, hashing, imageio, params as params_mod, receipts, schema
from scenic.stage import Ctx

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- fixtures


def _write_pano(path: Path, w: int = 256, h: int = 128) -> bytes:
    """Tiny smooth-gradient equirect pano: no people, no text/watermark."""
    xs = np.linspace(40.0, 200.0, w)[None, :]
    ys = np.linspace(30.0, 180.0, h)[:, None]
    r = np.broadcast_to(xs, (h, w))
    g = np.broadcast_to(ys, (h, w))
    b = (r + g) / 2.0
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    imageio.save_png(path, rgb)
    return path.read_bytes()


def _write_sidecar(pano: Path, obj: dict | None = None) -> Path:
    sc = pano.with_suffix(pano.suffix + ".license.json")
    if obj is None:
        obj = {
            "source": "synthetic fixture",
            "license_id": "CC0-1.0",
            "scope_note": "unit test pano, generated in tests/test_s0.py",
        }
    hashing.write_json(sc, obj)
    return sc


def _ctx(pano: Path) -> Ctx:
    return Ctx(
        repo_root=REPO_ROOT,
        pano_path=pano,
        sidecar_path=pano.with_suffix(pano.suffix + ".license.json"),
        params_path=REPO_ROOT / "params.yaml",
        weights_dir=REPO_ROOT / "weights",
    )


@pytest.fixture(scope="module")
def params() -> dict:
    return params_mod.load(REPO_ROOT / "params.yaml")


@pytest.fixture(scope="module")
def s0_run(tmp_path_factory, params):
    """Run the full stage ONCE on the synthetic pano and share the results.

    NOTE: this loads real RT-DETR weights (~80MB) and runs 10 detector
    forwards on CPU — slow but deliberate (asserts 0 persons on a smooth
    gradient with the shipping detector).
    """
    determinism.set_seed(int(params.get("seed", 0)))
    root = tmp_path_factory.mktemp("s0")
    run_dir = root / "run"
    run_dir.mkdir()
    pano = root / "pano.png"
    pano_bytes = _write_pano(pano)
    _write_sidecar(pano)
    s0_ingest.run(run_dir, params, _ctx(pano))
    return {
        "run_dir": run_dir,
        "out": run_dir / "s0_ingest" / "out",
        "pano_bytes": pano_bytes,
    }


# ------------------------------------------------------------------ tests


def test_receipt_exists_and_valid(s0_run):
    # read_receipt validates against the receipt schema (raises if invalid)
    rec = receipts.read_receipt(s0_run["run_dir"], "s0_ingest")
    assert rec["stage"] == "s0_ingest"
    assert set(rec["outputs"]) == {
        "pano",
        "pano_meta",
        "person_boxes",
        "person_mask",
        "watermark",
    }
    assert set(rec["inputs"]) == {"pano", "sidecar"}
    assert [wr["key"] for wr in rec["weights"]] == ["rtdetr_r18"]
    assert rec["notes"]["n_person_hits"] == 0
    # no absolute paths recorded
    for art in list(rec["inputs"].values()) + list(rec["outputs"].values()):
        assert not art["path"].startswith("/")


def test_pano_meta(s0_run, params):
    meta = schema.read_validated(s0_run["out"] / "pano_meta.json", "pano_meta")
    assert meta["source_sha256"] == hashing.sha256_bytes(s0_run["pano_bytes"])
    assert meta["width"] == 256
    assert meta["height"] == 128
    # sidecar carries no camera_height_m -> default from params
    assert meta["camera_height_m"] == params["camera_height_m_default"]
    assert meta["license"]["license_id"] == "CC0-1.0"
    # normalized master exists and round-trips to the same pixels/shape
    pano = imageio.load_rgb(s0_run["out"] / "pano.png")
    assert pano.shape == (128, 256, 3)


def test_watermark_gate_passes_on_smooth_pano(s0_run, params):
    wm = schema.read_validated(s0_run["out"] / "watermark.json", "watermark")
    assert wm["suspicious"] is False
    assert wm["edge_density"] <= params["s0"]["watermark_edge_density_max"]
    assert wm["band_pitch_deg"] == params["s0"]["nadir_band_pitch_deg"]
    rec = receipts.read_receipt(s0_run["run_dir"], "s0_ingest")
    gates = {g["gate"]: g for g in rec["gates"]}
    assert gates["watermark"]["pass"] is True
    assert gates["watermark"]["metrics"]["edge_density"] == wm["edge_density"]


def test_detector_finds_zero_persons(s0_run, params):
    # This ran the REAL pinned RT-DETR detector (~80MB weights, CPU) inside
    # the s0_run fixture — a smooth 256x128 gradient must yield 0 persons.
    pb = schema.read_validated(s0_run["out"] / "person_boxes.json", "person_boxes")
    assert pb["total_hits"] == 0
    n_views = params["s0"]["horizon_views"] + 2
    assert len(pb["views"]) == n_views
    assert all(v["boxes"] == [] for v in pb["views"])
    names = [v["name"] for v in pb["views"]]
    assert names[-2:] == ["up", "down"]


def test_person_mask_empty(s0_run):
    mask = imageio.load_mask_png(s0_run["out"] / "person_mask.png")
    assert mask.shape == (128, 256)
    assert not mask.any()


def test_missing_sidecar_raises(tmp_path, params):
    pano = tmp_path / "pano.png"
    _write_pano(pano)  # no sidecar written
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(SystemExit, match="license sidecar missing"):
        s0_ingest.run(run_dir, params, _ctx(pano))


def test_invalid_sidecar_raises(tmp_path, params):
    pano = tmp_path / "pano.png"
    _write_pano(pano)
    _write_sidecar(pano, {"source": "x"})  # missing required fields
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(SystemExit, match="failed schema license_sidecar"):
        s0_ingest.run(run_dir, params, _ctx(pano))


def test_non_2to1_aspect_raises(tmp_path, params):
    pano = tmp_path / "pano.png"
    _write_pano(pano, w=250, h=128)  # not 2:1
    _write_sidecar(pano)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="2:1"):
        s0_ingest.run(run_dir, params, _ctx(pano))


def test_box_reprojection_masks_pixels(params):
    """Unit-check the reprojection path without the detector: a fake box in
    the front view must mark equirect pixels near (yaw 0, pitch 0)."""
    view_px = int(params["s0"]["view_px"])
    fake_views = [
        {
            "name": "yaw000",
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "fov_deg": float(params["s0"]["horizon_fov_deg"]),
            "boxes": [
                {
                    "xyxy": [
                        view_px * 0.4,
                        view_px * 0.4,
                        view_px * 0.6,
                        view_px * 0.6,
                    ],
                    "score": 0.9,
                }
            ],
        }
    ]
    w, h = 256, 128
    mask = s0_ingest._person_mask(w, h, fake_views, view_px, dilate_px=0)
    assert mask.any()
    # center of the front view = equirect center pixel (theta=0 -> +Z)
    assert mask[h // 2, w // 2]
    # nothing behind the camera (yaw 180 -> u=0 column region)
    assert not mask[:, :10].any()
    # dilation strictly grows the mask
    dilated = s0_ingest._person_mask(
        w, h, fake_views, view_px, dilate_px=int(params["s0"]["mask_dilate_px"])
    )
    assert dilated.sum() > mask.sum()
    assert (dilated & mask).sum() == mask.sum()
