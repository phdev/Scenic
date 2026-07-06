"""Tests for pipeline/s2_depth.py. Runs the real DA-V2-Small on CPU (6 faces
per run, two runs for the determinism check) — expect tens of seconds."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scenic import determinism

determinism.set_env()  # before any torch import

from scenic import hashing, imageio, params as params_mod, receipts, schema
from scenic.stage import Ctx
from pipeline import s2_depth

REPO = Path(__file__).resolve().parent.parent


def _make_pano(w: int = 512, h: int = 256) -> np.ndarray:
    """Synthetic 2:1 pano: smooth sky gradient in the top half, high-contrast
    textured ground in the bottom half (no fixture generator in tools/ yet)."""
    uu, vv = np.meshgrid(
        np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64)
    )
    t = vv / (h - 1)
    img = np.empty((h, w, 3), dtype=np.float64)
    img[..., 0] = 0.35 + 0.30 * t
    img[..., 1] = 0.55 + 0.25 * t
    img[..., 2] = 0.95 - 0.15 * t
    ground = vv >= h // 2
    checker = (((uu // 8) + (vv // 8)) % 2).astype(np.float64)
    tex = 0.20 + 0.55 * checker + 0.15 * np.sin(uu * 0.7) * np.cos(vv * 0.9)
    for c, base in enumerate((0.85, 0.65, 0.45)):
        img[..., c] = np.where(ground, base * np.clip(tex, 0.0, 1.0), img[..., c])
    return (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _run_stage(run_dir: Path) -> None:
    s0_out = run_dir / "s0_ingest" / "out"
    s0_out.mkdir(parents=True)
    pano_path = s0_out / "pano.png"
    imageio.save_png(pano_path, _make_pano())
    p = params_mod.load(REPO / "params.yaml")
    determinism.enforce()
    determinism.set_seed(p.get("seed", 0))
    ctx = Ctx(
        repo_root=REPO,
        pano_path=pano_path,
        sidecar_path=pano_path.with_suffix(".png.license.json"),
        params_path=REPO / "params.yaml",
        weights_dir=REPO / "weights",
    )
    s2_depth.run(run_dir, p, ctx)


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> tuple[Path, Path]:
    d1 = tmp_path_factory.mktemp("s2_run_a")
    d2 = tmp_path_factory.mktemp("s2_run_b")
    _run_stage(d1)
    _run_stage(d2)
    return d1, d2


def test_depth_shape_dtype_and_ground_finite(runs):
    run_dir, _ = runs
    depth = imageio.load_npy(run_dir / "s2_depth" / "out" / "depth_rel.npy")
    # pano 512 wide -> out_w = min(2048, 512) = 512, out_h = 256
    assert depth.shape == (256, 512)
    assert depth.dtype == np.float32
    assert not np.isnan(depth).any()
    assert (depth[np.isfinite(depth)] > 0).all()
    # ground region (bottom quarter, pitch < 0) must be entirely finite
    assert np.isfinite(depth[192:]).all()


def test_sky_mask_top_half_only(runs):
    run_dir, _ = runs
    mask = imageio.load_mask_png(run_dir / "s2_depth" / "out" / "sky_mask.png")
    assert mask.shape == (256, 512)
    assert mask.sum() > 0, "expected a non-empty sky mask on the sky-gradient pano"
    # sky requires pitch > 0 deg -> rows 0..127 only; bottom quarter clean
    assert mask[128:].sum() == 0
    assert mask[192:].sum() == 0
    top = int(mask[:128].sum())
    assert top == int(mask.sum())  # "mostly in the top half" (here: entirely)
    # sky pixels are inf in the depth map, and only sky pixels are inf
    depth = imageio.load_npy(run_dir / "s2_depth" / "out" / "depth_rel.npy")
    assert np.array_equal(np.isinf(depth), mask)


def test_depth_meta_schema_and_affines(runs):
    run_dir, _ = runs
    meta = schema.read_validated(
        run_dir / "s2_depth" / "out" / "depth_meta.json", "depth_meta"
    )
    assert meta["backend"] == "depth_anything_v2_small"
    assert (meta["fused_w"], meta["fused_h"]) == (512, 256)
    assert (meta["out_w"], meta["out_h"]) == (512, 256)
    assert meta["overlap_residual_log"] >= 0.0
    assert meta["median_divisor"] > 0.0
    faces = meta["faces"]
    assert len(faces) == 6
    names = [f["name"] for f in faces]
    assert names == ["front", "right", "back", "left", "up", "down"]
    for f in faces:
        assert np.isfinite(f["affine_a"])
        assert np.isfinite(f["affine_b"])
    # face 0 anchored
    assert faces[0]["affine_a"] == 1.0
    assert faces[0]["affine_b"] == 0.0


def test_receipt_valid(runs):
    run_dir, _ = runs
    rec = receipts.read_receipt(run_dir, "s2_depth")  # schema-validated
    assert rec["stage"] == "s2_depth"
    assert set(rec["outputs"]) == {"depth_rel", "sky_mask", "depth_meta"}
    assert list(rec["inputs"]) == ["pano"]  # s0 fallback path in this fixture
    assert [w["key"] for w in rec["weights"]] == ["depth_anything_v2_small"]
    assert set(rec["params_used"]) == {"resolutions", "cubemap", "s2"}
    assert "overlap_residual_log" in rec["notes"]
    assert "median_divisor" in rec["notes"]
    # relative paths only
    for art in list(rec["inputs"].values()) + list(rec["outputs"].values()):
        assert not art["path"].startswith("/")


def test_determinism_bit_identical(runs):
    d1, d2 = runs
    for rel in ("depth_rel.npy", "sky_mask.png", "depth_meta.json"):
        h1 = hashing.sha256_file(d1 / "s2_depth" / "out" / rel)
        h2 = hashing.sha256_file(d2 / "s2_depth" / "out" / rel)
        assert h1 == h2, f"{rel} differs across identical runs"


def test_resolve_input_prefers_cleanplate(tmp_path):
    with pytest.raises(FileNotFoundError):
        s2_depth._resolve_input(tmp_path)
    s0 = tmp_path / "s0_ingest" / "out"
    s0.mkdir(parents=True)
    (s0 / "pano.png").write_bytes(b"x")
    key, path = s2_depth._resolve_input(tmp_path)
    assert key == "pano"
    assert path == s0 / "pano.png"
    s1 = tmp_path / "s1_cleanplate" / "out"
    s1.mkdir(parents=True)
    (s1 / "pano_clean.png").write_bytes(b"y")
    key, path = s2_depth._resolve_input(tmp_path)
    assert key == "pano_clean"
    assert path == s1 / "pano_clean.png"
