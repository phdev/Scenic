"""s3_layers tests: synthetic 256x128 scene — a vertical foreground slab at
2 m occluding a 10 m background, plus a sky band. Verifies edge detection,
the analytic band-width derivation, push-pull background fill quality,
schema validity, and bit-identical determinism across runs."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pipeline import s3_layers
from scenic import determinism, imageio, params as params_mod, receipts, schema
from scenic.stage import Ctx

REPO = Path(__file__).resolve().parent.parent

H, W = 128, 256
SKY_ROWS = 16
SLAB = (96, 160)  # columns of the 2 m foreground slab
D_FG, D_BG = 2.0, 10.0
FG_COLOR = np.array([200, 60, 60], np.uint8)
BG_COLOR = np.array([60, 200, 60], np.uint8)
SKY_COLOR = np.array([100, 150, 255], np.uint8)

OUT_FILES = [
    "fg_rgb.png",
    "fg_depth.npy",
    "fg_mask.png",
    "bg_rgb.png",
    "bg_depth.npy",
    "bg_mask.png",
    "layers.json",
]


def make_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.full((H, W), D_BG, np.float32)
    depth[SKY_ROWS:, SLAB[0] : SLAB[1]] = D_FG
    depth[:SKY_ROWS, :] = np.inf
    sky = np.zeros((H, W), bool)
    sky[:SKY_ROWS] = True
    rgb = np.empty((H, W, 3), np.uint8)
    rgb[:] = BG_COLOR
    rgb[:SKY_ROWS] = SKY_COLOR
    rgb[SKY_ROWS:, SLAB[0] : SLAB[1]] = FG_COLOR
    return depth, sky, rgb


def make_run(run_dir: Path, with_cleanplate: bool = False) -> None:
    depth, sky, rgb = make_scene()
    for d in ["s2b_scale/out", "s2_depth/out", "s0_ingest/out"]:
        (run_dir / d).mkdir(parents=True)
    imageio.save_npy(run_dir / "s2b_scale/out/depth_m.npy", depth)
    imageio.save_mask_png(run_dir / "s2_depth/out/sky_mask.png", sky)
    imageio.save_png(run_dir / "s0_ingest/out/pano.png", rgb)
    if with_cleanplate:
        (run_dir / "s1_cleanplate/out").mkdir(parents=True)
        imageio.save_png(run_dir / "s1_cleanplate/out/pano_clean.png", rgb)


def run_stage(run_dir: Path) -> dict:
    params = params_mod.load(REPO / "params.yaml")
    determinism.set_seed(params.get("seed", 0))
    ctx = Ctx(
        repo_root=REPO,
        pano_path=REPO / "params.yaml",  # unused by s3 (reads run dir)
        sidecar_path=REPO / "params.yaml",
        params_path=REPO / "params.yaml",
        weights_dir=REPO / "weights",
    )
    s3_layers.run(run_dir, params, ctx)
    return params


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("s3run") / "run"
    make_run(d)
    run_stage(d)
    return d


def out(run_dir: Path) -> Path:
    return run_dir / "s3_layers" / "out"


def test_outputs_exist_and_layers_schema_valid(run_dir):
    for name in OUT_FILES:
        assert (out(run_dir) / name).exists(), name
    layers = schema.read_validated(out(run_dir) / "layers.json", "layers")
    assert layers["edge_px_count"] > 0
    assert layers["bg_filled_px"] > 0


def test_edges_detected_at_slab_boundary(run_dir):
    layers = schema.read_validated(out(run_dir) / "layers.json", "layers")
    # one edge px per non-sky row on each side of the slab (forward diffs)
    assert layers["edge_px_count"] == 2 * (H - SKY_ROWS)


def test_band_px_matches_hand_computed_value(run_dir):
    layers = schema.read_validated(out(run_dir) / "layers.json", "layers")
    band_px = layers["band_px"]
    assert 2 <= band_px <= 64
    d = layers["band_derivation"]
    assert d["d_near"] == pytest.approx(D_FG, abs=1e-9)
    assert d["d_far"] == pytest.approx(D_BG, abs=1e-9)
    assert d["t_max"] == pytest.approx(1.0)
    # band_angle = 1.0 * |1/2 - 1/10| = 0.4 rad; angpix = pi/128
    assert d["band_angle_rad"] == pytest.approx(0.4, abs=1e-12)
    expected = int(np.clip(math.ceil(0.4 / (math.pi / H)) + 2, 2, 64))
    assert expected == 19
    assert band_px == expected
    assert d["band_px"] == band_px


def test_bg_fill_comes_from_far_side(run_dir):
    depth, sky, _ = make_scene()
    bg_mask = imageio.load_mask_png(out(run_dir) / "bg_mask.png")
    bg_depth = imageio.load_npy(out(run_dir) / "bg_depth.npy")
    bg_rgb = imageio.load_rgb(out(run_dir) / "bg_rgb.png")

    assert bg_mask.any()
    assert not (bg_mask & sky).any()  # band excludes sky
    # far-side band: pixels in the band whose original depth was background
    far_band = bg_mask & np.isfinite(depth) & (depth == D_BG)
    assert far_band.sum() > 0
    vals = bg_depth[far_band]
    assert vals.min() > 5.0
    assert abs(float(np.median(vals)) - D_BG) < 0.5
    # rgb filled from the background color, not the slab color
    fill = bg_rgb[far_band].astype(np.int64)
    assert np.abs(fill - BG_COLOR.astype(np.int64)).max() <= 5
    assert np.abs(fill - FG_COLOR.astype(np.int64)).max() > 50


def test_no_nan_and_fg_layout(run_dir):
    depth, sky, rgb = make_scene()
    bg_depth = imageio.load_npy(out(run_dir) / "bg_depth.npy")
    fg_depth = imageio.load_npy(out(run_dir) / "fg_depth.npy")
    fg_mask = imageio.load_mask_png(out(run_dir) / "fg_mask.png")
    fg_rgb = imageio.load_rgb(out(run_dir) / "fg_rgb.png")

    assert bg_depth.dtype == np.float32 and fg_depth.dtype == np.float32
    assert np.isfinite(bg_depth).all()  # filled everywhere, no NaN/inf
    assert not np.isnan(fg_depth).any()
    expected_fg = np.isfinite(depth) & ~sky
    assert (fg_mask == expected_fg).all()
    assert np.isinf(fg_depth[~expected_fg]).all()
    assert (fg_depth[expected_fg] == depth[expected_fg]).all()
    assert (fg_rgb == rgb).all()  # pano already at depth res


def test_receipt_written_with_params_used(run_dir):
    rec = receipts.read_receipt(run_dir, "s3_layers")
    assert set(rec["params_used"]) == {"head_box", "s3"}
    assert rec["weights"] == []
    assert rec["gates"] == []
    assert rec["notes"]["pano_source"] == "s0_ingest"
    assert set(rec["outputs"]) == {
        "fg_rgb", "fg_depth", "fg_mask", "bg_rgb", "bg_depth", "bg_mask", "layers",
    }
    for v in rec["outputs"].values():
        assert not v["path"].startswith("/")


def test_cleanplate_pano_preferred(tmp_path):
    d = tmp_path / "run"
    make_run(d, with_cleanplate=True)
    run_stage(d)
    rec = receipts.read_receipt(d, "s3_layers")
    assert rec["notes"]["pano_source"] == "s1_cleanplate"


def test_determinism_double_run_identical_bytes(run_dir, tmp_path):
    d2 = tmp_path / "run2"
    make_run(d2)
    run_stage(d2)
    for name in OUT_FILES:
        a = (out(run_dir) / name).read_bytes()
        b = (out(d2) / name).read_bytes()
        assert a == b, f"{name} differs across identical runs"
