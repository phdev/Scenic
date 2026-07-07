"""s3_layers tests (v2 "kill the flat blocks").

Synthetic 256x128 scene with three depth boundaries, each isolated by a sky
band so no spurious cross edges appear:

  * NEAR band (rows 12..59, bg 10 m):
      - slab A (cols 60..120) at 2 m   -> REAL occlusion (ratio 5 > 1.4,
        near enough to disocclude for a head-box pose) -> emits a band.
      - slab C (cols 160..200) at 13.8 m -> low ratio (13.8/10 = 1.38 < 1.4)
        while the log-gradient DOES pass -> must be REJECTED as an edge by the
        depth-ratio test (no band).
  * FAR band (rows 82..115, bg 200 m):
      - slab E (cols 60..120) at 600 m -> ratio 3 > 1.4 and log-gradient pass,
        but both surfaces are far so no head-box pose disoccludes it -> the
        visibility test must drop it (no band).

Verifies edge/ratio/visibility gating, the capped analytic band width, the
push-pull far-side fill, the bg_solid_angle gate + schema, no NaN, and
bit-identical determinism across runs.
"""
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

# row bands (half-open python ranges); sky separates every scene band
SKY_TOP = (0, 12)
NEAR = (12, 60)      # rows 12..59
SKY_MID = (60, 82)   # rows 60..81  (>= band_px gap so slab A can't leak down)
FAR = (82, 116)      # rows 82..115
SKY_BOT = (116, 128)

# column slabs
SLAB_A = (60, 121)   # cols 60..120, depth 2  (real occlusion)
SLAB_C = (160, 201)  # cols 160..200, depth 13.8 (ratio-fail)
SLAB_E = (60, 121)   # cols 60..120, depth 600 (far-far, visibility drops)

D_NEAR_BG = 10.0
D_FG = 2.0
D_C = 13.8
D_FAR_BG = 200.0
D_E = 600.0

BG_COLOR = np.array([60, 200, 60], np.uint8)      # near bg (green)
FG_COLOR = np.array([200, 60, 60], np.uint8)      # slab A (red)
C_COLOR = np.array([200, 200, 60], np.uint8)      # slab C (yellow)
FARBG_COLOR = np.array([60, 60, 200], np.uint8)   # far bg (blue)
E_COLOR = np.array([200, 60, 200], np.uint8)      # slab E (magenta)
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
    depth = np.full((H, W), np.inf, np.float32)
    rgb = np.empty((H, W, 3), np.uint8)
    rgb[:] = SKY_COLOR

    # near band
    depth[NEAR[0] : NEAR[1], :] = D_NEAR_BG
    rgb[NEAR[0] : NEAR[1], :] = BG_COLOR
    depth[NEAR[0] : NEAR[1], SLAB_A[0] : SLAB_A[1]] = D_FG
    rgb[NEAR[0] : NEAR[1], SLAB_A[0] : SLAB_A[1]] = FG_COLOR
    depth[NEAR[0] : NEAR[1], SLAB_C[0] : SLAB_C[1]] = D_C
    rgb[NEAR[0] : NEAR[1], SLAB_C[0] : SLAB_C[1]] = C_COLOR

    # far band
    depth[FAR[0] : FAR[1], :] = D_FAR_BG
    rgb[FAR[0] : FAR[1], :] = FARBG_COLOR
    depth[FAR[0] : FAR[1], SLAB_E[0] : SLAB_E[1]] = D_E
    rgb[FAR[0] : FAR[1], SLAB_E[0] : SLAB_E[1]] = E_COLOR

    sky = ~np.isfinite(depth)
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


# expected occlusion edges (grad AND ratio): slab A (2 edges/row over 48 rows)
# + slab E (2 edges/row over 34 rows); slab C fails the ratio test -> excluded.
NEAR_ROWS = NEAR[1] - NEAR[0]  # 48
FAR_ROWS = FAR[1] - FAR[0]     # 34
EXPECTED_EDGE_PX = 2 * NEAR_ROWS + 2 * FAR_ROWS   # 96 + 68 = 164
EXPECTED_VISIBLE_EDGE_PX = 2 * NEAR_ROWS          # 96 (slab A only)


def test_outputs_exist_and_layers_schema_valid(run_dir):
    for name in OUT_FILES:
        assert (out(run_dir) / name).exists(), name
    layers = schema.read_validated(out(run_dir) / "layers.json", "layers")
    assert layers["edge_px_count"] > 0
    assert layers["bg_filled_px"] > 0
    assert layers["bg_scale_clamp"] is True  # echoes params s3.bg_scale_clamp


def test_ratio_test_rejects_low_ratio_boundary(run_dir):
    layers = schema.read_validated(out(run_dir) / "layers.json", "layers")
    # slab C (ratio 1.38 < 1.4) passes the log-grad but must be dropped, so the
    # occlusion-edge count is exactly slab A + slab E (no slab C contribution).
    assert layers["edge_px_count"] == EXPECTED_EDGE_PX


def test_visibility_test_drops_far_far_boundary(run_dir):
    layers = schema.read_validated(out(run_dir) / "layers.json", "layers")
    # slab E is a valid occlusion edge but no head-box pose disoccludes it at
    # 200 m, so only slab A's edges survive the visibility test.
    assert layers["visible_edge_px_count"] == EXPECTED_VISIBLE_EDGE_PX
    assert layers["visible_edge_px_count"] < layers["edge_px_count"]

    bg_mask = imageio.load_mask_png(out(run_dir) / "bg_mask.png")
    # no band anywhere in the far band (rows 82..115) -> visibility removed it
    assert bg_mask[FAR[0] : FAR[1]].sum() == 0
    # no band in the sky bands either
    assert bg_mask[SKY_TOP[0] : SKY_TOP[1]].sum() == 0
    assert bg_mask[SKY_MID[0] : SKY_MID[1]].sum() == 0
    assert bg_mask[SKY_BOT[0] : SKY_BOT[1]].sum() == 0


def test_only_real_edge_produces_a_band(run_dir):
    bg_mask = imageio.load_mask_png(out(run_dir) / "bg_mask.png")
    assert bg_mask.any()  # the real occlusion emits a band
    # band confined to the near band rows and to slab A's neighbourhood
    # (cols 40..139); slab C (cols 160..200) gets NO band.
    assert bg_mask[NEAR[0] : NEAR[1]].sum() == int(bg_mask.sum())
    assert bg_mask[:, :40].sum() == 0
    assert bg_mask[:, 140:].sum() == 0  # excludes slab C entirely


def test_band_px_capped_and_hand_computed(run_dir):
    layers = schema.read_validated(out(run_dir) / "layers.json", "layers")
    band_px = layers["band_px"]
    d = layers["band_derivation"]
    assert 2 <= band_px <= d["band_px_max"]
    assert d["band_px_max"] == 64
    assert d["clamped"] is False
    # analytic derivation uses the VISIBLE (slab A) edges: near 2 m, far 10 m.
    assert d["d_near"] == pytest.approx(D_FG, abs=1e-9)
    assert d["d_far"] == pytest.approx(D_NEAR_BG, abs=1e-9)
    assert d["t_max"] == pytest.approx(1.0)  # max(0.5, 0.2, 1.0)
    # band_angle = 1.0 * |1/2 - 1/10| = 0.4 rad; angpix = pi/128
    assert d["band_angle_rad"] == pytest.approx(0.4, abs=1e-12)
    expected = int(np.clip(math.ceil(0.4 / (math.pi / H)) + 2, 2, 64))
    assert expected == 19
    assert band_px == expected
    assert d["band_px"] == band_px


def test_band_derivation_records_cap_when_clamped():
    # a genuinely huge analytic band (a very near edge) must clamp to band_px_max
    # and record clamped=True.
    depth = np.full((H, W), 100.0, np.float32)   # far bg
    depth[NEAR[0] : NEAR[1], SLAB_A[0] : SLAB_A[1]] = 0.05  # extremely near slab
    depth[: NEAR[0]] = np.inf
    depth[NEAR[1] :] = np.inf
    sky = ~np.isfinite(depth)
    rgb = np.zeros((H, W, 3), np.uint8)

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "run"
        for sub in ["s2b_scale/out", "s2_depth/out", "s0_ingest/out"]:
            (d / sub).mkdir(parents=True)
        imageio.save_npy(d / "s2b_scale/out/depth_m.npy", depth)
        imageio.save_mask_png(d / "s2_depth/out/sky_mask.png", sky)
        imageio.save_png(d / "s0_ingest/out/pano.png", rgb)
        run_stage(d)
        layers = schema.read_validated(d / "s3_layers/out/layers.json", "layers")
    assert layers["band_px"] == 64
    assert layers["band_derivation"]["clamped"] is True
    assert layers["band_derivation"]["raw_band_px"] > 64


def test_bg_fill_comes_from_far_side(run_dir):
    depth, _, _ = make_scene()
    bg_mask = imageio.load_mask_png(out(run_dir) / "bg_mask.png")
    bg_depth = imageio.load_npy(out(run_dir) / "bg_depth.npy")
    bg_rgb = imageio.load_rgb(out(run_dir) / "bg_rgb.png")

    # far-side band pixels: in the band, originally the near-bg (10 m, green)
    far_band = bg_mask & np.isfinite(depth) & (depth == D_NEAR_BG)
    assert far_band.sum() > 0
    vals = bg_depth[far_band]
    assert float(np.median(vals)) > 5.0  # closer to far 10 m than near 2 m
    # rgb filled toward the background (green), not the slab (red)
    fill = bg_rgb[far_band].astype(np.int64)
    med = np.median(fill, axis=0)
    assert np.abs(med - BG_COLOR.astype(np.int64)).max() < np.abs(
        med - FG_COLOR.astype(np.int64)
    ).max()


def test_bg_depth_metric_and_finite(run_dir):
    bg_depth = imageio.load_npy(out(run_dir) / "bg_depth.npy")
    assert bg_depth.dtype == np.float32
    assert np.isfinite(bg_depth).all()  # filled everywhere, no NaN/inf
    assert not np.isnan(bg_depth).any()


def test_no_nan_and_fg_layout(run_dir):
    depth, sky, rgb = make_scene()
    fg_depth = imageio.load_npy(out(run_dir) / "fg_depth.npy")
    fg_mask = imageio.load_mask_png(out(run_dir) / "fg_mask.png")
    fg_rgb = imageio.load_rgb(out(run_dir) / "fg_rgb.png")

    assert fg_depth.dtype == np.float32
    assert not np.isnan(fg_depth).any()
    expected_fg = np.isfinite(depth) & ~sky
    assert (fg_mask == expected_fg).all()  # fg = original content, minus sky
    assert np.isinf(fg_depth[~expected_fg]).all()
    assert (fg_depth[expected_fg] == depth[expected_fg]).all()
    assert (fg_rgb == rgb).all()  # pano already at depth res, ~sky preserved


def test_bg_solid_angle_gate_schema_and_value(run_dir):
    rec = receipts.read_receipt(run_dir, "s3_layers")
    layers = schema.read_validated(out(run_dir) / "layers.json", "layers")

    assert len(rec["gates"]) == 1
    gate = rec["gates"][0]
    schema.validate(gate, "gate_verdict")  # schema-valid gate verdict
    assert gate["gate"] == "bg_solid_angle"

    frac = layers["bg_solid_angle_frac"]
    assert isinstance(frac, float)
    assert 0.0 <= frac <= 1.0
    assert gate["metrics"]["bg_solid_angle_frac"] == pytest.approx(frac)
    assert gate["metrics"]["edge_px_count"] == layers["edge_px_count"]
    assert gate["metrics"]["band_px"] == layers["band_px"]
    assert gate["thresholds"]["max_frac"] == pytest.approx(0.05)
    # pass reflects the threshold comparison exactly
    assert gate["pass"] == (frac <= 0.05)


def test_receipt_written_with_params_used(run_dir):
    rec = receipts.read_receipt(run_dir, "s3_layers")
    assert set(rec["params_used"]) == {"head_box", "s3", "s7"}
    assert rec["params_used"]["s7"] == {"squat_y_m": -0.9}
    assert rec["weights"] == []
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
    # receipts (incl. the gate + params_hash) are byte-identical too
    ra = (run_dir / "s3_layers" / "receipt.json").read_bytes()
    rb = (d2 / "s3_layers" / "receipt.json").read_bytes()
    assert ra == rb
