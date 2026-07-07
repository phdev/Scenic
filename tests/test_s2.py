"""Tests for pipeline/s2_depth.py (v2: 8-face ring + global solve).

Two kinds of test:
  * a model-free unit test of the global-solve function on synthetic aligned
    faces (known affines recovered up to the global gauge), plus tile-planner
    and adjacency helpers — fast;
  * a full-stage run on a synthesized s0 pano that runs the real DA-V2-Small on
    CPU (10 faces of 518px, twice for the determinism check) — tens of seconds.
"""
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


# --------------------------------------------------------------------------- #
# Model-free unit tests
# --------------------------------------------------------------------------- #

def test_face_list_and_adjacency():
    p = params_mod.load(REPO / "params.yaml")
    faces = s2_depth._face_list(p)
    assert len(faces) == 10
    names = [f[0] for f in faces]
    assert names == [f"ring{k}" for k in range(8)] + ["zenith", "nadir"]
    # ring yaws are k*45deg, pitch 0; caps at +/-90deg pitch.
    for k in range(8):
        assert np.isclose(faces[k][1], 2 * np.pi * k / 8)
        assert faces[k][2] == 0.0
        assert faces[k][3] == p["faces"]["ring_fov_deg"]
    assert np.isclose(faces[8][2], np.pi / 2)
    assert np.isclose(faces[9][2], -np.pi / 2)

    adj = s2_depth._ring_adjacency(8)
    # 8 ring-neighbour pairs (incl wrap 7-0) + 8*2 ring-vs-cap pairs = 24.
    assert len(adj) == 24
    assert (7, 0) in adj  # wrap pair -> loop closure
    zi, ni = 8, 9
    for k in range(8):
        assert (k, (k + 1) % 8) in adj
        assert (k, zi) in adj
        assert (k, ni) in adj


def test_tile_starts_inert_and_active():
    # inert when the render fits the inference budget
    assert s2_depth._tile_starts(518, 518, 64) == [0]
    assert s2_depth._tile_starts(300, 518, 64) == [0]
    # active: covers the whole span, every tile has span == tile, last fits
    starts = s2_depth._tile_starts(1000, 518, 64)
    assert starts[0] == 0
    assert starts[-1] == 1000 - 518
    for s in starts:
        assert 0 <= s <= 1000 - 518
    # union of tiles covers [0, size)
    covered = np.zeros(1000, dtype=bool)
    for s in starts:
        covered[s : s + 518] = True
    assert covered.all()


def test_global_solve_recovers_affines_up_to_gauge():
    """Synthetic aligned faces: face i observes x_i = (t - b_true)/a_true on its
    covered pixels, so a_true*x_i + b_true == t. The solve must make all faces
    mutually consistent and recover the affines up to a single global
    (scale, offset) gauge: a_i/a_true_i and b_i - (a_i/a_true_i)*b_true_i must
    be CONSTANT across faces."""
    rng = np.random.default_rng(1234)  # test-local fixture data only
    F = 6
    N = 400
    # underlying smooth log-depth field
    t = np.linspace(-1.0, 1.5, N) + 0.2 * np.sin(np.arange(N) * 0.3)

    # chain adjacency 0-1-2-3-4-5-0 (cyclic => loop closure), each pair overlaps
    # on a contiguous window; every pixel covered by >=2 faces.
    adjacency = [(i, (i + 1) % F) for i in range(F)]

    a_true = np.array([1.0, 1.3, 0.8, 1.1, 0.9, 1.2])
    b_true = np.array([0.0, 0.4, -0.3, 0.2, -0.5, 0.1])

    x = np.zeros((F, N))
    w = np.zeros((F, N))
    # face i covers a sliding window of the ring; windows overlap neighbours.
    win = N // 3
    for i in range(F):
        c = int(i * N / F)
        idx = (np.arange(-win // 2, win // 2) + c) % N
        cover = np.zeros(N, dtype=bool)
        cover[idx] = True
        w[i, cover] = 1.0
        x[i, cover] = (t[cover] - b_true[i]) / a_true[i]
    sky = np.zeros((F, N), dtype=bool)

    sol = s2_depth.global_solve(
        x, w, sky, adjacency, huber_iters=10, huber_delta=0.15, reg=1e-4
    )
    a, b = sol["a"], sol["b"]
    assert np.isfinite(a).all() and np.isfinite(b).all()

    # residual + interface step are tiny (perfectly consistent input).
    assert sol["overlap_residual_log"] < 1e-4
    assert sol["max_interface_step_log"] < 1e-3

    # recovery up to a single global gauge (scale s, offset c):
    #   a_i = s * a_true_i ; b_i = s * b_true_i + c
    s_i = a / a_true
    c_i = b - s_i * b_true
    assert np.allclose(s_i, s_i[0], atol=1e-4), s_i
    assert np.allclose(c_i, c_i[0], atol=1e-4), c_i

    # applying the recovered affine reproduces a common field s*t + c per face.
    for i in range(F):
        cover = w[i] > 0
        aligned = a[i] * x[i][cover] + b[i]
        expect = s_i[0] * t[cover] + c_i[0]
        assert np.allclose(aligned, expect, atol=1e-4)


def test_sky_mask_helper_flags_far_smooth_upper():
    """Model-free unit test of the sky heuristic: a far, smooth patch in the
    upper hemisphere is flagged; the near lower hemisphere is not."""
    h, w = 64, 128
    depth = np.ones((h, w), dtype=np.float64)
    depth[:6, :] = 10.0  # far + uniform (smooth) block, 6/64 = 9.4% < 10%
    q_log = np.log(depth)
    sky = s2_depth._sky_mask(
        depth, q_log, sky_far_pct=90.0, sky_grad_max=0.02, sky_min_pitch_deg=0.0
    )
    assert sky.shape == (h, w)
    assert sky[:6].any(), "far smooth upper block should be flagged as sky"
    assert sky[h // 2 :].sum() == 0, "near lower hemisphere must stay clear"
    # a near, smooth field yields no sky at all.
    flat = np.ones((h, w), dtype=np.float64)
    assert s2_depth._sky_mask(flat, np.log(flat), 90.0, 0.02, 0.0).sum() == 0


def test_global_solve_no_overlap_is_identity():
    """A face with no overlaps at all is held at identity by the Tikhonov pull;
    the solver stays finite and never raises."""
    F = 3
    N = 50
    x = np.zeros((F, N))
    w = np.zeros((F, N))
    # only faces 0 and 1 overlap; face 2 covers nothing.
    w[0, :30] = 1.0
    w[1, 20:] = 1.0
    x[0, :30] = np.linspace(0, 1, 30)
    x[1, 20:] = np.linspace(0, 1, 30) * 0.5  # different scale on the overlap
    sky = np.zeros((F, N), dtype=bool)
    sol = s2_depth.global_solve(
        x, w, sky, [(0, 1)], huber_iters=6, huber_delta=0.15, reg=1e-3
    )
    assert np.isfinite(sol["a"]).all() and np.isfinite(sol["b"]).all()
    # untouched face 2 stays at identity.
    assert abs(sol["a"][2] - 1.0) < 1e-6
    assert abs(sol["b"][2] - 0.0) < 1e-6


# --------------------------------------------------------------------------- #
# Full-stage run (real DA-V2-Small on CPU)
# --------------------------------------------------------------------------- #

def _make_pano(w: int = 512, h: int = 256) -> np.ndarray:
    """Synthetic 2:1 pano: smooth sky gradient in the top half, high-contrast
    textured ground in the bottom half."""
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


def test_sky_mask_subset_of_upper_hemisphere(runs):
    """Whatever sky the real model detects on the synthetic pano, it must be a
    subset of the upper hemisphere and exactly the inf pixels of depth_rel.
    (Non-emptiness of the heuristic is covered model-free above; DA-V2's depth
    on a purely synthetic pano does not reliably cue 'far sky'.)"""
    run_dir, _ = runs
    mask = imageio.load_mask_png(run_dir / "s2_depth" / "out" / "sky_mask.png")
    assert mask.shape == (256, 512)
    # sky requires pitch > 0 deg -> upper hemisphere (rows 0..127) only.
    assert mask[128:].sum() == 0
    assert int(mask[:128].sum()) == int(mask.sum())  # entirely upper hemisphere
    # sky pixels are inf in the depth map, and only sky pixels are inf.
    depth = imageio.load_npy(run_dir / "s2_depth" / "out" / "depth_rel.npy")
    assert np.array_equal(np.isinf(depth), mask)


def test_depth_meta_ten_faces_and_metrics(runs):
    run_dir, _ = runs
    meta = schema.read_validated(
        run_dir / "s2_depth" / "out" / "depth_meta.json", "depth_meta"
    )
    assert meta["backend"] == "depth_anything_v2_small"
    assert (meta["fused_w"], meta["fused_h"]) == (512, 256)
    assert (meta["out_w"], meta["out_h"]) == (512, 256)
    assert meta["overlap_residual_log"] >= 0.0
    assert meta["median_divisor"] > 0.0
    assert meta["max_interface_step_log"] >= 0.0
    assert meta["mean_interface_step_log"] >= 0.0
    faces = meta["faces"]
    assert len(faces) == 10
    names = [f["name"] for f in faces]
    assert names == [f"ring{k}" for k in range(8)] + ["zenith", "nadir"]
    for f in faces:
        assert np.isfinite(f["affine_a"])
        assert np.isfinite(f["affine_b"])
        assert f["infer_px"] == 518  # inert tiling at 518px render


def test_interface_step_gate_present_and_valid(runs):
    run_dir, _ = runs
    rec = receipts.read_receipt(run_dir, "s2_depth")  # schema-validated
    gates = rec["gates"]
    assert len(gates) == 1
    g = gates[0]
    assert g["gate"] == "interface_step"
    schema.validate(g, "gate_verdict")
    assert isinstance(g["pass"], bool)
    assert "max_interface_step_log" in g["metrics"]
    assert "mean_interface_step_log" in g["metrics"]
    assert "interface_step_max_log" in g["thresholds"]
    # metric matches meta
    meta = schema.read_validated(
        run_dir / "s2_depth" / "out" / "depth_meta.json", "depth_meta"
    )
    assert g["metrics"]["max_interface_step_log"] == meta["max_interface_step_log"]


def test_receipt_valid(runs):
    run_dir, _ = runs
    rec = receipts.read_receipt(run_dir, "s2_depth")
    assert rec["stage"] == "s2_depth"
    assert set(rec["outputs"]) == {"depth_rel", "sky_mask", "depth_meta"}
    assert list(rec["inputs"]) == ["pano"]  # s0 fallback path in this fixture
    assert [w["key"] for w in rec["weights"]] == ["depth_anything_v2_small"]
    assert set(rec["params_used"]) == {"resolutions", "faces", "s2"}
    assert "overlap_residual_log" in rec["notes"]
    assert "max_interface_step_log" in rec["notes"]
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
