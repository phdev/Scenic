"""Tests for pipeline/s2b_scale.py (metric scale from ground plane + gates)."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from pipeline import s2b_scale
from scenic import determinism, geometry, hashing, imageio, receipts, schema
from scenic import params as params_mod
from scenic.stage import Ctx

REPO = Path(__file__).resolve().parent.parent

W, H = 256, 128
HEIGHT_REL = 0.4
CAMERA_HEIGHT_M = 1.6


@pytest.fixture()
def params() -> dict:
    return params_mod.load(REPO / "params.yaml")


def make_ctx() -> Ctx:
    return Ctx(
        repo_root=REPO,
        pano_path=REPO / "fixtures" / "synthetic.png",
        sidecar_path=REPO / "fixtures" / "synthetic.png.license.json",
        params_path=REPO / "params.yaml",
        weights_dir=REPO / "weights",
    )


def synth_depth(
    normal=(0.0, 1.0, 0.0),
    height_rel: float = HEIGHT_REL,
    ground_pitch_deg: float = -30.0,
    far_rel: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Equirect depth for an infinite plane n.p = -height_rel below the camera.
    Below ground_pitch_deg: exact plane depth. Between ground_pitch_deg and the
    horizon: large finite depth (far content). pitch >= 0: sky (inf)."""
    lon, lat = geometry.equirect_lonlat(W, H)
    dirs = geometry.lonlat_to_dirs(lon, lat)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    nd = dirs @ n
    pitch_deg = np.degrees(lat)
    depth = np.full((H, W), np.inf, dtype=np.float64)
    band = (pitch_deg >= ground_pitch_deg) & (pitch_deg < 0.0)
    depth[band] = far_rel
    ground = (pitch_deg < ground_pitch_deg) & (nd < -1e-9)
    depth[ground] = height_rel / -nd[ground]
    sky = pitch_deg >= 0.0
    return depth.astype(np.float32), sky


def make_run(tmp_path: Path, depth_rel: np.ndarray, sky: np.ndarray,
             name: str = "run") -> Path:
    run = tmp_path / name
    s2 = run / "s2_depth" / "out"
    s2.mkdir(parents=True)
    imageio.save_npy(s2 / "depth_rel.npy", depth_rel.astype(np.float32))
    imageio.save_mask_png(s2 / "sky_mask.png", sky)
    s0 = run / "s0_ingest" / "out"
    s0.mkdir(parents=True)
    meta = {
        "width": 2048,
        "height": 1024,
        "source_sha256": "0" * 64,
        "license": {"id": "test"},
        "camera_height_m": CAMERA_HEIGHT_M,
    }
    schema.write_validated(s0 / "pano_meta.json", meta, "pano_meta")
    return run


def gates_by_name(run: Path) -> dict:
    rec = receipts.read_receipt(run, "s2b_scale")
    return {g["gate"]: g for g in rec["gates"]}


# ---------------------------------------------------------------- unit: fit


def test_fit_recovers_plane_with_outliers():
    """IRLS Tukey fit + h_rel derivation verified against closed form."""
    rng = determinism.rng("s2b-fit-test")
    n_pts = 4000
    x = rng.uniform(-5, 5, n_pts)
    z = rng.uniform(-5, 5, n_pts)
    a, b, c = 0.05, -0.03, -1.7
    y = a * x + b * z + c + rng.normal(0.0, 0.005, n_pts)
    out = rng.random(n_pts) < 0.2  # one-sided outliers (clutter above ground)
    y = np.where(out, y + rng.uniform(0.5, 3.0, n_pts), y)
    coef, inliers, _res = s2b_scale.fit_ground_plane(
        np.stack([x, y, z], axis=1), 20
    )
    assert np.allclose(coef, [a, b, c], atol=2e-3)
    assert inliers.sum() >= (~out).sum() * 0.9  # clean points kept

    plane = s2b_scale.plane_from_coef(coef)
    # closed-form origin->plane distance for y = a*x + b*z + c
    norm = math.sqrt(a * a + 1.0 + b * b)
    assert plane["h_rel"] == pytest.approx(abs(c) / norm, rel=1e-2)
    # normal is unit, y-up, and matches (-a, 1, -b)/norm
    n = np.asarray(plane["normal"])
    assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-12)
    assert n[1] > 0
    assert np.allclose(n, np.array([-a, 1.0, -b]) / norm, atol=2e-3)
    # any point on the fitted plane satisfies n.p + d = 0
    p_on = np.array([1.0, coef[0] * 1.0 + coef[1] * 2.0 + coef[2], 2.0])
    fitted_n = np.asarray(plane["normal"])
    assert float(fitted_n @ p_on + plane["d"]) == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------ stage: perfect


def test_perfect_scene_scale_gates_and_depth(tmp_path, params):
    depth_rel, sky = synth_depth()
    run = make_run(tmp_path, depth_rel, sky)
    s2b_scale.run(run, params, make_ctx())

    scale = schema.read_validated(run / "s2b_scale" / "out" / "scale.json", "scale")
    assert scale["scale_source"] == "ground_plane"
    assert scale["scale_factor"] == pytest.approx(
        CAMERA_HEIGHT_M / HEIGHT_REL, rel=0.02
    )
    assert scale["tilt_deg"] < 1.0
    assert scale["residual_rel"] <= 0.01
    assert scale["camera_height_m"] == pytest.approx(CAMERA_HEIGHT_M)
    n = np.asarray(scale["plane"]["normal"])
    assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-9)
    assert n[1] > 0.999  # points up

    gates = gates_by_name(run)
    assert set(gates) == {"ground_plane", "min_content_distance"}
    assert gates["ground_plane"]["pass"] is True
    assert gates["min_content_distance"]["pass"] is True
    assert gates["min_content_distance"]["metrics"]["near_distance_m"] >= 6.0

    depth_m = imageio.load_npy(run / "s2b_scale" / "out" / "depth_m.npy")
    assert depth_m.dtype == np.float32
    fin = np.isfinite(depth_rel)
    assert np.array_equal(np.isinf(depth_m), np.isinf(depth_rel))  # inf stays inf
    expect = depth_rel[fin].astype(np.float64) * scale["scale_factor"]
    assert np.allclose(depth_m[fin], expect, rtol=1e-4)
    # ~4x depth_rel
    assert np.allclose(depth_m[fin], 4.0 * depth_rel[fin], rtol=0.02)


# ------------------------------------------------------------- stage: tilted


def test_tilted_plane_fails_ground_gate(tmp_path, params):
    t = math.radians(20.0)
    depth_rel, sky = synth_depth(normal=(math.sin(t), math.cos(t), 0.0))
    run = make_run(tmp_path, depth_rel, sky)
    s2b_scale.run(run, params, make_ctx())

    scale = schema.read_validated(run / "s2b_scale" / "out" / "scale.json", "scale")
    assert scale["tilt_deg"] == pytest.approx(20.0, abs=1.0)
    gates = gates_by_name(run)
    assert gates["ground_plane"]["pass"] is False
    # tilted plane keeps the same origin distance -> scale still ~4
    assert scale["scale_factor"] == pytest.approx(4.0, rel=0.02)


# ----------------------------------------------------------- stage: explicit


def test_explicit_scale_override(tmp_path, params):
    p = copy.deepcopy(params)
    p["s2b"]["explicit_scale"] = 2.5
    depth_rel, sky = synth_depth()
    run = make_run(tmp_path, depth_rel, sky)
    s2b_scale.run(run, p, make_ctx())

    scale = schema.read_validated(run / "s2b_scale" / "out" / "scale.json", "scale")
    assert scale["scale_source"] == "explicit"
    assert scale["scale_factor"] == pytest.approx(2.5)
    gates = gates_by_name(run)
    assert gates["ground_plane"]["pass"] is True  # explicit -> pass with details
    assert "explicit" in gates["ground_plane"]["details"]
    depth_m = imageio.load_npy(run / "s2b_scale" / "out" / "depth_m.npy")
    fin = np.isfinite(depth_rel)
    assert np.allclose(depth_m[fin], 2.5 * depth_rel[fin], rtol=1e-4)
    rec = receipts.read_receipt(run, "s2b_scale")
    assert rec["params_used"]["s2b"]["explicit_scale"] == 2.5


# ------------------------------------------------------- schema + receipt IO


def test_receipt_shape(tmp_path, params):
    depth_rel, sky = synth_depth()
    run = make_run(tmp_path, depth_rel, sky)
    s2b_scale.run(run, params, make_ctx())
    rec = receipts.read_receipt(run, "s2b_scale")  # schema-validates
    assert rec["stage"] == "s2b_scale"
    assert set(rec["inputs"]) == {"depth_rel", "pano_meta", "sky_mask"}
    assert set(rec["outputs"]) == {"depth_m", "scale"}
    assert rec["weights"] == []
    assert set(rec["params_used"]) == {"s2b", "min_content_distance_m"}
    for g in rec["gates"]:
        schema.validate(g, "gate_verdict")
    # paths recorded relative to run dir
    for entry in {**rec["inputs"], **rec["outputs"]}.values():
        assert not entry["path"].startswith("/")


# -------------------------------------------------------------- determinism


def test_determinism_two_runs_identical(tmp_path, params):
    depth_rel, sky = synth_depth()
    digests = []
    for name in ("a", "b"):
        run = make_run(tmp_path, depth_rel, sky, name=name)
        s2b_scale.run(run, params, make_ctx())
        out = run / "s2b_scale" / "out"
        digests.append(
            (
                hashing.sha256_file(out / "scale.json"),
                hashing.sha256_file(out / "depth_m.npy"),
            )
        )
    assert digests[0] == digests[1]
