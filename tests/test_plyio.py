"""Unit tests for scenic.plyio: 3DGS PLY round-trip, canonical quats,
validation, and color/opacity encodings."""
from __future__ import annotations

import numpy as np
import pytest

from scenic import determinism, plyio


def _make_splats(n: int = 17) -> plyio.SplatData:
    rng = determinism.rng("test_plyio_make_splats")
    quat = rng.normal(size=(n, 4)).astype(np.float32)
    return plyio.SplatData(
        xyz=rng.uniform(-10, 10, size=(n, 3)).astype(np.float32),
        normals=rng.normal(size=(n, 3)).astype(np.float32),
        f_dc=rng.uniform(-1.5, 1.5, size=(n, 3)).astype(np.float32),
        opacity_logit=rng.normal(size=(n,)).astype(np.float32),
        log_scales=rng.uniform(-5, 1, size=(n, 3)).astype(np.float32),
        quat_wxyz=quat,
        layer=rng.integers(0, 3, size=(n,)).astype(np.uint8),
        origin_stage=rng.integers(0, 9, size=(n,)).astype(np.uint8),
    )


# ---------------------------------------------------------------- round trip

def test_write_read_roundtrip_exact(tmp_path):
    s = _make_splats()
    p = tmp_path / "splats.ply"
    plyio.write_splats(p, s)
    r = plyio.read_splats(p)
    assert len(r) == len(s)
    np.testing.assert_array_equal(r.xyz, s.xyz)
    np.testing.assert_array_equal(r.normals, s.normals)
    np.testing.assert_array_equal(r.f_dc, s.f_dc)
    np.testing.assert_array_equal(r.opacity_logit, s.opacity_logit)
    np.testing.assert_array_equal(r.log_scales, s.log_scales)
    # quats are canonicalized on write; expected = exactly what the writer does
    expected_q = plyio.canonical_quat(s.quat_wxyz.astype(np.float32)).astype("<f4")
    np.testing.assert_array_equal(r.quat_wxyz, expected_q)
    np.testing.assert_array_equal(r.layer, s.layer)
    np.testing.assert_array_equal(r.origin_stage, s.origin_stage)
    assert r.xyz.dtype == np.float32
    assert r.layer.dtype == np.uint8 and r.origin_stage.dtype == np.uint8


def test_write_byte_stable(tmp_path):
    # determinism invariant: same in-memory splats -> identical bytes
    s = _make_splats()
    p1, p2 = tmp_path / "a.ply", tmp_path / "b.ply"
    plyio.write_splats(p1, s)
    plyio.write_splats(p2, s)
    assert p1.read_bytes() == p2.read_bytes()
    # NOTE: write(read(p)) is NOT byte-idempotent — canonical_quat
    # renormalizes already-unit quats and can drift 1 ULP in float32.
    # The pipeline always writes from source arrays, so double runs stay
    # bit-identical; we only require value stability within 1 ULP here.
    r1 = plyio.read_splats(p1)
    p3 = tmp_path / "c.ply"
    plyio.write_splats(p3, r1)
    r2 = plyio.read_splats(p3)
    np.testing.assert_array_equal(r2.xyz, r1.xyz)
    np.testing.assert_array_equal(r2.f_dc, r1.f_dc)
    np.testing.assert_allclose(r2.quat_wxyz, r1.quat_wxyz, atol=2e-7)


def test_layer_origin_stage_preserved(tmp_path):
    s = _make_splats(3)
    s.layer[:] = [plyio.LAYER_FG, plyio.LAYER_BG, plyio.LAYER_SHELL]
    s.origin_stage[:] = [0, 4, 255]
    p = tmp_path / "layers.ply"
    plyio.write_splats(p, s)
    r = plyio.read_splats(p)
    np.testing.assert_array_equal(r.layer, [0, 1, 2])
    np.testing.assert_array_equal(r.origin_stage, [0, 4, 255])


def test_written_quats_are_canonical(tmp_path):
    # even NON-canonical input quats come back unit with w >= 0
    s = _make_splats(8)
    rng = determinism.rng("test_plyio_noncanon")
    s = plyio.SplatData(
        s.xyz, s.normals, s.f_dc, s.opacity_logit, s.log_scales,
        (rng.normal(size=(8, 4)) * 3).astype(np.float32),
        s.layer, s.origin_stage,
    )
    p = tmp_path / "q.ply"
    plyio.write_splats(p, s)
    r = plyio.read_splats(p)
    assert np.all(r.quat_wxyz[:, 0] >= 0)
    assert np.allclose(np.linalg.norm(r.quat_wxyz, axis=-1), 1.0, atol=1e-6)


# ---------------------------------------------------------------- canonical_quat

def test_canonical_quat_unit_and_w_nonneg():
    rng = determinism.rng("test_plyio_quat")
    q = rng.normal(size=(50, 4)) * 4
    c = plyio.canonical_quat(q)
    assert np.all(c[..., 0] >= 0)
    assert np.allclose(np.linalg.norm(c, axis=-1), 1.0, atol=1e-12)


def test_canonical_quat_sign_invariance():
    rng = determinism.rng("test_plyio_quat_sign")
    q = rng.normal(size=(20, 4))
    np.testing.assert_allclose(
        plyio.canonical_quat(q), plyio.canonical_quat(-q), atol=1e-12
    )


def test_canonical_quat_preserves_already_canonical():
    q = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
    np.testing.assert_allclose(plyio.canonical_quat(q), q, atol=1e-12)


# ---------------------------------------------------------------- rejection

def test_nonfinite_rejected(tmp_path):
    p = tmp_path / "bad.ply"
    s = _make_splats(4)
    s.xyz[1, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        plyio.write_splats(p, s)
    s = _make_splats(4)
    s.log_scales[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        plyio.write_splats(p, s)
    s = _make_splats(4)
    s.opacity_logit[3] = -np.inf
    with pytest.raises(ValueError, match="non-finite"):
        plyio.write_splats(p, s)
    assert not p.exists()  # validation happens before any bytes hit disk


def test_wrong_shape_rejected(tmp_path):
    p = tmp_path / "bad.ply"
    s = _make_splats(4)
    s = plyio.SplatData(s.xyz[:, :2], s.normals, s.f_dc, s.opacity_logit,
                        s.log_scales, s.quat_wxyz, s.layer, s.origin_stage)
    with pytest.raises(ValueError, match="shape"):
        plyio.write_splats(p, s)
    s = _make_splats(4)
    s = plyio.SplatData(s.xyz, s.normals, s.f_dc, s.opacity_logit[:, None],
                        s.log_scales, s.quat_wxyz, s.layer, s.origin_stage)
    with pytest.raises(ValueError, match="shape"):
        plyio.write_splats(p, s)
    s = _make_splats(4)
    s = plyio.SplatData(s.xyz, s.normals, s.f_dc, s.opacity_logit,
                        s.log_scales, s.quat_wxyz[:, :3], s.layer,
                        s.origin_stage)
    with pytest.raises(ValueError, match="shape"):
        plyio.write_splats(p, s)
    # mismatched row count across fields
    s = _make_splats(4)
    s = plyio.SplatData(s.xyz[:3], s.normals, s.f_dc, s.opacity_logit,
                        s.log_scales, s.quat_wxyz, s.layer, s.origin_stage)
    with pytest.raises(ValueError, match="shape"):
        plyio.write_splats(p, s)


# ---------------------------------------------------------------- encodings

def test_rgb01_dc_roundtrip():
    rgb = np.linspace(0.0, 1.0, 30).reshape(10, 3)
    back = plyio.dc_to_rgb01(plyio.rgb01_to_dc(rgb))
    np.testing.assert_allclose(back, rgb, atol=1e-12)
    # spot-check the SH DC constant
    np.testing.assert_allclose(
        plyio.rgb01_to_dc(np.array([1.0])), (0.5) / plyio.SH_C0, atol=1e-12
    )
    # dc_to_rgb01 clips out-of-gamut
    assert plyio.dc_to_rgb01(np.array([100.0]))[0] == 1.0
    assert plyio.dc_to_rgb01(np.array([-100.0]))[0] == 0.0


def test_opacity_logit_roundtrip():
    a = np.linspace(0.01, 0.99, 25)
    back = plyio.logit_to_opacity(plyio.opacity_to_logit(a))
    np.testing.assert_allclose(back, a, atol=1e-12)
    # clipping keeps extremes finite and round-trips to the clip bounds
    ext = plyio.opacity_to_logit(np.array([0.0, 1.0]))
    assert np.all(np.isfinite(ext))
    np.testing.assert_allclose(
        plyio.logit_to_opacity(ext), [1e-6, 1 - 1e-6], rtol=1e-9
    )
