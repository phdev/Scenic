"""Unit tests for scenic.geometry against the docs/CONTRACTS.md conventions:
right-handed, +Y up, theta=0 -> +Z, equirect lon=(u+0.5)/W*2pi-pi,
lat=pi/2-(v+0.5)/H*pi."""
from __future__ import annotations

import numpy as np
import pytest

from scenic import determinism
from scenic import geometry as g


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------- equirect dirs

def test_equirect_dirs_unit_norm():
    dirs = g.equirect_dirs(32, 16)
    assert dirs.shape == (16, 32, 3)
    norms = np.linalg.norm(dirs, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-12)


def test_dirs_to_uv_inverts_equirect_dirs():
    w, h = 16, 8
    uv = g.dirs_to_uv(g.equirect_dirs(w, h), w, h)
    cols, rows = np.meshgrid(np.arange(w, dtype=np.float64),
                             np.arange(h, dtype=np.float64))
    assert np.allclose(uv[..., 0], cols, atol=1e-6)  # u = column index
    assert np.allclose(uv[..., 1], rows, atol=1e-6)  # v = row index


def test_lonlat_conventions():
    # lat=0, lon=0 -> +Z
    assert np.allclose(g.lonlat_to_dirs(np.float64(0.0), np.float64(0.0)),
                       [0, 0, 1], atol=1e-15)
    # lon=pi/2 -> +X
    assert np.allclose(g.lonlat_to_dirs(np.float64(np.pi / 2), np.float64(0.0)),
                       [1, 0, 0], atol=1e-15)
    # lat=pi/2 -> +Y
    assert np.allclose(g.lonlat_to_dirs(np.float64(0.0), np.float64(np.pi / 2)),
                       [0, 1, 0], atol=1e-15)


# ---------------------------------------------------------------- rotations

@pytest.mark.parametrize("yaw,pitch", [
    (0.0, 0.0), (0.3, -0.2), (np.pi / 2, np.pi / 4), (-1.1, 0.7), (np.pi, -np.pi / 2),
])
def test_rotation_yaw_pitch_orthonormal_det1(yaw, pitch):
    r = g.rotation_yaw_pitch(yaw, pitch)
    assert r.shape == (3, 3)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-12)


def test_rotation_yaw_half_pi_maps_z_to_x():
    r = g.rotation_yaw_pitch(np.pi / 2, 0.0)
    assert np.allclose(r @ np.array([0.0, 0.0, 1.0]), [1, 0, 0], atol=1e-12)


def test_rotation_pitch_half_pi_maps_z_to_y():
    r = g.rotation_yaw_pitch(0.0, np.pi / 2)
    assert np.allclose(r @ np.array([0.0, 0.0, 1.0]), [0, 1, 0], atol=1e-12)


# ---------------------------------------------------------------- perspective

def test_perspective_dirs_center_pixel_is_plus_z():
    # odd dims so an exact center pixel exists
    dirs = g.perspective_dirs(90.0, 5, 5, 0.0, 0.0)
    assert dirs.shape == (5, 5, 3)
    assert np.allclose(dirs[2, 2], [0, 0, 1], atol=1e-12)
    # all rays unit norm
    assert np.allclose(np.linalg.norm(dirs, axis=-1), 1.0, atol=1e-12)


def test_perspective_dirs_yawed_center():
    dirs = g.perspective_dirs(90.0, 5, 5, np.pi / 2, 0.0)
    assert np.allclose(dirs[2, 2], [1, 0, 0], atol=1e-12)


# ---------------------------------------------------------------- sampling

def test_sample_equirect_exact_on_constant_image():
    img = np.full((8, 16), 3.75, dtype=np.float64)
    rng = determinism.rng("test_geometry_sample_const")
    dirs = rng.normal(size=(200, 3))
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    out = g.sample_equirect(img, dirs)
    assert out.shape == (200,)
    assert np.allclose(out, 3.75, atol=1e-12)
    # multichannel path too
    img3 = np.stack([img, img * 2, img * 0 + 1], axis=-1)
    out3 = g.sample_equirect(img3, dirs)
    assert np.allclose(out3, [3.75, 7.5, 1.0], atol=1e-12)


def test_sample_equirect_interior_bilinear_exact():
    # img value = column index => bilinear sample at float u returns u
    w, h = 16, 8
    img = np.broadcast_to(np.arange(w, dtype=np.float64), (h, w)).copy()
    u_target = 3.25
    lon = (u_target + 0.5) / w * 2 * np.pi - np.pi
    d = g.lonlat_to_dirs(np.array([lon]), np.array([0.0]))
    assert np.allclose(g.sample_equirect(img, d)[0], u_target, atol=1e-9)


def test_sample_equirect_wraps_across_lon_seam():
    # value = column index; sampling between last column center and column 0
    # must blend col (w-1) with col 0 (wrap), not clamp/extrapolate.
    w, h = 16, 8
    img = np.broadcast_to(np.arange(w, dtype=np.float64), (h, w)).copy()
    u_target = w - 0.75  # 25% of the way from col w-1 toward col 0
    lon = (u_target + 0.5) / w * 2 * np.pi - np.pi
    d = g.lonlat_to_dirs(np.array([lon]), np.array([0.0]))
    got = g.sample_equirect(img, d)[0]
    expect = (w - 1) * 0.75 + 0 * 0.25  # wrap blend
    assert np.isclose(got, expect, atol=1e-9)
    assert not np.isclose(got, u_target, atol=1e-3)  # naive non-wrap value


# ---------------------------------------------------------------- face_project

def test_face_project_axis_hits_face_center():
    z = np.array([0.0, 0.0, 1.0])
    for name, yaw, pitch in g.CUBE_FACES:
        axis = g.rotation_yaw_pitch(yaw, pitch) @ z  # actual face axis
        uv, inside, cc = g.face_project(axis[None, :], yaw, pitch, 100.0)
        assert np.allclose(uv[0], [0.5, 0.5], atol=1e-12), name
        assert bool(inside[0]), name
        assert np.isclose(cc[0], 1.0, atol=1e-12), name


def test_face_project_in_frustum_mask_at_fov_edges():
    # fov=90 -> tan(fov/2)=1: cam-space |x/z|<=1 and |y/z|<=1 is inside.
    cams = np.array([
        [0.99, 0.0, 1.0],    # just inside right edge
        [1.01, 0.0, 1.0],    # just outside right edge
        [0.0, -0.99, 1.0],   # just inside bottom edge
        [0.0, 1.01, 1.0],    # just outside top edge
        [0.0, 0.0, -1.0],    # behind camera
        [0.0, 0.0, 1.0],     # center
    ])
    expect = np.array([True, False, True, False, False, True])
    for yaw, pitch in [(0.0, 0.0), (np.pi / 2, 0.0), (0.3, -0.4)]:
        r = g.rotation_yaw_pitch(yaw, pitch)
        world = cams @ r.T  # world = R @ cam
        uv, inside, cc = g.face_project(world, yaw, pitch, 90.0)
        assert np.array_equal(inside, expect), (yaw, pitch)
        # inside points map into [0,1]^2
        assert np.all(uv[inside] >= 0) and np.all(uv[inside] <= 1)
        # behind-camera center_cos reported as 0
        assert cc[4] == 0.0


def test_face_project_edge_uv_values():
    # front face, fov=90: cam x/z=1 at the right edge -> u01=1, v01=0.5
    uv, inside, _ = g.face_project(
        np.array([[1.0, 0.0, 1.0]]), 0.0, 0.0, 90.0
    )
    assert np.allclose(uv[0], [1.0, 0.5], atol=1e-12)
    assert bool(inside[0])


# ---------------------------------------------------------------- rendering

def test_render_perspective_horizontal_gradient_monotonic():
    w, h = 128, 64
    pano = np.broadcast_to(np.arange(w, dtype=np.float64), (h, w)).copy()
    out = g.render_perspective(pano, 90.0, 33, 33, 0.0, 0.0)
    assert out.shape == (33, 33)
    # front view (lon in +/-45 deg) is far from the seam: every row strictly
    # increases left->right.
    assert np.all(np.diff(out, axis=1) > 0)


# ---------------------------------------------------------------- normals

def test_normals_from_depth_constant_sphere_faces_camera():
    w, h = 64, 32
    dirs = g.equirect_dirs(w, h)
    depth = np.full((h, w), 2.5, dtype=np.float64)
    n = g.normals_from_depth(depth, dirs)
    assert n.shape == (h, w, 3)
    # unit norm everywhere
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-6)
    # oriented toward the camera everywhere (never outward)
    assert np.all(np.sum(n * dirs, axis=-1) <= 1e-12)
    # away from poles: n ~ -dirs
    band = slice(h // 4, 3 * h // 4)
    dot = np.sum(n[band] * (-dirs[band]), axis=-1)
    assert np.all(dot > 0.99)


def test_normals_from_depth_wraps_longitude_seam():
    """Regression (adversarial review): the x central differences must wrap
    the equirect longitude axis, so columns 0 and w-1 get genuine computed
    normals (they used to silently fall back to -dir despite valid depths).
    Discriminating check: a smooth non-symmetric wrap-continuous field rolled
    by half a width must yield the same normals once un-rolled."""
    w, h = 64, 32
    dirs = g.equirect_dirs(w, h)
    lon, lat = g.equirect_lonlat(w, h)
    depth = 2.0 + 0.4 * np.sin(lon) + 0.2 * np.cos(2 * lon + 0.7) + 0.3 * np.sin(lat)
    n = g.normals_from_depth(depth, dirs)
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0, atol=1e-6)
    s = w // 2
    n_roll = g.normals_from_depth(np.roll(depth, s, axis=1), np.roll(dirs, s, axis=1))
    assert np.allclose(n, np.roll(n_roll, -s, axis=1), atol=1e-12)
    # the seam columns really are computed normals, not the -dir fallback:
    # the field's lon-gradient at lon=+/-pi is ~0.4|cos(pi)| != 0, so the
    # true surface tilts away from the ray there.
    band = slice(h // 4, 3 * h // 4)
    assert np.abs(n[band, 0] + dirs[band, 0]).max() > 1e-2
    assert np.abs(n[band, -1] + dirs[band, -1]).max() > 1e-2


def test_normals_from_depth_invalid_depth_falls_back_to_minus_dir():
    w, h = 16, 8
    dirs = g.equirect_dirs(w, h)
    depth = np.full((h, w), np.inf)
    # note: normals_from_depth does not guard inf arithmetic with np.errstate
    # (face_project does), so inf depths emit RuntimeWarnings; the fallback
    # output is still correct.
    with np.errstate(invalid="ignore"):
        n = g.normals_from_depth(depth, dirs)
    assert np.allclose(n, -dirs, atol=1e-12)


# ---------------------------------------------------------------- misc

def test_angular_pixel_size():
    assert np.isclose(g.angular_pixel_size(1024), np.pi / 1024, atol=0.0)
    assert np.isclose(g.angular_pixel_size(1), np.pi, atol=0.0)


def test_pitch_of_dirs():
    assert np.isclose(g.pitch_of_dirs(np.array([0.0, 1.0, 0.0])), np.pi / 2)
    assert np.isclose(g.pitch_of_dirs(np.array([0.0, 0.0, 1.0])), 0.0)
    # norm-invariant
    assert np.isclose(g.pitch_of_dirs(np.array([0.0, 5.0, 0.0])), np.pi / 2)
