"""Tests for scenic/rasterizer.py (deterministic CPU 3DGS EWA rasterizer)."""
from __future__ import annotations

import numpy as np

from scenic import determinism, geometry
from scenic.plyio import SplatData, opacity_to_logit, rgb01_to_dc
from scenic.rasterizer import Camera, render

PX = 65          # odd -> exact center pixel at index 32 (u = w/2 - 0.5 = 32.0)
CENTER = 32
FOV = 60.0


def make_splats(
    points,
    colors,
    opacities,
    log_scales,
    quats=None,
) -> SplatData:
    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = xyz.shape[0]
    col = np.asarray(colors, dtype=np.float32).reshape(n, 3)
    op = np.asarray(opacities, dtype=np.float32).reshape(n)
    ls = np.asarray(log_scales, dtype=np.float32).reshape(n, 3)
    if quats is None:
        quats = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
    return SplatData(
        xyz=xyz,
        normals=np.zeros((n, 3), dtype=np.float32),
        f_dc=rgb01_to_dc(col).astype(np.float32),
        opacity_logit=opacity_to_logit(op).astype(np.float32),
        log_scales=ls,
        quat_wxyz=np.asarray(quats, dtype=np.float32).reshape(n, 4),
        layer=np.zeros(n, dtype=np.uint8),
        origin_stage=np.zeros(n, dtype=np.uint8),
    )


def origin_cam(yaw: float = 0.0, pitch: float = 0.0) -> Camera:
    return Camera(pos=np.zeros(3, dtype=np.float64), yaw=yaw, pitch=pitch)


def rgb_argmax(rgb: np.ndarray) -> tuple[int, int]:
    lum = rgb.astype(np.int64).sum(axis=-1)
    vy, ux = np.unravel_index(int(np.argmax(lum)), lum.shape)
    return int(vy), int(ux)


def test_single_splat_centered_blob():
    s = make_splats(
        [[0.0, 0.0, 5.0]], [[1.0, 1.0, 1.0]], [0.98],
        [[np.log(0.2)] * 3],
    )
    out = render(s, origin_cam(), PX, PX, FOV)
    # output contract: dtypes and shapes
    assert out["rgb"].shape == (PX, PX, 3) and out["rgb"].dtype == np.uint8
    assert out["alpha"].shape == (PX, PX) and out["alpha"].dtype == np.float32
    assert out["depth"].shape == (PX, PX) and out["depth"].dtype == np.float32
    # centered blob
    vy, ux = rgb_argmax(out["rgb"])
    assert abs(vy - CENTER) <= 1 and abs(ux - CENTER) <= 1
    assert out["alpha"].max() > 0.5
    assert abs(float(out["depth"][CENTER, CENTER]) - 5.0) <= 0.2
    # background: far corner untouched -> alpha 0, depth +inf
    assert out["alpha"][0, 0] == 0.0
    assert np.isinf(out["depth"][0, 0])


def test_occlusion_near_over_far():
    # far GREEN splat listed FIRST to prove the depth sort orders compositing
    s = make_splats(
        [[0.0, 0.0, 8.0], [0.0, 0.0, 3.0]],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        [0.98, 0.98],
        [[np.log(0.4)] * 3, [np.log(0.15)] * 3],
    )
    out = render(s, origin_cam(), PX, PX, FOV)
    r, g, b = (int(c) for c in out["rgb"][CENTER, CENTER])
    assert r > 200, f"near red splat should dominate center, got {(r, g, b)}"
    assert g < 40 and b < 10
    # expected depth ~ near depth (far leaks only through residual T)
    assert abs(float(out["depth"][CENTER, CENTER]) - 3.0) <= 0.5


def test_projection_matches_perspective_dirs():
    # round-trip against geometry: dir to point -> pixel whose ray matches
    w = h = 129
    fov = 70.0
    p = np.array([1.0, 0.5, 4.0])
    cam = Camera(pos=np.array([0.1, -0.2, 0.3]), yaw=0.3, pitch=-0.2)
    s = make_splats([p], [[1.0, 1.0, 1.0]], [0.98], [[np.log(0.04)] * 3])
    out = render(s, cam, w, h, fov)
    vy, ux = np.unravel_index(int(np.argmax(out["alpha"])), (h, w))

    dirs = geometry.perspective_dirs(fov, w, h, cam.yaw, cam.pitch)
    to_p = p - cam.pos
    to_p = to_p / np.linalg.norm(to_p)
    ey, ex = np.unravel_index(int(np.argmax(dirs @ to_p)), (h, w))
    assert abs(int(vy) - int(ey)) <= 1 and abs(int(ux) - int(ex)) <= 1, (
        f"rendered argmax {(vy, ux)} vs perspective_dirs {(ey, ex)}"
    )

    # analytic projection agrees with the geometry-derived pixel too
    r = geometry.rotation_yaw_pitch(cam.yaw, cam.pitch)
    pc = r.T @ (p - cam.pos)
    f = (w / 2.0) / np.tan(np.deg2rad(fov) / 2.0)
    ua = f * pc[0] / pc[2] + w / 2.0 - 0.5
    va = h / 2.0 - 0.5 - f * pc[1] / pc[2]
    assert abs(ua - ex) <= 0.51 and abs(va - ey) <= 0.51


def _random_scene(n: int = 200) -> SplatData:
    g = determinism.rng("test-rasterizer-scene")
    pts = g.normal(0.0, 2.0, size=(n, 3))
    pts[:, 2] += 6.0  # mostly in front of the camera
    quats = g.normal(size=(n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return make_splats(
        pts,
        g.uniform(0.0, 1.0, size=(n, 3)),
        g.uniform(0.3, 0.99, size=n),
        g.uniform(np.log(0.05), np.log(0.5), size=(n, 3)),
        quats=quats,
    )


def test_determinism_byte_identical():
    s = _random_scene()
    cam = Camera(pos=np.array([0.2, 0.1, -0.3]), yaw=0.1, pitch=-0.05)
    a = render(s, cam, 96, 80, 75.0)
    b = render(s, cam, 96, 80, 75.0)
    assert a["rgb"].tobytes() == b["rgb"].tobytes()
    assert a["alpha"].tobytes() == b["alpha"].tobytes()
    assert a["depth"].tobytes() == b["depth"].tobytes()
    # sanity: the scene actually renders something
    assert a["alpha"].max() > 0.5


def test_override_rgb():
    s = make_splats(
        [[0.0, 0.0, 5.0]], [[1.0, 1.0, 1.0]], [0.98],
        [[np.log(0.2)] * 3],
    )
    plain = render(s, origin_cam(), PX, PX, FOV)
    magenta = render(
        s, origin_cam(), PX, PX, FOV,
        override_rgb=np.array([[1.0, 0.0, 1.0]]),
    )
    pr = plain["rgb"][CENTER, CENTER]
    mr = magenta["rgb"][CENTER, CENTER]
    assert int(pr[1]) > 200  # white splat: green high
    assert int(mr[0]) > 200 and int(mr[1]) < 10 and int(mr[2]) > 200
    # alpha/depth unchanged by a pure color override
    assert plain["alpha"].tobytes() == magenta["alpha"].tobytes()
    assert plain["depth"].tobytes() == magenta["depth"].tobytes()


def test_yaw_pitch_moves_projection():
    s = make_splats(
        [[0.0, 0.0, 5.0]], [[1.0, 1.0, 1.0]], [0.98],
        [[np.log(0.1)] * 3],
    )
    zhat = np.array([0.0, 0.0, 1.0])

    def argmax_px(yaw: float, pitch: float) -> tuple[int, int]:
        out = render(s, origin_cam(yaw, pitch), PX, PX, FOV)
        vy, ux = np.unravel_index(int(np.argmax(out["alpha"])), (PX, PX))
        return int(vy), int(ux)

    def expected_px(yaw: float, pitch: float) -> tuple[int, int]:
        dirs = geometry.perspective_dirs(FOV, PX, PX, yaw, pitch)
        ey, ex = np.unravel_index(int(np.argmax(dirs @ zhat)), (PX, PX))
        return int(ey), int(ex)

    v0, u0 = argmax_px(0.0, 0.0)
    assert (v0, u0) == (CENTER, CENTER)

    for yaw, pitch in [(0.15, 0.0), (-0.15, 0.0), (0.0, 0.15), (0.0, -0.15)]:
        got = argmax_px(yaw, pitch)
        want = expected_px(yaw, pitch)
        assert abs(got[0] - want[0]) <= 1 and abs(got[1] - want[1]) <= 1, (
            f"yaw={yaw} pitch={pitch}: render {got} vs geometry {want}"
        )

    # explicit directions per the shared geometry convention
    # (+yaw turns camera toward +X -> +Z point moves LEFT, u decreases)
    _, u_yaw = argmax_px(0.15, 0.0)
    assert u_yaw < CENTER
    _, u_nyaw = argmax_px(-0.15, 0.0)
    assert u_nyaw > CENTER
    # rotation_yaw_pitch(0, +p) points the camera axis toward +Y (looks up),
    # so the +Z point moves DOWN in the image (v increases); verified to
    # match geometry.perspective_dirs above.
    v_pitch, _ = argmax_px(0.0, 0.15)
    assert v_pitch > CENTER
    v_npitch, _ = argmax_px(0.0, -0.15)
    assert v_npitch < CENTER


def test_behind_camera_and_empty():
    s = make_splats(
        [[0.0, 0.0, -5.0]], [[1.0, 1.0, 1.0]], [0.98],
        [[np.log(0.2)] * 3],
    )
    out = render(s, origin_cam(), 32, 32, FOV)
    assert out["alpha"].max() == 0.0
    assert np.all(np.isinf(out["depth"]))
    assert out["rgb"].sum() == 0

    empty = make_splats(
        np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3))
    )
    out = render(empty, origin_cam(), 32, 32, FOV)
    assert out["alpha"].max() == 0.0
    assert np.all(np.isinf(out["depth"]))
