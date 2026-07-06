"""s4_place tests. Oracles per docs/CONTRACTS.md:
  (a) SPHERE — constant fg depth R ==> every splat at |xyz| ~ R, with
      class-stride radii and canonical unit quats;
  (b) ROOM — closed-form box distances (6 x 3 x 8 m, camera 1.5 m above the
      floor) ==> every splat inside the box bounds +-1%, floor splats at
      y ~ -1.5 +-2%;
  (c) determinism — double run is byte-identical;
  (d) stride_multiplier=2 ==> < 0.5x the splats of multiplier 1 (roughly
      quadratic reduction).
Plus shell placement and quaternion round-trip unit tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline import s4_place
from scenic import determinism, geometry, imageio, params as params_mod
from scenic import plyio, receipts, schema
from scenic.stage import Ctx

REPO = Path(__file__).resolve().parent.parent

H, W = 64, 128
SPHERE_R = 7.0
FG_COLOR = np.array([120, 80, 200], np.uint8)
BG_COLOR = np.array([40, 160, 90], np.uint8)
BOX_X, BOX_Y, BOX_Z = 3.0, 1.5, 4.0  # half-extents; 6 x 3 x 8 m room


# ------------------------------------------------------------- fixtures


def _params() -> dict:
    return params_mod.load(REPO / "params.yaml")


def _ctx() -> Ctx:
    return Ctx(
        repo_root=REPO,
        pano_path=REPO / "params.yaml",  # unused by s4 (reads the run dir)
        sidecar_path=REPO / "params.yaml",
        params_path=REPO / "params.yaml",
        weights_dir=REPO / "weights",
    )


def make_run(
    run_dir: Path,
    fg_depth: np.ndarray,
    *,
    sky: np.ndarray | None = None,
    fg_rgb: np.ndarray | None = None,
    bg_depth: np.ndarray | None = None,
    bg_rgb: np.ndarray | None = None,
    bg_mask: np.ndarray | None = None,
) -> None:
    h, w = fg_depth.shape
    s3d = run_dir / "s3_layers" / "out"
    s2d = run_dir / "s2_depth" / "out"
    s3d.mkdir(parents=True)
    s2d.mkdir(parents=True)
    if sky is None:
        sky = np.zeros((h, w), bool)
    if fg_rgb is None:
        fg_rgb = np.broadcast_to(FG_COLOR, (h, w, 3)).copy()
    if bg_depth is None:
        bg_depth = np.full((h, w), np.inf, np.float32)
    if bg_rgb is None:
        bg_rgb = np.broadcast_to(BG_COLOR, (h, w, 3)).copy()
    if bg_mask is None:
        bg_mask = np.zeros((h, w), bool)
    imageio.save_png(s3d / "fg_rgb.png", fg_rgb.astype(np.uint8))
    imageio.save_npy(s3d / "fg_depth.npy", fg_depth.astype(np.float32))
    imageio.save_mask_png(s3d / "fg_mask.png", np.isfinite(fg_depth) & ~sky)
    imageio.save_png(s3d / "bg_rgb.png", bg_rgb.astype(np.uint8))
    imageio.save_npy(s3d / "bg_depth.npy", bg_depth.astype(np.float32))
    imageio.save_mask_png(s3d / "bg_mask.png", bg_mask)
    schema.write_validated(
        s3d / "layers.json",
        {
            "band_px": 2,
            "band_derivation": {},
            "edge_px_count": 0,
            "bg_filled_px": int(bg_mask.sum()),
        },
        "layers",
    )
    imageio.save_mask_png(s2d / "sky_mask.png", sky)


def run_stage(run_dir: Path, stride_multiplier: float = 1.0) -> dict:
    params = _params()
    determinism.set_seed(params.get("seed", 0))
    s4_place.run(run_dir, params, _ctx(), stride_multiplier=stride_multiplier)
    return params


def _room_depth() -> np.ndarray:
    """Exact per-direction distance to the interior of the 6x3x8 box with the
    camera at the origin (1.5 m above the floor, i.e. mid-height)."""
    dirs = geometry.equirect_dirs(W, H)
    ax = np.abs(dirs[..., 0])
    ay = np.abs(dirs[..., 1])
    az = np.abs(dirs[..., 2])
    with np.errstate(divide="ignore"):
        tx = np.where(ax > 1e-12, BOX_X / ax, np.inf)
        ty = np.where(ay > 1e-12, BOX_Y / ay, np.inf)
        tz = np.where(az > 1e-12, BOX_Z / az, np.inf)
    return np.minimum(np.minimum(tx, ty), tz).astype(np.float32)


@pytest.fixture(scope="module")
def sphere_run(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("s4sphere") / "run"
    make_run(d, np.full((H, W), SPHERE_R, np.float32))
    run_stage(d)
    return d


@pytest.fixture(scope="module")
def room_run(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("s4room") / "run"
    depth = _room_depth()
    bg_mask = np.zeros((H, W), bool)
    bg_mask[20:28, 30:50] = True  # synthesized band patch on the same geometry
    make_run(d, depth, bg_depth=depth.copy(), bg_mask=bg_mask)
    run_stage(d)
    return d


def _out(run_dir: Path) -> Path:
    return run_dir / "s4_place" / "out"


def _read(run_dir: Path):
    splats = plyio.read_splats(_out(run_dir) / "splats.ply")
    meta = schema.read_validated(_out(run_dir) / "splats_meta.json", "splats_meta")
    return splats, meta


# ------------------------------------------------------------- (a) sphere


def test_sphere_all_splats_on_sphere(sphere_run):
    splats, meta = _read(sphere_run)
    assert meta["count"] == len(splats) > 0
    assert meta["counts_by_layer"]["fg"] > 0
    assert meta["counts_by_layer"]["bg"] == 0
    assert meta["counts_by_layer"]["shell"] == 0
    r = np.linalg.norm(splats.xyz.astype(np.float64), axis=1)
    assert np.all(r >= SPHERE_R - 1e-3)
    assert np.all(r <= SPHERE_R + 1e-3)
    assert np.all(splats.layer == plyio.LAYER_FG)
    assert np.all(splats.origin_stage == 4)


def test_sphere_radii_match_class_strides(sphere_run):
    splats, meta = _read(sphere_run)
    p = _params()["s4"]
    angpix = geometry.angular_pixel_size(H)
    xyz = splats.xyz.astype(np.float64)
    rnorm = np.linalg.norm(xyz, axis=1)
    pitch_deg = np.degrees(np.arcsin(np.clip(xyz[:, 1] / rnorm, -1, 1)))
    # constant depth => no edges; ground stride 1, base stride 2 at mult=1
    expect_stride = np.where(pitch_deg < p["ground_band_pitch_deg"], 1, 2)
    assert meta["strides"] == {"edge": 1, "ground": 1, "base": 2, "bg": 2, "shell": 4}
    expect_r = SPHERE_R * angpix * expect_stride * p["scale_multiplier"]
    got_r = np.exp(splats.log_scales[:, 0].astype(np.float64))
    assert np.allclose(got_r, expect_r, rtol=1e-4)
    assert {1, 2} == set(np.unique(expect_stride))  # both classes sampled
    # isotropic tangent axes + flattened normal axis
    assert np.allclose(splats.log_scales[:, 1], splats.log_scales[:, 0], atol=1e-6)
    got_r2 = np.exp(splats.log_scales[:, 2].astype(np.float64))
    assert np.allclose(got_r2, expect_r * p["flatten_ratio"], rtol=1e-4)


def test_sphere_quats_unit_canonical(sphere_run):
    splats, _ = _read(sphere_run)
    q = splats.quat_wxyz.astype(np.float64)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)
    assert np.all(q[:, 0] >= -1e-7)


def test_sphere_colors_and_opacity(sphere_run):
    splats, _ = _read(sphere_run)
    p = _params()["s4"]
    rgb = plyio.dc_to_rgb01(splats.f_dc.astype(np.float64)) * 255.0
    assert np.allclose(rgb, FG_COLOR.astype(np.float64), atol=0.51)
    op = plyio.logit_to_opacity(splats.opacity_logit.astype(np.float64))
    assert np.allclose(op, p["fg_opacity"], atol=1e-4)


def test_sphere_receipt_written(sphere_run):
    rec = receipts.read_receipt(sphere_run, "s4_place")
    assert rec["stage"] == "s4_place"
    assert "splats" in rec["outputs"] and "splats_meta" in rec["outputs"]
    for key in ["fg_rgb", "fg_depth", "bg_rgb", "bg_depth", "sky_mask", "layers"]:
        assert key in rec["inputs"], key
    assert rec["notes"]["strides"]["base"] == 2
    assert rec["notes"]["stride_multiplier"] == 1.0
    assert rec["params_used"]["s4"]["base_stride"] == 2
    assert rec["weights"] == []


# ------------------------------------------------------------- (b) room


def test_room_splats_inside_box(room_run):
    splats, meta = _read(room_run)
    assert meta["counts_by_layer"]["fg"] > 0
    assert meta["counts_by_layer"]["bg"] > 0  # the bg_mask patch got sampled
    assert meta["counts_by_layer"]["shell"] == 0
    xyz = splats.xyz.astype(np.float64)
    assert np.all(np.abs(xyz[:, 0]) <= BOX_X * 1.01)
    assert np.all(np.abs(xyz[:, 1]) <= BOX_Y * 1.01)
    assert np.all(np.abs(xyz[:, 2]) <= BOX_Z * 1.01)


def test_room_floor_splats_at_floor_height(room_run):
    splats, _ = _read(room_run)
    xyz = splats.xyz.astype(np.float64)
    rnorm = np.linalg.norm(xyz, axis=1)
    d = xyz / rnorm[:, None]
    with np.errstate(divide="ignore"):
        tx = np.where(np.abs(d[:, 0]) > 1e-12, BOX_X / np.abs(d[:, 0]), np.inf)
        ty = np.where(np.abs(d[:, 1]) > 1e-12, BOX_Y / np.abs(d[:, 1]), np.inf)
        tz = np.where(np.abs(d[:, 2]) > 1e-12, BOX_Z / np.abs(d[:, 2]), np.inf)
    floor = (ty < 0.999 * tx) & (ty < 0.999 * tz) & (d[:, 1] < 0)
    assert floor.sum() > 0
    y = xyz[floor, 1]
    assert np.all(np.abs(y - (-BOX_Y)) <= BOX_Y * 0.02)  # y ~ -1.5 +- 2%


def test_room_radii_are_stride_multiples_of_pixel_size(room_run):
    splats, meta = _read(room_run)
    p = _params()["s4"]
    angpix = geometry.angular_pixel_size(H)
    depth = np.linalg.norm(splats.xyz.astype(np.float64), axis=1)
    ratio = np.exp(splats.log_scales[:, 0].astype(np.float64)) / (
        depth * angpix * p["scale_multiplier"]
    )
    strides = meta["strides"]
    allowed = {strides["edge"], strides["ground"], strides["base"], strides["bg"]}
    assert np.allclose(ratio, np.round(ratio), rtol=1e-3, atol=1e-3)
    assert set(np.round(ratio).astype(int)) <= allowed


# ------------------------------------------------------------- (c) determinism


def test_determinism_double_run_byte_identical(tmp_path):
    outputs = []
    for name in ["a", "b"]:
        d = tmp_path / name / "run"
        depth = _room_depth()
        bg_mask = np.zeros((H, W), bool)
        bg_mask[20:28, 30:50] = True
        make_run(d, depth, bg_depth=depth.copy(), bg_mask=bg_mask)
        run_stage(d)
        outputs.append(
            (
                (_out(d) / "splats.ply").read_bytes(),
                (_out(d) / "splats_meta.json").read_bytes(),
            )
        )
    assert outputs[0][0] == outputs[1][0]
    assert outputs[0][1] == outputs[1][1]


# ------------------------------------------------------------- (d) stride cap


def test_stride_multiplier_reduces_counts_quadratically(tmp_path):
    counts = {}
    for mult in [1.0, 2.0]:
        d = tmp_path / f"m{int(mult)}" / "run"
        make_run(d, np.full((H, W), SPHERE_R, np.float32))
        run_stage(d, stride_multiplier=mult)
        _, meta = _read(d)
        assert meta["stride_multiplier"] == mult
        counts[mult] = meta["count"]
    assert counts[2.0] > 0
    assert counts[2.0] < 0.5 * counts[1.0]


# ------------------------------------------------------------- shell


def test_shell_sky_and_far_content(tmp_path):
    d = tmp_path / "run"
    p = _params()["s4"]
    depth = np.full((H, W), SPHERE_R, np.float32)
    sky = np.zeros((H, W), bool)
    sky[:8] = True
    depth[:8] = np.inf                    # sky has no depth
    depth[8:12] = 150.0                   # finite but beyond shell_radius/2
    make_run(d, depth, sky=sky)
    run_stage(d)
    splats, meta = _read(d)
    shell_r = p["shell_radius_m"]
    sh = splats.layer == plyio.LAYER_SHELL
    assert meta["counts_by_layer"]["shell"] == int(sh.sum()) > 0
    xyz = splats.xyz.astype(np.float64)
    rnorm = np.linalg.norm(xyz, axis=1)
    assert np.allclose(rnorm[sh], shell_r, atol=1e-2)
    # textured shell faces the camera
    assert np.allclose(
        splats.normals[sh].astype(np.float64), -xyz[sh] / shell_r, atol=1e-4
    )
    op = plyio.logit_to_opacity(splats.opacity_logit[sh].astype(np.float64))
    assert np.allclose(op, 0.995, atol=1e-4)
    angpix = geometry.angular_pixel_size(H)
    expect_r = shell_r * angpix * meta["strides"]["shell"] * p["scale_multiplier"]
    assert np.allclose(np.exp(splats.log_scales[sh, 0].astype(np.float64)),
                       expect_r, rtol=1e-4)
    # fg splats exist at both depths and never inside the sky band
    fg = splats.layer == plyio.LAYER_FG
    assert fg.sum() > 0
    near7 = np.isclose(rnorm[fg], SPHERE_R, atol=1e-3)
    near150 = np.isclose(rnorm[fg], 150.0, atol=1e-2)
    assert np.all(near7 | near150)
    assert near150.sum() > 0


# ------------------------------------------------------------- quats


def _random_unit_vectors(n: int) -> np.ndarray:
    g = determinism.rng("test-s4-quat")
    v = g.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_frames_are_proper_rotations_with_normal_column():
    n_vec = np.concatenate(
        [
            _random_unit_vectors(256),
            np.array(
                [
                    [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0],
                    [0, 0, 1], [0, 0, -1],
                    [0.05, 0.99, 0.05], [0.05, -0.99, 0.05],
                ],
                dtype=np.float64,
            ),
        ]
    )
    n_vec = n_vec / np.linalg.norm(n_vec, axis=1, keepdims=True)
    R = s4_place.frames_from_normals(n_vec)
    eye = np.einsum("nij,nkj->nik", R, R)  # R @ R.T
    assert np.allclose(eye, np.eye(3)[None], atol=1e-12)
    assert np.allclose(np.linalg.det(R), 1.0, atol=1e-12)
    assert np.allclose(R[:, :, 2], n_vec, atol=1e-12)  # third column = n


def test_quat_matrix_round_trip():
    R = s4_place.frames_from_normals(_random_unit_vectors(512))
    q = s4_place.quat_from_rotmats(R)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-12)
    assert np.all(q[:, 0] >= 0.0)
    R2 = s4_place.rotmat_from_quat(q)
    assert np.allclose(R2, R, atol=1e-9)


def test_quat_round_trip_w_near_zero():
    # rotations by pi (w == 0) hit every non-w pivot branch
    g = determinism.rng("test-s4-quat-pi")
    axes = g.normal(size=(64, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    q_in = np.concatenate([np.zeros((64, 1)), axes], axis=1)
    R = s4_place.rotmat_from_quat(q_in)
    q_out = s4_place.quat_from_rotmats(R)
    assert np.allclose(np.linalg.norm(q_out, axis=1), 1.0, atol=1e-12)
    assert np.all(q_out[:, 0] >= 0.0)
    assert np.allclose(s4_place.rotmat_from_quat(q_out), R, atol=1e-9)
