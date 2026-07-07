"""s4_place tests. v2 "shell-inward + density/scale" oracles per
docs/CONTRACTS.md "S4 place":

  (a) SPHERE — constant fg depth R with min_content < R < shell_distance
      ==> every splat at |xyz| ~ R, all LAYER_FG (no shell routing);
      constant fg depth 100 (> shell_distance 50) ==> ALL routed to the
      shell at radius 200;
  (b) NEAR routing — constant fg depth 3 (< min_content 6) ==> ALL shell;
  (c) FEATHER — depths spanning [46,50] ==> fg opacities ramp to ~0 at 50;
  (d) COLOR VARIANCE — flat pano vs noisy pano ==> noisy retains more splats
      (color_var_retained_frac higher);
  (e) DETERMINISM — double run is byte-identical;
  (f) STRIDE — stride_multiplier=2 reduces the count;
plus bg scale clamp, shell placement, and quaternion round-trip unit tests."""
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
SPHERE_R = 20.0                       # min_content(6) < R < shell_distance(50)
FG_COLOR = np.array([120, 80, 200], np.uint8)
BG_COLOR = np.array([40, 160, 90], np.uint8)
# Big room so every wall distance sits in [min_content, shell_distance-feather]
BOX_X, BOX_Y, BOX_Z = 12.0, 8.0, 16.0  # half-extents; 24 x 16 x 32 m room


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
    """Exact per-direction distance to the interior of the big box, camera at
    the origin (centered)."""
    dirs = geometry.equirect_dirs(W, H)
    ax = np.abs(dirs[..., 0])
    ay = np.abs(dirs[..., 1])
    az = np.abs(dirs[..., 2])
    with np.errstate(divide="ignore"):
        tx = np.where(ax > 1e-12, BOX_X / ax, np.inf)
        ty = np.where(ay > 1e-12, BOX_Y / ay, np.inf)
        tz = np.where(az > 1e-12, BOX_Z / az, np.inf)
    return np.minimum(np.minimum(tx, ty), tz).astype(np.float32)


def _checkerboard_rgb(h: int, w: int) -> np.ndarray:
    """High local-variance texture: alternating black/white per pixel."""
    r = np.arange(h)[:, None]
    c = np.arange(w)[None, :]
    val = np.where((r + c) % 2 == 0, 255, 0).astype(np.uint8)
    return np.repeat(val[..., None], 3, axis=2)


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


def test_sphere_all_fg_on_sphere(sphere_run):
    splats, meta = _read(sphere_run)
    assert meta["count"] == len(splats) > 0
    assert meta["counts_by_layer"]["fg"] > 0
    assert meta["counts_by_layer"]["bg"] == 0
    assert meta["counts_by_layer"]["shell"] == 0  # R well inside the frontier
    assert meta["near_shell_px"] == 0
    assert meta["far_shell_px"] == 0
    assert meta["feather_px"] == 0
    r = np.linalg.norm(splats.xyz.astype(np.float64), axis=1)
    assert np.all(r >= SPHERE_R - 1e-3)
    assert np.all(r <= SPHERE_R + 1e-3)
    assert np.all(splats.layer == plyio.LAYER_FG)
    assert np.all(splats.origin_stage == 4)


def test_sphere_strides_and_radii(sphere_run):
    splats, meta = _read(sphere_run)
    p = _params()["s4"]
    assert meta["strides"] == {"edge": 1, "ground": 1, "base": 2, "bg": 2, "shell": 4}
    angpix = geometry.angular_pixel_size(H)
    depth = np.linalg.norm(splats.xyz.astype(np.float64), axis=1)
    ratio = np.exp(splats.log_scales[:, 0].astype(np.float64)) / (
        depth * angpix * p["scale_multiplier"]
    )
    assert np.allclose(ratio, np.round(ratio), rtol=1e-3, atol=1e-3)
    strides_seen = set(np.round(ratio).astype(int))
    assert strides_seen <= {1, 2}
    assert {1, 2} == strides_seen  # both ground(1) and base(2) sampled
    # isotropic tangent axes + flattened normal axis
    assert np.allclose(splats.log_scales[:, 1], splats.log_scales[:, 0], atol=1e-6)
    got_r2 = np.exp(splats.log_scales[:, 2].astype(np.float64))
    got_r = np.exp(splats.log_scales[:, 0].astype(np.float64))
    assert np.allclose(got_r2, got_r * p["flatten_ratio"], rtol=1e-4)


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
    assert np.allclose(op, p["fg_opacity"], atol=1e-4)  # no feather at R=20


def test_sphere_receipt_written(sphere_run):
    rec = receipts.read_receipt(sphere_run, "s4_place")
    assert rec["stage"] == "s4_place"
    assert "splats" in rec["outputs"] and "splats_meta" in rec["outputs"]
    for key in ["fg_rgb", "fg_depth", "bg_rgb", "bg_depth", "sky_mask", "layers"]:
        assert key in rec["inputs"], key
    assert rec["notes"]["strides"]["base"] == 2
    assert rec["notes"]["stride_multiplier"] == 1.0
    assert rec["params_used"]["s4"]["base_stride"] == 2
    assert rec["params_used"]["min_content_distance_m"] == 6.0
    assert rec["notes"]["near_shell_px"] == 0
    assert rec["weights"] == []


# --------------------------------------------- (a) far routing -> shell


def test_far_content_all_routed_to_shell(tmp_path):
    d = tmp_path / "run"
    p = _params()["s4"]
    depth = np.full((H, W), 100.0, np.float32)  # > shell_distance 50
    make_run(d, depth)
    run_stage(d)
    splats, meta = _read(d)
    assert meta["counts_by_layer"]["fg"] == 0
    assert meta["counts_by_layer"]["bg"] == 0
    assert meta["counts_by_layer"]["shell"] == len(splats) > 0
    assert meta["far_shell_px"] == H * W
    assert meta["near_shell_px"] == 0
    r = np.linalg.norm(splats.xyz.astype(np.float64), axis=1)
    assert np.allclose(r, p["shell_radius_m"], atol=1e-2)
    assert np.all(splats.layer == plyio.LAYER_SHELL)


# ------------------------------------------------------------- (b) near routing


def test_near_content_kept_as_fg_splats(tmp_path):
    # Near content (d < min_content 6m) STAYS as fg splats at true depth (real
    # geometry the viewer must see); it is NOT routed to the shell. Its
    # discomfort is flagged by the min_content_distance + stereo gates off the
    # depth map, not hidden behind a backdrop.
    d = tmp_path / "run"
    depth = np.full((H, W), 3.0, np.float32)  # < min_content 6
    make_run(d, depth)
    run_stage(d)
    splats, meta = _read(d)
    assert meta["counts_by_layer"]["fg"] > 0
    assert meta["counts_by_layer"]["shell"] == 0
    assert meta["near_shell_px"] == 0
    assert meta["near_fg_px"] == H * W
    assert meta["far_shell_px"] == 0
    fg = splats.layer == plyio.LAYER_FG
    r = np.linalg.norm(splats.xyz[fg].astype(np.float64), axis=1)
    assert np.allclose(r, 3.0, atol=1e-2)  # placed at true depth, not shell radius


# ------------------------------------------------------------- (c) feather


def test_feather_ramps_opacity_to_zero_at_frontier(tmp_path):
    d = tmp_path / "run"
    p = _params()["s4"]
    shell_dist = float(p["shell_distance_m"])   # 50
    feather_m = float(p["feather_m"])           # 5
    # depth increases with row from 46 -> 50 (entirely inside the feather band)
    col_depth = np.linspace(46.0, shell_dist, H, dtype=np.float64)
    depth = np.broadcast_to(col_depth[:, None], (H, W)).astype(np.float32)
    make_run(d, depth)
    run_stage(d)
    splats, meta = _read(d)
    assert meta["feather_px"] > 0
    fg = splats.layer == plyio.LAYER_FG
    assert fg.sum() > 0
    xyz = splats.xyz.astype(np.float64)
    dd = np.linalg.norm(xyz[fg], axis=1)
    op = plyio.logit_to_opacity(splats.opacity_logit[fg].astype(np.float64))
    expect = p["fg_opacity"] * np.clip((shell_dist - dd) / feather_m, 0.0, 1.0)
    assert np.allclose(op, expect, atol=3e-3)   # opacity ramps with depth
    assert op.min() < 0.05                       # ~0 near the frontier (d~50)
    assert op.max() > 0.60                       # ~full near d=46
    # the shell backdrop is present behind the fading splats (d > 45)
    assert meta["counts_by_layer"]["shell"] > 0


# ------------------------------------------------------------- (d) color variance


def _run_texture(tmp_path: Path, name: str, fg_rgb: np.ndarray) -> dict:
    d = tmp_path / name / "run"
    depth = np.full((H, W), SPHERE_R, np.float32)  # all content, no feather
    make_run(d, depth, fg_rgb=fg_rgb)
    run_stage(d)
    _, meta = _read(d)
    return meta


def test_color_variance_retains_more_in_textured(tmp_path):
    flat = _run_texture(tmp_path, "flat", np.broadcast_to(FG_COLOR, (H, W, 3)).copy())
    noisy = _run_texture(tmp_path, "noisy", _checkerboard_rgb(H, W))
    # noisy (high local variance) keeps ~everything; flat is decimated to ~1/boost
    assert noisy["color_var_retained_frac"] > flat["color_var_retained_frac"]
    assert noisy["counts_by_layer"]["fg"] > flat["counts_by_layer"]["fg"]
    boost = _params()["s4"]["color_var_boost"]
    assert flat["color_var_retained_frac"] < 0.5           # decimated
    assert flat["color_var_retained_frac"] > 0.9 / boost   # ~1/boost survive
    assert noisy["color_var_retained_frac"] > 0.9          # textured kept


# ------------------------------------------------------------- (e) determinism


def test_determinism_double_run_byte_identical(tmp_path):
    outputs = []
    for name in ["a", "b"]:
        d = tmp_path / name / "run"
        depth = _room_depth()
        bg_mask = np.zeros((H, W), bool)
        bg_mask[20:28, 30:50] = True
        make_run(d, depth, bg_depth=depth.copy(), bg_mask=bg_mask,
                 fg_rgb=_checkerboard_rgb(H, W))
        run_stage(d)
        outputs.append(
            (
                (_out(d) / "splats.ply").read_bytes(),
                (_out(d) / "splats_meta.json").read_bytes(),
            )
        )
    assert outputs[0][0] == outputs[1][0]
    assert outputs[0][1] == outputs[1][1]


# ------------------------------------------------------------- (f) stride cap


def test_stride_multiplier_reduces_counts(tmp_path):
    counts = {}
    for mult in [1.0, 2.0]:
        d = tmp_path / f"m{int(mult)}" / "run"
        make_run(d, np.full((H, W), SPHERE_R, np.float32),
                 fg_rgb=_checkerboard_rgb(H, W))
        run_stage(d, stride_multiplier=mult)
        _, meta = _read(d)
        assert meta["stride_multiplier"] == mult
        counts[mult] = meta["count"]
    assert counts[2.0] > 0
    assert counts[2.0] < 0.5 * counts[1.0]


# ------------------------------------------------------------- room / bg clamp


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
    dvec = xyz / rnorm[:, None]
    with np.errstate(divide="ignore"):
        tx = np.where(np.abs(dvec[:, 0]) > 1e-12, BOX_X / np.abs(dvec[:, 0]), np.inf)
        ty = np.where(np.abs(dvec[:, 1]) > 1e-12, BOX_Y / np.abs(dvec[:, 1]), np.inf)
        tz = np.where(np.abs(dvec[:, 2]) > 1e-12, BOX_Z / np.abs(dvec[:, 2]), np.inf)
    floor = (ty < 0.999 * tx) & (ty < 0.999 * tz) & (dvec[:, 1] < 0)
    assert floor.sum() > 0
    y = xyz[floor, 1]
    assert np.all(np.abs(y - (-BOX_Y)) <= BOX_Y * 0.02)  # y ~ -BOX_Y +- 2%


def test_bg_scale_clamped_to_fg_equivalent(room_run):
    splats, meta = _read(room_run)
    p = _params()["s4"]
    angpix = geometry.angular_pixel_size(H)
    strides = meta["strides"]
    bg = splats.layer == plyio.LAYER_BG
    assert bg.sum() > 0
    depth = np.linalg.norm(splats.xyz[bg].astype(np.float64), axis=1)
    got_r = np.exp(splats.log_scales[bg, 0].astype(np.float64))
    ratio = got_r / (depth * angpix * p["scale_multiplier"])
    # clamp => bg stride never exceeds the bg base stride and is an integer
    # in {edge, ground, base, bg}; NEVER inflated beyond the fg-equivalent.
    assert np.allclose(ratio, np.round(ratio), rtol=1e-3, atol=1e-3)
    allowed = {strides["edge"], strides["ground"], strides["base"], strides["bg"]}
    assert set(np.round(ratio).astype(int)) <= allowed
    assert np.all(np.round(ratio).astype(int) <= strides["bg"])  # never inflated


# ------------------------------------------------------------- shell


def test_shell_sky_and_far_content(tmp_path):
    d = tmp_path / "run"
    p = _params()["s4"]
    depth = np.full((H, W), SPHERE_R, np.float32)
    sky = np.zeros((H, W), bool)
    sky[:8] = True
    depth[:8] = np.inf                    # sky has no depth
    depth[8:12] = 150.0                   # finite but beyond shell_distance
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
    # fg splats exist only at the content depth (R), never in sky / far
    fg = splats.layer == plyio.LAYER_FG
    assert fg.sum() > 0
    assert np.allclose(rnorm[fg], SPHERE_R, atol=1e-3)
    assert meta["far_shell_px"] == 4 * W  # 4 rows at depth 150


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


# ------------------------------------------------------------- color-var helpers


def test_hash_uniform_is_deterministic_and_in_range():
    u1 = s4_place._hash_uniform(H, W)
    u2 = s4_place._hash_uniform(H, W)
    assert np.array_equal(u1, u2)
    assert u1.shape == (H, W)
    assert u1.min() >= 0.0 and u1.max() < 1.0


def test_local_color_variance_flat_is_zero_textured_is_high():
    flat = np.full((H, W, 3), 0.5, dtype=np.float64)
    v_flat = s4_place._local_color_variance(flat, 3)
    assert np.allclose(v_flat, 0.0, atol=1e-12)
    noisy = _checkerboard_rgb(H, W).astype(np.float64) / 255.0
    v_noisy = s4_place._local_color_variance(noisy, 3)
    ref = _params()["s4"]["color_var_ref"]
    assert v_noisy.mean() > ref  # well above the reference -> full boost
