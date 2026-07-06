"""s6_compress tests.

Main fixture: a synthetic s4 output (~2k splats) with exactly-known groups —
a dense fg grid cluster, low-opacity splats, isolated outliers, bg duplicates
sitting on fg positions, a separate surviving bg cluster, and a sparse shell
ring — so every prune/merge count is closed-form.

Cap test: a small (128x64) run dir with s3-style inputs; real s3_layers
generates genuine layer artifacts, then s4 placement is re-run by s6's cap
loop. If pipeline.s4_place is not implemented yet, a faithful stub honoring
the documented `run(run_dir, params, ctx, stride_multiplier=)` contract is
installed as pipeline.s4_place (see notes in the stub).
"""
from __future__ import annotations

import copy
import importlib.util
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pipeline import s6_compress
from scenic import determinism, geometry, imageio, params as params_mod
from scenic import plyio, receipts, schema
from scenic.stage import Ctx

REPO = Path(__file__).resolve().parent.parent

# ------------------------------------------------------- synthetic s4 output

N_DENSE = 1900       # fg grid cluster, spacing 0.05, survives everything
N_LOWOP = 30         # fg, sigmoid(opacity) ~ 0.0025 < floor 0.05 -> step 1
N_OUTLIER = 5        # fg, ~60 m out, isolated -> step 2
N_BG_DUP = 40        # bg, 1 mm from fg splats -> step 3 merge
N_BG_KEEP = 60       # bg grid cluster far from fg, survives
N_SHELL = 50         # sparse ring at 200 m, exempt from isolation prune

IN_COUNT = N_DENSE + N_LOWOP + N_OUTLIER + N_BG_DUP + N_BG_KEEP + N_SHELL
LOG_SCALE = float(np.log(0.05))   # exp(max fg scale)=0.05 -> merge thresh 12.5mm
OP_HI = 3.89182       # logit(0.98)
OP_LO = -6.0          # sigmoid ~ 0.0025 < 0.05 opacity floor


def _grid(n: int, center: np.ndarray, spacing: float) -> np.ndarray:
    side = int(np.ceil(n ** (1.0 / 3.0))) + 1
    ax = np.arange(side, dtype=np.float64) * spacing
    ax -= ax.mean()
    g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1).reshape(-1, 3)
    return (g[:n] + center).astype(np.float32)


def _mk(xyz: np.ndarray, layer: int, opacity_logit: float,
        rgb: tuple[float, float, float]) -> plyio.SplatData:
    n = xyz.shape[0]
    return plyio.SplatData(
        xyz=xyz.astype(np.float32),
        normals=np.zeros((n, 3), np.float32),
        f_dc=np.tile(plyio.rgb01_to_dc(np.array(rgb, np.float32)), (n, 1)),
        opacity_logit=np.full(n, opacity_logit, np.float32),
        log_scales=np.full((n, 3), LOG_SCALE, np.float32),
        quat_wxyz=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        layer=np.full(n, layer, np.uint8),
        origin_stage=np.full(n, 4, np.uint8),
    )


def make_synthetic_splats() -> plyio.SplatData:
    dense_xyz = _grid(N_DENSE, np.array([0.0, 0.0, 2.0]), 0.05)
    dense = _mk(dense_xyz, plyio.LAYER_FG, OP_HI, (0.8, 0.2, 0.2))
    lowop = _mk(dense_xyz[100:100 + N_LOWOP] + np.array([0, 0.001, 0], np.float32),
                plyio.LAYER_FG, OP_LO, (0.8, 0.2, 0.2))
    out_dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                         [0, -1, 0], [0, 0, -1]], np.float32)
    outliers = _mk(out_dirs * 60.0, plyio.LAYER_FG, OP_HI, (0.2, 0.2, 0.8))
    bg_dup = _mk(dense_xyz[:N_BG_DUP] + np.array([0.001, 0, 0], np.float32),
                 plyio.LAYER_BG, OP_HI, (0.2, 0.8, 0.2))
    bg_keep = _mk(_grid(N_BG_KEEP, np.array([0.0, 0.0, -3.0]), 0.05),
                  plyio.LAYER_BG, OP_HI, (0.2, 0.8, 0.2))
    ang = 2 * np.pi * np.arange(N_SHELL, dtype=np.float64) / N_SHELL
    shell_xyz = np.stack(
        [200 * np.sin(ang), np.zeros(N_SHELL), 200 * np.cos(ang)], axis=1)
    shell = _mk(shell_xyz.astype(np.float32), plyio.LAYER_SHELL, OP_HI,
                (0.5, 0.5, 0.5))
    return plyio.SplatData.concat([dense, lowop, outliers, bg_dup, bg_keep, shell])


def make_ctx() -> Ctx:
    return Ctx(
        repo_root=REPO,
        pano_path=REPO / "params.yaml",   # unused by s6 (reads run dir)
        sidecar_path=REPO / "params.yaml",
        params_path=REPO / "params.yaml",
        weights_dir=REPO / "weights",
    )


def make_prune_run(run_dir: Path) -> None:
    (run_dir / "s4_place" / "out").mkdir(parents=True)
    plyio.write_splats(run_dir / "s4_place" / "out" / "splats.ply",
                       make_synthetic_splats())


def run_stage(run_dir: Path, params: dict | None = None) -> dict:
    if params is None:
        params = params_mod.load(REPO / "params.yaml")
    determinism.set_seed(params.get("seed", 0))
    s6_compress.run(run_dir, params, make_ctx())
    return params


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("s6run") / "run"
    make_prune_run(d)
    run_stage(d)
    return d


def out(run_dir: Path) -> Path:
    return run_dir / "s6_compress" / "out"


def compress(run_dir: Path) -> dict:
    return schema.read_validated(out(run_dir) / "compress.json", "compress")


# ------------------------------------------------------------------- tests


def test_outputs_exist_and_schema_valid(run_dir):
    for name in ["scene.ply", "scene_std.ply", "scene.sog", "compress.json"]:
        assert (out(run_dir) / name).exists(), name
    c = compress(run_dir)
    assert c["sog_tool"] == "splat-transform@2.7.1+ziprenorm"


def test_counts_exact_and_monotonic(run_dir):
    c = compress(run_dir)
    assert c["in_count"] == IN_COUNT
    assert c["after_opacity_floor"] == IN_COUNT - N_LOWOP
    assert c["after_isolation_prune"] == IN_COUNT - N_LOWOP - N_OUTLIER
    assert c["after_merge"] == IN_COUNT - N_LOWOP - N_OUTLIER - N_BG_DUP
    assert c["final_count"] == c["after_merge"]
    assert (c["in_count"] >= c["after_opacity_floor"]
            >= c["after_isolation_prune"] >= c["after_merge"]
            >= 0)
    assert c["stride_retries"] == []  # cap not exceeded


def test_low_opacity_dropped(run_dir):
    s = plyio.read_splats(out(run_dir) / "scene.ply")
    floor = params_mod.load(REPO / "params.yaml")["s6"]["opacity_floor"]
    assert (plyio.logit_to_opacity(s.opacity_logit) >= floor).all()


def test_outliers_pruned_shell_kept(run_dir):
    s = plyio.read_splats(out(run_dir) / "scene.ply")
    nonshell = s.layer != plyio.LAYER_SHELL
    # every non-shell survivor is in one of the two clusters (< 4 m)
    assert (np.linalg.norm(s.xyz[nonshell], axis=1) < 4.0).all()
    # the sparse shell is exempt from isolation pruning: all 50 survive
    assert int((s.layer == plyio.LAYER_SHELL).sum()) == N_SHELL


def test_bg_duplicates_merged_keepers_kept(run_dir):
    s = plyio.read_splats(out(run_dir) / "scene.ply")
    assert int((s.layer == plyio.LAYER_FG).sum()) == N_DENSE
    assert int((s.layer == plyio.LAYER_BG).sum()) == N_BG_KEEP
    # no surviving bg splat sits on top of an fg splat
    from scipy.spatial import cKDTree
    fg = s.xyz[s.layer == plyio.LAYER_FG]
    bg = s.xyz[s.layer == plyio.LAYER_BG]
    d, _ = cKDTree(fg).query(bg, k=1, workers=1)
    thresh = 0.25 * float(np.exp(LOG_SCALE))
    assert (d >= thresh).all()


def test_scene_std_ply_standard_layout(run_dir):
    raw = (out(run_dir) / "scene_std.ply").read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii").splitlines()
    props = [ln.split()[2] for ln in header if ln.startswith("property")]
    assert props == s6_compress._STD_FLOAT_PROPS  # no layer/origin_stage
    assert all(ln.split()[1] == "float"
               for ln in header if ln.startswith("property"))
    n = int(next(ln for ln in header if ln.startswith("element vertex")).split()[2])
    c = compress(run_dir)
    assert n == c["final_count"]
    body = np.frombuffer(raw[end:], dtype="<f4").reshape(n, 17)
    s = plyio.read_splats(out(run_dir) / "scene.ply")
    assert np.array_equal(body[:, 0:3], s.xyz)
    assert np.array_equal(body[:, 9], s.opacity_logit)


def test_sog_valid_normalized_zip(run_dir):
    sog = out(run_dir) / "scene.sog"
    assert sog.stat().st_size > 0
    assert zipfile.is_zipfile(sog)
    c = compress(run_dir)
    assert c["sog_bytes"] == sog.stat().st_size
    with zipfile.ZipFile(sog) as z:
        infos = z.infolist()
        assert len(infos) > 0
        assert [i.filename for i in infos] == sorted(i.filename for i in infos)
        for i in infos:
            assert i.date_time == (1980, 1, 1, 0, 0, 0)
            # CPython zipfile force-sets 0o600<<16 for zero attrs; the point
            # is a FIXED value, not the specific bits (see _renorm_zip)
            assert i.external_attr == 0o600 << 16
            assert i.compress_type == zipfile.ZIP_DEFLATED
        assert z.testzip() is None


def test_double_run_byte_identical(run_dir, tmp_path):
    d2 = tmp_path / "run2"
    make_prune_run(d2)
    run_stage(d2)
    for name in ["scene.ply", "scene_std.ply", "scene.sog", "compress.json"]:
        a = (out(run_dir) / name).read_bytes()
        b = (out(d2) / name).read_bytes()
        assert a == b, f"{name} differs across identical runs"


def test_double_conversion_same_ply_identical_sog(run_dir, tmp_path):
    """Step-6 contract: converting the SAME std ply twice yields identical
    bytes after zip normalization."""
    std = out(run_dir) / "scene_std.ply"
    s1, s2 = tmp_path / "a.sog", tmp_path / "b.sog"
    s6_compress.ply_to_sog(std, s1, REPO)
    s6_compress.ply_to_sog(std, s2, REPO)
    assert s1.read_bytes() == s2.read_bytes()
    assert s1.read_bytes() == (out(run_dir) / "scene.sog").read_bytes()


def test_sog_tool_failure_raises_with_stderr(tmp_path):
    bad = tmp_path / "bad.ply"
    bad.write_bytes(b"not a ply at all")
    with pytest.raises(RuntimeError, match="splat-transform failed"):
        s6_compress.ply_to_sog(bad, tmp_path / "bad.sog", REPO)


def test_receipt(run_dir):
    rec = receipts.read_receipt(run_dir, "s6_compress")
    assert set(rec["params_used"]) == {"s6", "splat_cap"}
    assert rec["weights"] == []
    assert rec["gates"] == []
    assert set(rec["outputs"]) == {"scene", "scene_std", "sog", "compress"}
    assert set(rec["inputs"]) == {"splats"}
    for v in list(rec["outputs"].values()) + list(rec["inputs"].values()):
        assert not v["path"].startswith("/")
    n = rec["notes"]["counts"]
    c = compress(run_dir)
    assert n["final_count"] == c["final_count"]
    assert n["in_count"] == c["in_count"]
    assert rec["notes"]["stride_retry_count"] == 0


# ----------------------------------------------------------------- cap test
#
# Builds a 128x64 run dir with s3-style inputs, runs the real s3_layers to
# produce genuine layer artifacts, seeds an initial s4 output, then runs s6
# with a tiny splat_cap so the cap loop re-runs s4 placement with growing
# stride multipliers.

CW, CH = 128, 64          # small equirect (W x H)
C_SKY = 8                 # sky rows
C_SLAB = (48, 80)         # fg slab columns at 2 m over a 10 m background


def _make_s3_inputs(run_dir: Path) -> None:
    depth = np.full((CH, CW), 10.0, np.float32)
    depth[C_SKY:, C_SLAB[0]:C_SLAB[1]] = 2.0
    depth[:C_SKY] = np.inf
    sky = np.zeros((CH, CW), bool)
    sky[:C_SKY] = True
    rgb = np.full((CH, CW, 3), 90, np.uint8)
    rgb[C_SKY:, C_SLAB[0]:C_SLAB[1]] = (200, 60, 60)
    for d in ["s2b_scale/out", "s2_depth/out", "s0_ingest/out"]:
        (run_dir / d).mkdir(parents=True)
    imageio.save_npy(run_dir / "s2b_scale/out/depth_m.npy", depth)
    schema.write_validated(
        run_dir / "s2b_scale/out/scale.json",
        {"scale_factor": 1.0, "camera_height_m": 1.6,
         "scale_source": "ground_plane",
         "plane": {"normal": [0.0, 1.0, 0.0], "d": 1.6},
         "residual_rel": 0.01, "tilt_deg": 0.0},
        "scale",
    )
    imageio.save_mask_png(run_dir / "s2_depth/out/sky_mask.png", sky)
    imageio.save_png(run_dir / "s0_ingest/out/pano.png", rgb)


def _stub_s4_run(run_dir: Path, params: dict, ctx: Ctx,
                 stride_multiplier: float = 1.0) -> None:
    """Minimal stand-in for pipeline.s4_place honoring the documented
    contract: reads s3 layer outputs, places one splat per stride-subsampled
    valid pixel (+ a sparse shell ring), rewrites s4 outputs + receipt.
    Used only when the real module does not exist yet."""
    run_dir = Path(run_dir)
    o = ctx.out(run_dir, "s4_place")
    p4 = params["s4"]
    stride = max(1, int(round(float(p4["base_stride"]) * stride_multiplier)))
    s3 = run_dir / "s3_layers" / "out"
    parts, by_layer = [], {}
    for name, layer, opac in [("fg", plyio.LAYER_FG, p4["fg_opacity"]),
                              ("bg", plyio.LAYER_BG, p4["bg_opacity"])]:
        depth = imageio.load_npy(s3 / f"{name}_depth.npy").astype(np.float64)
        rgb01 = imageio.load_rgb(s3 / f"{name}_rgb.png").astype(np.float64) / 255
        h, w = depth.shape
        dirs = geometry.equirect_dirs(w, h)
        d_s = depth[::stride, ::stride]
        m = np.isfinite(d_s)
        xyz = (dirs[::stride, ::stride][m] * d_s[m][:, None]).astype(np.float32)
        n = xyz.shape[0]
        by_layer[name] = n
        parts.append(plyio.SplatData(
            xyz=xyz,
            normals=np.zeros((n, 3), np.float32),
            f_dc=plyio.rgb01_to_dc(
                rgb01[::stride, ::stride][m].astype(np.float32)),
            opacity_logit=plyio.opacity_to_logit(
                np.full(n, opac, np.float32)),
            log_scales=np.full((n, 3), LOG_SCALE, np.float32),
            quat_wxyz=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
            layer=np.full(n, layer, np.uint8),
            origin_stage=np.full(n, 4, np.uint8),
        ))
    n_shell = 16
    ang = 2 * np.pi * np.arange(n_shell) / n_shell
    r = float(p4["shell_radius_m"])
    shell_xyz = np.stack([r * np.sin(ang), np.zeros(n_shell),
                          r * np.cos(ang)], axis=1).astype(np.float32)
    parts.append(plyio.SplatData(
        xyz=shell_xyz, normals=np.zeros((n_shell, 3), np.float32),
        f_dc=np.zeros((n_shell, 3), np.float32),
        opacity_logit=plyio.opacity_to_logit(
            np.full(n_shell, p4["bg_opacity"], np.float32)),
        log_scales=np.full((n_shell, 3), np.log(10.0), np.float32),
        quat_wxyz=np.tile(np.array([1, 0, 0, 0], np.float32), (n_shell, 1)),
        layer=np.full(n_shell, plyio.LAYER_SHELL, np.uint8),
        origin_stage=np.full(n_shell, 4, np.uint8),
    ))
    by_layer["shell"] = n_shell
    s = plyio.SplatData.concat(parts)
    plyio.write_splats(o / "splats.ply", s)
    schema.write_validated(
        o / "splats_meta.json",
        {"count": len(s), "counts_by_layer": by_layer,
         "stride_multiplier": float(stride_multiplier)},
        "splats_meta",
    )
    receipts.write_receipt(
        run_dir, "s4_place",
        inputs={"fg_depth": s3 / "fg_depth.npy", "bg_depth": s3 / "bg_depth.npy"},
        outputs={"splats": o / "splats.ply",
                 "splats_meta": o / "splats_meta.json"},
        params_used={"s4": p4},
        weights_used=[],
        notes={"stride_multiplier": float(stride_multiplier)},
    )


def _get_s4(monkeypatch):
    """Real pipeline.s4_place when available; otherwise install the stub."""
    if importlib.util.find_spec("pipeline.s4_place") is not None:
        return importlib.import_module("pipeline.s4_place")
    mod = types.ModuleType("pipeline.s4_place")
    mod.run = _stub_s4_run
    monkeypatch.setitem(sys.modules, "pipeline.s4_place", mod)
    return mod


def test_cap_enforcement_stride_retries(tmp_path, monkeypatch):
    from pipeline import s3_layers

    d = tmp_path / "caprun"
    _make_s3_inputs(d)
    params = copy.deepcopy(params_mod.load(REPO / "params.yaml"))
    determinism.set_seed(params.get("seed", 0))
    ctx = make_ctx()
    s3_layers.run(d, params, ctx)

    s4 = _get_s4(monkeypatch)
    s4.run(d, params, ctx, stride_multiplier=1.0)
    initial = len(plyio.read_splats(d / "s4_place" / "out" / "splats.ply"))
    assert initial > 200

    params["splat_cap"] = 200  # tiny cap -> forces stride retries
    s6_compress.run(d, params, ctx)

    c = compress(d)
    retries = c["stride_retries"]
    assert len(retries) > 0
    assert len(retries) <= params["s6"]["max_stride_retries"]
    for i, r in enumerate(retries):
        assert r["multiplier"] == pytest.approx(1.5 ** (i + 1))
        assert set(r) == {"multiplier", "count_before", "count_after"}
    # retries strictly reduce the working set overall
    assert retries[-1]["count_after"] < retries[0]["count_before"]
    assert c["final_count"] == retries[-1]["count_after"]
    # s4 outputs + receipt were rewritten at the last multiplier
    meta = schema.read_validated(
        d / "s4_place" / "out" / "splats_meta.json", "splats_meta")
    assert meta["stride_multiplier"] == pytest.approx(retries[-1]["multiplier"])
    rec = receipts.read_receipt(d, "s6_compress")
    assert rec["notes"]["stride_retry_count"] == len(retries)
    # loop exit condition: under cap, or retry budget exhausted
    assert (c["final_count"] <= params["splat_cap"]
            or len(retries) == params["s6"]["max_stride_retries"])
