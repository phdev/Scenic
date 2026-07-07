"""s6_compress tests (two-profile: review + quest).

Prune fixture: a synthetic s4 output (~2k splats) with exactly-known groups —
a dense fg grid cluster, low-opacity splats, isolated outliers, bg duplicates
sitting on fg positions, a separate surviving bg cluster, and a sparse shell
ring — so every prune/merge count is closed-form. Both profiles have caps far
above the survivor count, so neither retries: review and quest are identical
and the primary (quest) aliases are byte copies.

Cap test: a fake run dir is seeded with a small synthetic splats.ply and s4 is
STUBBED (monkeypatched into sys.modules) — the real pipeline.s4_place is being
rewritten in parallel and must never be invoked here. The stub honors the
documented `run(run_dir, params, ctx, stride_multiplier=)` contract and thins
its dense grid as the multiplier grows, so s6's per-profile cap loop is
exercised deterministically.
"""
from __future__ import annotations

import copy
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pipeline import s6_compress
from scenic import determinism, params as params_mod
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
SURVIVORS = IN_COUNT - N_LOWOP - N_OUTLIER - N_BG_DUP
LOG_SCALE = float(np.log(0.05))   # exp(max fg scale)=0.05 -> merge thresh 12.5mm
OP_HI = 3.89182       # logit(0.98)
OP_LO = -6.0          # sigmoid ~ 0.0025 < 0.05 opacity floor

PROFILES = ("review", "quest")
ALIAS_FILES = ["scene.ply", "scene_std.ply", "scene.sog"]
DBL_FILES = [  # byte-determinism set: per-profile ply + sog + compress.json
    "scene_review.ply", "scene_review_std.ply", "scene_review.sog",
    "scene_quest.ply", "scene_quest_std.ply", "scene_quest.sog",
    "compress.json",
]


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


# ------------------------------------------------------------- prune fixture


def test_outputs_exist_and_schema_valid(run_dir):
    names = ["compress.json"] + ALIAS_FILES + [
        f"scene_{p}{suf}" for p in PROFILES
        for suf in (".ply", "_std.ply", ".sog")
    ]
    for name in names:
        assert (out(run_dir) / name).exists(), name
    c = compress(run_dir)
    assert c["sog_tool"] == "splat-transform@2.7.1+ziprenorm"
    assert c["primary_profile"] == "quest"
    assert c["viewer_profile"] == "review"
    assert set(c["profiles"]) == set(PROFILES)


def test_counts_exact_and_monotonic(run_dir):
    c = compress(run_dir)
    for name in PROFILES:
        p = c["profiles"][name]
        assert p["in_count"] == IN_COUNT
        assert p["after_opacity_floor"] == IN_COUNT - N_LOWOP
        assert p["after_isolation_prune"] == IN_COUNT - N_LOWOP - N_OUTLIER
        assert p["after_merge"] == SURVIVORS
        assert p["final_count"] == SURVIVORS
        assert (p["in_count"] >= p["after_opacity_floor"]
                >= p["after_isolation_prune"] >= p["after_merge"] >= 0)
        assert p["stride_retries"] == []          # caps not exceeded
        assert p["ply_bytes"] > 0 and p["sog_bytes"] > 0
    # review is the inspection profile: its cap is >= the quest ship cap
    assert c["profiles"]["review"]["cap"] >= c["profiles"]["quest"]["cap"]
    assert c["profiles"]["review"]["target"] >= c["profiles"]["quest"]["target"]


def test_primary_aliases_byte_equal_quest(run_dir):
    q = {".ply": "scene_quest.ply", "_std.ply": "scene_quest_std.ply",
         ".sog": "scene_quest.sog"}
    for alias, quest in [("scene.ply", q[".ply"]),
                         ("scene_std.ply", q["_std.ply"]),
                         ("scene.sog", q[".sog"])]:
        a = (out(run_dir) / alias).read_bytes()
        b = (out(run_dir) / quest).read_bytes()
        assert a == b, f"{alias} not a byte copy of {quest}"


def test_review_and_quest_identical_when_no_retry(run_dir):
    # both profiles start from the same s4 output and neither coarsens, so the
    # per-profile PLYs are byte-identical.
    for suf in (".ply", "_std.ply", ".sog"):
        assert (out(run_dir) / f"scene_review{suf}").read_bytes() == \
               (out(run_dir) / f"scene_quest{suf}").read_bytes()


def test_low_opacity_dropped(run_dir):
    s = plyio.read_splats(out(run_dir) / "scene.ply")
    floor = params_mod.load(REPO / "params.yaml")["s6"]["opacity_floor"]
    assert (plyio.logit_to_opacity(s.opacity_logit) >= floor).all()


def test_outliers_pruned_shell_kept(run_dir):
    s = plyio.read_splats(out(run_dir) / "scene.ply")
    nonshell = s.layer != plyio.LAYER_SHELL
    assert (np.linalg.norm(s.xyz[nonshell], axis=1) < 4.0).all()
    assert int((s.layer == plyio.LAYER_SHELL).sum()) == N_SHELL


def test_bg_duplicates_merged_keepers_kept(run_dir):
    s = plyio.read_splats(out(run_dir) / "scene.ply")
    assert int((s.layer == plyio.LAYER_FG).sum()) == N_DENSE
    assert int((s.layer == plyio.LAYER_BG).sum()) == N_BG_KEEP
    from scipy.spatial import cKDTree
    fg = s.xyz[s.layer == plyio.LAYER_FG]
    bg = s.xyz[s.layer == plyio.LAYER_BG]
    d, _ = cKDTree(fg).query(bg, k=1, workers=1)
    thresh = 0.25 * float(np.exp(LOG_SCALE))
    assert (d >= thresh).all()


@pytest.mark.parametrize("std_name", ["scene_std.ply", "scene_quest_std.ply",
                                      "scene_review_std.ply"])
def test_scene_std_ply_standard_layout(run_dir, std_name):
    raw = (out(run_dir) / std_name).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii").splitlines()
    props = [ln.split()[2] for ln in header if ln.startswith("property")]
    assert props == s6_compress._STD_FLOAT_PROPS  # no layer/origin_stage
    assert all(ln.split()[1] == "float"
               for ln in header if ln.startswith("property"))
    n = int(next(ln for ln in header
                 if ln.startswith("element vertex")).split()[2])
    assert n == SURVIVORS
    body = np.frombuffer(raw[end:], dtype="<f4").reshape(n, 17)
    s = plyio.read_splats(out(run_dir) / "scene.ply")
    assert np.array_equal(body[:, 0:3], s.xyz)
    assert np.array_equal(body[:, 9], s.opacity_logit)


@pytest.mark.parametrize("sog_name", ["scene.sog", "scene_review.sog",
                                      "scene_quest.sog"])
def test_sog_valid_normalized_zip(run_dir, sog_name):
    sog = out(run_dir) / sog_name
    assert sog.stat().st_size > 0
    assert zipfile.is_zipfile(sog)
    with zipfile.ZipFile(sog) as z:
        infos = z.infolist()
        assert len(infos) > 0
        assert [i.filename for i in infos] == sorted(i.filename for i in infos)
        for i in infos:
            assert i.date_time == (1980, 1, 1, 0, 0, 0)
            assert i.external_attr == 0o600 << 16
            assert i.compress_type == zipfile.ZIP_DEFLATED
        assert z.testzip() is None


def test_compress_sog_bytes_match_files(run_dir):
    c = compress(run_dir)
    for name in PROFILES:
        assert c["profiles"][name]["sog_bytes"] == \
            (out(run_dir) / f"scene_{name}.sog").stat().st_size
        assert c["profiles"][name]["ply_bytes"] == \
            (out(run_dir) / f"scene_{name}.ply").stat().st_size


def test_double_run_byte_identical(run_dir, tmp_path):
    d2 = tmp_path / "run2"
    make_prune_run(d2)
    run_stage(d2)
    for name in DBL_FILES + ALIAS_FILES:
        a = (out(run_dir) / name).read_bytes()
        b = (out(d2) / name).read_bytes()
        assert a == b, f"{name} differs across identical runs"


def test_double_conversion_same_ply_identical_sog(run_dir, tmp_path):
    std = out(run_dir) / "scene_quest_std.ply"
    s1, s2 = tmp_path / "a.sog", tmp_path / "b.sog"
    s6_compress.ply_to_sog(std, s1, REPO)
    s6_compress.ply_to_sog(std, s2, REPO)
    assert s1.read_bytes() == s2.read_bytes()
    assert s1.read_bytes() == (out(run_dir) / "scene_quest.sog").read_bytes()
    assert s1.read_bytes() == (out(run_dir) / "scene.sog").read_bytes()


def test_sog_tool_failure_raises_with_stderr(tmp_path):
    bad = tmp_path / "bad.ply"
    bad.write_bytes(b"not a ply at all")
    with pytest.raises(RuntimeError, match="splat-transform failed"):
        s6_compress.ply_to_sog(bad, tmp_path / "bad.sog", REPO)


def test_receipt(run_dir):
    rec = receipts.read_receipt(run_dir, "s6_compress")
    assert set(rec["params_used"]) == {"s6"}
    assert rec["weights"] == []
    assert rec["gates"] == []
    assert set(rec["inputs"]) == {"splats"}
    expect_outputs = {"scene", "scene_std", "sog", "compress"}
    expect_outputs |= {f"scene_{p}" for p in PROFILES}
    expect_outputs |= {f"scene_{p}_std" for p in PROFILES}
    expect_outputs |= {f"scene_{p}_sog" for p in PROFILES}
    assert set(rec["outputs"]) == expect_outputs
    for v in list(rec["outputs"].values()) + list(rec["inputs"].values()):
        assert not v["path"].startswith("/")
    c = compress(run_dir)
    assert set(rec["notes"]["profiles"]) == set(PROFILES)
    for name in PROFILES:
        n = rec["notes"]["profiles"][name]
        assert n["final_count"] == c["profiles"][name]["final_count"]
        assert n["sog_bytes"] == c["profiles"][name]["sog_bytes"]
        assert n["stride_retry_count"] == 0
    assert rec["notes"]["primary_profile"] == "quest"
    assert rec["notes"]["viewer_profile"] == "review"


# ----------------------------------------------------------------- cap test
#
# A fake run dir seeded with a small dense splats.ply; s4 is STUBBED (never the
# real module). The stub thins a 1000-point dense grid by a stride that grows
# with the multiplier, so a tight quest cap forces the per-profile cap loop.

N_BASE = 1000        # 10^3 dense grid, spacing 0.02 (isolation prune never fires)
N_STUB_SHELL = 16


def _stub_base_grid() -> np.ndarray:
    side = int(round(N_BASE ** (1.0 / 3.0)))   # 10
    ax = np.arange(side, dtype=np.float64) * 0.02
    ax -= ax.mean()
    g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1).reshape(-1, 3)
    g[:, 2] += 2.0
    return g


def _stub_s4_run(run_dir: Path, params: dict, ctx: Ctx,
                 stride_multiplier: float = 1.0) -> None:
    """Stand-in for pipeline.s4_place honoring
    run(run_dir, params, ctx, stride_multiplier=). Thins a dense grid by
    stride = round(base_stride * multiplier); count shrinks as the multiplier
    grows. Writes splats.ply + splats_meta.json + a receipt (rewrites s4)."""
    run_dir = Path(run_dir)
    o = ctx.out(run_dir, "s4_place")
    stride = max(1, int(round(float(params["s4"]["base_stride"])
                              * float(stride_multiplier))))
    base = _stub_base_grid()[::stride]
    n = base.shape[0]
    fg = _mk(base.astype(np.float32), plyio.LAYER_FG, OP_HI, (0.8, 0.2, 0.2))
    ang = 2 * np.pi * np.arange(N_STUB_SHELL, dtype=np.float64) / N_STUB_SHELL
    shell_xyz = np.stack([200 * np.sin(ang), np.zeros(N_STUB_SHELL),
                          200 * np.cos(ang)], axis=1)
    shell = _mk(shell_xyz.astype(np.float32), plyio.LAYER_SHELL, OP_HI,
                (0.5, 0.5, 0.5))
    s = plyio.SplatData.concat([fg, shell])
    plyio.write_splats(o / "splats.ply", s)
    schema.write_validated(
        o / "splats_meta.json",
        {"count": len(s),
         "counts_by_layer": {"fg": n, "bg": 0, "shell": N_STUB_SHELL},
         "stride_multiplier": float(stride_multiplier)},
        "splats_meta",
    )
    receipts.write_receipt(
        run_dir, "s4_place",
        inputs={},
        outputs={"splats": o / "splats.ply",
                 "splats_meta": o / "splats_meta.json"},
        params_used={"s4": params["s4"]},
        weights_used=[],
        notes={"stride_multiplier": float(stride_multiplier)},
    )


def _install_stub_s4(monkeypatch) -> None:
    mod = types.ModuleType("pipeline.s4_place")
    mod.run = _stub_s4_run
    monkeypatch.setitem(sys.modules, "pipeline.s4_place", mod)


def _cap_params() -> dict:
    params = copy.deepcopy(params_mod.load(REPO / "params.yaml"))
    params["s6"]["profiles"]["review"]["cap"] = 600   # >= seeded survivors
    params["s6"]["profiles"]["review"]["target"] = 500
    params["s6"]["profiles"]["quest"]["cap"] = 200     # forces stride retries
    params["s6"]["profiles"]["quest"]["target"] = 150
    return params


def test_cap_enforcement_stride_retries(tmp_path, monkeypatch):
    _install_stub_s4(monkeypatch)
    d = tmp_path / "caprun"
    (d / "s4_place" / "out").mkdir(parents=True)
    ctx = make_ctx()
    params = _cap_params()
    determinism.set_seed(params.get("seed", 0))

    _stub_s4_run(d, params, ctx, stride_multiplier=1.0)   # seed splats.ply
    initial = len(plyio.read_splats(d / "s4_place" / "out" / "splats.ply"))
    assert initial > 400

    s6_compress.run(d, params, ctx)
    c = compress(d)

    # review's cap (600) accommodates the seeded survivors -> no retries.
    rev = c["profiles"]["review"]
    assert rev["stride_retries"] == []
    assert rev["final_count"] <= params["s6"]["profiles"]["review"]["cap"]

    # quest's tight cap (200) forces retries.
    q = c["profiles"]["quest"]
    retries = q["stride_retries"]
    assert len(retries) > 0
    assert len(retries) <= params["s6"]["max_stride_retries"]
    for i, r in enumerate(retries):
        assert r["multiplier"] == pytest.approx(1.5 ** (i + 1))
        assert set(r) == {"multiplier", "count_before", "count_after"}
    # counts strictly shrink across the retry chain
    assert retries[-1]["count_after"] < retries[0]["count_before"]
    assert q["final_count"] == retries[-1]["count_after"]
    assert (q["final_count"] <= params["s6"]["profiles"]["quest"]["cap"]
            or len(retries) == params["s6"]["max_stride_retries"])

    # review's inspection cap is at least the quest ship cap.
    assert rev["cap"] >= q["cap"]

    # s4's final on-disk placement is the LAST (quest) coarsening.
    meta = schema.read_validated(
        d / "s4_place" / "out" / "splats_meta.json", "splats_meta")
    assert meta["stride_multiplier"] == pytest.approx(retries[-1]["multiplier"])

    # primary aliases are byte copies of the quest files.
    for alias, quest in [("scene.ply", "scene_quest.ply"),
                         ("scene_std.ply", "scene_quest_std.ply"),
                         ("scene.sog", "scene_quest.sog")]:
        assert (out(d) / alias).read_bytes() == (out(d) / quest).read_bytes()

    rec = receipts.read_receipt(d, "s6_compress")
    assert rec["notes"]["profiles"]["quest"]["stride_retry_count"] == len(retries)
    assert rec["notes"]["profiles"]["review"]["stride_retry_count"] == 0

    # sog files are valid normalized zips.
    for name in ("scene_review.sog", "scene_quest.sog", "scene.sog"):
        with zipfile.ZipFile(out(d) / name) as z:
            assert z.testzip() is None
            assert [i.filename for i in z.infolist()] == \
                sorted(i.filename for i in z.infolist())


def test_cap_double_run_byte_identical(tmp_path, monkeypatch):
    _install_stub_s4(monkeypatch)
    ctx = make_ctx()
    outs = []
    for tag in ("a", "b"):
        d = tmp_path / tag
        (d / "s4_place" / "out").mkdir(parents=True)
        params = _cap_params()
        determinism.set_seed(params.get("seed", 0))
        _stub_s4_run(d, params, ctx, stride_multiplier=1.0)
        s6_compress.run(d, params, ctx)
        outs.append(out(d))
    for name in DBL_FILES + ALIAS_FILES:
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes(), name
