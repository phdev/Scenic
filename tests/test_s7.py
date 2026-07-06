"""s7_gates + gates/* tests.

Synthetic scenes built directly as SplatData (no upstream stages):

- PASS scene: a closed lat-long sphere shell (r=10, mildly checkered gray)
  enclosing a ground disk at y=-1.6, with a LAYER_SHELL sphere at r=30
  hidden behind the content. Every gate must pass.
- Wedge scenes: the same scene minus all content splats in a solid angle
  below the horizon (lon 70..110 deg, pitch -35..-8 deg). With the shell
  present the hole gate must fail via magenta; without it, via alpha —
  falsifiability both ways.
- Near scene: PASS scene + one opaque splat 0.3 m ahead -> stereo near-limit
  fails.
- Budgets: fake s6 dir (scene.ply + compress.json + scene.sog stub bytes);
  pass at real params, fail under a monkeypatched cap / sog limit.

render_px is reduced via params override so CPU renders stay fast.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from gates import GATE_ORDER, budgets, hole, jitter, stereo
from pipeline import s7_gates
from scenic import determinism, params as params_mod, plyio, receipts, schema
from scenic.stage import Ctx

REPO = Path(__file__).resolve().parent.parent
SOG_STUB = b"not-a-real-sog" * 37  # 518 bytes; only the size matters here


# ------------------------------------------------------------ scene builders


def _splats(xyz, rgb01, layer, opacity=0.98, scale=0.45) -> plyio.SplatData:
    xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
    n = xyz.shape[0]
    col = np.asarray(rgb01, np.float32)
    if col.ndim == 1:
        col = np.tile(col, (n, 1))
    return plyio.SplatData(
        xyz=xyz,
        normals=np.zeros((n, 3), np.float32),
        f_dc=plyio.rgb01_to_dc(col).astype(np.float32),
        opacity_logit=plyio.opacity_to_logit(
            np.full(n, opacity, np.float32)
        ).astype(np.float32),
        log_scales=np.full((n, 3), np.log(scale), np.float32),
        quat_wxyz=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        layer=np.full(n, layer, np.uint8),
        origin_stage=np.full(n, 6, np.uint8),
    )


def _sphere_pts(r: float, nt: int, npi: int) -> tuple[np.ndarray, np.ndarray]:
    """Lat-long point sphere + checker shade (0.40/0.55 gray levels)."""
    th = (np.arange(nt) + 0.5) / nt * 2 * np.pi - np.pi
    ph = (np.arange(npi) + 0.5) / npi * np.pi - np.pi / 2
    tt, pp = np.meshgrid(th, ph)
    d = np.stack(
        [np.cos(pp) * np.sin(tt), np.sin(pp), np.cos(pp) * np.cos(tt)], axis=-1
    ).reshape(-1, 3)
    checker = (np.add.outer(np.arange(npi), np.arange(nt)) % 2).reshape(-1)
    return d * r, 0.40 + 0.15 * checker.astype(np.float64)


def _ground_pts(spacing=0.35, y=-1.6, radius=10.0):
    g = np.arange(-radius, radius + 1e-9, spacing)
    gx, gz = np.meshgrid(g, g)
    m = gx * gx + gz * gz <= radius * radius
    ix = np.rint(gx / spacing).astype(int) + np.rint(gz / spacing).astype(int)
    shade = 0.35 + 0.2 * ((ix % 2) == 0)
    pts = np.stack([gx[m], np.full(int(m.sum()), y), gz[m]], axis=-1)
    return pts, shade[m]


# The wedge: a solid angle below the horizon, centered on lon 90 (so it sits
# dead-center in the yaw-090 views).
WEDGE_LON = (70.0, 110.0)
WEDGE_PITCH = (-35.0, -8.0)


def _cut_wedge(xyz: np.ndarray, shade: np.ndarray):
    """Drop points whose direction from the origin falls in the wedge."""
    r = np.linalg.norm(xyz, axis=1)
    lon = np.degrees(np.arctan2(xyz[:, 0], xyz[:, 2]))
    pitch = np.degrees(np.arcsin(xyz[:, 1] / np.maximum(r, 1e-9)))
    cut = (
        (lon > WEDGE_LON[0]) & (lon < WEDGE_LON[1])
        & (pitch > WEDGE_PITCH[0]) & (pitch < WEDGE_PITCH[1])
    )
    return xyz[~cut], shade[~cut]


def make_scene(
    wedge=False, shell=True, near_splat=False, nt=100, npi=50,
    ground_spacing=0.35,
) -> plyio.SplatData:
    sp, s_shade = _sphere_pts(10.0, nt, npi)
    gp, g_shade = _ground_pts(ground_spacing)
    if wedge:
        sp, s_shade = _cut_wedge(sp, s_shade)
        gp, g_shade = _cut_wedge(gp, g_shade)
    # Ground scale tracks its grid spacing: the ground passes ~1.6 m under
    # the camera, and oversized splats there have grazing EWA footprints
    # that smear a false "blanket" over the lower image rows.
    parts = [
        _splats(sp, np.stack([s_shade] * 3, axis=1), plyio.LAYER_BG,
                scale=0.45),
        _splats(gp, np.stack([g_shade] * 3, axis=1), plyio.LAYER_BG,
                scale=0.5 * ground_spacing),
    ]
    if shell:
        shp, _ = _sphere_pts(30.0, 60, 30)
        parts.append(_splats(shp, (0.5, 0.5, 0.5), plyio.LAYER_SHELL,
                             scale=2.5))
    if near_splat:
        parts.append(
            _splats([[0.0, 0.0, 0.3]], (0.5, 0.5, 0.5), plyio.LAYER_FG,
                    opacity=0.99, scale=0.08)
        )
    return plyio.SplatData.concat(parts)


# --------------------------------------------------------------- run helpers


def stage_params(px: int = 96, **s7_over) -> dict:
    p = copy.deepcopy(params_mod.load(REPO / "params.yaml"))
    p["s7"]["render_px"] = px  # keep CPU renders fast in tests
    p["s7"].update(s7_over)
    return p


def make_ctx() -> Ctx:
    return Ctx(
        repo_root=REPO,
        pano_path=REPO / "params.yaml",   # unused by s7 (reads the run dir)
        sidecar_path=REPO / "params.yaml",
        params_path=REPO / "params.yaml",
        weights_dir=REPO / "weights",
    )


def make_s6_dir(run_dir: Path, scene: plyio.SplatData,
                sog: bytes = SOG_STUB) -> None:
    out = run_dir / "s6_compress" / "out"
    out.mkdir(parents=True)
    plyio.write_splats(out / "scene.ply", scene)
    (out / "scene.sog").write_bytes(sog)
    n = len(scene)
    schema.write_validated(
        out / "compress.json",
        {
            "in_count": n, "after_opacity_floor": n,
            "after_isolation_prune": n, "after_merge": n, "final_count": n,
            "stride_retries": [], "sog_bytes": len(sog),
            "sog_tool": "test-stub",
        },
        "compress",
    )


@pytest.fixture(scope="module")
def stage_run(tmp_path_factory):
    """One full s7_gates run over the closed PASS scene."""
    d = tmp_path_factory.mktemp("s7run") / "run"
    scene = make_scene()
    make_s6_dir(d, scene)
    params = stage_params()
    determinism.set_seed(params.get("seed", 0))
    s7_gates.run(d, params, make_ctx())
    return d, params, scene


def verdict(run_dir: Path, gate: str) -> dict:
    return schema.read_validated(
        run_dir / "s7_gates" / "out" / "verdicts" / f"{gate}.json",
        "gate_verdict",
    )


# ------------------------------------------------------- stage-level checks


def test_stage_verdicts_exist_and_schema_valid(stage_run):
    d, _, _ = stage_run
    for g in GATE_ORDER:
        v = verdict(d, g)  # read_validated => schema gate_verdict enforced
        assert v["gate"] == g
        assert isinstance(v["pass"], bool)


def test_stage_representative_renders(stage_run):
    d, _, _ = stage_run
    rdir = d / "s7_gates" / "out" / "renders"
    for yaw in ("000", "090", "180", "270"):
        assert (rdir / f"center_yaw{yaw}_magenta.png").exists()
        assert (rdir / f"center_yaw{yaw}_normal.png").exists()
    for extra in ("jitter_base.png", "jitter_offset.png",
                  "stereo_yaw000_left.png", "stereo_yaw000_right.png",
                  "hole_worst_magenta.png"):
        assert (rdir / extra).exists()


def test_stage_receipt(stage_run):
    d, params, _ = stage_run
    rec = receipts.read_receipt(d, "s7_gates")
    assert [g["gate"] for g in rec["gates"]] == list(GATE_ORDER)
    for g in rec["gates"]:
        schema.validate(g, "gate_verdict")
    assert [w["key"] for w in rec["weights"]] == ["rtdetr_r18"]
    assert set(rec["params_used"]) == {
        "head_box", "s7", "splat_cap", "splat_target", "sog_max_mb"
    }
    assert rec["params_used"]["s7"] == params["s7"]
    assert set(rec["inputs"]) == {"scene", "compress"}
    assert set(rec["outputs"]) == {f"verdict_{g}" for g in GATE_ORDER}
    for v in list(rec["inputs"].values()) + list(rec["outputs"].values()):
        assert not v["path"].startswith("/")
    rdir = d / "s7_gates" / "out" / "renders"
    assert rec["notes"]["renders"] == sorted(
        p.name for p in rdir.glob("*.png"))
    assert len(rec["notes"]["renders"]) > 0
    assert rec["notes"]["all_pass"] is True
    # receipt gates match the on-disk verdict files
    for g, emb in zip(GATE_ORDER, rec["gates"]):
        assert emb == verdict(d, g)


# ------------------------------------------------------------------- hole


def test_hole_passes_closed_scene(stage_run):
    d, params, _ = stage_run
    v = verdict(d, "hole")
    assert v["pass"] is True
    m = v["metrics"]
    assert m["worst_magenta_below_skyline_frac"] <= params["s7"]["hole_max_frac"]
    assert m["worst_alpha_below_skyline_frac"] <= params["s7"]["hole_max_frac"]
    assert m["worst_blob_px"] <= params["s7"]["hole_blob_max_px"]
    assert m["n_views"] == 28
    assert len(v["details"]["per_view"]) == 28


def test_hole_fails_on_wedge_magenta(tmp_path):
    """Falsifiability: deleting a below-horizon wedge exposes the magenta
    shell and the gate must fail on the frac rule."""
    scene = make_scene(wedge=True, shell=True)
    params = stage_params()
    v = hole.run_gate(scene, params, tmp_path)
    schema.validate(v, "gate_verdict")
    assert v["pass"] is False
    assert (v["metrics"]["worst_magenta_below_skyline_frac"]
            > params["s7"]["hole_max_frac"])
    # the wedge is centered on lon 90: the center-pose yaw-090 view sees it
    per = {pv["view"]: pv for pv in v["details"]["per_view"]}
    assert per["center_yaw090"]["magenta_frac"] > params["s7"]["hole_max_frac"]
    # ... and views facing away from it stay clean
    assert per["center_yaw270"]["magenta_frac"] <= params["s7"]["hole_max_frac"]
    assert (tmp_path / "renders" / "hole_worst_magenta.png").exists()


def test_hole_blob_rule_fails_independently(tmp_path):
    """Disable the frac rule; a small central blob budget must still fail on
    the wedge (it sits in the central window of the yaw-090 views)."""
    scene = make_scene(wedge=True, shell=True)
    params = stage_params(hole_max_frac=1.0, hole_blob_max_px=50)
    v = hole.run_gate(scene, params, tmp_path)
    assert v["pass"] is False
    assert v["metrics"]["worst_blob_px"] > 50


def test_hole_fails_on_wedge_alpha_without_shell(tmp_path):
    """Without a shell the wedge is a true void: alpha<0.05 pixels must fail
    the gate via the separate alpha metric."""
    scene = make_scene(wedge=True, shell=False)
    params = stage_params()
    v = hole.run_gate(scene, params, tmp_path)
    assert v["pass"] is False
    assert (v["metrics"]["worst_alpha_below_skyline_frac"]
            > params["s7"]["hole_max_frac"])
    # no shell anywhere -> no magenta at all
    assert v["metrics"]["worst_magenta_below_skyline_frac"] == 0.0


# ------------------------------------------------------------------ jitter


def test_jitter_passes_static_scene(stage_run):
    d, params, _ = stage_run
    v = verdict(d, "jitter")
    assert v["pass"] is True
    assert 0.0 <= v["metrics"]["energy"] <= params["s7"]["jitter_energy_max"]
    assert v["thresholds"]["jitter_offset_m"] == params["s7"]["jitter_offset_m"]


def test_jitter_threshold_falsifiable(tmp_path):
    """The comparison is live: an impossible threshold must fail."""
    scene = make_scene(nt=24, npi=12, ground_spacing=1.0)
    params = stage_params(px=64, jitter_energy_max=-1.0)
    v = jitter.run_gate(scene, params, tmp_path)
    schema.validate(v, "gate_verdict")
    assert v["pass"] is False
    assert v["metrics"]["energy"] >= 0.0


def test_jitter_deterministic(tmp_path):
    scene = make_scene(nt=24, npi=12, ground_spacing=1.0)
    params = stage_params(px=64)
    a = jitter.run_gate(scene, params, tmp_path / "a")
    b = jitter.run_gate(scene, params, tmp_path / "b")
    assert a == b


# ------------------------------------------------------------------ stereo


def test_stereo_passes(stage_run):
    d, params, _ = stage_run
    v = verdict(d, "stereo")
    assert v["pass"] is True
    m = v["metrics"]
    assert m["vdisp_max_px"] <= params["s7"]["stereo_vdisp_max_px"]
    assert m["min_depth_m"] >= params["s7"]["stereo_near_depth_min_m"]
    assert m["order_min_frac"] >= params["s7"]["stereo_order_min_frac"]
    assert len(v["details"]["per_yaw"]) == 4
    # the scene's nearest central content is the ground several meters out
    assert m["min_depth_m"] > 2.0


def test_stereo_near_limit_fails(tmp_path):
    """A splat 0.3 m ahead violates the near limit (min 0.75 m)."""
    scene = make_scene(near_splat=True)
    params = stage_params()
    v = stereo.run_gate(scene, params, tmp_path)
    schema.validate(v, "gate_verdict")
    assert v["pass"] is False
    assert v["metrics"]["min_depth_m"] < params["s7"]["stereo_near_depth_min_m"]
    assert (tmp_path / "renders" / "stereo_yaw000_left.png").exists()


# ------------------------------------------------------------------ people


def test_people_passes(stage_run):
    d, params, _ = stage_run
    v = verdict(d, "people")
    assert v["pass"] is True
    assert v["metrics"]["n_detections"] == 0
    assert 0.0 <= v["metrics"]["max_score"] < params["s7"]["people_score_min"]
    assert len(v["details"]["per_view"]) == 28


# ----------------------------------------------------------------- budgets


def test_budgets_passes(stage_run):
    d, params, scene = stage_run
    v = verdict(d, "budgets")
    assert v["pass"] is True
    m = v["metrics"]
    assert m["final_count"] == len(scene)
    assert m["sog_bytes"] == len(SOG_STUB)
    assert m["count_vs_target_ratio"] == pytest.approx(
        len(scene) / params["splat_target"])
    assert v["thresholds"]["splat_cap"] == params["splat_cap"]
    assert v["thresholds"]["sog_max_bytes"] == params["sog_max_mb"] * 1024 * 1024
    assert v["details"]["count_matches_compress"] is True
    assert v["details"]["sog_bytes_matches_compress"] is True


def test_budgets_default_run_dir_derivation(stage_run):
    """run_gate(splats, params, outdir) alone must find the s6 artifacts via
    the standard <run>/s7_gates/out layout."""
    d, params, _ = stage_run
    splats = plyio.read_splats(d / "s6_compress" / "out" / "scene.ply")
    v = budgets.run_gate(splats, params, d / "s7_gates" / "out")
    assert v == verdict(d, "budgets")


def test_budgets_cap_violation_fails(stage_run):
    d, params, scene = stage_run
    splats = plyio.read_splats(d / "s6_compress" / "out" / "scene.ply")
    p = copy.deepcopy(params)
    p["splat_cap"] = 10  # monkeypatched cap far below the scene count
    v = budgets.run_gate(splats, p, d / "s7_gates" / "out", run_dir=d)
    schema.validate(v, "gate_verdict")
    assert v["pass"] is False
    assert v["metrics"]["final_count"] == len(scene) > 10


def test_budgets_sog_violation_fails(stage_run):
    d, params, _ = stage_run
    splats = plyio.read_splats(d / "s6_compress" / "out" / "scene.ply")
    p = copy.deepcopy(params)
    p["sog_max_mb"] = 0  # 0 MiB budget: any non-empty .sog fails
    v = budgets.run_gate(splats, p, d / "s7_gates" / "out", run_dir=d)
    assert v["pass"] is False
    assert v["metrics"]["sog_bytes"] > v["thresholds"]["sog_max_bytes"]


# ------------------------------------------------------ stage error handling


def test_stage_missing_s6_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        s7_gates.run(tmp_path / "empty_run", stage_params(), make_ctx())


# ---------------------------------------------------------------- determinism


def test_stage_double_run_byte_identical(tmp_path):
    """Same scene bytes + params => byte-identical verdicts, renders and
    receipt across two independent stage runs (tiny scene to keep the 2x28
    detector calls cheap)."""
    scene = make_scene(nt=24, npi=12, ground_spacing=1.0)
    params = stage_params(px=64)
    dirs = [tmp_path / "run_a", tmp_path / "run_b"]
    for d in dirs:
        make_s6_dir(d, scene)
        determinism.set_seed(params.get("seed", 0))
        s7_gates.run(d, params, make_ctx())

    def files(d: Path) -> list[Path]:
        stage = d / "s7_gates"
        out = [stage / "receipt.json"]
        out += sorted((stage / "out" / "verdicts").glob("*.json"))
        out += sorted((stage / "out" / "renders").glob("*.png"))
        return out

    fa, fb = files(dirs[0]), files(dirs[1])
    assert [p.name for p in fa] == [p.name for p in fb]
    assert len(fa) >= 1 + 5 + 8
    for a, b in zip(fa, fb):
        assert a.read_bytes() == b.read_bytes(), f"{a.name} differs"
