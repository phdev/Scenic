"""s7_gates + gates/* tests (v2 quality pass).

Synthetic scenes built directly as SplatData (no upstream stages):

- PASS scene: a closed lat-long sphere shell (r=10, mildly checkered gray)
  enclosing a ground disk at y=-1.6 (genuine nadir coverage for the
  straight-down views), with a LAYER_SHELL sphere at r=30 hidden behind the
  content. All six gates pass (fidelity thresholds are relaxed to -1 in
  tests since the synthetic pano is not the synthetic scene).
- Wedge scenes: the same scene minus all content splats in a solid angle
  below the horizon. With the shell present the hole gate must fail via
  magenta; without it, via alpha — falsifiability both ways.
- Nadir-cut scene: PASS scene minus everything below pitch -50 deg (no
  shell behind) — invisible to the pitch-0 view ring, must fail via the
  down view (the nadir-blind-cone regression).
- Near scenes: PASS scene + one opaque splat 0.3 m away, dead ahead or at
  azimuth 45 deg (the near-limit-blind-wedge regression) -> stereo
  near-limit fails.
- Budgets: fake s6 dir (scene.ply + two-profile compress.json + scene.sog
  stub); pass at real params, fail under a monkeypatched cap / sog limit;
  reads the QUEST profile.
- Fidelity: a fake run dir with s1_cleanplate/out/pano_clean.png (REQUIRED —
  no s0 fallback); SSIM per tile, advisory LPIPS unavailable, falsifiable
  via an impossible threshold.

render_px is reduced via params override so CPU renders stay fast.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from gates import GATE_ORDER, LAYER_ITEMS, budgets, fidelity, hole, jitter, stereo
from pipeline import s7_gates
from scenic import determinism, imageio, params as params_mod, plyio, receipts, schema
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


def _cut_below_pitch(splats: plyio.SplatData, pitch_deg: float) -> plyio.SplatData:
    """Drop every splat whose direction from the origin sits below pitch_deg
    (the nadir cone — invisible to the pitch-0 fov-90 view ring)."""
    xyz = splats.xyz.astype(np.float64)
    r = np.maximum(np.linalg.norm(xyz, axis=1), 1e-9)
    pitch = np.degrees(np.arcsin(np.clip(xyz[:, 1] / r, -1.0, 1.0)))
    return splats.take(pitch >= pitch_deg)


def make_scene(
    wedge=False, shell=True, near_splat=False, nt=100, npi=50,
    ground_spacing=0.35, fg=False,
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
        _splats(sp, np.stack([s_shade] * 3, axis=1),
                plyio.LAYER_FG if fg else plyio.LAYER_BG, scale=0.45),
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


def stage_params(px: int = 96, fid_px: int = 48, fid_over: dict | None = None,
                 **s7_over) -> dict:
    p = copy.deepcopy(params_mod.load(REPO / "params.yaml"))
    p["s7"]["render_px"] = px  # keep CPU renders fast in tests
    # Small, always-pass fidelity by default: the synthetic pano is unrelated
    # to the synthetic scene, so relax the SSIM floors to -1 (always true).
    p["s7"]["fidelity"].update(
        {
            "tiles_lon": 4,
            "tiles_lat": 2,
            "render_px": fid_px,
            "ssim_worst_tile_min": -1.0,
            "ssim_mean_min": -1.0,
        }
    )
    if fid_over:
        p["s7"]["fidelity"].update(fid_over)
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


def _profile(n, final, sog_b, ply_bytes, target, cap, sog_max_mb):
    return {
        "in_count": n, "after_opacity_floor": n,
        "after_isolation_prune": n, "after_merge": final,
        "final_count": final, "stride_retries": [],
        "ply_bytes": int(ply_bytes), "sog_bytes": int(sog_b),
        "target": target, "cap": cap, "sog_max_mb": sog_max_mb,
    }


def make_s6_dir(run_dir: Path, scene: plyio.SplatData,
                sog: bytes = SOG_STUB, *, quest_final: int | None = None,
                quest_sog: int | None = None) -> None:
    out = run_dir / "s6_compress" / "out"
    out.mkdir(parents=True)
    plyio.write_splats(out / "scene.ply", scene)
    (out / "scene.sog").write_bytes(sog)
    ply_bytes = (out / "scene.ply").stat().st_size
    n = len(scene)
    qf = n if quest_final is None else quest_final
    qs = len(sog) if quest_sog is None else quest_sog
    compress = {
        "sog_tool": "test-stub",
        "primary_profile": "quest",
        "viewer_profile": "review",
        "profiles": {
            "review": _profile(n, n, len(sog) * 3, ply_bytes,
                                1500000, 2000000, 0),
            "quest": _profile(n, qf, qs, ply_bytes, 600000, 1000000, 60),
        },
    }
    schema.write_validated(out / "compress.json", compress, "compress")


def make_source_pano(run_dir: Path, kind: str = "clean",
                     w: int = 256, h: int = 128) -> Path:
    """Deterministic equirect source pano (s1 clean plate — the fidelity
    gate's required source — or the s0 ingest master)."""
    if kind == "ingest":
        d = run_dir / "s0_ingest" / "out"
        name = "pano.png"
    else:
        d = run_dir / "s1_cleanplate" / "out"
        name = "pano_clean.png"
    d.mkdir(parents=True, exist_ok=True)
    j, i = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    checker = (((i * 8) // w + (j * 4) // h) % 2).astype(np.uint8)
    r = (60 + 120 * checker).astype(np.uint8)
    g = (i % 256).astype(np.uint8)
    b = ((j * 2) % 256).astype(np.uint8)
    imageio.save_png(d / name, np.stack([r, g, b], axis=-1))
    return d / name


@pytest.fixture(scope="module")
def stage_run(tmp_path_factory):
    """One full s7_gates run over the closed PASS scene."""
    d = tmp_path_factory.mktemp("s7run") / "run"
    scene = make_scene()
    make_s6_dir(d, scene)
    make_source_pano(d)
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
    assert GATE_ORDER == (
        "hole", "jitter", "stereo", "people", "budgets", "fidelity_at_origin"
    )
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
        for layer in ("fg", "bg", "shell"):
            assert (rdir / f"center_yaw{yaw}_layer_{layer}.png").exists()
    # the straight-down nadir view is part of the standard matrix
    assert (rdir / "center_down_magenta.png").exists()
    assert (rdir / "center_down_normal.png").exists()
    for extra in ("jitter_base.png", "jitter_offset.png",
                  "stereo_yaw000_left.png", "stereo_yaw000_right.png",
                  "hole_worst_magenta.png", "fidelity_worst_render.png",
                  "fidelity_worst_source.png"):
        assert (rdir / extra).exists()


def test_stage_receipt(stage_run):
    d, params, scene = stage_run
    rec = receipts.read_receipt(d, "s7_gates")
    assert [g["gate"] for g in rec["gates"]] == list(GATE_ORDER)
    for g in rec["gates"]:
        schema.validate(g, "gate_verdict")
    assert [w["key"] for w in rec["weights"]] == ["rtdetr_r18"]
    assert set(rec["params_used"]) == {
        "head_box", "s7", "splat_cap", "splat_target", "sog_max_mb"
    }
    assert rec["params_used"]["s7"] == params["s7"]
    # provenance: the receipt covers everything the stage read (incl. the
    # gated .sog + the fidelity source pano) and every render it produced
    assert set(rec["inputs"]) == {"scene", "compress", "sog", "pano_clean"}
    assert rec["inputs"]["pano_clean"]["path"] == (
        "s1_cleanplate/out/pano_clean.png")
    render_keys = {
        f"render_{n[:-len('.png')]}" for n in rec["notes"]["renders"]
    }
    assert set(rec["outputs"]) == (
        {f"verdict_{g}" for g in GATE_ORDER} | render_keys
    )
    assert len(render_keys) == len(rec["notes"]["renders"])  # stems unique
    for v in list(rec["inputs"].values()) + list(rec["outputs"].values()):
        assert not v["path"].startswith("/")
    rdir = d / "s7_gates" / "out" / "renders"
    assert rec["notes"]["renders"] == sorted(
        p.name for p in rdir.glob("*.png"))
    assert len(rec["notes"]["renders"]) > 0
    assert rec["notes"]["all_pass"] is True
    # layer forensics notes
    lc = rec["notes"]["layer_counts"]
    la = rec["notes"]["layer_solid_angle"]
    assert set(lc) == {"fg", "bg", "shell"} == set(la)
    # this scene has no fg splats; bg + shell are populated
    assert lc["fg"] == 0
    assert lc["bg"] > 0 and lc["shell"] > 0
    assert lc["bg"] + lc["shell"] == len(scene)
    assert la["fg"] == 0.0
    assert 0.0 < la["bg"] <= 1.0 and 0.0 < la["shell"] <= 1.0
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
    # 7 poses x (4 pitch-0 yaws + 1 straight-down view)
    assert m["n_views"] == 35
    assert len(v["details"]["per_view"]) == 35
    views = {pv["view"] for pv in v["details"]["per_view"]}
    assert "center_down" in views and "squat_down" in views


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


def test_hole_fails_on_nadir_cut(tmp_path):
    """Regression (nadir blind cone): deleting ALL splats below pitch -50 deg
    with no shell behind is invisible to the pitch-0 fov-90 ring (its rays
    only reach ~-45 deg) — the straight-down views must fail the gate."""
    scene = _cut_below_pitch(make_scene(), -50.0)
    params = stage_params()
    v = hole.run_gate(scene, params, tmp_path)
    schema.validate(v, "gate_verdict")
    assert v["pass"] is False
    per = {pv["view"]: pv for pv in v["details"]["per_view"]}
    # the hole shows up as an alpha void in the down view...
    assert per["center_down"]["alpha_frac"] > params["s7"]["hole_max_frac"]
    # ... while the pitch-0 ring still sees nothing (the old blind spot)
    for yaw in ("000", "090", "180", "270"):
        pv = per[f"center_yaw{yaw}"]
        assert pv["alpha_frac"] <= params["s7"]["hole_max_frac"]
        assert pv["magenta_frac"] <= params["s7"]["hole_max_frac"]


# ------------------------------------------------------------------ jitter


def test_jitter_passes_static_scene(stage_run):
    d, params, _ = stage_run
    v = verdict(d, "jitter")
    assert v["pass"] is True
    assert 0.0 <= v["metrics"]["energy"] <= params["s7"]["jitter_energy_max"]
    assert v["thresholds"]["jitter_offset_m"] == params["s7"]["jitter_offset_m"]
    # all 5 center-pose views certified; the metric is the worst of them
    per = v["details"]["per_view"]
    assert [pv["view"] for pv in per] == [
        "center_yaw000", "center_yaw090", "center_yaw180", "center_yaw270",
        "center_down",
    ]
    assert v["metrics"]["energy"] == max(pv["energy"] for pv in per)
    assert v["details"]["worst_view"] in {pv["view"] for pv in per}


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
    # 4 full yaw entries + 2 near-limit-only pitched entries (down, up)
    per = v["details"]["per_yaw"]
    assert len(per) == 6
    ring = [e for e in per if not e["near_only"]]
    polar = [e for e in per if e["near_only"]]
    assert [e["pitch_deg"] for e in ring] == [0.0] * 4
    assert [e["pitch_deg"] for e in polar] == [-90.0, 90.0]
    for e in ring:
        assert {"vdisp_px", "order_frac", "n_order_px"} <= set(e)
    for e in polar:  # near-only rows: no vdisp / order fields
        assert "vdisp_px" not in e and "order_frac" not in e
        assert e["min_depth_m"] >= params["s7"]["stereo_near_depth_min_m"]
    # the nearest content overall is the ground ~1.6 m below (down view)
    assert 1.0 < m["min_depth_m"] < 2.5


def test_stereo_near_limit_fails(tmp_path):
    """A splat 0.3 m ahead violates the near limit (min 0.75 m)."""
    scene = make_scene(near_splat=True)
    params = stage_params()
    v = stereo.run_gate(scene, params, tmp_path)
    schema.validate(v, "gate_verdict")
    assert v["pass"] is False
    assert v["metrics"]["min_depth_m"] < params["s7"]["stereo_near_depth_min_m"]
    assert (tmp_path / "renders" / "stereo_yaw000_left.png").exists()


def test_stereo_near_limit_fails_off_axis(tmp_path):
    """Regression (near-limit blind wedge): an opaque splat 0.3 m away at
    azimuth 45 deg sits outside every view's old central-half window (only
    +-26.5 deg per yaw was checked) — the full-frame near limit must now
    record min_depth_m < 0.75 and fail the gate."""
    az = np.deg2rad(45.0)
    near = _splats(
        [[0.3 * np.sin(az), 0.0, 0.3 * np.cos(az)]], (0.5, 0.5, 0.5),
        plyio.LAYER_FG, opacity=0.99, scale=0.08,
    )
    scene = plyio.SplatData.concat([make_scene(), near])
    params = stage_params()
    v = stereo.run_gate(scene, params, tmp_path)
    schema.validate(v, "gate_verdict")
    assert v["pass"] is False
    assert v["metrics"]["min_depth_m"] < params["s7"]["stereo_near_depth_min_m"]


# ------------------------------------------------------------------ people


def test_people_passes(stage_run):
    d, params, _ = stage_run
    v = verdict(d, "people")
    assert v["pass"] is True
    assert v["metrics"]["n_detections"] == 0
    assert 0.0 <= v["metrics"]["max_score"] < params["s7"]["people_score_min"]
    # 7 poses x (4 yaws + down), same matrix as the hole gate
    assert len(v["details"]["per_view"]) == 35


# ----------------------------------------------------------------- budgets


def test_budgets_passes(stage_run):
    d, params, scene = stage_run
    v = verdict(d, "budgets")
    assert v["pass"] is True
    m = v["metrics"]
    # budgets reads the QUEST profile's final_count / sog_bytes
    assert m["final_count"] == len(scene)
    assert m["sog_bytes"] == len(SOG_STUB)
    assert m["scene_ply_count"] == len(scene)
    assert m["scene_sog_bytes"] == len(SOG_STUB)
    # review profile recorded, non-failing
    assert m["review_final_count"] == len(scene)
    assert m["review_sog_bytes"] == len(SOG_STUB) * 3
    assert m["count_vs_target_ratio"] == pytest.approx(
        len(scene) / params["splat_target"])
    assert v["thresholds"]["splat_cap"] == params["splat_cap"]
    assert v["thresholds"]["sog_max_bytes"] == params["sog_max_mb"] * 1024 * 1024
    assert v["details"]["count_matches_scene"] is True
    assert v["details"]["sog_bytes_matches_compress"] is True
    assert v["details"]["primary_profile"] == "quest"


def test_budgets_default_run_dir_derivation(stage_run):
    """run_gate(splats, params, outdir) alone must find the s6 artifacts via
    the standard <run>/s7_gates/out layout."""
    d, params, _ = stage_run
    splats = plyio.read_splats(d / "s6_compress" / "out" / "scene.ply")
    v = budgets.run_gate(splats, params, d / "s7_gates" / "out")
    assert v == verdict(d, "budgets")


def test_budgets_reads_quest_not_review(tmp_path):
    """The gate must key off the quest profile: a bloated quest count fails
    even when the review count is fine."""
    scene = make_scene(nt=24, npi=12, ground_spacing=1.0)
    d = tmp_path / "run"
    make_s6_dir(d, scene, quest_final=5_000_000)  # over the 1M cap
    splats = plyio.read_splats(d / "s6_compress" / "out" / "scene.ply")
    params = stage_params()
    v = budgets.run_gate(splats, params, d / "s7_gates" / "out", run_dir=d)
    schema.validate(v, "gate_verdict")
    assert v["pass"] is False
    assert v["metrics"]["final_count"] == 5_000_000
    # the on-disk ship .ply is smaller -> cross-check flags the mismatch
    assert v["metrics"]["scene_ply_count"] == len(scene)
    assert v["details"]["count_matches_scene"] is False


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


# ---------------------------------------------------------------- fidelity


def _fidelity_run_dir(tmp_path, kind="clean", nt=24, npi=12):
    """Fake run dir with a source pano; returns (run_dir, scene, out_dir)."""
    d = tmp_path / "run"
    scene = make_scene(nt=nt, npi=npi, ground_spacing=1.0)
    make_source_pano(d, kind=kind)
    out = d / "s7_gates" / "out"
    out.mkdir(parents=True)
    return d, scene, out


def test_fidelity_reports_and_advisory_unavailable(tmp_path):
    d, scene, out = _fidelity_run_dir(tmp_path)
    params = stage_params(fid_px=48)
    v = fidelity.run_gate(scene, params, out, run_dir=d)
    schema.validate(v, "gate_verdict")
    assert v["gate"] == "fidelity_at_origin"
    m = v["metrics"]
    assert "ssim_worst_tile" in m and "ssim_mean" in m
    assert -1.0 <= m["ssim_worst_tile"] <= 1.0
    assert -1.0 <= m["ssim_mean"] <= 1.0
    assert m["ssim_worst_tile"] <= m["ssim_mean"] + 1e-9
    assert m["n_tiles"] == 4 * 2
    assert isinstance(m["worst_tile"], str) and m["worst_tile"]
    # LPIPS advisory unavailable by default (weights never in enforced tree)
    assert m["lpips"] == "advisory_unavailable"
    assert "lpips_mean" not in m
    assert v["details"]["lpips_reason"]
    assert v["thresholds"] == {
        "ssim_worst_tile_min": -1.0, "ssim_mean_min": -1.0
    }
    # relaxed thresholds -> pass; diagnostic tiles written
    assert v["pass"] is True
    assert (out / "renders" / "fidelity_worst_render.png").exists()
    assert (out / "renders" / "fidelity_worst_source.png").exists()
    assert len(v["details"]["per_tile"]) == 8


def test_fidelity_falsifiable(tmp_path):
    """SSIM is enforced: an impossible worst-tile floor (>1) must fail."""
    d, scene, out = _fidelity_run_dir(tmp_path)
    params = stage_params(fid_px=48, fid_over={"ssim_worst_tile_min": 1.5})
    v = fidelity.run_gate(scene, params, out, run_dir=d)
    assert v["pass"] is False
    assert v["metrics"]["ssim_worst_tile"] < 1.5


def test_fidelity_mean_floor_falsifiable(tmp_path):
    """The mean floor is independently enforced."""
    d, scene, out = _fidelity_run_dir(tmp_path)
    params = stage_params(fid_px=48, fid_over={"ssim_mean_min": 1.5})
    v = fidelity.run_gate(scene, params, out, run_dir=d)
    assert v["pass"] is False
    assert v["metrics"]["ssim_mean"] < 1.5


def test_fidelity_deterministic(tmp_path):
    d, scene, out = _fidelity_run_dir(tmp_path)
    params = stage_params(fid_px=48)
    a = fidelity.run_gate(scene, params, out, run_dir=d)
    b = fidelity.run_gate(scene, params, out, run_dir=d)
    assert a == b


def test_fidelity_requires_clean_plate(tmp_path):
    """The s1 clean plate is REQUIRED: an s0 ingest master alone must raise
    (a complete run always writes pano_clean.png; a silent s0 fallback would
    score fidelity against the UNCLEANED pano on broken runs)."""
    d = tmp_path / "run"
    with pytest.raises(FileNotFoundError):
        fidelity._source_pano_path(d)
    make_source_pano(d, kind="ingest")
    with pytest.raises(FileNotFoundError):
        fidelity._source_pano_path(d)
    make_source_pano(d, kind="clean")
    assert fidelity._source_pano_path(d).name == "pano_clean.png"


def test_fidelity_uses_clean_plate_when_present(tmp_path):
    """When only the clean plate exists the gate uses it (no s0 pano)."""
    d = tmp_path / "run"
    scene = make_scene(nt=24, npi=12, ground_spacing=1.0)
    make_source_pano(d, kind="clean")
    out = d / "s7_gates" / "out"
    out.mkdir(parents=True)
    params = stage_params(fid_px=48)
    v = fidelity.run_gate(scene, params, out, run_dir=d)
    schema.validate(v, "gate_verdict")
    assert v["metrics"]["n_tiles"] == 8


def test_fidelity_missing_pano_raises(tmp_path):
    d = tmp_path / "run"
    (d / "s6_compress" / "out").mkdir(parents=True)  # unrelated dir
    out = d / "s7_gates" / "out"
    out.mkdir(parents=True)
    scene = make_scene(nt=12, npi=6, ground_spacing=1.0)
    with pytest.raises(FileNotFoundError):
        fidelity.run_gate(scene, stage_params(fid_px=32), out, run_dir=d)


# ------------------------------------------------------------------ layers


def test_layer_forensics_renders_and_counts(tmp_path):
    """A scene with all three layers: every fg/bg/shell origin render is
    emitted and the receipt counts partition the scene."""
    d = tmp_path / "run"
    # fg=True routes the sphere to LAYER_FG so every layer is populated
    scene = make_scene(nt=24, npi=12, ground_spacing=1.0, fg=True)
    make_s6_dir(d, scene)
    make_source_pano(d)
    params = stage_params(px=64)
    determinism.set_seed(params.get("seed", 0))
    s7_gates.run(d, params, make_ctx())
    rdir = d / "s7_gates" / "out" / "renders"
    for yaw in ("000", "090", "180", "270"):
        for layer in ("fg", "bg", "shell"):
            assert (rdir / f"center_yaw{yaw}_layer_{layer}.png").exists()
    rec = receipts.read_receipt(d, "s7_gates")
    lc = rec["notes"]["layer_counts"]
    assert lc["fg"] > 0 and lc["bg"] > 0 and lc["shell"] > 0
    assert lc["fg"] + lc["bg"] + lc["shell"] == len(scene)


def test_layers_disabled_skips_forensics(tmp_path):
    d = tmp_path / "run"
    scene = make_scene(nt=24, npi=12, ground_spacing=1.0)
    make_s6_dir(d, scene)
    make_source_pano(d)
    params = stage_params(px=64, layers=False)
    determinism.set_seed(params.get("seed", 0))
    s7_gates.run(d, params, make_ctx())
    rdir = d / "s7_gates" / "out" / "renders"
    assert not list(rdir.glob("*_layer_*.png"))
    rec = receipts.read_receipt(d, "s7_gates")
    assert rec["notes"]["layer_counts"] == {}
    assert rec["notes"]["layer_solid_angle"] == {}


# ------------------------------------------------------ stage error handling


def test_stage_missing_s6_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        s7_gates.run(tmp_path / "empty_run", stage_params(), make_ctx())


# ---------------------------------------------------------------- determinism


def test_stage_double_run_byte_identical(tmp_path):
    """Same scene bytes + params => byte-identical verdicts, renders and
    receipt across two independent stage runs (tiny scene to keep the 2x35
    detector calls cheap)."""
    scene = make_scene(nt=24, npi=12, ground_spacing=1.0)
    params = stage_params(px=64)
    dirs = [tmp_path / "run_a", tmp_path / "run_b"]
    for d in dirs:
        make_s6_dir(d, scene)
        make_source_pano(d)
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
    assert len(fa) >= 1 + 6 + 8
    for a, b in zip(fa, fb):
        assert a.read_bytes() == b.read_bytes(), f"{a.name} differs"
