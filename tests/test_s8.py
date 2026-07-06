"""s8_review tests.

Fixture: a fake run dir with a small synthetic scene.ply (a colored ring of
splats around the origin, one color per quadrant), five minimal schema-valid
s7 verdicts (one deliberately failing), a schema-valid compress.json and a
scene.sog stub. No s7 renders -> s8 renders the 4 standard views itself.

Covered: outputs + schema validity, page contents (five gate names, base64
data URIs, params hash, SuperSplat note, PASS+FAIL), double-run byte-identical
index.html across DIFFERENT run dirs (no absolute paths leak), the
runs/_accepted side-by-side flow, byte-for-byte reuse of s7 renders, receipt
shape, and hard errors on missing inputs.
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

import numpy as np
import pytest

from pipeline import s8_review
from scenic import determinism, hashing, imageio, params as params_mod
from scenic import plyio, receipts, schema
from scenic.stage import Ctx

REPO = Path(__file__).resolve().parent.parent

GATES = ("budgets", "hole", "jitter", "people", "stereo")
YAWS = (0, 90, 180, 270)
PX = 64                      # small render for fast tests
N_RING = 48
SOG_STUB = b"SOGSTUB" * 100  # scene.sog placeholder (s8 only reads its size)


# ------------------------------------------------------------ fixture build


def make_splats() -> plyio.SplatData:
    """Ring of splats at radius 5 in the y=0 plane, one color per quadrant."""
    ang = 2.0 * np.pi * np.arange(N_RING, dtype=np.float64) / N_RING
    xyz = np.stack(
        [5.0 * np.sin(ang), np.zeros(N_RING), 5.0 * np.cos(ang)], axis=1
    ).astype(np.float32)
    colors = np.array(
        [[0.9, 0.2, 0.2], [0.2, 0.9, 0.2], [0.2, 0.2, 0.9], [0.9, 0.9, 0.2]],
        np.float32,
    )
    quadrant = (ang / (np.pi / 2.0)).astype(int) % 4
    return plyio.SplatData(
        xyz=xyz,
        normals=np.zeros((N_RING, 3), np.float32),
        f_dc=plyio.rgb01_to_dc(colors[quadrant]),
        opacity_logit=plyio.opacity_to_logit(np.full(N_RING, 0.95, np.float32)),
        log_scales=np.full((N_RING, 3), np.log(0.4), np.float32),
        quat_wxyz=np.tile(np.array([1, 0, 0, 0], np.float32), (N_RING, 1)),
        layer=np.full(N_RING, plyio.LAYER_BG, np.uint8),
        origin_stage=np.full(N_RING, 4, np.uint8),
    )


def make_run(d: Path) -> None:
    s6 = d / "s6_compress" / "out"
    s6.mkdir(parents=True)
    plyio.write_splats(s6 / "scene.ply", make_splats())
    (s6 / "scene.sog").write_bytes(SOG_STUB)
    schema.write_validated(
        s6 / "compress.json",
        {
            "in_count": N_RING,
            "after_opacity_floor": N_RING,
            "after_isolation_prune": N_RING,
            "after_merge": N_RING,
            "final_count": N_RING,
            "stride_retries": [],
            "sog_bytes": len(SOG_STUB),
            "sog_tool": "stub",
        },
        "compress",
    )
    vdir = d / "s7_gates" / "out" / "verdicts"
    vdir.mkdir(parents=True)
    for i, g in enumerate(GATES):
        schema.write_validated(
            vdir / f"{g}.json",
            {
                "gate": g,
                "pass": g != "hole",  # one failing gate -> FAIL row rendered
                "metrics": {"metric_a": i, "frac": 0.25},
                "thresholds": {"max": 1.0},
                "details": f"synthetic {g} verdict",
            },
            "gate_verdict",
        )


def make_ctx() -> Ctx:
    return Ctx(
        repo_root=REPO,
        pano_path=REPO / "params.yaml",   # unused by s8 (reads run dir)
        sidecar_path=REPO / "params.yaml",
        params_path=REPO / "params.yaml",
        weights_dir=REPO / "weights",
    )


def small_params() -> dict:
    p = copy.deepcopy(params_mod.load(REPO / "params.yaml"))
    p["s7"]["render_px"] = PX
    return p


def run_stage(d: Path) -> dict:
    params = small_params()
    determinism.set_seed(params.get("seed", 0))
    s8_review.run(d, params, make_ctx())
    return params


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("s8run") / "run"
    make_run(d)
    run_stage(d)
    return d


def out(d: Path) -> Path:
    return d / "s8_review" / "out"


# ------------------------------------------------------------------- tests


def test_outputs_exist(run_dir):
    assert (out(run_dir) / "index.html").exists()
    assert (out(run_dir) / "review.json").exists()
    for y in YAWS:
        assert (out(run_dir) / "thumbs" / f"{y}.png").exists()


def test_index_contents(run_dir):
    text = (out(run_dir) / "index.html").read_text()
    for g in GATES:
        assert g in text, f"gate name {g} missing from page"
    assert "data:image/png;base64," in text
    assert hashing.sha256_json(small_params()) in text  # run params hash
    assert str(N_RING) in text                          # splat counts
    assert f"{len(SOG_STUB)} bytes" in text             # sog size
    assert "superspl.at" in text                        # SuperSplat footer
    assert "PASS" in text and "FAIL" in text            # hole gate fails
    # self-contained + deterministic: no network fetches, no absolute paths
    assert "http" not in text.replace("https://superspl.at", "")
    assert str(run_dir) not in text


def test_review_json_schema_valid(run_dir):
    r = schema.read_validated(out(run_dir) / "review.json", "review")
    assert r["page"] == "index.html"
    assert r["compared_to_accepted"] is False
    assert [p["yaw_deg"] for p in r["poses"]] == list(YAWS)
    assert all(p["source"] == "rendered" for p in r["poses"])
    assert all(p["px"] == PX for p in r["poses"])


def test_thumbs_are_real_renders(run_dir):
    for y in YAWS:
        arr = imageio.load_rgb(out(run_dir) / "thumbs" / f"{y}.png")
        assert arr.shape == (PX, PX, 3)
        assert arr.max() > 0, f"yaw {y} render is all black"


def test_thumbs_embedded_in_page(run_dir):
    import base64

    text = (out(run_dir) / "index.html").read_text()
    for y in YAWS:
        png = (out(run_dir) / "thumbs" / f"{y}.png").read_bytes()
        assert base64.b64encode(png).decode("ascii") in text


def test_double_run_byte_identical(tmp_path):
    d1, d2 = tmp_path / "a" / "run", tmp_path / "b" / "run"
    for d in (d1, d2):
        make_run(d)
        run_stage(d)
    assert (out(d1) / "index.html").read_bytes() == (
        out(d2) / "index.html"
    ).read_bytes()
    assert (out(d1) / "review.json").read_bytes() == (
        out(d2) / "review.json"
    ).read_bytes()
    for y in YAWS:
        assert (out(d1) / "thumbs" / f"{y}.png").read_bytes() == (
            out(d2) / "thumbs" / f"{y}.png"
        ).read_bytes()


def test_accepted_side_by_side(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_run(a)
    run_stage(a)
    # promote run a as the accepted baseline (sibling _accepted dir)
    shutil.copytree(a / "s8_review", tmp_path / "_accepted" / "s8_review")
    make_run(b)
    run_stage(b)
    r = schema.read_validated(out(b) / "review.json", "review")
    assert r["compared_to_accepted"] is True
    text = (out(b) / "index.html").read_text()
    assert "accepted" in text
    # 4 this-run images + 4 accepted images, no placeholders
    assert text.count("data:image/png;base64,") == 8
    assert "no accepted baseline" not in text


def test_accepted_incomplete_ignored(tmp_path):
    d = tmp_path / "run"
    make_run(d)
    # _accepted exists but is missing thumbs -> not comparable, no error
    acc = tmp_path / "_accepted" / "s8_review" / "out"
    acc.mkdir(parents=True)
    schema.write_validated(
        acc / "review.json",
        {"page": "index.html", "poses": [], "compared_to_accepted": False},
        "review",
    )
    run_stage(d)
    r = schema.read_validated(out(d) / "review.json", "review")
    assert r["compared_to_accepted"] is False
    assert "no accepted baseline" in (out(d) / "index.html").read_text()


def test_reuses_s7_renders_byte_identical(tmp_path):
    d = tmp_path / "run"
    make_run(d)
    rdir = d / "s7_gates" / "out" / "renders"
    rdir.mkdir(parents=True)
    fills = {0: 40, 90: 90, 180: 160, 270: 220}
    for y in YAWS:
        arr = np.full((8, 8, 3), fills[y], np.uint8)
        imageio.save_png(rdir / f"center_yaw{y:03d}.png", arr)
    run_stage(d)
    r = schema.read_validated(out(d) / "review.json", "review")
    assert all(p["source"] == "s7_renders" for p in r["poses"])
    for y in YAWS:
        assert (out(d) / "thumbs" / f"{y}.png").read_bytes() == (
            rdir / f"center_yaw{y:03d}.png"
        ).read_bytes()


def test_receipt(run_dir):
    rec = receipts.read_receipt(run_dir, "s8_review")
    assert set(rec["inputs"]) == {"scene"} | {f"verdict_{g}" for g in GATES}
    assert set(rec["outputs"]) == {"index", "review"} | {
        f"thumb_{y}" for y in YAWS
    }
    assert rec["params_used"] == {
        "s7": {"render_px": PX, "render_fov_deg": 90.0}
    }
    assert rec["weights"] == []
    assert rec["gates"] == []
    assert rec["notes"]["compared_to_accepted"] is False
    assert rec["notes"]["sog_bytes"] == len(SOG_STUB)
    for v in list(rec["inputs"].values()) + list(rec["outputs"].values()):
        assert not v["path"].startswith("/")


def test_missing_verdict_raises(tmp_path):
    d = tmp_path / "run"
    make_run(d)
    (d / "s7_gates" / "out" / "verdicts" / "jitter.json").unlink()
    with pytest.raises(FileNotFoundError, match="verdict"):
        run_stage(d)


def test_missing_scene_raises(tmp_path):
    d = tmp_path / "run"
    make_run(d)
    (d / "s6_compress" / "out" / "scene.ply").unlink()
    with pytest.raises(FileNotFoundError, match="s6 output"):
        run_stage(d)


def test_wrong_gate_name_raises(tmp_path):
    d = tmp_path / "run"
    make_run(d)
    vpath = d / "s7_gates" / "out" / "verdicts" / "hole.json"
    schema.write_validated(
        vpath,
        {"gate": "jitter", "pass": True, "metrics": {}, "thresholds": {}},
        "gate_verdict",
    )
    with pytest.raises(ValueError, match="declares gate"):
        run_stage(d)
