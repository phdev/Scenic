"""s8_review tests (v2 quality pass: toggles + SOG quant).

Fixture: a fake run dir with
  * a two-profile schema-valid compress.json (profiles.review + profiles.quest,
    each with ply_bytes/sog_bytes),
  * six schema-valid s7 verdicts (one deliberately failing),
  * scene.ply (primary/quest alias) + scene_review.*/scene_quest.* files,
  * s7 layer renders center_yaw{NNN}_layer_{fg,bg,shell}.png.

The 4 standard views are ALWAYS rendered from scene.ply; stale
center_yaw{NNN}.png debris in s7's renders dir must be ignored.

Covered: outputs + schema validity, page contents (all six gate names, base64
data URIs, params hash, SuperSplat footer, PASS+FAIL, the fg/bg/shell toggle,
the sog byte-ratio, the copied scene_review.sog viewer reference), review.json
with layers + sog_ssim + profiles, double-run byte-identical index.html across
DIFFERENT run dirs, the runs/_accepted side-by-side flow (incl. the baseline
files recorded as receipt inputs), missing-layer placeholder, stale-debris
immunity, receipt shape (incl. layer_* inputs with run-relative s7 paths), and
hard errors on missing inputs.
"""
from __future__ import annotations

import base64
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

GATES = ("budgets", "fidelity_at_origin", "hole", "jitter", "people", "stereo")
YAWS = (0, 90, 180, 270)
LAYERS = ("fg", "bg", "shell")
PX = 64                      # small render for fast tests
N_RING = 48

# distinct byte streams per profile so byte counts + ratios are meaningful
REVIEW_SOG = b"REVIEWSOG" * 40
QUEST_SOG = b"QUESTSOG" * 30
REVIEW_PLY_BYTES = 900000
QUEST_PLY_BYTES = 360000


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


def _profile(final: int, sog_bytes: int, ply_bytes: int, target: int,
             cap: int, sog_max_mb: float) -> dict:
    return {
        "in_count": final,
        "after_opacity_floor": final,
        "after_isolation_prune": final,
        "after_merge": final,
        "final_count": final,
        "stride_retries": [],
        "ply_bytes": ply_bytes,
        "sog_bytes": sog_bytes,
        "target": target,
        "cap": cap,
        "sog_max_mb": sog_max_mb,
    }


def make_run(d: Path, *, with_layer_renders: bool = True,
             drop_layer: tuple[int, str] | None = None) -> None:
    s6 = d / "s6_compress" / "out"
    s6.mkdir(parents=True)
    splats = make_splats()
    # primary alias (quest) + per-profile ply/sog
    plyio.write_splats(s6 / "scene.ply", splats)
    plyio.write_splats(s6 / "scene_quest.ply", splats)
    plyio.write_splats(s6 / "scene_review.ply", splats)
    (s6 / "scene.sog").write_bytes(QUEST_SOG)
    (s6 / "scene_quest.sog").write_bytes(QUEST_SOG)
    (s6 / "scene_review.sog").write_bytes(REVIEW_SOG)
    schema.write_validated(
        s6 / "compress.json",
        {
            "sog_tool": "stub",
            "primary_profile": "quest",
            "viewer_profile": "review",
            "profiles": {
                "review": _profile(
                    N_RING, len(REVIEW_SOG), REVIEW_PLY_BYTES,
                    1500000, 2000000, 0,
                ),
                "quest": _profile(
                    N_RING, len(QUEST_SOG), QUEST_PLY_BYTES,
                    600000, 1000000, 60,
                ),
            },
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
    if with_layer_renders:
        rdir = d / "s7_gates" / "out" / "renders"
        rdir.mkdir(parents=True, exist_ok=True)
        for y in YAWS:
            for li, layer in enumerate(LAYERS):
                if drop_layer is not None and drop_layer == (y, layer):
                    continue
                fill = 30 + 20 * li + (y // 90) * 3
                arr = np.full((8, 8, 3), fill, np.uint8)
                imageio.save_png(
                    rdir / f"center_yaw{y:03d}_layer_{layer}.png", arr
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
    assert (out(run_dir) / "scene_review.sog").exists()
    for y in YAWS:
        assert (out(run_dir) / "thumbs" / f"{y}.png").exists()


def test_all_six_gate_names_on_page(run_dir):
    text = (out(run_dir) / "index.html").read_text()
    for g in GATES:
        assert g in text, f"gate name {g} missing from page"
    # explicitly ensure the v2-added gates are present
    assert "fidelity_at_origin" in text


def test_index_core_contents(run_dir):
    text = (out(run_dir) / "index.html").read_text()
    assert "data:image/png;base64," in text
    assert hashing.sha256_json(small_params()) in text  # run params hash
    assert str(N_RING) in text                          # splat counts
    # both profile sog sizes surfaced in header
    assert f"{len(REVIEW_SOG)} bytes" in text
    assert f"{len(QUEST_SOG)} bytes" in text
    assert "superspl.at" in text                        # SuperSplat footer
    assert "PASS" in text and "FAIL" in text            # hole gate fails
    # self-contained + deterministic: no network fetches, no absolute paths
    assert "http" not in text.replace("https://superspl.at", "")
    assert str(run_dir) not in text


def test_layer_toggle_present(run_dir):
    text = (out(run_dir) / "index.html").read_text()
    # fg/bg/shell toggle: buttons + the setLayer JS + gallery data-layer state
    for layer in LAYERS:
        assert f'data-layer="{layer}"' in text
        assert f"setLayer('{layer}')" in text
    assert "function setLayer" in text
    assert 'id="gallery"' in text
    # every layer render is embedded (4 yaws * 3 layers = 12 images present)
    d = run_dir
    for y in YAWS:
        for layer in LAYERS:
            png = (
                d / "s7_gates" / "out" / "renders"
                / f"center_yaw{y:03d}_layer_{layer}.png"
            ).read_bytes()
            assert base64.b64encode(png).decode("ascii") in text


def test_sog_byte_ratio_shown(run_dir):
    text = (out(run_dir) / "index.html").read_text()
    assert "SOG byte ratio" in text
    # per-profile ply/sog ratio values
    review_ratio = f"{REVIEW_PLY_BYTES / len(REVIEW_SOG):.2f}x"
    quest_ratio = f"{QUEST_PLY_BYTES / len(QUEST_SOG):.2f}x"
    assert review_ratio in text
    assert quest_ratio in text
    # raw byte counts present
    assert str(REVIEW_PLY_BYTES) in text
    assert str(QUEST_PLY_BYTES) in text
    # sog_ssim skipped -> decode-unavailable note surfaced
    assert "decode unavailable" in text


def test_viewer_references_review_sog(run_dir):
    text = (out(run_dir) / "index.html").read_text()
    assert "scene_review.sog" in text
    assert "viewer_profile" in text
    # the copied sog is a byte copy of s6's review sog
    assert (out(run_dir) / "scene_review.sog").read_bytes() == REVIEW_SOG


def test_review_json_schema_valid(run_dir):
    r = schema.read_validated(out(run_dir) / "review.json", "review")
    assert r["page"] == "index.html"
    assert r["compared_to_accepted"] is False
    assert [p["yaw_deg"] for p in r["poses"]] == list(YAWS)
    assert all(p["source"] == "rendered" for p in r["poses"])
    assert all(p["px"] == PX for p in r["poses"])
    # v2 additions
    assert r["sog_ssim"] is None
    assert isinstance(r["layers"], list)
    expected_layers = sorted(
        f"center_yaw{y:03d}_layer_{layer}.png"
        for y in YAWS
        for layer in LAYERS
    )
    assert r["layers"] == expected_layers
    assert set(r["profiles"]) == {"review", "quest"}
    assert r["profiles"]["review"]["final_count"] == N_RING
    assert r["profiles"]["review"]["sog_bytes"] == len(REVIEW_SOG)
    assert r["profiles"]["quest"]["sog_bytes"] == len(QUEST_SOG)


def test_thumbs_are_real_renders(run_dir):
    for y in YAWS:
        arr = imageio.load_rgb(out(run_dir) / "thumbs" / f"{y}.png")
        assert arr.shape == (PX, PX, 3)
        assert arr.max() > 0, f"yaw {y} render is all black"


def test_thumbs_embedded_in_page(run_dir):
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
    assert (out(d1) / "scene_review.sog").read_bytes() == (
        out(d2) / "scene_review.sog"
    ).read_bytes()
    for y in YAWS:
        assert (out(d1) / "thumbs" / f"{y}.png").read_bytes() == (
            out(d2) / "thumbs" / f"{y}.png"
        ).read_bytes()


def test_missing_layer_render_placeholder(tmp_path):
    d = tmp_path / "run"
    make_run(d, drop_layer=(90, "bg"))
    run_stage(d)
    text = (out(d) / "index.html").read_text()
    assert "bg missing" in text            # placeholder rendered
    r = schema.read_validated(out(d) / "review.json", "review")
    assert "center_yaw090_layer_bg.png" not in r["layers"]
    assert len(r["layers"]) == len(YAWS) * len(LAYERS) - 1
    # only EXISTING renders are receipt inputs (receipts record what was read)
    rec = receipts.read_receipt(d, "s8_review")
    assert "layer_90_bg" not in rec["inputs"]
    layer_keys = {k for k in rec["inputs"] if k.startswith("layer_")}
    assert len(layer_keys) == len(YAWS) * len(LAYERS) - 1


def test_no_layer_renders_all_placeholders(tmp_path):
    d = tmp_path / "run"
    make_run(d, with_layer_renders=False)
    run_stage(d)
    text = (out(d) / "index.html").read_text()
    # 12 placeholders (4 yaws * 3 layers), page still self-contained
    assert text.count("missing</div>") == len(YAWS) * len(LAYERS)
    r = schema.read_validated(out(d) / "review.json", "review")
    assert r["layers"] == []
    rec = receipts.read_receipt(d, "s8_review")
    assert not any(k.startswith("layer_") for k in rec["inputs"])


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
    assert "no accepted baseline" not in text
    # the 4 accepted thumbs are embedded alongside the 4 this-run thumbs
    for y in YAWS:
        png = (out(b) / "thumbs" / f"{y}.png").read_bytes()
        assert base64.b64encode(png).decode("ascii") in text


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


def test_stale_s7_render_debris_ignored(tmp_path):
    # No current s7 output matches the standard view names (hole writes the
    # magenta-dyed center_yawNNN_magenta.png, people center_yawNNN_normal.png),
    # so a center_yawNNN.png in the renders dir can only be stale debris from a
    # reused run dir — it must NEVER become the shipped thumbnail. s8 always
    # renders the 4 views from scene.ply.
    d = tmp_path / "run"
    make_run(d)
    rdir = d / "s7_gates" / "out" / "renders"
    fills = {0: 40, 90: 90, 180: 160, 270: 220}
    for y in YAWS:
        arr = np.full((8, 8, 3), fills[y], np.uint8)
        imageio.save_png(rdir / f"center_yaw{y:03d}.png", arr)
    run_stage(d)
    r = schema.read_validated(out(d) / "review.json", "review")
    assert all(p["source"] == "rendered" for p in r["poses"])
    for y in YAWS:
        thumb = imageio.load_rgb(out(d) / "thumbs" / f"{y}.png")
        assert thumb.shape == (PX, PX, 3)  # real render, not the 8x8 debris
        assert (out(d) / "thumbs" / f"{y}.png").read_bytes() != (
            rdir / f"center_yaw{y:03d}.png"
        ).read_bytes()
    # the debris is not embedded in the page either
    text = (out(d) / "index.html").read_text()
    for y in YAWS:
        png = (rdir / f"center_yaw{y:03d}.png").read_bytes()
        assert base64.b64encode(png).decode("ascii") not in text


def test_receipt(run_dir):
    rec = receipts.read_receipt(run_dir, "s8_review")
    assert set(rec["inputs"]) == (
        {"scene", "scene_review_sog", "compress"}
        | {f"verdict_{g}" for g in GATES}
        | {f"layer_{y}_{layer}" for y in YAWS for layer in LAYERS}
    )
    assert set(rec["outputs"]) == {"index", "review", "viewer_sog"} | {
        f"thumb_{y}" for y in YAWS
    }
    assert rec["params_used"] == {
        "s7": {"render_px": PX, "render_fov_deg": 90.0}
    }
    assert rec["weights"] == []
    assert rec["gates"] == []
    assert rec["notes"]["compared_to_accepted"] is False
    assert rec["notes"]["viewer_profile"] == "review"
    assert rec["notes"]["primary_profile"] == "quest"
    assert rec["notes"]["sog_ssim"] is None
    for v in list(rec["inputs"].values()) + list(rec["outputs"].values()):
        assert not v["path"].startswith("/")


def test_receipt_records_layer_render_inputs(run_dir):
    # every s7 layer render embedded in index.html is a receipt input, recorded
    # under its real run-relative path (so the manifest hash-coherence check
    # can match it against s7's recorded output) with the real byte hash.
    rec = receipts.read_receipt(run_dir, "s8_review")
    for y in YAWS:
        for layer in LAYERS:
            entry = rec["inputs"][f"layer_{y}_{layer}"]
            rel = f"s7_gates/out/renders/center_yaw{y:03d}_layer_{layer}.png"
            assert entry["path"] == rel
            assert entry["sha256"] == hashing.sha256_file(run_dir / rel)


def test_receipt_records_accepted_baseline_inputs(tmp_path):
    # the accepted-baseline bytes are embedded in the hashed index.html, so
    # promoting a baseline must be attributable: when compared=True the five
    # baseline files are receipt inputs (external/<name> entries, real sha256).
    a, b = tmp_path / "a", tmp_path / "b"
    make_run(a)
    run_stage(a)
    rec_a = receipts.read_receipt(a, "s8_review")
    assert not any(k.startswith("accepted_") for k in rec_a["inputs"])
    shutil.copytree(a / "s8_review", tmp_path / "_accepted" / "s8_review")
    make_run(b)
    run_stage(b)
    rec_b = receipts.read_receipt(b, "s8_review")
    acc_out = tmp_path / "_accepted" / "s8_review" / "out"
    expected = {"accepted_review": acc_out / "review.json"}
    for y in YAWS:
        expected[f"accepted_thumb_{y}"] = acc_out / "thumbs" / f"{y}.png"
    for key, src in expected.items():
        entry = rec_b["inputs"][key]
        assert entry["path"] == f"external/{src.name}"
        assert entry["sha256"] == hashing.sha256_file(src)


def test_missing_verdict_raises(tmp_path):
    d = tmp_path / "run"
    make_run(d)
    (d / "s7_gates" / "out" / "verdicts" / "fidelity_at_origin.json").unlink()
    with pytest.raises(FileNotFoundError, match="verdict"):
        run_stage(d)


def test_missing_scene_raises(tmp_path):
    d = tmp_path / "run"
    make_run(d)
    (d / "s6_compress" / "out" / "scene.ply").unlink()
    with pytest.raises(FileNotFoundError, match="s6 output"):
        run_stage(d)


def test_missing_review_sog_raises(tmp_path):
    d = tmp_path / "run"
    make_run(d)
    (d / "s6_compress" / "out" / "scene_review.sog").unlink()
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
