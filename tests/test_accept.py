"""Tests for tools/accept_run.py — promoting a run to the _accepted baseline.

Covers: slim promotion contents, s8's comparison contract being satisfied,
refusal of unshippable/incomplete runs, --allow-failed-gates override,
baseline replacement, and self-promotion refusal.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "accept_run", REPO_ROOT / "tools" / "accept_run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


accept_run = _load_tool()

from pipeline import s8_review  # noqa: E402
from scenic import imageio, manifest, receipts, schema  # noqa: E402

GATES = ("budgets", "hole", "jitter", "people", "stereo")
YAWS = (0, 90, 180, 270)


# ------------------------------------------------------------ fixture build


def _verdict(gate: str, ok: bool) -> dict:
    return {
        "gate": gate,
        "pass": ok,
        "metrics": {"m": 1.0},
        "thresholds": {"t": 2.0},
        "details": f"synthetic {gate} verdict",
    }


def make_completed_run(d: Path, name: str, all_pass: bool = True) -> Path:
    """A run with a full receipt chain + the s8 baseline artifacts.

    Fails the stereo gate when all_pass=False (shippable=false). The s8
    receipt records index.html/review.json as outputs so accept's
    verify_disk pass has real hashes to check."""
    run = d / name
    run.mkdir(parents=True)
    (run / "params.snapshot.yaml").write_text("seed: 0\n")
    out = run / "s8_review" / "out"
    (out / "thumbs").mkdir(parents=True)
    (out / "index.html").write_text(f"<html><body>{name}</body></html>\n")
    schema.write_validated(
        out / "review.json",
        {"page": "index.html", "poses": [], "compared_to_accepted": False},
        "review",
    )
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[..., 0] = sum(name.encode()) % 256  # distinct thumbs per run name
    for y in YAWS:
        imageio.save_png(out / "thumbs" / f"{y}.png", rgb)
    for stage in manifest.stage_order():
        gates = []
        outputs: dict = {}
        if stage == "s7_gates":
            gates = [_verdict(g, ok=(all_pass or g != "stereo")) for g in GATES]
        if stage == "s8_review":
            outputs = {"index": out / "index.html", "review": out / "review.json"}
        receipts.write_receipt(
            run, stage, inputs={}, outputs=outputs, params_used={}, gates=gates
        )
    # a heavyweight artifact that must NOT be promoted (slim baseline)
    s4 = run / "s4_place" / "out"
    s4.mkdir(parents=True)
    (s4 / "splats.ply").write_bytes(b"\x00" * 4096)
    manifest.build(run)
    return run


# ------------------------------------------------------------------- tests


def test_accept_promotes_slim_baseline(tmp_path):
    run = make_completed_run(tmp_path, "a")
    assert accept_run.main([str(run)]) == 0
    acc = tmp_path / "_accepted"
    for rel in (
        "accepted.json",
        "manifest.json",
        "params.snapshot.yaml",
        "s8_review/receipt.json",
        "s8_review/out/review.json",
        "s8_review/out/index.html",
        *[f"s8_review/out/thumbs/{y}.png" for y in YAWS],
    ):
        assert (acc / rel).exists(), rel
    # slim: no heavyweight stage outputs promoted
    assert not (acc / "s4_place").exists()

    rec = schema.read_validated(acc / "accepted.json", "accepted")
    assert rec["source_run"] == "a"
    assert rec["shippable"] is True
    assert rec["gate_summary"] == {"total": 5, "passed": 5, "all_pass": True}
    assert rec["manifest_hash"] == manifest.manifest_hash(run)


def test_accepted_satisfies_s8_comparison_contract(tmp_path):
    run = make_completed_run(tmp_path, "a")
    accept_run.main([str(run)])
    # any sibling run dir now sees the baseline as comparable
    thumbs, compared = s8_review._load_accepted(tmp_path / "b")
    assert compared is True
    assert sorted(thumbs) == sorted(YAWS)
    for y in YAWS:
        assert thumbs[y] == (run / "s8_review/out/thumbs" / f"{y}.png").read_bytes()


def test_refuses_unshippable_without_force(tmp_path):
    run = make_completed_run(tmp_path, "a", all_pass=False)
    with pytest.raises(SystemExit, match="shippable=false"):
        accept_run.main([str(run)])
    assert not (tmp_path / "_accepted").exists()


def test_allow_failed_gates_promotes_and_records_honestly(tmp_path):
    run = make_completed_run(tmp_path, "a", all_pass=False)
    assert accept_run.main([str(run), "--allow-failed-gates"]) == 0
    rec = schema.read_validated(tmp_path / "_accepted/accepted.json", "accepted")
    assert rec["shippable"] is False
    assert rec["gate_summary"] == {"total": 5, "passed": 4, "all_pass": False}


def test_refuses_incomplete_receipt_chain(tmp_path):
    run = make_completed_run(tmp_path, "a")
    (run / "s2_depth" / "receipt.json").unlink()
    with pytest.raises(SystemExit, match="incomplete receipt chain"):
        accept_run.main([str(run)])
    assert not (tmp_path / "_accepted").exists()


def test_refuses_missing_s8_artifacts(tmp_path):
    run = make_completed_run(tmp_path, "a")
    (run / "s8_review/out/thumbs/90.png").unlink()
    with pytest.raises(SystemExit, match="baseline artifacts missing"):
        accept_run.main([str(run)])
    assert not (tmp_path / "_accepted").exists()


def test_rederives_manifest_from_receipts(tmp_path):
    """A hand-edited manifest.json is overwritten with receipt truth."""
    run = make_completed_run(tmp_path, "a", all_pass=False)
    forged = schema.read_validated(run / "manifest.json", "manifest")
    forged["shippable"] = True
    forged["gate_summary"]["all_pass"] = True
    forged["gate_summary"]["passed"] = 5
    schema.write_validated(run / "manifest.json", forged, "manifest")
    with pytest.raises(SystemExit, match="shippable=false"):
        accept_run.main([str(run)])
    fixed = schema.read_validated(run / "manifest.json", "manifest")
    assert fixed["shippable"] is False


def test_replaces_previous_baseline(tmp_path):
    run_a = make_completed_run(tmp_path, "a")
    run_b = make_completed_run(tmp_path, "b")
    accept_run.main([str(run_a)])
    accept_run.main([str(run_b)])
    acc = tmp_path / "_accepted"
    rec = schema.read_validated(acc / "accepted.json", "accepted")
    assert rec["source_run"] == "b"
    assert rec["manifest_hash"] == manifest.manifest_hash(run_b)
    for y in YAWS:
        assert (acc / f"s8_review/out/thumbs/{y}.png").read_bytes() == (
            run_b / f"s8_review/out/thumbs/{y}.png"
        ).read_bytes()
    assert not (tmp_path / "_accepted.incoming").exists()


def test_refuses_self_promotion(tmp_path):
    run = make_completed_run(tmp_path, "a")
    accept_run.main([str(run)])
    with pytest.raises(SystemExit, match="onto itself"):
        accept_run.main([str(tmp_path / "_accepted")])


def test_refuses_tampered_artifact(tmp_path):
    """verify_disk: a recorded output modified after the run refuses."""
    run = make_completed_run(tmp_path, "a")
    (run / "s8_review/out/index.html").write_text("<html>tampered</html>\n")
    with pytest.raises(SystemExit, match="does not match the file on disk"):
        accept_run.main([str(run)])
    assert not (tmp_path / "_accepted").exists()


def test_trailing_slash_and_dotdot_resolved(tmp_path):
    run = make_completed_run(tmp_path, "a")
    assert accept_run.main([str(run) + "/"]) == 0
    rec = schema.read_validated(tmp_path / "_accepted/accepted.json", "accepted")
    assert rec["source_run"] == "a"
    # '..' resolves away instead of naming staging inside the run: the
    # resolved dir (tmp_path) has no receipts, so this refuses cleanly
    # instead of recursively copytree-ing into the run itself.
    with pytest.raises(SystemExit, match="incomplete receipt chain"):
        accept_run.main([str(run) + "/.."])
    assert not any(p.name.startswith("_accepted.incoming") for p in run.iterdir())
