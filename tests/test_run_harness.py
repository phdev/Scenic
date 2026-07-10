"""Tests for the scenic.run harness guarantees.

The harness must prove each receipt comes from THIS invocation (stale
receipts cleared before the stage runs), start every stage with a clean
out/ dir, refuse --only re-runs under different params, and remove the
now-stale manifest after an --only re-run.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pipeline.registry as registry  # noqa: E402
from scenic import receipts  # noqa: E402
from scenic import run as run_mod  # noqa: E402

STAGE = "s0_ingest"
PANO = REPO_ROOT / "fixtures" / "test.jpg"
PARAMS = REPO_ROOT / "params.yaml"


def _fake_stage(writes_receipt: bool = True) -> types.SimpleNamespace:
    def _run(run_dir: Path, params: dict, ctx) -> None:
        out = ctx.out(run_dir, STAGE)
        (out / "made.txt").write_text("x")
        if writes_receipt:
            receipts.write_receipt(
                run_dir, STAGE, inputs={}, outputs={}, params_used={}
            )

    return types.SimpleNamespace(run=_run)


@pytest.fixture()
def single_stage(monkeypatch):
    stage = _fake_stage()
    monkeypatch.setattr(registry, "STAGES", [(STAGE, "fake")])
    monkeypatch.setattr(registry, "get_stage", lambda name: stage)
    return stage


def test_stale_receipt_does_not_mask_a_receiptless_stage(
    tmp_path, monkeypatch
):
    stage = _fake_stage(writes_receipt=False)
    monkeypatch.setattr(registry, "STAGES", [(STAGE, "fake")])
    monkeypatch.setattr(registry, "get_stage", lambda name: stage)
    out = tmp_path / "run"
    # a receipt left by a previous invocation
    receipts.write_receipt(out, STAGE, inputs={}, outputs={}, params_used={})
    with pytest.raises(RuntimeError, match="did not write a receipt"):
        run_mod.run_pipeline(PANO, out, PARAMS)


def test_stage_out_dir_cleared_before_run(tmp_path, single_stage):
    out = tmp_path / "run"
    stale = out / STAGE / "out" / "stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("debris from an older code version")
    run_mod.run_pipeline(PANO, out, PARAMS)
    assert not stale.exists()
    assert (out / STAGE / "out" / "made.txt").exists()


def test_only_refuses_params_mismatch(tmp_path, single_stage):
    out = tmp_path / "run"
    out.mkdir()
    (out / "params.snapshot.yaml").write_text("seed: 999  # different\n")
    with pytest.raises(SystemExit, match="params differ"):
        run_mod.run_pipeline(PANO, out, PARAMS, only=STAGE)


def test_only_removes_stale_manifest(tmp_path, single_stage):
    out = tmp_path / "run"
    run_mod.run_pipeline(PANO, out, PARAMS)  # full run writes manifest.json
    assert (out / "manifest.json").exists()
    run_mod.run_pipeline(PANO, out, PARAMS, only=STAGE)
    assert not (out / "manifest.json").exists()
