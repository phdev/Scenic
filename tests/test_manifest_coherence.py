"""Tests for manifest.build chain-coherence checking.

Existence of every receipt is necessary but not sufficient: a single-stage
re-run or a hand-edit can leave receipts that disagree about the same
artifact. build() must refuse stale mixed chains, receipts parked in the
wrong stage dir, inputs with no recorded producer, and (verify_disk=True)
outputs that no longer match the files on disk.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scenic import manifest, receipts, schema  # noqa: E402


def _verdict(ok: bool = True) -> dict:
    return {
        "gate": "hole",
        "pass": ok,
        "metrics": {"m": 0.0},
        "thresholds": {"t": 1.0},
        "details": "synthetic",
    }


def make_chain(d: Path, producer_records_output: bool = True) -> Path:
    """A full receipt chain where s0 produces a file and s2 consumes it.

    All other stages have empty inputs/outputs; s7 carries one passing gate
    so the manifest is shippable."""
    run = d / "run"
    f = run / "s0_ingest" / "out" / "blob.bin"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"artifact-v1")
    for stage in manifest.stage_order():
        inputs: dict = {}
        outputs: dict = {}
        gates: list = []
        if stage == "s0_ingest" and producer_records_output:
            outputs = {"blob": f}
        if stage == "s2_depth":
            inputs = {"blob": f}
        if stage == "s7_gates":
            gates = [_verdict()]
        receipts.write_receipt(
            run, stage, inputs=inputs, outputs=outputs, params_used={},
            gates=gates,
        )
    return run


def test_coherent_chain_builds(tmp_path):
    run = make_chain(tmp_path)
    m = manifest.build(run)
    assert m["shippable"] is True


def test_stale_chain_refused(tmp_path):
    """Producer's recorded output hash != consumer's recorded input hash."""
    run = tmp_path / "run"
    f = run / "s0_ingest" / "out" / "blob.bin"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"artifact-v1")
    for stage in manifest.stage_order():
        if stage == "s2_depth":
            f.write_bytes(b"artifact-v2")  # mutated between receipts
            receipts.write_receipt(
                run, stage, inputs={"blob": f}, outputs={}, params_used={}
            )
            continue
        outputs = {"blob": f} if stage == "s0_ingest" else {}
        receipts.write_receipt(
            run, stage, inputs={}, outputs=outputs, params_used={}
        )
    with pytest.raises(RuntimeError, match="hash differs"):
        manifest.build(run)


def test_input_without_producer_refused(tmp_path):
    run = make_chain(tmp_path, producer_records_output=False)
    with pytest.raises(RuntimeError, match="no recorded producer"):
        manifest.build(run)


def test_receipt_in_wrong_stage_dir_refused(tmp_path):
    run = make_chain(tmp_path)
    shutil.copyfile(
        run / "s0_ingest" / "receipt.json",
        run / "s1_cleanplate" / "receipt.json",
    )
    with pytest.raises(RuntimeError, match="declares stage"):
        manifest.build(run)


def test_verify_disk_catches_tampered_output(tmp_path):
    run = make_chain(tmp_path)
    manifest.build(run)  # coherent as recorded
    (run / "s0_ingest" / "out" / "blob.bin").write_bytes(b"tampered")
    manifest.build(run)  # cross-receipt only: still coherent
    with pytest.raises(RuntimeError, match="does not match the file on disk"):
        manifest.build(run, verify_disk=True)


def test_verify_disk_catches_deleted_output(tmp_path):
    run = make_chain(tmp_path)
    (run / "s0_ingest" / "out" / "blob.bin").unlink()
    with pytest.raises(RuntimeError, match="does not match the file on disk"):
        manifest.build(run, verify_disk=True)


def test_run_root_params_snapshot_is_allowed_input(tmp_path):
    """params.snapshot.yaml is a legitimate producer-less run-root input."""
    run = tmp_path / "run"
    snap = run / "params.snapshot.yaml"
    run.mkdir(parents=True)
    snap.write_text("seed: 0\n")
    for stage in manifest.stage_order():
        inputs = {"params": snap} if stage == "s0_ingest" else {}
        gates = [_verdict()] if stage == "s7_gates" else []
        receipts.write_receipt(
            run, stage, inputs=inputs, outputs={}, params_used={}, gates=gates
        )
    m = manifest.build(run)
    assert m["shippable"] is True


def test_non_boolean_gate_pass_rejected_by_receipt_schema():
    """A schema-valid receipt can no longer smuggle pass:'FAIL' (truthy)."""
    rec = {
        "stage": "s7_gates",
        "code": {"git_sha": "unknown", "dirty": True},
        "inputs": {},
        "outputs": {},
        "params_used": {},
        "params_hash": "0" * 64,
        "weights": [],
        "gates": [
            {
                "gate": "hole",
                "pass": "FAIL",
                "metrics": {},
                "thresholds": {},
                "details": "x",
            }
        ],
        "notes": {},
    }
    with pytest.raises(Exception, match="'FAIL' is not of type 'boolean'"):
        schema.validate(rec, "receipt")


def test_traversal_and_backslash_paths_rejected_by_schema():
    base = {
        "stage": "s0_ingest",
        "code": {"git_sha": "unknown", "dirty": True},
        "inputs": {},
        "outputs": {},
        "params_used": {},
        "params_hash": "0" * 64,
        "weights": [],
        "gates": [],
        "notes": {},
    }
    for bad in ("../escape.bin", "a/../../b.bin", "C:\\evil.bin", "a\\b.bin"):
        rec = dict(base)
        rec["outputs"] = {"f": {"path": bad, "sha256": "0" * 64}}
        with pytest.raises(Exception):
            schema.validate(rec, "receipt")
    ok = dict(base)
    ok["outputs"] = {"f": {"path": "s0_ingest/out/pano.png", "sha256": "0" * 64}}
    schema.validate(ok, "receipt")
