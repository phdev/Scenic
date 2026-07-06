"""Unit tests for scenic.receipts + scenic.manifest against a fake run dir.

pipeline.registry is imported (static data only); stage MODULES are never
imported — manifest.build on an incomplete run dir raises before needing any.
"""
from __future__ import annotations

import jsonschema
import pytest

from scenic import hashing, manifest, receipts, schema

GOOD_GATE = {
    "gate": "budgets",
    "pass": True,
    "metrics": {"splat_count": 1000},
    "thresholds": {"splat_cap": 1000000},
    "details": "under cap",
}


def _fake_run(tmp_path):
    """run dir with one internal input, one output, one external input."""
    run_dir = tmp_path / "run"
    external = tmp_path / "pano.png"  # OUTSIDE run_dir
    external.write_bytes(b"not really a png")
    out_dir = run_dir / "s0_ingest" / "out"
    out_dir.mkdir(parents=True)
    internal_in = run_dir / "params.snapshot.yaml"
    internal_in.write_text("seed: 0\n")
    output = out_dir / "pano.png"
    output.write_bytes(b"normalized pano bytes")
    return run_dir, external, internal_in, output


def _write_s0(run_dir, external, internal_in, output, gates=None):
    return receipts.write_receipt(
        run_dir,
        "s0_ingest",
        inputs={"pano": external, "params_snapshot": internal_in},
        outputs={"pano_master": output},
        params_used={"s0": {"person_score_min": 0.5}},
        gates=gates,
    )


# ---------------------------------------------------------------- write_receipt

def test_write_receipt_relative_paths_and_hashes(tmp_path):
    run_dir, external, internal_in, output = _fake_run(tmp_path)
    rec = _write_s0(run_dir, external, internal_in, output)

    # internal paths recorded relative to run_dir (posix), no absolute paths
    assert rec["outputs"]["pano_master"]["path"] == "s0_ingest/out/pano.png"
    assert rec["inputs"]["params_snapshot"]["path"] == "params.snapshot.yaml"
    # external inputs become external/<name>
    assert rec["inputs"]["pano"]["path"] == "external/pano.png"
    for m in (rec["inputs"], rec["outputs"]):
        for v in m.values():
            assert not v["path"].startswith("/")

    # sha256s are the real file hashes
    assert rec["inputs"]["pano"]["sha256"] == hashing.sha256_file(external)
    assert rec["inputs"]["params_snapshot"]["sha256"] == hashing.sha256_file(
        internal_in
    )
    assert rec["outputs"]["pano_master"]["sha256"] == hashing.sha256_file(output)

    # params_used echoed + hashed canonically
    assert rec["params_used"] == {"s0": {"person_score_min": 0.5}}
    assert rec["params_hash"] == hashing.sha256_json(rec["params_used"])
    assert rec["weights"] == []
    assert rec["gates"] == [] and rec["notes"] == {}


def test_write_receipt_validates_and_roundtrips(tmp_path):
    run_dir, external, internal_in, output = _fake_run(tmp_path)
    rec = _write_s0(run_dir, external, internal_in, output)
    p = run_dir / "s0_ingest" / "receipt.json"
    assert p.exists()
    schema.validate(rec, "receipt")  # returned dict is schema-valid
    assert receipts.read_receipt(run_dir, "s0_ingest") == rec
    # on-disk form is the canonical write_json layout (deterministic bytes)
    rewritten = tmp_path / "again.json"
    hashing.write_json(rewritten, rec)
    assert rewritten.read_bytes() == p.read_bytes()


def test_write_receipt_bad_stage_name_rejected_by_schema(tmp_path):
    run_dir, external, internal_in, output = _fake_run(tmp_path)
    with pytest.raises(jsonschema.ValidationError):
        receipts.write_receipt(
            run_dir,
            "S0-Ingest",  # violates ^s[0-9]+b?_[a-z_]+$
            inputs={},
            outputs={"pano_master": output},
            params_used={},
        )


# ---------------------------------------------------------------- gates

def test_gates_recorded_in_receipt(tmp_path):
    run_dir, external, internal_in, output = _fake_run(tmp_path)
    rec = _write_s0(run_dir, external, internal_in, output, gates=[GOOD_GATE])
    assert rec["gates"] == [GOOD_GATE]
    assert receipts.read_receipt(run_dir, "s0_ingest")["gates"] == [GOOD_GATE]


@pytest.mark.parametrize(
    "bad_gate",
    [
        {"gate": "budgets", "pass": True, "metrics": {}},  # missing thresholds
        {"gate": "not_a_real_gate", "pass": True, "metrics": {}, "thresholds": {}},
        {"gate": "hole", "pass": "yes", "metrics": {}, "thresholds": {}},  # bad type
        {"gate": "hole", "pass": True, "metrics": {}, "thresholds": {},
         "extra_key": 1},  # additionalProperties: false
    ],
)
def test_bad_gate_dict_raises(tmp_path, bad_gate):
    run_dir, external, internal_in, output = _fake_run(tmp_path)
    with pytest.raises(jsonschema.ValidationError):
        _write_s0(run_dir, external, internal_in, output, gates=[bad_gate])
    # nothing written on failure
    assert not (run_dir / "s0_ingest" / "receipt.json").exists()


# ---------------------------------------------------------------- manifest

def test_manifest_build_raises_on_missing_stage_receipt(tmp_path):
    run_dir, external, internal_in, output = _fake_run(tmp_path)
    _write_s0(run_dir, external, internal_in, output, gates=[GOOD_GATE])
    # registry lists 9 stages; only s0_ingest has a receipt -> unshippable
    with pytest.raises(RuntimeError, match="missing"):
        manifest.build(run_dir)
    assert not (run_dir / "manifest.json").exists()


def test_manifest_build_raises_on_empty_run_dir(tmp_path):
    empty = tmp_path / "empty_run"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="incomplete receipt chain"):
        manifest.build(empty)


def test_stage_order_matches_registry():
    order = manifest.stage_order()
    assert order[0] == "s0_ingest"
    assert "s5" not in " ".join(order)  # s5 reserved, never registered
    assert len(order) == len(set(order))
    from pipeline.registry import STAGES

    assert order == [name for name, _ in STAGES]


def test_manifest_build_single_stage(tmp_path, monkeypatch):
    run_dir, external, internal_in, output = _fake_run(tmp_path)
    _write_s0(run_dir, external, internal_in, output, gates=[GOOD_GATE])
    monkeypatch.setattr(manifest, "stage_order", lambda: ["s0_ingest"])
    m = manifest.build(run_dir)
    assert (run_dir / "manifest.json").exists()
    schema.validate(m, "manifest")
    assert m["schema"] == "scenic-manifest-v1"
    assert len(m["stages"]) == 1 and m["stages"][0]["stage"] == "s0_ingest"
    assert m["gate_summary"] == {"total": 1, "passed": 1, "all_pass": True}
    assert m["shippable"] is True
    assert len(manifest.manifest_hash(run_dir)) == 64


def test_manifest_failing_gate_not_shippable(tmp_path, monkeypatch):
    run_dir, external, internal_in, output = _fake_run(tmp_path)
    failing = dict(GOOD_GATE, **{"pass": False})
    _write_s0(run_dir, external, internal_in, output, gates=[GOOD_GATE, failing])
    monkeypatch.setattr(manifest, "stage_order", lambda: ["s0_ingest"])
    m = manifest.build(run_dir)
    assert m["gate_summary"] == {"total": 2, "passed": 1, "all_pass": False}
    assert m["shippable"] is False


def test_manifest_no_gates_not_shippable(tmp_path, monkeypatch):
    run_dir, external, internal_in, output = _fake_run(tmp_path)
    _write_s0(run_dir, external, internal_in, output)  # zero gates
    monkeypatch.setattr(manifest, "stage_order", lambda: ["s0_ingest"])
    m = manifest.build(run_dir)
    assert m["gate_summary"] == {"total": 0, "passed": 0, "all_pass": False}
    assert m["shippable"] is False
