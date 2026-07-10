"""Run manifest: ordered aggregation of every stage receipt. An incomplete
OR incoherent chain raises — unshippable by definition.

Coherence is checked, not assumed: receipt existence alone would accept a
stale mixed chain (e.g. a single-stage `--only` re-run that rewrote one
stage's artifacts while downstream receipts still attest the old bytes).
build() therefore also verifies that every in-run-dir input a stage recorded
carries the same sha256 the producing stage recorded for that path, and
(with verify_disk=True, used at promotion boundaries) that every recorded
output still matches the file on disk."""
from __future__ import annotations

from pathlib import Path

from scenic import hashing, schema

# Run-root files stages may legitimately read that no stage produces.
_RUN_ROOT_INPUTS = ("params.snapshot.yaml",)


def stage_order() -> list[str]:
    from pipeline.registry import STAGES

    return [name for name, _ in STAGES]


def _check_coherence(
    run_dir: Path, receipts: dict[str, dict], verify_disk: bool
) -> None:
    problems: list[str] = []
    for name, rec in receipts.items():
        if rec["stage"] != name:
            problems.append(
                f"{name}/receipt.json declares stage {rec['stage']!r}"
            )
    produced: dict[str, tuple[str, str]] = {}  # rel path -> (stage, sha256)
    for name in stage_order():
        rec = receipts[name]
        for key in sorted(rec["inputs"]):
            ent = rec["inputs"][key]
            path = ent["path"]
            if path.startswith("external/") or path in _RUN_ROOT_INPUTS:
                continue
            prod = produced.get(path)
            if prod is None:
                problems.append(
                    f"{name} input {key} ({path}) has no recorded producer"
                )
            elif prod[1] != ent["sha256"]:
                problems.append(
                    f"{name} input {key} ({path}) hash differs from the "
                    f"{prod[0]} output (stale chain?)"
                )
        for key in sorted(rec["outputs"]):
            ent = rec["outputs"][key]
            produced[ent["path"]] = (name, ent["sha256"])
            if verify_disk:
                f = run_dir / ent["path"]
                if not f.exists() or hashing.sha256_file(f) != ent["sha256"]:
                    problems.append(
                        f"{name} output {key} ({ent['path']}) does not match "
                        f"the file on disk"
                    )
    if problems:
        raise RuntimeError(
            "incoherent receipt chain (unshippable): " + "; ".join(problems)
        )


def build(run_dir: Path, verify_disk: bool = False) -> dict:
    run_dir = Path(run_dir)
    receipts = {}
    missing = []
    for name in stage_order():
        p = run_dir / name / "receipt.json"
        if not p.exists():
            missing.append(name)
            continue
        receipts[name] = schema.read_validated(p, "receipt")
    if missing:
        raise RuntimeError(
            f"incomplete receipt chain (unshippable): missing {missing}"
        )
    _check_coherence(run_dir, receipts, verify_disk)
    gates = [g for r in receipts.values() for g in r["gates"]]
    manifest = {
        "schema": "scenic-manifest-v1",
        "stages": [receipts[n] for n in stage_order()],
        "gate_summary": {
            "total": len(gates),
            "passed": sum(1 for g in gates if g["pass"] is True),
            "all_pass": all(g["pass"] is True for g in gates) if gates else False,
        },
        "shippable": bool(gates) and all(g["pass"] is True for g in gates),
    }
    schema.write_validated(run_dir / "manifest.json", manifest, "manifest")
    return manifest


def manifest_hash(run_dir: Path) -> str:
    m = schema.read_validated(Path(run_dir) / "manifest.json", "manifest")
    return hashing.sha256_json(m)
