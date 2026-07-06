"""Run manifest: ordered aggregation of every stage receipt. An incomplete
chain raises — unshippable by definition."""
from __future__ import annotations

from pathlib import Path

from scenic import hashing, schema


def stage_order() -> list[str]:
    from pipeline.registry import STAGES

    return [name for name, _ in STAGES]


def build(run_dir: Path) -> dict:
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
    gates = [g for r in receipts.values() for g in r["gates"]]
    manifest = {
        "schema": "scenic-manifest-v1",
        "stages": [receipts[n] for n in stage_order()],
        "gate_summary": {
            "total": len(gates),
            "passed": sum(1 for g in gates if g["pass"]),
            "all_pass": all(g["pass"] for g in gates) if gates else False,
        },
        "shippable": bool(gates) and all(g["pass"] for g in gates),
    }
    schema.write_validated(run_dir / "manifest.json", manifest, "manifest")
    return manifest


def manifest_hash(run_dir: Path) -> str:
    m = schema.read_validated(Path(run_dir) / "manifest.json", "manifest")
    return hashing.sha256_json(m)
