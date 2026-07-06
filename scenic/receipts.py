"""Per-stage provenance receipts. A receipt records exactly what a stage
consumed and produced. No timestamps, no absolute paths — receipts must be
byte-identical across identical runs."""
from __future__ import annotations

import subprocess
from pathlib import Path

from scenic import hashing, schema


def git_state(repo_root: Path) -> dict:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True
    )
    return {
        "git_sha": sha.stdout.strip() if sha.returncode == 0 else "unknown",
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else True,
    }


def _rel(path: Path, run_dir: Path) -> str:
    p = Path(path).resolve()
    run_dir = run_dir.resolve()
    try:
        return p.relative_to(run_dir).as_posix()
    except ValueError:
        # inputs outside the run dir (source pano, params file) keep name only
        return f"external/{p.name}"


def write_receipt(
    run_dir: Path,
    stage_name: str,
    *,
    inputs: dict[str, Path],
    outputs: dict[str, Path],
    params_used: dict,
    weights_used: list[str] | None = None,
    gates: list[dict] | None = None,
    notes: dict | None = None,
) -> dict:
    from scenic import weights as weights_mod

    run_dir = Path(run_dir)
    wrec = []
    for key in weights_used or []:
        pin = weights_mod.load_pins()[key]
        wrec.append(
            {
                "key": key,
                "repo": pin["repo"],
                "license": pin["license"],
                "license_url": pin["license_url"],
                "files_sha256": pin["files"],
            }
        )
    repo_root = Path(__file__).resolve().parent.parent
    receipt = {
        "stage": stage_name,
        "code": git_state(repo_root),
        "inputs": {
            k: {"path": _rel(v, run_dir), "sha256": hashing.sha256_file(v)}
            for k, v in sorted(inputs.items())
        },
        "outputs": {
            k: {"path": _rel(v, run_dir), "sha256": hashing.sha256_file(v)}
            for k, v in sorted(outputs.items())
        },
        "params_used": params_used,
        "params_hash": hashing.sha256_json(params_used),
        "weights": wrec,
        "gates": gates or [],
        "notes": notes or {},
    }
    for g in receipt["gates"]:
        schema.validate(g, "gate_verdict")
    stage_dir = run_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    schema.write_validated(stage_dir / "receipt.json", receipt, "receipt")
    return receipt


def read_receipt(run_dir: Path, stage_name: str) -> dict:
    return schema.read_validated(Path(run_dir) / stage_name / "receipt.json", "receipt")
