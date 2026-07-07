"""Promote a completed run to the runs/_accepted baseline used by s8_review.

The baseline is a SLIM snapshot, not a full run copy: s8_review/ (receipt +
out/ — exactly what s8's side-by-side comparison contract reads), the run's
manifest.json and params.snapshot.yaml for provenance, plus accepted.json
(schema `accepted`: source run name, manifest hash, gate summary). No
timestamps, no absolute paths.

Integrity: the manifest is RE-DERIVED from the stage receipts via
scenic.manifest.build before promoting — receipts are the source of truth,
so an incomplete receipt chain refuses and a hand-edited manifest.json is
overwritten with what the receipts actually say. A run with failing gates
(shippable=false) is refused unless --allow-failed-gates; gates record
verdicts, humans decide.

The swap is staged: the snapshot is built in _accepted.incoming, then the
old _accepted is removed and the staging dir renamed into place.

Usage: uv run python tools/accept_run.py runs/<name> [--allow-failed-gates]
       (or: make accept RUN=runs/<name> [FORCE=1])
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scenic import manifest, schema  # noqa: E402

ACCEPTED_NAME = "_accepted"
STAGING_NAME = "_accepted.incoming"
THUMB_YAWS = (0, 90, 180, 270)  # s8_review's standard views


def check_run(run_dir: Path) -> dict:
    """Validate the run is promotable; return its (re-derived) manifest."""
    if not run_dir.is_dir():
        raise SystemExit(f"not a run dir: {run_dir}")
    try:
        man = manifest.build(run_dir)
    except RuntimeError as e:  # incomplete receipt chain
        raise SystemExit(f"refusing to accept {run_dir.name}: {e}") from e

    s8_out = run_dir / "s8_review" / "out"
    required = [s8_out / "review.json", s8_out / "index.html"]
    required += [s8_out / "thumbs" / f"{y}.png" for y in THUMB_YAWS]
    missing = [p for p in required if not p.exists()]
    if missing:
        names = ", ".join(str(p.relative_to(run_dir)) for p in missing)
        raise SystemExit(
            f"refusing to accept {run_dir.name}: s8 baseline artifacts "
            f"missing ({names})"
        )
    schema.read_validated(s8_out / "review.json", "review")
    return man


def promote(run_dir: Path, man: dict) -> Path:
    """Stage the slim baseline snapshot, then swap it into _accepted."""
    parent = run_dir.parent
    dest = parent / ACCEPTED_NAME
    staging = parent / STAGING_NAME
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    shutil.copytree(run_dir / "s8_review", staging / "s8_review")
    shutil.copyfile(run_dir / "manifest.json", staging / "manifest.json")
    snap = run_dir / "params.snapshot.yaml"
    if snap.exists():
        shutil.copyfile(snap, staging / "params.snapshot.yaml")

    record = {
        "schema": "scenic-accepted-v1",
        "source_run": run_dir.name,
        "manifest_hash": manifest.manifest_hash(run_dir),
        "shippable": man["shippable"],
        "gate_summary": man["gate_summary"],
    }
    schema.write_validated(staging / "accepted.json", record, "accepted")

    if dest.exists():
        shutil.rmtree(dest)
    staging.rename(dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Promote a completed run to the runs/_accepted baseline."
    )
    ap.add_argument("run_dir", type=Path, help="run directory, e.g. runs/machu2")
    ap.add_argument(
        "--allow-failed-gates",
        action="store_true",
        help="promote even if shippable=false (failing gates)",
    )
    args = ap.parse_args(argv)
    run_dir = args.run_dir
    if run_dir.name in (ACCEPTED_NAME, STAGING_NAME):
        raise SystemExit("refusing to promote the baseline onto itself")

    man = check_run(run_dir)
    gs = man["gate_summary"]
    print(
        f"run {run_dir.name}: gates {gs['passed']}/{gs['total']} passed, "
        f"shippable={str(man['shippable']).lower()}"
    )
    if not man["shippable"] and not args.allow_failed_gates:
        raise SystemExit(
            "refusing to accept: shippable=false. Re-run with "
            "--allow-failed-gates (make accept FORCE=1) to promote anyway."
        )

    dest = promote(run_dir, man)
    print(f"ACCEPTED: {run_dir.name} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
