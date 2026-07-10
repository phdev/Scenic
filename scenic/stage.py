"""Stage context passed by the harness to every stage's run().

Ctx.out only ensures the dir exists; CLEARING stale state (receipt.json and
the out/ tree) is the harness's job (scenic/run.py), done once per stage
invocation — Ctx.out may be called repeatedly within a stage (e.g. s6's
re-entrant s4 re-run) and must never wipe in-progress work."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Ctx:
    repo_root: Path
    pano_path: Path            # source pano (outside run dir)
    sidecar_path: Path         # <pano>.license.json
    params_path: Path
    weights_dir: Path
    extras: dict = field(default_factory=dict)

    def out(self, run_dir: Path, stage: str) -> Path:
        d = Path(run_dir) / stage / "out"
        d.mkdir(parents=True, exist_ok=True)
        return d
