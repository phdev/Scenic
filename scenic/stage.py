"""Stage context passed by the harness to every stage's run()."""
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
