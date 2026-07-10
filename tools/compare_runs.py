"""Assert two runs produced bit-identical manifests (and report divergence
stage-by-stage if not)."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scenic import hashing, manifest  # noqa: E402

# Receipt fields diffed per stage when manifests diverge. inputs/outputs get
# per-key hash reporting; the rest are compared as canonical JSON.
_SCALAR_FIELDS = ("params_used", "params_hash", "weights", "gates", "notes", "code")


def _diff_stage(ra: dict, rb: dict) -> None:
    print(f"  stage {ra['stage']} diverges:")
    for kind in ("inputs", "outputs"):
        for k in sorted(set(ra[kind]) | set(rb[kind])):
            va = ra[kind].get(k, {}).get("sha256")
            vb = rb[kind].get(k, {}).get("sha256")
            if va != vb:
                print(f"    {kind}.{k}: {va} != {vb}")
    for field in _SCALAR_FIELDS:
        if hashing.canonical_json(ra.get(field)) != hashing.canonical_json(
            rb.get(field)
        ):
            print(f"    {field} differs")


def main(a: str, b: str) -> int:
    ha = manifest.manifest_hash(Path(a))
    hb = manifest.manifest_hash(Path(b))
    if ha == hb:
        print(f"DETERMINISM OK: manifest {ha}")
        return 0
    print(f"DETERMINISM FAILED: {ha} != {hb}")
    ma = hashing.read_json(Path(a) / "manifest.json")
    mb = hashing.read_json(Path(b) / "manifest.json")
    if len(ma["stages"]) != len(mb["stages"]):
        print(
            f"  stage count differs: {len(ma['stages'])} != {len(mb['stages'])}"
        )
    for ra, rb in itertools.zip_longest(ma["stages"], mb["stages"]):
        if ra is None or rb is None:
            present = ra or rb
            print(f"  stage {present['stage']} present in only one run")
            continue
        if hashing.sha256_json(ra) == hashing.sha256_json(rb):
            continue
        _diff_stage(ra, rb)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
