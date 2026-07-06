"""Assert two runs produced bit-identical manifests (and report divergence
stage-by-stage if not)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scenic import hashing, manifest  # noqa: E402


def main(a: str, b: str) -> int:
    ha = manifest.manifest_hash(Path(a))
    hb = manifest.manifest_hash(Path(b))
    if ha == hb:
        print(f"DETERMINISM OK: manifest {ha}")
        return 0
    print(f"DETERMINISM FAILED: {ha} != {hb}")
    ma = hashing.read_json(Path(a) / "manifest.json")
    mb = hashing.read_json(Path(b) / "manifest.json")
    for ra, rb in zip(ma["stages"], mb["stages"]):
        if hashing.sha256_json(ra) == hashing.sha256_json(rb):
            continue
        print(f"  stage {ra['stage']} diverges:")
        for kind in ("inputs", "outputs"):
            for k in sorted(set(ra[kind]) | set(rb[kind])):
                va = ra[kind].get(k, {}).get("sha256")
                vb = rb[kind].get(k, {}).get("sha256")
                if va != vb:
                    print(f"    {kind}.{k}: {va} != {vb}")
        if ra.get("notes") != rb.get("notes"):
            print("    notes differ")
        if ra.get("gates") != rb.get("gates"):
            print("    gates differ")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
