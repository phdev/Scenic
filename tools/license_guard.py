"""CI license guard. Fails unless:
- every file under weights/ (except LICENSES.md, pins.json) is pinned in
  weights/pins.json with an allowed license AND documented in LICENSES.md;
- no forbidden deps (AGPL family / known-NC packages) in the lockfile.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scenic import hashing, schema  # noqa: E402

ALLOWED = {"Apache-2.0", "MIT", "BSD-3-Clause"}
FORBIDDEN_PACKAGES = {"ultralytics", "yolov5", "yolov8"}  # AGPL family
META = {"LICENSES.md", "pins.json", ".gitkeep"}


def main() -> int:
    errors: list[str] = []
    wdir = REPO_ROOT / "weights"
    pins = schema.read_validated(wdir / "pins.json", "pins")
    licenses_md = (wdir / "LICENSES.md").read_text() if (wdir / "LICENSES.md").exists() else ""

    pinned: dict[Path, str] = {}
    for key, pin in pins.items():
        if pin["license"] not in ALLOWED:
            errors.append(f"pins.json: {key} license {pin['license']} not allowed")
        if key not in licenses_md or pin["repo"] not in licenses_md:
            errors.append(f"LICENSES.md: missing entry for {key} ({pin['repo']})")
        for rel, want in pin["files"].items():
            pinned[wdir / key / rel] = want

    for f in sorted(wdir.rglob("*")):
        if not f.is_file() or f.name in META:
            continue
        if f not in pinned:
            errors.append(f"unpinned file in weights/: {f.relative_to(REPO_ROOT)}")
        elif f.exists():
            got = hashing.sha256_file(f)
            if got != pinned[f]:
                errors.append(f"hash mismatch: {f.relative_to(REPO_ROOT)}")

    lock = REPO_ROOT / "uv.lock"
    if lock.exists():
        text = lock.read_text()
        for pkg in sorted(FORBIDDEN_PACKAGES):
            if re.search(rf'name = "{re.escape(pkg)}"', text):
                errors.append(f"forbidden dependency in uv.lock: {pkg}")
    else:
        errors.append("uv.lock missing")

    if errors:
        print("LICENSE GUARD FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"license guard OK: {len(pins)} pinned weight sets, all {sorted(ALLOWED)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
