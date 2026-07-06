"""Static no-network guard (CI step, after license_guard).

Contract (docs/CONTRACTS.md, CLAUDE.md): no network at stage runtime. This
guard AST-parses every .py under pipeline/, gates/, and scenic/ and flags any
Import / ImportFrom of network-capable modules: urllib (any submodule),
requests, http.client, and socket.

Exemptions:
- scenic/weights.py (weight loading may reference offline-mode plumbing)
- tools/ is not scanned at all (fetch_weights.py legitimately downloads at
  setup time)

AST-based on purpose: string grep false-positives on comments/docstrings and
on license/schema URLs; an import is the actual capability we forbid.

Exit nonzero on findings; hard errors (unparseable file) also fail.

    uv run python tools/check_no_network.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories scanned for forbidden imports. gates/ may not exist yet while
# stages land in parallel; missing dirs are skipped (not an error).
SCAN_DIRS = ("pipeline", "gates", "scenic")

EXEMPT = {Path("scenic") / "weights.py"}

# Forbidden import roots. A dotted module matches if it equals the entry or
# starts with "<entry>.", so `urllib` catches urllib.request etc., while
# `http.client` catches http.client but not http.server (add if ever needed).
FORBIDDEN = ("urllib", "requests", "http.client", "socket")


def _matches(module: str) -> str | None:
    for f in FORBIDDEN:
        if module == f or module.startswith(f + "."):
            return f
    return None


def _check_file(path: Path) -> list[str]:
    rel = path.relative_to(REPO_ROOT)
    tree = ast.parse(path.read_text(), filename=str(rel))
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                hit = _matches(alias.name)
                if hit:
                    findings.append(
                        f"{rel}:{node.lineno}: import {alias.name} (forbidden: {hit})"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — always in-repo, never stdlib net
                continue
            module = node.module or ""
            hit = _matches(module)
            if hit:
                findings.append(
                    f"{rel}:{node.lineno}: from {module} import ... (forbidden: {hit})"
                )
                continue
            # `from urllib import request`, `from http import client`
            for alias in node.names:
                full = f"{module}.{alias.name}" if module else alias.name
                hit = _matches(full)
                if hit:
                    findings.append(
                        f"{rel}:{node.lineno}: from {module} import {alias.name}"
                        f" (forbidden: {hit})"
                    )
    return findings


def main() -> int:
    findings: list[str] = []
    scanned = 0
    for d in SCAN_DIRS:
        root = REPO_ROOT / d
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if path.relative_to(REPO_ROOT) in EXEMPT:
                continue
            scanned += 1
            findings.extend(_check_file(path))
    if findings:
        print(f"NO-NETWORK GUARD FAILED ({len(findings)} finding(s)):")
        for f in findings:
            print(f"  {f}")
        return 1
    print(f"no-network guard OK ({scanned} files scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
