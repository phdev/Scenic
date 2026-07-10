"""Static no-network guard (CI step, after license_guard).

Contract (docs/CONTRACTS.md, CLAUDE.md): no network at stage runtime. This
guard AST-parses every .py under pipeline/, gates/, and scenic/ and flags:

- Import / ImportFrom of network-capable modules (FORBIDDEN below): the
  stdlib clients (urllib, http — client AND server, socket, socketserver,
  ftplib, smtplib, poplib, imaplib, telnetlib, xmlrpc) plus the third-party
  clients installed or installable via the lockfile (urllib3, requests,
  huggingface_hub, aiohttp, httpx, pycurl, websockets). urllib3 matters: it
  is a transitive dep in uv.lock, imports fine, and is NOT neutralized by
  HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE.
- Dynamic imports with LITERAL names: importlib.import_module("X") and
  __import__("X") where the first argument is a string constant matching
  FORBIDDEN (a bare import_module("X") from `from importlib import
  import_module` is caught too).

Honest boundary of static analysis: dynamic imports with NON-literal names
(importlib.import_module(some_var)) and subprocess-based network (curl, wget)
are invisible here. The complementary RUNTIME layer is
scenic.determinism.block_network() — a sys.addaudithook that raises on the
"socket.connect" audit event — which the harness wires in at process start.

Exemptions:
- scenic/weights.py (weight loading may reference offline-mode plumbing;
  huggingface_hub in FORBIDDEN still stops OTHER files from quietly growing
  a hub dependency)
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
# starts with "<entry>.", so `urllib` catches urllib.request etc. and `http`
# catches http.client AND http.server. (`httpx` does not match `http` — no
# dot — hence its own entry.)
FORBIDDEN = (
    "urllib",
    "urllib3",
    "requests",
    "http",
    "socket",
    "socketserver",
    "ftplib",
    "smtplib",
    "poplib",
    "imaplib",
    "telnetlib",
    "xmlrpc",
    "huggingface_hub",
    "aiohttp",
    "httpx",
    "pycurl",
    "websockets",
)


def _matches(module: str) -> str | None:
    for f in FORBIDDEN:
        if module == f or module.startswith(f + "."):
            return f
    return None


def _dynamic_import_literal(node: ast.Call) -> str | None:
    """Return the literal module name if `node` is importlib.import_module("X"),
    import_module("X"), or __import__("X"); None otherwise. Non-literal first
    arguments are out of scope for static analysis (see module docstring)."""
    func = node.func
    is_dyn = (
        isinstance(func, ast.Name) and func.id in ("__import__", "import_module")
    ) or (
        isinstance(func, ast.Attribute)
        and func.attr == "import_module"
        and isinstance(func.value, ast.Name)
        and func.value.id == "importlib"
    )
    if not is_dyn or not node.args:
        return None
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    return None


def check_source(source: str, display_name: str) -> list[str]:
    """Core checker: findings for one file's source text. Import-testable."""
    tree = ast.parse(source, filename=display_name)
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                hit = _matches(alias.name)
                if hit:
                    findings.append(
                        f"{display_name}:{node.lineno}: import {alias.name}"
                        f" (forbidden: {hit})"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — always in-repo, never stdlib net
                continue
            module = node.module or ""
            hit = _matches(module)
            if hit:
                findings.append(
                    f"{display_name}:{node.lineno}: from {module} import ..."
                    f" (forbidden: {hit})"
                )
                continue
            # `from urllib import request`, `from http import client`
            for alias in node.names:
                full = f"{module}.{alias.name}" if module else alias.name
                hit = _matches(full)
                if hit:
                    findings.append(
                        f"{display_name}:{node.lineno}: from {module} import"
                        f" {alias.name} (forbidden: {hit})"
                    )
        elif isinstance(node, ast.Call):
            name = _dynamic_import_literal(node)
            if name is None:
                continue
            hit = _matches(name)
            if hit:
                callee = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else "importlib.import_module"
                )
                findings.append(
                    f"{display_name}:{node.lineno}: {callee}({name!r})"
                    f" (forbidden: {hit})"
                )
    return findings


def _check_file(path: Path) -> list[str]:
    rel = path.relative_to(REPO_ROOT)
    return check_source(path.read_text(), str(rel))


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
