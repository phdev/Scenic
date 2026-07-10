"""Unit tests for the tools/check_no_network.py checker core.

Feeds source snippets straight through check_source() (the import-testable
core) — the broadened FORBIDDEN list (urllib3, whole http package,
huggingface_hub, ...) and the literal dynamic-import detection
(importlib.import_module("X") / __import__("X")) each get a positive and,
where the boundary matters, a negative case. Also runs the real main() over
the actual repo: the ship path MUST be clean under the broadened list — if
this fails, do not weaken the list; the printed findings name the offender.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_guard():
    # tools/ is not a package; load the module straight from its file.
    spec = importlib.util.spec_from_file_location(
        "check_no_network", REPO_ROOT / "tools" / "check_no_network.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GUARD = _load_guard()


def _findings(source: str) -> list[str]:
    return GUARD.check_source(source, "snippet.py")


# ---------------------------------------------------------------------------
# static imports — broadened FORBIDDEN list
# ---------------------------------------------------------------------------


def test_urllib3_import_flagged():
    # urllib3 is a live transitive dep (uv.lock) and NOT neutralized by the
    # HF/transformers offline env vars — the original guard missed it.
    (f,) = _findings("import urllib3\n")
    assert "snippet.py:1" in f and "urllib3" in f


def test_from_urllib_import_request_flagged():
    (f,) = _findings("from urllib import request\n")
    assert "urllib" in f


def test_http_server_import_flagged():
    # Whole http package is forbidden now, not just http.client.
    (f,) = _findings("import http.server\n")
    assert "http.server" in f and "forbidden: http" in f


def test_huggingface_hub_import_flagged():
    (f,) = _findings("import huggingface_hub\n")
    assert "huggingface_hub" in f


# ---------------------------------------------------------------------------
# dynamic imports — literal names only
# ---------------------------------------------------------------------------


def test_import_module_literal_flagged():
    src = "import importlib\nimportlib.import_module('urllib.request')\n"
    (f,) = _findings(src)
    assert "snippet.py:2" in f and "urllib.request" in f


def test_dunder_import_literal_flagged():
    (f,) = _findings("__import__('socket')\n")
    assert "socket" in f


def test_benign_import_not_flagged():
    assert _findings("import numpy\nfrom pathlib import Path\n") == []


def test_non_literal_dynamic_import_not_flagged():
    # Documented boundary of static analysis: non-literal names pass the
    # scan (the runtime block_network() hook is the complementary layer).
    src = "import importlib\nsome_var = 'pipeline.s4_place'\nimportlib.import_module(some_var)\n"
    assert _findings(src) == []


# ---------------------------------------------------------------------------
# the real repo must be clean under the broadened list
# ---------------------------------------------------------------------------


def test_real_repo_clean_under_broadened_list(capsys):
    rc = GUARD.main()
    out = capsys.readouterr().out
    assert rc == 0, f"ship path not clean under broadened FORBIDDEN list:\n{out}"
    assert "no-network guard OK" in out
