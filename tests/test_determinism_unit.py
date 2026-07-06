"""Unit-level determinism sweeps that need no pipeline stages.

Covers the three cross-cutting invariants CI leans on:
1. scenic.determinism.rng(tag) streams are stable across calls AND across
   fresh processes (values frozen below — if these change, every receipt
   hash in every run changes; that is a breaking event, not a test update).
2. hashing.canonical_json is byte-stable for tricky floats (repr shortest
   round-trip). NOTE the -0.0 behavior: repr(-0.0) == '-0.0', so canonical
   JSON of -0.0 differs from 0.0 and they hash differently. That is the
   documented, accepted behavior — producers must normalize -0.0 themselves
   if they want 0.0's hash (float(x) + 0.0 does it).
3. After determinism.enforce(), a fixed torch conv on fixed input yields a
   bit-identical result across two fresh subprocesses.

Also guards tools/check_no_network.py (the CI no-network static gate).
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from scenic import determinism, hashing

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 1. rng(tag) stream stability
# ---------------------------------------------------------------------------

# Frozen expected streams for seed=0 (params.yaml default). Computed once via
# rng(tag).random() x3, then hardcoded. DO NOT regenerate casually: a change
# here means the seeding scheme changed and all run artifacts shift.
EXPECTED_STREAMS = {
    "alpha": [0.41656667312768636, 0.6051181772338435, 0.08317190301126998],
    "s4_place": [0.9154302436081907, 0.5068430011579771, 0.5734779685923777],
    "gate/jitter": [0.9032524630001642, 0.8983912791554756, 0.6603913065336663],
}


@pytest.mark.parametrize("tag", sorted(EXPECTED_STREAMS))
def test_rng_stream_frozen(tag):
    determinism.set_seed(0)
    g = determinism.rng(tag)
    got = [g.random() for _ in range(3)]
    assert got == EXPECTED_STREAMS[tag]


def test_rng_stream_stable_across_calls():
    determinism.set_seed(0)
    a = determinism.rng("alpha").random(3)
    b = determinism.rng("alpha").random(3)
    assert a.tolist() == b.tolist() == EXPECTED_STREAMS["alpha"]


def test_rng_streams_independent_per_tag():
    determinism.set_seed(0)
    assert (
        determinism.rng("alpha").random(3).tolist()
        != determinism.rng("s4_place").random(3).tolist()
    )


def test_rng_depends_on_seed():
    determinism.set_seed(1)
    try:
        shifted = determinism.rng("alpha").random(3).tolist()
    finally:
        determinism.set_seed(0)  # never leak seed state to other tests
    assert shifted != EXPECTED_STREAMS["alpha"]


def test_rng_stream_stable_across_processes():
    snippet = (
        "from scenic import determinism\n"
        "determinism.set_seed(0)\n"
        "g = determinism.rng('alpha')\n"
        "print(repr([g.random() for _ in range(3)]))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert ast.literal_eval(out.stdout.strip()) == EXPECTED_STREAMS["alpha"]


# ---------------------------------------------------------------------------
# 2. canonical_json byte stability for tricky floats
# ---------------------------------------------------------------------------


def test_canonical_json_tricky_floats_frozen():
    assert hashing.canonical_json({"x": 1e-7}) == b'{"x":1e-07}'
    assert (
        hashing.canonical_json({"x": 0.1 + 0.2}) == b'{"x":0.30000000000000004}'
    )
    # -0.0: repr distinguishes it from 0.0; canonical JSON therefore does too.
    # Documented behavior — do not "fix" in hashing; normalize at producers.
    assert hashing.canonical_json({"x": -0.0}) == b'{"x":-0.0}'
    assert hashing.canonical_json({"x": 0.0}) == b'{"x":0.0}'
    assert hashing.sha256_json({"x": -0.0}) != hashing.sha256_json({"x": 0.0})


def test_canonical_json_composite_frozen():
    obj = {"b": [1e-7, 0.1 + 0.2, -0.0], "a": {"z": 1.0, "y": True}}
    assert (
        hashing.canonical_json(obj)
        == b'{"a":{"y":true,"z":1.0},"b":[1e-07,0.30000000000000004,-0.0]}'
    )
    assert (
        hashing.sha256_json(obj)
        == "702188b29deac1dc374ce91de5e8ebb9dfb94f058a7b76a73e94d4980030e129"
    )


def test_canonical_json_key_order_irrelevant():
    a = {"k1": 1.5, "k2": [2.5, 3.5]}
    b = {"k2": [2.5, 3.5], "k1": 1.5}
    assert hashing.canonical_json(a) == hashing.canonical_json(b)


# ---------------------------------------------------------------------------
# 3. torch determinism smoke (fresh subprocesses)
# ---------------------------------------------------------------------------

_TORCH_SNIPPET = """
import hashlib
from scenic import determinism
determinism.enforce()
import torch
with torch.no_grad():
    conv = torch.nn.Conv2d(3, 4, 3, bias=True)
    w = torch.arange(conv.weight.numel(), dtype=torch.float32)
    conv.weight.copy_(torch.sin(w * 0.1).reshape(conv.weight.shape))
    conv.bias.copy_(torch.cos(torch.arange(conv.bias.numel(), dtype=torch.float32)))
    x = torch.sin(torch.arange(3 * 16 * 16, dtype=torch.float32) * 0.01)
    y = conv(x.reshape(1, 3, 16, 16))
print(hashlib.sha256(y.numpy().tobytes()).hexdigest())
"""


def _run_torch_snippet() -> str:
    out = subprocess.run(
        [sys.executable, "-c", _TORCH_SNIPPET],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    h = out.stdout.strip().splitlines()[-1]
    assert len(h) == 64, f"expected sha256 hex on stdout, got: {out.stdout!r}"
    return h


def test_torch_conv_identical_across_fresh_processes():
    assert _run_torch_snippet() == _run_torch_snippet()


# ---------------------------------------------------------------------------
# tools/check_no_network.py guard behavior
# ---------------------------------------------------------------------------


def _run_guard(cwd=REPO_ROOT):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "check_no_network.py")],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_no_network_guard_passes_on_repo():
    out = _run_guard()
    assert out.returncode == 0, out.stdout + out.stderr


def test_no_network_guard_catches_forbidden_imports(tmp_path):
    # The guard resolves the repo root from its own file location, so to test
    # the failure path we copy it into a synthetic mini-repo.
    tools = tmp_path / "tools"
    tools.mkdir()
    guard = (REPO_ROOT / "tools" / "check_no_network.py").read_text()
    (tools / "check_no_network.py").write_text(guard)
    pl = tmp_path / "pipeline"
    pl.mkdir()
    (pl / "s9_bad.py").write_text(
        "import urllib.request\n"
        "from http import client\n"
        "import socket\n"
        "import requests\n"
    )
    gt = tmp_path / "gates"
    gt.mkdir()
    (gt / "bad_gate.py").write_text("from urllib.request import urlopen\n")
    # scenic/weights.py is exempt even with a forbidden import
    sc = tmp_path / "scenic"
    sc.mkdir()
    (sc / "weights.py").write_text("import urllib.request\n")
    (sc / "ok.py").write_text("import json\nfrom pathlib import Path\n")

    out = subprocess.run(
        [sys.executable, str(tools / "check_no_network.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 1
    assert "s9_bad.py:1" in out.stdout and "urllib.request" in out.stdout
    assert "s9_bad.py:2" in out.stdout  # from http import client
    assert "s9_bad.py:3" in out.stdout  # socket
    assert "s9_bad.py:4" in out.stdout  # requests
    assert "bad_gate.py:1" in out.stdout
    assert "weights.py" not in out.stdout  # exempt
    assert "ok.py" not in out.stdout


def test_no_network_guard_ignores_comments_and_docstrings(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "check_no_network.py").write_text(
        (REPO_ROOT / "tools" / "check_no_network.py").read_text()
    )
    pl = tmp_path / "pipeline"
    pl.mkdir()
    (pl / "s1_ok.py").write_text(
        '"""Docstring mentioning urllib and https://example.com is fine."""\n'
        "# import requests  <- comment, fine\n"
        "URL = 'https://example.com'  # string constant, not an import\n"
    )
    out = subprocess.run(
        [sys.executable, str(tools / "check_no_network.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
