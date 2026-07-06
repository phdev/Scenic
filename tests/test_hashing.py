"""Unit tests for scenic.hashing: canonical JSON, hashing, JSON artifact IO."""
from __future__ import annotations

import hashlib
import math

import pytest

from scenic import hashing


# ---------------------------------------------------------------- canonical_json

def test_canonical_json_sorted_keys():
    out = hashing.canonical_json({"b": 1, "a": 2, "aa": 3})
    assert out == b'{"a":2,"aa":3,"b":1}'


def test_canonical_json_nested_sorted_and_compact():
    obj = {"z": {"y": 1, "x": [1, 2, {"b": 0, "a": 0}]}, "a": True}
    out = hashing.canonical_json(obj)
    # compact separators: no spaces anywhere outside strings
    assert b" " not in out
    assert out == b'{"a":true,"z":{"x":[1,2,{"a":0,"b":0}],"y":1}}'


def test_canonical_json_ascii_escaped():
    out = hashing.canonical_json({"k": "é☃"})
    # ensure_ascii: non-ascii escaped, output is pure ASCII bytes
    assert out == b'{"k":"\\u00e9\\u2603"}'
    out.decode("ascii")  # must not raise


def test_canonical_json_float_repr():
    # floats serialized via repr (shortest round-trip)
    assert hashing.canonical_json({"f": 0.1}) == b'{"f":0.1}'
    assert hashing.canonical_json({"f": 1e-07}) == b'{"f":1e-07}'


def test_canonical_json_tuple_treated_as_list():
    assert hashing.canonical_json({"t": (1, 2)}) == b'{"t":[1,2]}'


# ---------------------------------------------------------------- sha256_json

def test_sha256_json_stable_across_key_order():
    a = {"x": 1, "y": [1.5, "s"], "z": {"k": None}}
    b = {"z": {"k": None}, "y": [1.5, "s"], "x": 1}
    assert hashing.sha256_json(a) == hashing.sha256_json(b)


def test_sha256_json_matches_manual_hash():
    obj = {"a": [1, 2, 3], "b": "text"}
    expect = hashlib.sha256(hashing.canonical_json(obj)).hexdigest()
    assert hashing.sha256_json(obj) == expect
    assert len(expect) == 64


def test_sha256_json_differs_on_value_change():
    assert hashing.sha256_json({"a": 1}) != hashing.sha256_json({"a": 2})


# ---------------------------------------------------------------- rejection paths

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_inf_rejected_top_level(bad):
    with pytest.raises(ValueError):
        hashing.canonical_json(bad)


@pytest.mark.parametrize(
    "obj",
    [
        {"a": float("nan")},
        {"a": [1.0, float("inf")]},
        {"a": {"b": (float("-inf"),)}},
        [{"deep": [[float("nan")]]}],
    ],
)
def test_nan_inf_rejected_nested(obj):
    with pytest.raises(ValueError):
        hashing.canonical_json(obj)
    with pytest.raises(ValueError):
        hashing.sha256_json(obj)


def test_nan_inf_rejected_in_write_json(tmp_path):
    with pytest.raises(ValueError):
        hashing.write_json(tmp_path / "x.json", {"a": math.nan})
    assert not (tmp_path / "x.json").exists()


@pytest.mark.parametrize("obj", [{1: "x"}, {"ok": {2.5: "y"}}, {"ok": [{None: 1}]}])
def test_non_string_keys_rejected(obj):
    with pytest.raises(TypeError):
        hashing.canonical_json(obj)


# ---------------------------------------------------------------- file IO

def test_write_read_json_roundtrip(tmp_path):
    obj = {"b": [1, 2.5, "s", None, True], "a": {"nested": {"deep": -3}}}
    p = tmp_path / "artifact.json"
    hashing.write_json(p, obj)
    assert hashing.read_json(p) == obj


def test_write_json_deterministic_bytes(tmp_path):
    obj = {"b": 1, "a": {"y": 2.25, "x": [3]}}
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    hashing.write_json(p1, obj)
    hashing.write_json(p2, {"a": {"x": [3], "y": 2.25}, "b": 1})  # other key order
    b1, b2 = p1.read_bytes(), p2.read_bytes()
    assert b1 == b2
    assert b1.endswith(b"\n")


def test_sha256_file_matches_hashlib(tmp_path):
    data = bytes(range(256)) * 1000  # 256 KB, exercises chunked read
    p = tmp_path / "blob.bin"
    p.write_bytes(data)
    assert hashing.sha256_file(p) == hashlib.sha256(data).hexdigest()
    # small chunk size still identical
    assert hashing.sha256_file(p, chunk=7) == hashlib.sha256(data).hexdigest()


def test_sha256_bytes_matches_hashlib():
    b = b"scenic determinism"
    assert hashing.sha256_bytes(b) == hashlib.sha256(b).hexdigest()
