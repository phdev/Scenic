"""Hashing + canonical JSON. Canonical form: sorted keys, compact separators,
floats via repr (shortest round-trip), NaN/Inf forbidden."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            c = f.read(chunk)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def _check_floats(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError("NaN/Inf not allowed in canonical JSON")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"non-string key {k!r}")
            _check_floats(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _check_floats(v)


def canonical_json(obj) -> bytes:
    _check_floats(obj)
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def sha256_json(obj) -> str:
    return sha256_bytes(canonical_json(obj))


def write_json(path: Path | str, obj) -> None:
    """Deterministic human-readable JSON artifact (sorted keys, 2-space indent)."""
    _check_floats(obj)
    Path(path).write_text(
        json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    )


def read_json(path: Path | str):
    return json.loads(Path(path).read_text())
