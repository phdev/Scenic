"""JSON-schema validation for all on-disk JSON artifacts."""
from __future__ import annotations

import functools
from pathlib import Path

import jsonschema

from scenic import hashing

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


@functools.lru_cache(maxsize=None)
def _load(name: str) -> dict:
    p = SCHEMA_DIR / f"{name}.schema.json"
    if not p.exists():
        raise FileNotFoundError(f"missing schema {p}")
    return hashing.read_json(p)


def validate(obj, name: str) -> None:
    jsonschema.validate(
        obj, _load(name), format_checker=jsonschema.FormatChecker()
    )


def write_validated(path, obj, name: str) -> None:
    validate(obj, name)
    hashing.write_json(path, obj)


def read_validated(path, name: str):
    obj = hashing.read_json(path)
    validate(obj, name)
    return obj
