"""Params loading + hashing. params.yaml is part of every receipt."""
from __future__ import annotations

from pathlib import Path

import yaml

from scenic import hashing


def load(path: Path | str) -> dict:
    with open(path) as f:
        p = yaml.safe_load(f)
    if not isinstance(p, dict):
        raise ValueError("params must be a mapping")
    return p


def params_hash(p: dict) -> str:
    return hashing.sha256_json(p)
