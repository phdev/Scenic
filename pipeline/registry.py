"""Ordered stage registry. One stage = one module (single owner)."""
from __future__ import annotations

import importlib

STAGES: list[tuple[str, str]] = [
    ("s0_ingest", "pipeline.s0_ingest"),
    ("s1_cleanplate", "pipeline.s1_cleanplate"),
    ("s2_depth", "pipeline.s2_depth"),
    ("s2b_scale", "pipeline.s2b_scale"),
    ("s3_layers", "pipeline.s3_layers"),
    ("s4_place", "pipeline.s4_place"),
    # s5 reserved — no optimization ever in the ship path
    ("s6_compress", "pipeline.s6_compress"),
    ("s7_gates", "pipeline.s7_gates"),
    ("s8_review", "pipeline.s8_review"),
]


def get_stage(name: str):
    mod = dict(STAGES)[name]
    return importlib.import_module(mod)
