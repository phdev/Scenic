"""Budgets gate: ship-size invariants.

- splat count: len(scene.ply splats) <= params.splat_cap  (FAIL above cap)
- .sog size:   s6_compress/out/scene.sog file size <= params.sog_max_mb MiB
               (FAIL above; compress.json's recorded sog_bytes and
               final_count are cross-checked and recorded, not failing)
- splat count vs params.splat_target: recorded as a metric only.

No renders (nothing visual to diagnose). run_gate keeps the common
(splats, params, outdir) signature; the s6 artifacts are located from
run_dir, which defaults to outdir/../../ (the standard
<run>/s7_gates/out layout) and can be passed explicitly.
"""
from __future__ import annotations

from pathlib import Path

from scenic import schema
from scenic.plyio import SplatData

MIB = 1024 * 1024


def run_gate(
    splats: SplatData,
    params: dict,
    outdir: Path | str,
    run_dir: Path | str | None = None,
) -> dict:
    outdir = Path(outdir)
    if run_dir is None:
        run_dir = outdir.resolve().parent.parent
    s6_out = Path(run_dir) / "s6_compress" / "out"
    compress_path = s6_out / "compress.json"
    sog_path = s6_out / "scene.sog"
    if not compress_path.exists():
        raise FileNotFoundError(f"budgets gate: missing {compress_path}")
    if not sog_path.exists():
        raise FileNotFoundError(f"budgets gate: missing {sog_path}")
    compress = schema.read_validated(compress_path, "compress")

    cap = int(params["splat_cap"])
    target = int(params["splat_target"])
    sog_max_mb = float(params["sog_max_mb"])
    sog_max_bytes = int(sog_max_mb * MIB)

    final_count = len(splats)
    sog_bytes = int(sog_path.stat().st_size)
    passed = final_count <= cap and sog_bytes <= sog_max_bytes
    return {
        "gate": "budgets",
        "pass": bool(passed),
        "metrics": {
            "final_count": int(final_count),
            "sog_bytes": int(sog_bytes),
            "count_vs_target_ratio": float(final_count) / float(target),
            "compress_final_count": int(compress["final_count"]),
            "compress_sog_bytes": int(compress["sog_bytes"]),
        },
        "thresholds": {
            "splat_cap": cap,
            "sog_max_bytes": sog_max_bytes,
            "sog_max_mb": sog_max_mb,
            "splat_target": target,
        },
        "details": {
            "count_matches_compress": bool(
                final_count == int(compress["final_count"])
            ),
            "sog_bytes_matches_compress": bool(
                sog_bytes == int(compress["sog_bytes"])
            ),
        },
    }
