"""Budgets gate: the ship-size invariants, read from the s6 compress profiles.

The v2 two-profile compress.json carries a `quest` (ship) profile and a
`review` (inspection) profile. This gate enforces the QUEST budget — the
shipped scene.ply/.sog is a byte copy of the quest profile:

- splat count: profiles.quest.final_count <= params.splat_cap  (FAIL above)
- .sog size:   profiles.quest.sog_bytes <= params.sog_max_mb MiB (FAIL above)

The `review` profile's final_count/sog_bytes are recorded as non-failing
metrics (it is deliberately unbounded). As cross-checks (non-failing details)
we also compare the quest numbers against the actual on-disk scene.ply splat
count and scene.sog byte size.

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
    if not compress_path.exists():
        raise FileNotFoundError(f"budgets gate: missing {compress_path}")
    compress = schema.read_validated(compress_path, "compress")

    profiles = compress["profiles"]
    if "quest" not in profiles:
        raise KeyError("budgets gate: compress.json has no 'quest' profile")
    quest = profiles["quest"]

    cap = int(params["splat_cap"])
    target = int(params["splat_target"])
    sog_max_mb = float(params["sog_max_mb"])
    sog_max_bytes = int(sog_max_mb * MIB)

    final_count = int(quest["final_count"])
    sog_bytes = int(quest["sog_bytes"])
    passed = final_count <= cap and sog_bytes <= sog_max_bytes

    metrics: dict = {
        "final_count": int(final_count),
        "sog_bytes": int(sog_bytes),
        "count_vs_target_ratio": float(final_count) / float(target),
        # actual shipped scene.ply, cross-checked below (non-failing)
        "scene_ply_count": int(len(splats)),
    }
    # review profile: recorded, never failing.
    review = profiles.get("review")
    if review is not None:
        metrics["review_final_count"] = int(review["final_count"])
        metrics["review_sog_bytes"] = int(review["sog_bytes"])

    # cross-check quest numbers against the actual on-disk ship artifacts.
    sog_path = s6_out / "scene.sog"
    details: dict = {
        "primary_profile": compress.get("primary_profile", "quest"),
        "count_matches_scene": bool(final_count == int(len(splats))),
    }
    if sog_path.exists():
        actual_sog_bytes = int(sog_path.stat().st_size)
        metrics["scene_sog_bytes"] = int(actual_sog_bytes)
        details["sog_bytes_matches_compress"] = bool(
            sog_bytes == actual_sog_bytes
        )

    return {
        "gate": "budgets",
        "pass": bool(passed),
        "metrics": metrics,
        "thresholds": {
            "splat_cap": cap,
            "sog_max_bytes": sog_max_bytes,
            "sog_max_mb": sog_max_mb,
            "splat_target": target,
        },
        "details": details,
    }
