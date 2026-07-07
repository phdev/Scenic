"""tools/sweep.py — deterministic parameter sweep over the Scenic pipeline.

Grid over {s4.scale_multiplier, s4.base_stride, s3.edge_depth_ratio_min,
s3.band_px_max}. For each cell: deep-copy params, apply the override, write a
temp params yaml, run the full pipeline into runs/_sweep/<cell>, and read the
`fidelity_at_origin` verdict from that run's s7 gates. Cells are ranked by
`ssim_mean` (desc), tie-broken by `ssim_worst_tile` (desc), then by grid index
(stable). A ranked table is printed to stdout and written to
runs/_sweep/report.json.

Determinism: no wall-clock, no absolute paths in report.json, a stable sort,
canonical JSON. The report is a pure function of (fixture, grid, per-cell
fidelity verdicts) — and the verdicts are themselves deterministic given the
pinned weights + params. This is a developer tool, not a pipeline stage, so it
writes its report with hashing.write_json (there is no sweep-report schema) and
never mutates params.yaml (each cell gets its own temp override yaml).

    uv run python tools/sweep.py --pano fixtures/ci_tiny.jpg [--limit N]
    make sweep FIXTURE=fixtures/ci_tiny.jpg
"""
from __future__ import annotations

import argparse
import copy
import itertools
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scenic import hashing, params as params_mod, schema  # noqa: E402

SWEEP_DIR = REPO_ROOT / "runs" / "_sweep"
REPORT_NAME = "report.json"
FIDELITY_VERDICT_REL = Path("s7_gates/out/verdicts/fidelity_at_origin.json")
COMPRESS_REL = Path("s6_compress/out/compress.json")

# Grid axes: dotted param key -> default value list. Kept small + configurable
# via CLI flags. Order here is the grid iteration order (outermost first).
DEFAULT_GRID: dict[str, list] = {
    "s4.scale_multiplier": [0.7, 0.85, 1.0],
    "s4.base_stride": [2, 3],
    "s3.edge_depth_ratio_min": [1.3, 1.4, 1.6],
    "s3.band_px_max": [32, 64],
}

# Metric key candidates (primary first). The contract names these ssim_mean /
# ssim_worst_tile; the extra aliases are a defensive fallback for the s7 gate's
# exact naming and never change the ranking order (first-present wins).
_MEAN_KEYS = ("ssim_mean", "mean_ssim", "ssim")
_WORST_KEYS = ("ssim_worst_tile", "worst_tile_ssim", "ssim_worst")


# --------------------------------------------------------------------------- #
# Pure, deterministic core (unit-tested without running the pipeline)
# --------------------------------------------------------------------------- #
def make_grid(grid_spec: dict[str, list]) -> list[dict]:
    """Cartesian product of the axes -> ordered list of override dicts.

    Deterministic: iterates axes in grid_spec insertion order, outermost axis
    first, matching itertools.product semantics.
    """
    keys = list(grid_spec.keys())
    value_lists = [list(grid_spec[k]) for k in keys]
    cells: list[dict] = []
    for combo in itertools.product(*value_lists):
        cells.append({k: v for k, v in zip(keys, combo)})
    return cells


def apply_overrides(params: dict, override: dict) -> dict:
    """Return a DEEP COPY of `params` with `override` applied.

    `override` maps dotted paths (e.g. "s4.scale_multiplier") to values. The
    original `params` dict is never mutated. Unknown paths raise KeyError so a
    typo'd axis fails loudly rather than silently no-op'ing.
    """
    out = copy.deepcopy(params)
    for dotted, value in override.items():
        parts = dotted.split(".")
        node = out
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"unknown param path {dotted!r}")
            node = node[part]
        leaf = parts[-1]
        if not isinstance(node, dict) or leaf not in node:
            raise KeyError(f"unknown param path {dotted!r}")
        node[leaf] = value
    return out


def rank_cells(scored: list[dict]) -> list[dict]:
    """Rank scored cells: ssim_mean desc, tie-break ssim_worst_tile desc.

    Final tie-break is the grid `index` so equal-scoring cells keep grid order
    (fully deterministic). Each scored cell must have keys: index, overrides,
    ssim_mean, ssim_worst_tile, final_count. Returns cell dicts shaped for the
    report (overrides, ssim_mean, ssim_worst_tile, final_count, rank).
    """
    ordered = sorted(
        scored,
        key=lambda c: (-c["ssim_mean"], -c["ssim_worst_tile"], c["index"]),
    )
    ranked: list[dict] = []
    for r, c in enumerate(ordered, start=1):
        ranked.append(
            {
                "overrides": copy.deepcopy(c["overrides"]),
                "ssim_mean": float(c["ssim_mean"]),
                "ssim_worst_tile": float(c["ssim_worst_tile"]),
                "final_count": int(c["final_count"]),
                "rank": r,
            }
        )
    return ranked


def run_sweep(
    fixture: str, grid_spec: dict[str, list], score_fn, limit: int | None = None
) -> dict:
    """Build the ranked report by scoring every grid cell via `score_fn`.

    `score_fn(index, override) -> {ssim_mean, ssim_worst_tile, final_count}` is
    injected so the ranking/report logic is testable without running the model.
    `grid_spec` is not mutated. Deterministic given the grid + score_fn.
    """
    grid = make_grid(grid_spec)
    if limit is not None:
        grid = grid[:limit]

    scored: list[dict] = []
    for index, override in enumerate(grid):
        res = score_fn(index, override)
        scored.append(
            {
                "index": index,
                "overrides": override,
                "ssim_mean": float(res["ssim_mean"]),
                "ssim_worst_tile": float(res["ssim_worst_tile"]),
                "final_count": int(res["final_count"]),
            }
        )

    ranked = rank_cells(scored)
    best = copy.deepcopy(ranked[0]) if ranked else None
    return {
        "fixture": fixture,
        "grid": {k: list(v) for k, v in grid_spec.items()},
        "cells": ranked,
        "best": best,
    }


def _fmt_val(v) -> str:
    return f"{v}"


def format_table(report: dict) -> str:
    """Human-readable ranked table for stdout (deterministic formatting)."""
    lines = [
        f"sweep fixture={report['fixture']}  cells={len(report['cells'])}"
    ]
    header = (
        f"{'rank':>4}  {'ssim_mean':>10}  {'worst_tile':>10}  "
        f"{'final_cnt':>10}  overrides"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for c in report["cells"]:
        ov = ", ".join(
            f"{k}={_fmt_val(v)}" for k, v in c["overrides"].items()
        )
        lines.append(
            f"{c['rank']:>4}  {c['ssim_mean']:>10.4f}  "
            f"{c['ssim_worst_tile']:>10.4f}  {c['final_count']:>10}  {ov}"
        )
    best = report.get("best")
    if best is not None:
        ov = ", ".join(
            f"{k}={_fmt_val(v)}" for k, v in best["overrides"].items()
        )
        lines.append(
            f"best (rank 1): {ov}  "
            f"[ssim_mean={best['ssim_mean']:.4f}, "
            f"worst={best['ssim_worst_tile']:.4f}]"
        )
    else:
        lines.append("best: (no cells evaluated)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Pipeline-backed scoring (the part that actually runs the model)
# --------------------------------------------------------------------------- #
def _metric(metrics: dict, keys: tuple[str, ...]) -> float:
    for k in keys:
        if k in metrics:
            return float(metrics[k])
    raise KeyError(
        f"sweep: fidelity verdict metrics missing any of {keys}; "
        f"present keys: {sorted(metrics)}"
    )


def read_cell_scores(cell_dir: Path) -> dict:
    """Read a finished cell run: fidelity SSIM + primary-profile final_count."""
    vpath = cell_dir / FIDELITY_VERDICT_REL
    if not vpath.exists():
        raise FileNotFoundError(f"sweep: missing fidelity verdict {vpath}")
    verdict = schema.read_validated(vpath, "gate_verdict")
    metrics = verdict.get("metrics", {})
    ssim_mean = _metric(metrics, _MEAN_KEYS)
    ssim_worst = _metric(metrics, _WORST_KEYS)

    cpath = cell_dir / COMPRESS_REL
    if not cpath.exists():
        raise FileNotFoundError(f"sweep: missing compress.json {cpath}")
    comp = hashing.read_json(cpath)
    prof = comp.get("primary_profile", "quest")
    final_count = int(comp["profiles"][prof]["final_count"])

    return {
        "ssim_mean": ssim_mean,
        "ssim_worst_tile": ssim_worst,
        "final_count": final_count,
    }


def pipeline_score_fn(pano: Path, base_params: dict, workdir: Path):
    """Factory: a score_fn that runs the real pipeline for each grid cell."""
    from scenic.run import run_pipeline

    def score(index: int, override: dict) -> dict:
        cell_dir = workdir / f"cell_{index:03d}"
        if cell_dir.exists():
            shutil.rmtree(cell_dir)
        cell_dir.mkdir(parents=True, exist_ok=True)

        cell_params = apply_overrides(base_params, override)
        pfile = cell_dir / "params.override.yaml"
        with open(pfile, "w") as f:
            yaml.safe_dump(cell_params, f, sort_keys=True, default_flow_style=False)

        run_pipeline(pano, cell_dir, pfile)
        return read_cell_scores(cell_dir)

    return score


def _grid_from_args(args) -> dict[str, list]:
    return {
        "s4.scale_multiplier": list(args.scale_multiplier),
        "s4.base_stride": list(args.base_stride),
        "s3.edge_depth_ratio_min": list(args.edge_depth_ratio_min),
        "s3.band_px_max": list(args.band_px_max),
    }


def _fixture_label(pano: Path) -> str:
    """Repo-relative, path-clean fixture label (never an absolute path)."""
    p = Path(pano)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.name


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Deterministic Scenic pipeline parameter sweep."
    )
    ap.add_argument("--pano", required=True, type=Path)
    ap.add_argument("--params", type=Path, default=REPO_ROOT / "params.yaml")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N grid cells (smoke sweep)",
    )
    ap.add_argument(
        "--scale-multiplier",
        type=_parse_floats,
        default=list(DEFAULT_GRID["s4.scale_multiplier"]),
        help="comma list, e.g. 0.7,0.85,1.0",
    )
    ap.add_argument(
        "--base-stride",
        type=_parse_ints,
        default=list(DEFAULT_GRID["s4.base_stride"]),
        help="comma list, e.g. 2,3",
    )
    ap.add_argument(
        "--edge-depth-ratio-min",
        type=_parse_floats,
        default=list(DEFAULT_GRID["s3.edge_depth_ratio_min"]),
        help="comma list, e.g. 1.3,1.4,1.6",
    )
    ap.add_argument(
        "--band-px-max",
        type=_parse_ints,
        default=list(DEFAULT_GRID["s3.band_px_max"]),
        help="comma list, e.g. 32,64",
    )
    args = ap.parse_args(argv)

    if not args.pano.exists():
        raise SystemExit(f"sweep: pano not found: {args.pano}")

    grid_spec = _grid_from_args(args)
    base_params = params_mod.load(args.params)

    if SWEEP_DIR.exists():
        shutil.rmtree(SWEEP_DIR)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    fixture_label = _fixture_label(args.pano)
    score_fn = pipeline_score_fn(args.pano, base_params, SWEEP_DIR)
    report = run_sweep(fixture_label, grid_spec, score_fn, limit=args.limit)

    report_path = SWEEP_DIR / REPORT_NAME
    hashing.write_json(report_path, report)

    print(format_table(report))
    print(f"[sweep] wrote {report_path.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
