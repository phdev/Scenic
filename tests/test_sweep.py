"""Tests for tools/sweep.py — grid generation, override application, ranking.

The real pipeline (and depth model) is NEVER run here: `run_sweep` takes an
injected `score_fn`, so every test uses a fast deterministic stub. Covers:
grid Cartesian product + order, deep-copy override application (original
params unchanged), ranking by ssim_mean desc / ssim_worst_tile desc with a
stable grid-index tie-break, report shape + best cell, --limit truncation,
determinism (byte-identical canonical JSON across runs), the dotted axis paths
resolving against the real params.yaml, and the Makefile `sweep` target.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "sweep", REPO_ROOT / "tools" / "sweep.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sweep = _load_tool()

from scenic import hashing, params as params_mod  # noqa: E402


# --------------------------------------------------------------------------- #
# make_grid
# --------------------------------------------------------------------------- #
def test_make_grid_count_and_keys():
    spec = {
        "s4.scale_multiplier": [0.7, 0.85, 1.0],
        "s4.base_stride": [2, 3],
        "s3.edge_depth_ratio_min": [1.3, 1.4, 1.6],
        "s3.band_px_max": [32, 64],
    }
    grid = sweep.make_grid(spec)
    assert len(grid) == 3 * 2 * 3 * 2  # full Cartesian product
    for cell in grid:
        assert list(cell.keys()) == list(spec.keys())
        assert cell["s4.scale_multiplier"] in spec["s4.scale_multiplier"]
        assert cell["s3.band_px_max"] in spec["s3.band_px_max"]


def test_make_grid_order_is_deterministic_product():
    spec = {"a": [1, 2], "b": [10, 20, 30]}
    grid = sweep.make_grid(spec)
    # outermost axis first, matching itertools.product
    assert grid == [
        {"a": 1, "b": 10},
        {"a": 1, "b": 20},
        {"a": 1, "b": 30},
        {"a": 2, "b": 10},
        {"a": 2, "b": 20},
        {"a": 2, "b": 30},
    ]


# --------------------------------------------------------------------------- #
# apply_overrides
# --------------------------------------------------------------------------- #
def test_apply_overrides_deep_copies_and_sets_nested():
    base = {
        "s3": {"edge_depth_ratio_min": 1.4, "band_px_max": 64},
        "s4": {"scale_multiplier": 0.85, "base_stride": 2},
    }
    original = copy.deepcopy(base)
    override = {
        "s4.scale_multiplier": 0.7,
        "s4.base_stride": 3,
        "s3.edge_depth_ratio_min": 1.6,
        "s3.band_px_max": 32,
    }
    new = sweep.apply_overrides(base, override)

    # original untouched (deep copy, not aliased)
    assert base == original
    assert base["s4"]["scale_multiplier"] == 0.85
    assert base["s3"]["band_px_max"] == 64

    # override applied to the copy
    assert new["s4"]["scale_multiplier"] == 0.7
    assert new["s4"]["base_stride"] == 3
    assert new["s3"]["edge_depth_ratio_min"] == 1.6
    assert new["s3"]["band_px_max"] == 32

    # mutating a nested container in the copy does not touch base
    new["s4"]["scale_multiplier"] = 999
    assert base["s4"]["scale_multiplier"] == 0.85


def test_apply_overrides_unknown_path_raises():
    base = {"s4": {"base_stride": 2}}
    with pytest.raises(KeyError):
        sweep.apply_overrides(base, {"s4.nonexistent": 1})
    with pytest.raises(KeyError):
        sweep.apply_overrides(base, {"sX.base_stride": 1})


def test_default_grid_paths_exist_in_real_params():
    """Every dotted axis in DEFAULT_GRID must resolve against params.yaml."""
    real = params_mod.load(REPO_ROOT / "params.yaml")
    first_cell = sweep.make_grid(sweep.DEFAULT_GRID)[0]
    updated = sweep.apply_overrides(real, first_cell)  # raises if a path is bad
    assert updated["s4"]["scale_multiplier"] == first_cell["s4.scale_multiplier"]
    assert updated["s3"]["band_px_max"] == first_cell["s3.band_px_max"]
    # real params object untouched
    assert real is not updated


# --------------------------------------------------------------------------- #
# rank_cells
# --------------------------------------------------------------------------- #
def test_rank_cells_mean_then_worst_then_stable_index():
    scored = [
        {"index": 0, "overrides": {"a": 1}, "ssim_mean": 0.5,
         "ssim_worst_tile": 0.20, "final_count": 10},
        {"index": 1, "overrides": {"a": 2}, "ssim_mean": 0.5,
         "ssim_worst_tile": 0.40, "final_count": 20},
        {"index": 2, "overrides": {"a": 3}, "ssim_mean": 0.9,
         "ssim_worst_tile": 0.10, "final_count": 30},
        {"index": 3, "overrides": {"a": 4}, "ssim_mean": 0.5,
         "ssim_worst_tile": 0.40, "final_count": 40},
    ]
    ranked = sweep.rank_cells(scored)

    # highest mean first; then the two mean=0.5/worst=0.40 cells keep grid
    # order (index 1 before index 3); worst=0.20 last.
    assert [c["overrides"]["a"] for c in ranked] == [3, 2, 4, 1]
    assert [c["rank"] for c in ranked] == [1, 2, 3, 4]
    # report cells expose exactly the contract keys
    for c in ranked:
        assert set(c.keys()) == {
            "overrides", "ssim_mean", "ssim_worst_tile", "final_count", "rank"
        }


def test_rank_cells_empty():
    assert sweep.rank_cells([]) == []


# --------------------------------------------------------------------------- #
# run_sweep (with an injected deterministic stub score_fn)
# --------------------------------------------------------------------------- #
def _stub_scores(table):
    """Build a score_fn from an index -> (mean, worst, count) table."""
    calls = []

    def score(index, override):
        calls.append((index, dict(override)))
        m, w, fc = table[index]
        return {"ssim_mean": m, "ssim_worst_tile": w, "final_count": fc}

    return score, calls


def test_run_sweep_report_shape_and_ranking():
    spec = {"s3.band_px_max": [32, 48, 64]}  # 3 cells: indices 0,1,2
    table = {0: (0.70, 0.30, 111), 1: (0.90, 0.50, 222), 2: (0.80, 0.40, 333)}
    score_fn, calls = _stub_scores(table)

    report = sweep.run_sweep("fixtures/ci_tiny.jpg", spec, score_fn)

    # every cell scored once, in grid order
    assert [c[0] for c in calls] == [0, 1, 2]

    assert report["fixture"] == "fixtures/ci_tiny.jpg"
    assert report["grid"] == {"s3.band_px_max": [32, 48, 64]}
    assert set(report.keys()) == {"fixture", "grid", "cells", "best"}

    # ranked by ssim_mean desc: index1(0.9) > index2(0.8) > index0(0.7)
    assert [c["ssim_mean"] for c in report["cells"]] == [0.90, 0.80, 0.70]
    assert [c["rank"] for c in report["cells"]] == [1, 2, 3]
    assert [c["overrides"]["s3.band_px_max"] for c in report["cells"]] == [
        48, 64, 32
    ]

    # best is the rank-1 cell
    assert report["best"]["rank"] == 1
    assert report["best"]["overrides"] == {"s3.band_px_max": 48}
    assert report["best"]["ssim_mean"] == 0.90


def test_run_sweep_does_not_mutate_grid_spec():
    spec = {"s3.band_px_max": [32, 64]}
    spec_copy = copy.deepcopy(spec)
    table = {0: (0.5, 0.5, 1), 1: (0.6, 0.6, 2)}
    score_fn, _ = _stub_scores(table)
    sweep.run_sweep("f", spec, score_fn)
    assert spec == spec_copy


def test_run_sweep_limit_truncates():
    spec = {"a": [1, 2, 3, 4]}
    table = {i: (0.5 + 0.01 * i, 0.3, i) for i in range(4)}
    score_fn, calls = _stub_scores(table)
    report = sweep.run_sweep("f", spec, score_fn, limit=2)
    assert len(report["cells"]) == 2
    assert [c[0] for c in calls] == [0, 1]  # only first two evaluated


def test_run_sweep_limit_zero_gives_no_best():
    spec = {"a": [1, 2]}
    score_fn, _ = _stub_scores({})
    report = sweep.run_sweep("f", spec, score_fn, limit=0)
    assert report["cells"] == []
    assert report["best"] is None


def test_run_sweep_deterministic_canonical_json():
    spec = {"s4.base_stride": [2, 3], "s3.band_px_max": [32, 64]}
    table = {
        0: (0.61, 0.31, 100),
        1: (0.61, 0.31, 100),  # exact tie with 0 -> stable order
        2: (0.90, 0.10, 400),
        3: (0.72, 0.55, 250),
    }
    score_a, _ = _stub_scores(table)
    score_b, _ = _stub_scores(table)
    report_a = sweep.run_sweep("fixtures/ci_tiny.jpg", spec, score_a)
    report_b = sweep.run_sweep("fixtures/ci_tiny.jpg", spec, score_b)
    assert hashing.canonical_json(report_a) == hashing.canonical_json(report_b)
    # writable as a deterministic artifact (no NaN/Inf, string keys)
    _ = hashing.canonical_json(report_a)
    # tie between cell 0 and cell 1 keeps grid order
    tied = [c for c in report_a["cells"] if c["ssim_mean"] == 0.61]
    assert [c["overrides"] for c in tied] == [
        {"s4.base_stride": 2, "s3.band_px_max": 32},
        {"s4.base_stride": 2, "s3.band_px_max": 64},
    ]


# --------------------------------------------------------------------------- #
# metric extraction / formatting helpers
# --------------------------------------------------------------------------- #
def test_metric_primary_and_alias():
    assert sweep._metric({"ssim_mean": 0.7}, sweep._MEAN_KEYS) == 0.7
    assert sweep._metric({"ssim": 0.4}, sweep._MEAN_KEYS) == 0.4
    with pytest.raises(KeyError):
        sweep._metric({"nope": 1}, sweep._MEAN_KEYS)


def test_format_table_contains_rows():
    spec = {"a": [1, 2]}
    table = {0: (0.5, 0.2, 10), 1: (0.9, 0.4, 20)}
    score_fn, _ = _stub_scores(table)
    report = sweep.run_sweep("fx", spec, score_fn)
    txt = sweep.format_table(report)
    assert "fixture=fx" in txt
    assert "rank" in txt and "ssim_mean" in txt
    assert "best (rank 1)" in txt
    # rank-1 (mean 0.9) row is printed before rank-2
    assert txt.index("0.9000") < txt.index("0.5000")


def test_fixture_label_is_relative():
    label = sweep._fixture_label(REPO_ROOT / "fixtures" / "ci_tiny.jpg")
    assert label == "fixtures/ci_tiny.jpg"
    assert not Path(label).is_absolute()


# --------------------------------------------------------------------------- #
# Makefile target
# --------------------------------------------------------------------------- #
def test_makefile_has_sweep_target():
    text = (REPO_ROOT / "Makefile").read_text()
    lines = text.splitlines()

    # .PHONY declares sweep
    phony = [ln for ln in lines if ln.startswith(".PHONY")]
    assert phony and any("sweep" in ln for ln in phony)

    # a well-formed `sweep:` target that runs tools/sweep.py on $(FIXTURE)
    assert "FIXTURE ?=" in text
    assert any(ln.startswith("sweep:") for ln in lines)
    recipe = text.split("sweep:", 1)[1]
    # the recipe line (next non-empty) invokes the tool with the fixture
    body = recipe.splitlines()[1] if len(recipe.splitlines()) > 1 else ""
    assert "tools/sweep.py" in body
    assert "--pano" in body
    assert "$(FIXTURE)" in body
