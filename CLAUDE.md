# Scenic

Deterministic pano → layered Gaussian-splat "scenic bubble" pipeline for
Quest-class VR. One licensed equirect 360 pano in; `.ply` master + `.sog`
derivative + machine-readable provenance receipts out.

**Read `docs/CONTRACTS.md` before touching any stage.** It defines the run
layout, geometry conventions, receipt API, and determinism rules.

## Hard rules

- **No NC weights/code, ever.** Allowed weight licenses: Apache-2.0, MIT,
  BSD-3-Clause; enforced by `tools/license_guard.py` in CI. DAP is CC-BY-NC →
  lives only in the separate `scenic-bench` repo. No AGPL deps (ultralytics).
- **Determinism is tested, not intended**: `make determinism-check` runs the
  pipeline twice and asserts bit-identical manifests. No timestamps, no
  absolute paths, no unseeded RNG, no wall-clock in any artifact or receipt.
- **S5 is reserved. No gsplat training/optimization in the ship path.**
- Stages are single-owner modules communicating only via schema-validated
  on-disk artifacts. No network at stage runtime (weights pre-fetched +
  hash-pinned in `weights/pins.json`).

## Commands

- `make setup` — uv sync + npm ci (splat-transform for .sog)
- `make fetch-weights` — download/verify pinned weights (setup-time only)
- `make fixtures` — generate synthetic test panos (sphere / room oracles)
- `make run PANO=fixtures/test.jpg [OUT=runs/x]` — full pipeline
- `make determinism-check` — the acceptance gate (double run, hash-equal)
- `make test` / `make license-guard`

## Layout

- `scenic/` core libs (hashing, receipts, manifest, geometry, plyio,
  rasterizer, determinism, weights, run harness)
- `pipeline/sN_name.py` stages (registry order: s0_ingest, s1_cleanplate,
  s2_depth, s2b_scale, s3_layers, s4_place, s6_compress, s7_gates, s8_review)
- `gates/*.py` gate implementations (hole, jitter, stereo, people, budgets)
- `schemas/*.schema.json` all JSON artifact schemas
- `weights/` pinned weights (gitignored) + `pins.json` + `LICENSES.md` (committed)
- `tools/` fetch_weights, license_guard, make_fixtures, compare_runs
- `runs/<name>/` per-run artifacts + receipts + `manifest.json`

## Depth backend

Ship backend: Depth-Anything-V2-**Small** (Apache-2.0) via 6-face cubemap +
joint affine log-depth alignment (face0 anchor) + feathered equirect fusion.
The backend sits behind `scenic.weights.load_depth_model()`; future cleared
models (e.g. MoGe-2, card-checked) slot in there. DA-V2 Base/Large are
CC-BY-NC — forbidden.
