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
- **Receipts are checked, not trusted**: `manifest.build` refuses incomplete
  AND incoherent chains (wrong stage dir, in-run input without a recorded
  producer, producer/consumer hash mismatch from `--only` staleness);
  `make accept` additionally re-hashes every recorded output on disk. The
  harness clears each stage's `receipt.json` + `out/` before running it.
- **S5 is reserved. No gsplat training/optimization in the ship path.**
- Stages are single-owner modules communicating only via schema-validated
  on-disk artifacts. No network at stage runtime (weights pre-fetched +
  hash-pinned in `weights/pins.json`; static import guard + a runtime
  `socket.connect` audit hook enforce it).

## Commands

- `make setup` — uv sync + npm ci (splat-transform for .sog)
- `make fetch-weights` — download/verify pinned weights (setup-time only)
- `make fixtures` — generate synthetic test panos (sphere / room oracles)
- `make run PANO=fixtures/test.jpg [OUT=runs/x]` — full pipeline
- `make determinism-check` — the acceptance gate (double run, hash-equal)
- `make accept RUN=runs/<name> [FORCE=1]` — promote a run to the
  `runs/_accepted` baseline (refuses incoherent/tampered/unshippable runs;
  FORCE=1 overrides the shippable check only, recorded honestly)
- `make test` / `make license-guard`

## Layout

- `scenic/` core libs (hashing, receipts, manifest, geometry, plyio,
  rasterizer, determinism, weights, run harness)
- `pipeline/sN_name.py` stages (registry order: s0_ingest, s1_cleanplate,
  s2_depth, s2b_scale, s3_layers, s4_place, s6_compress, s7_gates, s8_review)
- `gates/*.py` gate implementations (hole, jitter, stereo, people, budgets,
  fidelity_at_origin)
- `scenic/metrics.py` deterministic SSIM / solid-angle / fidelity tile grid
- `schemas/*.schema.json` all JSON artifact schemas
- `weights/` pinned weights (gitignored) + `pins.json` + `LICENSES.md` (committed)
- `tools/` fetch_weights, license_guard, check_no_network, make_fixtures,
  compare_runs, accept_run, sweep
- `runs/<name>/` per-run artifacts + receipts + `manifest.json`

## Depth backend

Ship backend: Depth-Anything-V2-**Small** (Apache-2.0) via an **8-face horizon
ring (100° FOV, 45° spacing) + zenith/nadir caps**, one global least-squares
log-depth alignment over all adjacent overlaps (Huber IRLS, **hard face-0
anchor** to prevent the a=0 constant-depth collapse), feathered equirect
fusion, then median normalise. The backend sits behind
`scenic.weights.load_depth_model()`; future cleared models (e.g. MoGe-2,
card-checked) slot in there. DA-V2 Base/Large are CC-BY-NC — forbidden.

## v2 quality pass (see docs/CONTRACTS.md "v2 QUALITY PASS")

- **Shell-inward**: fg splats for `d <= shell_distance_m` (50); sky + far →
  textured shell at `shell_radius_m` (200). Near content (< `min_content`)
  STAYS as fg splats at true depth (real geometry; its discomfort is flagged
  by the min_content/stereo gates, not hidden behind a backdrop).
- **S3 constraints**: occlusion edge needs depth-ratio > `edge_depth_ratio_min`
  (1.4) AND a head-box visibility test; band hard-capped; `bg_solid_angle`
  gate (≤5%). Kills the flat blocks.
- **Two S6 profiles**: `review` (1.5M, hi-q, s8 viewer + SOG-quant) and
  `quest` (600k/60MB, the shipped/gated asset). `scene.ply/.sog` = quest.
- **Gates added**: `interface_step` (s2; weighted-median ring content seam +
  depth-range collapse guard), `bg_solid_angle` (s3), `fidelity_at_origin`
  (s7; per-tile SSIM enforced, LPIPS advisory-only — VGG/ImageNet provenance
  OPEN, weights never in the enforced tree, see weights/LICENSES.md).
- **Layer forensics**: s7 renders fg/bg/shell-only origin views + per-layer
  counts & solid-angle; s8 surfaces them as toggles.
- **Gate view matrix**: 7 poses × (4 yaws + straight-down) = 35 views for
  hole/people (nadir blind cone closed); stereo near-limit is full-frame
  (±45°/view → no azimuth wedges) + pitch ±90 near-only pairs; jitter gates
  the worst of 5 center-pose view pairs.
- **`make sweep`**: grid over placement params ranked by fidelity SSIM.

The Machu Picchu vantage is the **adversarial** fixture: it correctly FAILs
`min_content_distance` + `stereo` (near walls, elevated). Do not relax those.
