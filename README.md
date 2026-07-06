# Scenic

Deterministic pano → layered Gaussian-splat "scenic bubble" pipeline.

One licensed equirectangular 360° pano in; a layered 3DGS asset out (`.ply`
master + `.sog` derivative) comfortable within a small head-box on
Quest-class hardware — plus a machine-readable provenance receipt for every
run. Invariants are **tested, not intended**:

- **Determinism**: same input bytes + pinned weights + params ⇒ bit-identical
  artifacts at every stage. CI runs the pipeline twice on a fixture pano and
  asserts manifest hash equality (`make determinism-check`).
- **Provenance**: every stage writes a receipt (input/output hashes, code git
  SHA, weight sha256 + license id, params, gate verdicts); a run manifest
  aggregates them. No complete receipt chain ⇒ unshippable by definition.
- **License posture**: no non-commercial weights or code, ever. Allowed
  weight licenses: Apache-2.0 / MIT / BSD-3-Clause, enforced in CI by
  `tools/license_guard.py`. Ship depth backend: Depth-Anything-V2-**Small**
  (Apache-2.0). Person detector: RT-DETR r18 (Apache-2.0). NC benchmarking
  lives in a separate repo, never here.

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| S0 ingest | `pipeline/s0_ingest.py` | hash pano, require license sidecar, nadir watermark heuristic, RT-DETR person detection on 10 perspective renders → equirect masks |
| S1 cleanplate | `pipeline/s1_cleanplate.py` | semi-auto human 2D edit loop; re-entry gates: detector == 0 hits AND diff confined to dilated masks |
| S2 depth | `pipeline/s2_depth.py` | 6 cubemap faces → DA-V2-Small → joint affine log-depth alignment (face0 anchor) → feathered equirect fusion → guided upsample; heuristic sky mask |
| S2b scale | `pipeline/s2b_scale.py` | deterministic IRLS ground-plane fit on the nadir cone → metric scale from camera height; gates: plane quality, min content distance |
| S3 layers | `pipeline/s3_layers.py` | occlusion edges from log-depth gradient; analytic inpaint band from head-box; deterministic push-pull background fill |
| S4 place | `pipeline/s4_place.py` | importance-sampled splat placement (denser at edges + ground), DC-only SH, textured far shell; per-splat layer/provenance PLY props |
| S5 | — | **reserved — no optimization in the ship path, ever** |
| S6 compress | `pipeline/s6_compress.py` | opacity floor, radius-relative kNN isolation prune, cross-layer merge, cap via logged stride re-run; `.sog` via pinned splat-transform + zip renorm |
| S7 gates | `pipeline/s7_gates.py`, `gates/*.py` | falsifiable verdicts from headless CPU splat renders: hole (magenta shell, 7 head-box poses incl. squat), jitter (P vs P+1mm), stereo (63mm), people (detector on final renders), budgets |
| S8 review | `pipeline/s8_review.py` | self-contained static review page, this run vs last accepted |

## Quick start

```bash
make setup            # uv sync + npm ci (splat-transform)
make fetch-weights    # download + sha256-verify pinned weights (setup-time only)
make fixtures         # regenerate the procedural fixture panos
make run PANO=fixtures/test.jpg OUT=runs/demo
make determinism-check   # the acceptance gate: two runs, bit-identical manifests
open runs/demo/s8_review/out/index.html
```

Each run directory contains per-stage `receipt.json` files, gate verdict
JSONs, headless renders, `scene.ply` / `scene.sog`, and `manifest.json`
(`shippable: true` only when every gate passed).

Read `docs/CONTRACTS.md` before touching any stage.
