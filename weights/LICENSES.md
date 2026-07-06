# Weight licenses (ship-path registry)

Every file under `weights/` MUST have an entry here AND a sha256 pin in
`weights/pins.json`. Allowed licenses: Apache-2.0, MIT, BSD-3-Clause.
CI enforces this via `tools/license_guard.py`. Forbidden anywhere in this
repo: AGPL dependencies (e.g. ultralytics), CC-BY-NC weights (e.g.
Depth-Anything V2 Large, DAP), research-only weights (SHARP/UniSHARP).
Non-commercial benchmarking lives in the separate `scenic-bench` repo, never
here.

## depth_anything_v2_small

- Repo: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
- Revision pinned: `5426e4f0f36572d16453bbda7a8389317b1bef99`
- License: **Apache-2.0** (verbatim tag on the HF model card: `license: apache-2.0`)
- License URL: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/blob/main/README.md
- Verified: 2026-07-05 via HF API card data at fetch time (fetch refuses
  non-allowlisted card licenses).
- Note: only the **Small** DA-V2 checkpoint is Apache-2.0. Base/Large are
  CC-BY-NC-4.0 and are FORBIDDEN here.
- Files + sha256: see `pins.json` key `depth_anything_v2_small`.

## rtdetr_r18

- Repo: https://huggingface.co/PekingU/rtdetr_r18vd
- Revision pinned: `ac77a11ff0170a41b771c03264987f8ce2b0d753`
- License: **Apache-2.0** (verbatim tag on the HF model card: `license: apache-2.0`)
- License URL: https://huggingface.co/PekingU/rtdetr_r18vd/blob/main/README.md
- Verified: 2026-07-05 via HF API card data at fetch time.
- Files + sha256: see `pins.json` key `rtdetr_r18`.
