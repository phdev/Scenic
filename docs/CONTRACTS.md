# Scenic contracts (read this before writing any stage)

Deterministic pano → layered 3DGS pipeline. HARD INVARIANTS:

1. **Determinism**: same input bytes + pinned weights + params ⇒ bit-identical
   artifacts. No timestamps, no absolute paths, no `set` iteration, no
   unseeded RNG, no wall-clock anywhere in artifacts or receipts. Torch: CPU
   only, single thread, `scenic.determinism.enforce()` is called by the
   harness before any stage runs.
2. **Provenance**: every stage writes `receipt.json` via
   `scenic.receipts.write_receipt` (schema-validated). Manifest aggregates.
3. **One stage = one module.** Stages read ONLY prior stages' `out/` dirs and
   `fixtures`/params; communicate ONLY via on-disk artifacts. No network.

## Run layout

```
runs/<name>/
  params.snapshot.yaml
  manifest.json                    # built last by scenic.manifest
  s0_ingest/{receipt.json, out/}
  s1_cleanplate/…  s2_depth/…  s2b_scale/…  s3_layers/…  s4_place/…
  s6_compress/…  s7_gates/…  s8_review/…
```

Stage module: `pipeline/sN_name.py` exposing
`run(run_dir: pathlib.Path, params: dict, ctx: scenic.stage.Ctx) -> None`.
Register in `pipeline/registry.py` STAGES list (ordered). The harness
(`scenic/run.py`) creates `<run_dir>/<stage>/out/`, calls `run`, then the
stage MUST have called `write_receipt` exactly once. CLI for any single
stage: `uv run python -m scenic.run --run-dir runs/x --pano P --only s2_depth`.

## Core APIs (scenic/)

- `hashing.sha256_file(path) -> str`, `hashing.sha256_bytes(b)`,
  `hashing.canonical_json(obj) -> bytes` (sorted keys, no whitespace drift,
  floats via repr), `hashing.sha256_json(obj)`.
- `schema.validate(obj, "name")` validates against `schemas/name.schema.json`.
- `receipts.write_receipt(run_dir, stage_name, *, inputs: dict[str,Path],
  outputs: dict[str,Path], params_used: dict, weights_used: list[str] = [],
  gates: list[dict] = [], notes: dict = {})` — paths are hashed and recorded
  RELATIVE to run_dir; `weights_used` are keys into weights/pins.json (hash +
  license id get embedded); `gates` entries must validate as gate_verdict.
- `manifest.build(run_dir)` — aggregates receipts in registry order into
  `manifest.json`; raises if any stage receipt missing (incomplete chain =
  unshippable). `manifest.manifest_hash(run_dir) -> str`.
- `params.load(path) -> dict` (+ `params_hash`).
- `determinism.enforce()` — env vars, torch single-thread CPU deterministic,
  seeds. `determinism.rng(tag: str) -> np.random.Generator` — seeded from
  (params seed, tag); NEVER use global np.random.
- `weights.load_pins() -> dict` — weights/pins.json {key: {repo, files:
  {relpath: sha256}, license, license_url}}. `weights.local_dir(key) -> Path`
  (verifies hashes; raises if missing/mismatch). `weights.load_depth_model()`,
  `weights.load_person_detector()` return (model, processor) on CPU, eval.
- `imageio.load_rgb(path) -> np.ndarray uint8 HxWx3` (EXIF/ICC ignored),
  `imageio.save_png(path, arr)` (no metadata), `imageio.save_npy/load_npy`,
  `imageio.save_mask_png/load_mask_png` (uint8 0/255 -> bool).

## Geometry (scenic/geometry.py) — fixed conventions

Right-handed, **+Y up**, camera at origin, θ=0 → **+Z**.
Equirect WxH: lon θ = (u+0.5)/W·2π − π; lat φ = π/2 − (v+0.5)/H·π.
`dir = [cosφ·sinθ, sinφ, cosφ·cosθ]` (float64 math, artifacts float32).

- `equirect_dirs(w, h) -> (h,w,3)`; `dirs_to_uv(dirs, w, h) -> (…,2)` float
  pixel coords (u right, v down, +0.5 center convention).
- `rotation_yaw_pitch(yaw_rad, pitch_rad) -> 3x3` (world = R @ cam; cam looks
  +Z, x right, y up; positive pitch looks up).
- `perspective_dirs(fov_deg, w, h, yaw_rad, pitch_rad) -> (h,w,3)` world dirs.
- `sample_equirect(img_f32 HxWxC, dirs) -> (...,C)` bilinear, lon wraps,
  lat clamps.
- `render_perspective(img, fov_deg, w, h, yaw, pitch)` = sample(perspective_dirs).
- `CUBE_FACES`: 6 (name, yaw, pitch): front(0,0) right(π/2,0) back(π,0)
  left(−π/2,0) up(0,π/2) down(0,−π/2).
- `face_project(dirs, yaw, pitch, fov_deg) -> (uv in [0,1]^2, in_frustum mask,
  center_cos)`: project world dirs into a face; `center_cos` = cosine of angle
  to face axis (feather weight base).
- `angular_pixel_size(h) = π/h` radians/pixel at equator.
- `pitch_of_dirs(dirs)` -> lat in rad.

Depth arrays: float32 HxW **radial distance along the ray** (relative until
s2b, meters after). Invalid/sky = np.inf.

## Weights (already fetched to ./weights, pinned in weights/pins.json)

- key `depth_anything_v2_small`: HF `depth-anything/Depth-Anything-V2-Small-hf`
  (Apache-2.0). transformers `AutoModelForDepthEstimation` +
  `AutoImageProcessor` from local dir. Outputs RELATIVE DISPARITY (bigger =
  closer). Convert: depth_rel = 1/(disp + 1e-6) then align.
- key `rtdetr_r18`: HF `PekingU/rtdetr_r18vd` (Apache-2.0),
  `RTDetrForObjectDetection` + `RTDetrImageProcessor`; COCO person = label 0
  — use `model.config.id2label` to find "person", don't hardcode.
- NO other weights. NEVER add a weight without a LICENSES.md entry
  ({Apache-2.0, MIT, BSD-3-Clause}) + pins.json hash. No network at stage
  runtime: transformers must load with local_files_only semantics.

## Stage IO summary (out/ files; every JSON has a schema in schemas/)

- s0_ingest: `pano.png` (normalized master), `pano_meta.json`,
  `person_mask.png`, `person_boxes.json`, `watermark.json`. Fails if license
  sidecar `<pano>.license.json` missing/invalid (schema license_sidecar).
- s1_cleanplate: `pano_clean.png`, `cleanplate.json` (mode: passthrough |
  edited; gates re-run detector==0 and diff-containment when edited).
  Package emit for humans: `package/` (pano + overlay) when persons found.
- s2_depth: `depth_rel.npy` (sampling res), `sky_mask.png`, `depth_meta.json`
  (incl. per-face affine coefficients, overlap residuals).
- s2b_scale: `depth_m.npy`, `scale.json` {scale_factor, plane:{normal,d},
  residual_rel, tilt_deg, camera_height_m, gates…}.
- s3_layers: `fg_rgb.png fg_depth.npy fg_mask.png bg_rgb.png bg_depth.npy
  layers.json` (band_px analytic derivation recorded).
- s4_place: `splats.ply` (3DGS PLY + extra uchar props `layer` 0=fg 1=bg
  2=shell and `origin_stage`), `splats_meta.json`.
- s6_compress: `scene.ply`, `scene.sog`, `compress.json` (counts per step,
  stride retries).
- s7_gates: `verdicts/{hole,jitter,stereo,people,budgets}.json` (schema
  gate_verdict: {gate, pass, metrics{}, thresholds{}, details}) +
  `renders/*.png`; receipt embeds all five verdicts in `gates`.
- s8_review: `index.html` (static, self-contained: base64 PNG renders this
  run vs runs/_accepted if present, metrics table), `review.json`.

## PLY (scenic/plyio.py)

Binary little-endian 3DGS layout: x y z nx ny nz f_dc_0..2 opacity scale_0..2
rot_0..3 (all float32) + uchar `layer`, uchar `origin_stage`. Order of
elements is the deterministic placement order. `write_splats(path, SplatData)`,
`read_splats(path)`. SplatData: dataclass of np arrays (xyz, normals, f_dc,
opacity_logit, log_scales, quat_wxyz, layer, origin_stage).
f_dc = (rgb01 − 0.5)/0.28209479177387814; opacity stored as logit;
scales stored as ln(meters); quat wxyz unit, w≥0 canonical sign.

## Rasterizer (scenic/rasterizer.py)

`render(splats: SplatData, cam: Camera, px_w, px_h, fov_deg,
override_rgb: np.ndarray|None = None) -> dict(rgb uint8, alpha f32,
depth f32)`. Camera{pos(3), yaw, pitch}. EWA projection of 3D gaussians,
stable depth sort (key: depth then index), front-to-back per-splat bbox
compositing, 3σ cutoff, transmittance early-out. Deterministic float32.

## Testing

Every module gets `tests/test_<module>.py`. Oracles:
- sphere pano radius R ⇒ every splat |xyz| ≈ R (naive path, no layers).
- synthetic room ⇒ positions within closed-form wall/floor/ceiling bounds.
Geometry round-trips: dirs→uv→dirs; perspective↔equirect consistency.
Run: `uv run pytest tests/test_x.py -q` from repo root.
