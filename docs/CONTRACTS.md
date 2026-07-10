# Scenic contracts (read this before writing any stage)

Deterministic pano → layered 3DGS pipeline. HARD INVARIANTS:

1. **Determinism**: same input bytes + pinned weights + params ⇒ bit-identical
   artifacts. No timestamps, no absolute paths, no `set` iteration, no
   unseeded RNG, no wall-clock anywhere in artifacts or receipts. Torch: CPU
   only, single thread, `scenic.determinism.enforce()` is called by the
   harness before any stage runs (thread-pinning env vars apply at
   `scenic.determinism` IMPORT time, before numpy loads BLAS).
2. **Provenance**: every stage writes `receipt.json` via
   `scenic.receipts.write_receipt` (schema-validated; receipts record
   EXACTLY what the stage consumed and produced). Manifest aggregates AND
   checks chain coherence (see manifest.build below).
3. **One stage = one module.** Stages read ONLY prior stages' `out/` dirs and
   `fixtures`/params; communicate ONLY via on-disk artifacts. No network at
   stage runtime — enforced statically (tools/check_no_network.py, imports +
   literal dynamic imports) and at runtime (`determinism.block_network()`
   audit hook raises on any in-process socket.connect; wired by the harness).

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
(`scenic/run.py`) CLEARS the stage's prior state (`receipt.json` + `out/`)
so the receipt provably comes from this invocation and `out/` holds only
files this execution wrote, then calls `run`; the stage MUST have called
`write_receipt` exactly once. CLI for any single stage:
`uv run python -m scenic.run --run-dir runs/x --pano P --only s2_depth`.
`--only` refuses to run if the params file differs from the run's
`params.snapshot.yaml` (a mixed-params chain is unshippable) and deletes
`manifest.json` afterwards — the next full build re-derives it and the
coherence check refuses any stale mix a single-stage re-run left behind.

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
- `manifest.build(run_dir, verify_disk=False)` — aggregates receipts in
  registry order into `manifest.json`; raises if any stage receipt is
  missing (incomplete chain = unshippable) OR the chain is INCOHERENT:
  receipt `stage` must match its directory, every in-run-dir input must
  have a recorded producer, and its sha256 must equal what the producer
  recorded for that path (a stale mixed chain from an `--only` re-run or a
  hand-edit is unshippable). `verify_disk=True` additionally re-hashes every
  recorded output against the file on disk (used at promotion boundaries).
  Gate counting uses `pass is True` — the receipt schema enforces boolean.
  `manifest.manifest_hash(run_dir) -> str`.
- `params.load(path) -> dict` (+ `params_hash`).
- `determinism.enforce()` — env vars, torch single-thread CPU deterministic,
  seeds. `determinism.block_network()` — idempotent audit hook, raises on
  any in-process socket.connect (subprocess tools unaffected); the harness
  installs it at pipeline start. `determinism.rng(tag: str) ->
  np.random.Generator` — seeded from (params seed, tag); NEVER use global
  np.random.
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
  (incl. per-face affine coefficients, overlap residuals). Input is
  `s1_cleanplate/out/pano_clean.png` — REQUIRED (s1 always writes it; a
  missing clean plate is a broken run, never a silent s0 fallback).
- s2b_scale: `depth_m.npy`, `scale.json` {scale_factor, plane:{normal,d},
  residual_rel, tilt_deg, camera_height_m, gates…}.
- s3_layers: `fg_rgb.png fg_depth.npy fg_mask.png bg_rgb.png bg_depth.npy
  bg_mask.png layers.json` (band_px analytic derivation recorded). Input
  pano is the s1 clean plate — REQUIRED, same rule as s2.
- s4_place: `splats.ply` (3DGS PLY + extra uchar props `layer` 0=fg 1=bg
  2=shell and `origin_stage`), `splats_meta.json`.
- s6_compress: per-profile `scene_{review,quest}.ply/_std.ply/.sog` + primary
  aliases `scene.ply/scene_std.ply/scene.sog`, `compress.json` (per-profile
  counts, stride retries; see the v2 section).
- s7_gates: `verdicts/{budgets,fidelity_at_origin,hole,jitter,people,
  stereo}.json` (schema gate_verdict: {gate, pass, metrics{}, thresholds{},
  details}) + `renders/*.png`; receipt embeds all six verdicts in `gates`,
  records the renders as hashed outputs, and records scene.sog + the s1
  clean plate (fidelity's source) as inputs.
- s8_review: `index.html` (static, self-contained: base64 PNG renders this
  run vs runs/_accepted if present, metrics table), `review.json`,
  `thumbs/`, `scene_review.sog`. Everything whose bytes reach index.html is
  a receipt input — including the accepted-baseline files (as
  `external/<name>`) and the s7 layer renders.

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

## v2 QUALITY PASS (Machu Picchu review fixes) — authoritative deltas

This section overrides the pre-v2 stage descriptions where they conflict.
Determinism, single-owner modules, schema-validated on-disk IO, and
"gates record verdicts, never abort" all still hold. Every new number is a
`params.yaml` key hashed into receipts. Backend-independent — the depth model
is unchanged (DA-V2-Small).

### metrics (scenic/metrics.py — shared, already written, DO NOT edit)

- `ssim(a, b) -> float`, `ssim_map(a, b)` — deterministic SSIM; accepts uint8
  or float01, any channel count (grayscales internally).
- `solid_angle_fraction(mask_HxW_bool) -> float` — cos-lat-weighted sphere
  coverage fraction of an equirect boolean mask.
- `equirect_tile_views(tiles_lon, tiles_lat) -> [(name, yaw_deg, pitch_deg,
  fov_deg)]` — deterministic perspective view grid for fidelity tiling.
- `lpips_advisory_available() -> (bool, reason)` — False by default; LPIPS is
  advisory-only, weights never in the enforced tree.

### S2 depth (pipeline/s2_depth.py) — 8-face ring + global solve

- Face layout replaces the 6-cube with `params.faces`: an 8-face horizon ring
  (yaw = k*45deg for k in 0..7, pitch 0, fov `ring_fov_deg`=100) PLUS zenith
  (pitch +90) and nadir (pitch -90) caps at `cap_fov_deg` = 10 faces total.
  Larger overlaps than the cube. Define the face list internally in s2 (name,
  yaw, pitch, fov); reuse geometry.face_project / render_perspective.
- Per-face inference at `faces.max_infer_px` (tile the face into overlapping
  sub-tiles and mosaic the disparity if a face render exceeds max_infer_px;
  at 518 no tiling occurs — record actual infer_px per face in depth_meta).
- Alignment: ONE global least-squares over ALL adjacent-face overlaps (ring
  neighbors incl. the wrap pair 7-0, and every ring face vs both caps) in
  log-depth, minimizing sum over overlap pixels of Huber_delta(
  (a_i x_i + b_i) - (a_j x_j + b_j)) via fixed-iteration IRLS
  (`s2.huber_iters`, `s2.huber_delta_log`). SKY pixels excluded from overlap
  rows (per-face sky via the same heuristic as the fused sky mask, computed on
  the face). Gauge fixed by a Tikhonov pull toward identity (`s2.affine_reg`)
  AND a final renormalization so the MEDIAN of the fused relative depth is 1
  (median gauge). Loop closure is automatic from including the cyclic ring
  adjacency. Receipt/depth_meta: per-face (affine_a, affine_b, residual_log,
  infer_px) for all 10 faces, `overlap_residual_log`, and
  `max_interface_step_log` (max over overlap pixels of the aligned log-depth
  difference, robust 99th percentile).
- NEW GATE `interface_step` (emitted in s2's receipt, like s2b's gates).
  **REVISED (integration): `max_interface_step_log` is the confidence-WEIGHTED
  MEDIAN ring-ring content seam** (not a 99th-pct max). Per-face monocular
  depth genuinely disagrees in the far field (each face independently guesses
  distant/sky depth) — inherent to the backend and irrelevant to a shell-based
  bubble — so the metric excludes the far/sky tail (top decile of aligned
  depth) and reports a robust central seam. The gate ALSO requires
  `depth_dynamic_range` (p99/p1 finite depth) >= 2.0, so the opposite failure
  mode (a collapsed near-constant fusion, which has a LOW seam) is still
  caught. pass iff weighted-median-seam <= `s2.interface_step_max_log` AND
  range >= 2. Never aborts.
- The global solve HARD-ANCHORS face 0 to identity (a0=1, b0=0). Without a hard
  anchor the pairwise log-depth-difference objective has a trivial a_i=0
  (constant-depth) minimiser that a weak Tikhonov cannot outweigh; it collapses
  the whole depth map to a constant. Post-normalise, the anchor and a "median
  gauge" are identical, so the anchor is the correct, collapse-proof choice.
- depth_meta `faces` now has 6..12 entries (schema updated). Downstream
  consumers are unchanged (they read equirect depth_rel.npy).

### S3 layers (pipeline/s3_layers.py) — kill the flat blocks

- Occlusion edge now requires BOTH the log-grad threshold AND a depth ratio:
  `d_far/d_near > s3.edge_depth_ratio_min` (1.4). Compute per-edge-pixel
  near/far from the dominant-diff neighbor (as today) and drop edges whose
  ratio is below threshold BEFORE building the band.
- Band width hard-capped: `band_px = min(ceil(analytic)+band_extra_px,
  s3.band_px_max)`. Analytic value from (head_box t_max, d_near, d_far) as
  today. Record the cap in band_derivation.
- Visibility test: emit background ONLY where the foreground edge actually
  occludes some head-box pose. For each edge pixel, the disocclusion is real
  iff the near surface, viewed from at least one head-box extreme pose
  (the 6 lateral/vertical extremes + squat from gates.head_box_poses via
  params — reimplement the pose list locally in s3 to preserve single-owner),
  shifts by >= 1 px against the far surface. Practical deterministic test:
  angular parallax of the near point between center and the pose,
  |t_perp| / d_near (rad), must exceed one pixel angular_pixel_size(H) for the
  band to be emitted at that edge; restrict bg_region to edges passing this.
- Clamp bg splat scale to fg-equivalent at same depth: record in layers.json a
  `bg_scale_clamp` flag; the actual clamp is applied in S4 (see below), s3
  just guarantees bg_depth is the metric far-side depth so S4 can size bg
  splats like fg at that depth.
- Receipt + layers.json: add `bg_solid_angle_frac` =
  metrics.solid_angle_fraction(bg_region). 
- NEW GATE `bg_solid_angle` (emitted in s3's receipt): pass iff
  bg_solid_angle_frac <= `s3.bg_solid_angle_max_frac` (0.05). metrics
  {bg_solid_angle_frac, edge_px_count, band_px}, thresholds {max_frac}.

### S4 place (pipeline/s4_place.py) — shell-inward + density/scale

- `scale_multiplier` default is now 0.85 (param already changed).
- Shell-inward routing by METRIC depth d (fg_depth is metric):
  * splats are placed for finite, non-sky pixels with `d <= shell_distance_m`
    (50). **REVISED (integration, evidence-driven): near content (d <
    min_content_distance) STAYS as fg splats at true depth**, it is NOT routed
    to the shell. Routing near content to the shell made the hole gate read
    "shell below skyline" as 100% holes and killed origin fidelity; keeping it
    as real geometry fixes both, and its discomfort is still flagged honestly
    by the min_content_distance + stereo gates off the depth map. Record
    near_fg_px (near content kept as fg) + far_shell_px in splats_meta;
    near_shell_px stays as a field, now always 0.
  * pixels with d > shell_distance_m OR sky OR non-finite route to the TEXTURED
    SHELL at shell_radius_m (200), full pano texture, normals facing camera.
  * Feather the frontier: fg splats with d in
    [shell_distance_m - feather_m, shell_distance_m] get opacity multiplied by
    (shell_distance_m - d)/feather_m (ramp to 0 at the boundary); the shell
    additionally covers every direction with d > (shell_distance_m -
    feather_m) so a backdrop is always present behind a fading splat (no hard
    seam).
- Importance sampling weighted by local color variance: in ADDITION to the
  edge>ground>base class strides, bias density toward textured regions. Deterministic:
  compute local color variance v (max over channels of the variance in a
  `s4.color_var_window_px` half-window box, on fg_rgb01) and a per-pixel boost
  b = 1 + (color_var_boost-1) * clip(v / color_var_ref, 0, 1). Apply by
  keeping a class-selected pixel iff `(hash(r,c) deterministic in [0,1)) <
  b/color_var_boost` — i.e. flat areas (b~1) are decimated by up to
  1/color_var_boost, textured areas (b~color_var_boost) are kept. Use a fixed
  deterministic hash of (r,c) (e.g. from scenic.determinism.rng is NOT allowed
  per-pixel; instead use a closed-form: frac(sin dot) is non-portable — use
  `((r*73856093) ^ (c*19349663)) & 0xffffff) / 0x1000000` as the uniform).
  Record the effective retained fraction in splats_meta.
- bg splat scale clamp: size bg splats as fg-equivalent at their depth:
  radius = d * angpix * stride * scale_mult (same formula as fg), NOT inflated
  by the bg stride if that would exceed the fg-equivalent; clamp to the fg
  radius at that depth.
- splats_meta additions: near_shell_px, far_shell_px, feather_px,
  color_var_retained_frac. counts_by_layer unchanged (fg/bg/shell).

### S6 compress (pipeline/s6_compress.py) — two profiles

- Two profiles from `params.s6.profiles`: `review` (target 1.5M, cap 2M, sog
  unbounded) and `quest` (600k/1M/60MB). For each profile, run the existing
  opacity-floor -> isolation-prune -> merge pipeline, and enforce the cap via
  s4 stride re-run (unchanged mechanism; s4 natural density is tuned to the
  review target so review uses multiplier ~1 and quest coarsens). Run the
  profiles in FIXED order (review then quest) so s4's final on-disk state is
  deterministic; leave s4 outputs as the last (quest) placement.
- Outputs: `scene_review.ply/.sog(+std)`, `scene_quest.ply/.sog(+std)`, and
  PRIMARY aliases `scene.ply/.sog(+std)` = a byte copy of the
  `primary_profile` (quest) — s7 gates load scene.ply (the shipped asset).
- compress.json NEW SHAPE (schema updated):
  ```
  {"sog_tool": "...", "primary_profile": "quest", "viewer_profile": "review",
   "profiles": {"review": {in_count, after_opacity_floor,
     after_isolation_prune, after_merge, final_count, stride_retries,
     ply_bytes, sog_bytes, target, cap, sog_max_mb},
     "quest": {...same...}}}
  ```
- s7 budgets reads `profiles.quest` (the ship budget); s8 reads
  `profiles.review` for the viewer + `profiles.quest` for the header, and
  loads scene_review.sog into the inline viewer.

### S7 gates (pipeline/s7_gates.py, gates/) — forensics + fidelity

- View matrix: each of the 7 head-box poses is certified over FIVE views —
  the 4 pitch-0 yaws PLUS one straight-down view (yaw 0, pitch −90, named
  `{pose}_down`) — 35 views total for hole/people. The down view closes the
  nadir blind cone (pitch-0 fov-90 frusta only reach ~−45°; the nadir is
  where cleanplate/tripod defects live). The hole gate's below-skyline mask
  is per-pitch (all-True for the down view). Jitter runs its base/offset
  pair on all 5 center-pose views and gates on the worst energy. Stereo's
  near-limit is measured over the FULL frame minus a 2-px border (the old
  central-half window left four ~37° azimuth blind wedges) across the 4 yaw
  eye-pairs plus two near-limit-only pairs at pitch ±90.
- Layer forensics (`params.s7.layers`): render origin (center pose, the 4 yaw
  ring) THREE extra times per yaw with only fg, only bg, only shell splats
  (filter by layer), save to out/renders/ as
  `center_yaw{NNN}_layer_{fg,bg,shell}.png`. In s7's receipt notes add
  `layer_counts` {fg,bg,shell} and `layer_solid_angle` {fg,bg,shell} =
  solid_angle_fraction of the equirect mask each layer's splats project to
  from the origin (approx: mark the direction bins the splats occupy).
- NEW GATE `fidelity_at_origin` (gates/fidelity.py): for each view from
  metrics.equirect_tile_views(s7.fidelity.tiles_lon, tiles_lat) at
  s7.fidelity.render_px, render the PRIMARY splats from the origin and sample
  the SOURCE pano (the s1 clean plate — REQUIRED, never a silent s0
  fallback) into the same perspective; SSIM per tile. Report worst tile + mean. PASS iff worst >=
  ssim_worst_tile_min AND mean >= ssim_mean_min (SSIM enforced). LPIPS:
  if lpips_advisory_available() compute + report as an advisory metric
  `lpips_mean`; else metrics get `lpips: "advisory_unavailable"` and a reason
  in details. LPIPS NEVER affects pass. gate name `fidelity_at_origin`.
- gates/fidelity.py needs run_dir (like budgets) to find the source pano; wire
  `fidelity.run_gate(splats, params, out, run_dir=run_dir)` in s7.
- GATE_ORDER becomes ("hole","jitter","stereo","people","budgets",
  "fidelity_at_origin"). budgets reads compress profiles.quest.

### S8 review (pipeline/s8_review.py, docs/) — toggles + SOG quant

- Layer toggle links: surface the s7 layer renders
  (center_yaw*_layer_{fg,bg,shell}.png) on the page as a small gallery with
  fg/bg/shell toggle (inline, base64, no network).
- SOG quant check: origin render of `scene_review.ply` vs `scene_review.sog`
  is not directly renderable (.sog is packed); instead render scene_review.ply
  from origin (4 yaws) AND decode-and-render is out of scope — compare the
  review PLY origin render vs the QUEST/primary PLY origin render is NOT the
  ask. The ask: measure .ply-vs-.sog compression loss. Deterministic approach:
  load scene_review.sog back via the same reader path splat-transform round-
  trips (if a python .sog reader is unavailable, SKIP the pixel compare and
  instead record the byte sizes + a note). MINIMUM viable + deterministic:
  record ply_bytes vs sog_bytes ratio per profile from compress.json AND, if a
  .sog->splat decode is available in-repo, SSIM(origin render ply, origin
  render sog) into review.json.sog_ssim; else review.json.sog_ssim = null with
  reason. Show the number (or "decode unavailable") on the page.
- The inline viewer loads `scene_review.sog` (copy the review sog to the run's
  s8 out and reference it; for the docs/ deploy the human copies it).
- review.json additions: `layers` (the layer render filenames), `sog_ssim`
  (float or null), `profiles` (echo compress profiles final_count + sog_bytes).

### tools/sweep.py + `make sweep`

- Grid over {s4.scale_multiplier, s4.base_stride, s3.edge_depth_ratio_min,
  s3.band_px_max} on a fixture; for each cell write a temp params override,
  run the pipeline (or a fast subset up to s7 fidelity), rank cells by
  fidelity_at_origin mean SSIM (desc), tie-break worst-tile SSIM. Emit a
  ranked table (stdout + runs/_sweep/report.json). Deterministic given the
  grid. `make sweep FIXTURE=fixtures/ci_tiny.jpg`.

## Baseline acceptance (runs/_accepted, `make accept`)

`make accept RUN=runs/<name> [FORCE=1]` (tools/accept_run.py) promotes a
completed run to the `runs/_accepted` baseline that s8_review compares
against. It re-derives the manifest from the receipts with
`manifest.build(run_dir, verify_disk=True)` — an incomplete/incoherent chain
or a tampered artifact refuses; a hand-edited manifest.json is overwritten
with receipt truth. `shippable=false` runs refuse unless FORCE=1 (gates
record verdicts, humans decide; the promotion is recorded honestly in
`accepted.json`, schema `accepted`). The baseline is a SLIM snapshot
(s8_review + manifest.json + params.snapshot.yaml + accepted.json), staged
in a per-process dir and swapped atomically; it is deliberately NOT a full
run copy. s8 records the baseline files it embeds as receipt inputs, so a
baseline change is attributable in any manifest diff.

## Testing

Every module gets `tests/test_<module>.py`. Oracles:
- sphere pano radius R ⇒ every splat |xyz| ≈ R (naive path, no layers).
- synthetic room ⇒ positions within closed-form wall/floor/ceiling bounds.
Geometry round-trips: dirs→uv→dirs; perspective↔equirect consistency.
Run: `uv run pytest tests/test_x.py -q` from repo root.
