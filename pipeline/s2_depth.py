"""S2 depth: 8-face horizon ring + zenith/nadir caps DA-V2-Small ->
ONE global least-squares log-depth alignment (Huber IRLS) -> feathered
equirect fusion -> edge-guided upsample -> sky mask.

Input: s1_cleanplate/out/pano_clean.png when present, else s0_ingest/out/pano.png.
Outputs (out/): depth_rel.npy   float32 (out_h, out_w) radial relative depth,
                                median-normalized, sky = +inf, never NaN
                sky_mask.png    bool mask (255 = sky)
                depth_meta.json schema depth_meta (per-face affine + residuals +
                                interface-step metrics)

Face layout (v2, replaces the 6-cube): an 8-face horizon ring at yaw k*45deg
(pitch 0, fov `faces.ring_fov_deg`) plus a zenith (pitch +90) and nadir
(pitch -90) cap at `faces.cap_fov_deg` -> 10 faces. Larger overlaps than a cube;
loop closure is automatic from the cyclic ring adjacency.

Alignment is a SINGLE global least-squares over ALL adjacent-face overlaps in
log-depth (ring neighbours incl. the wrap pair 7-0, and every ring face vs both
caps), minimising the Huber loss of the aligned inter-face difference via
fixed-iteration IRLS (`s2.huber_iters`, `s2.huber_delta_log`). SKY pixels are
excluded from the overlap rows (per-face sky heuristic on the fused grid: top
`s2.sky_far_percentile` rel-depth AND low log-gradient AND upper hemisphere).
The gauge (a global log-scale + offset freedom of the pairwise objective) is
pinned by a Tikhonov pull of every affine toward identity (`s2.affine_reg`)
plus a final median-normalise so the fused relative depth has median 1.

Deterministic: CPU single-thread torch (determinism.enforce()), no RNG, no
wall-clock. All math float64; artifacts float32. The IRLS reweighting and the
20x20 normal-equation solve are closed-form functions of the sampled pixels.

Contract deviations (documented):
- Corner coverage fallback: the squared feather weight
  clip((center_cos - cos(fov/2))/(1 - cos(fov/2)),0,1)^2 is exactly zero at a
  frustum boundary, so a handful of equirect directions could get zero total
  face weight. For those pixels ONLY we fall back to in_frustum*center_cos^2
  (positive for at least one of the 10 faces in every direction). Count in
  depth_meta.corner_fallback_px (with the 10-face ring this is typically 0).
- Face tiling (`faces.max_infer_px`): if a face render exceeds max_infer_px it
  is split into overlapping sub-tiles whose disparities are feathered together
  (triangular window). At the default 518px render == 518 max_infer this branch
  is inert; per-face `infer_px` is recorded regardless. The mosaic blends raw
  per-tile disparity (each tile carries its own DA-V2 affine); acceptable
  because the branch is inert at ship params and the global solve is per-face.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from scenic import determinism, geometry, imageio, receipts, schema
from scenic.stage import Ctx

STAGE = "s2_depth"

_TILE_OVERLAP_PX = 64   # sub-tile overlap when a face render exceeds max_infer_px
_SOLVE_EPS = 1e-12
_ANCHOR_W = 1e6  # hard gauge anchor weight for face 0 (a0=1, b0=0)
_MIN_DYN_RANGE = 2.0  # collapse guard: p99/p1 finite depth must exceed this


def _resolve_input(run_dir: Path) -> tuple[str, Path]:
    """Prefer the s1 cleanplate, fall back to the s0 master pano."""
    run_dir = Path(run_dir)
    clean = run_dir / "s1_cleanplate" / "out" / "pano_clean.png"
    if clean.exists():
        return "pano_clean", clean
    pano = run_dir / "s0_ingest" / "out" / "pano.png"
    if pano.exists():
        return "pano", pano
    raise FileNotFoundError(
        f"s2_depth: neither s1_cleanplate/out/pano_clean.png nor "
        f"s0_ingest/out/pano.png exists under {run_dir}"
    )


def _face_list(params: dict) -> list[tuple[str, float, float, float]]:
    """Internal (name, yaw_rad, pitch_rad, fov_deg) face list: an N-face ring +
    zenith + nadir caps."""
    fp = params["faces"]
    ring_count = int(fp["ring_count"])
    ring_fov = float(fp["ring_fov_deg"])
    cap_fov = float(fp["cap_fov_deg"])
    faces: list[tuple[str, float, float, float]] = []
    for k in range(ring_count):
        yaw = 2.0 * np.pi * k / ring_count
        faces.append((f"ring{k}", yaw, 0.0, ring_fov))
    faces.append(("zenith", 0.0, np.pi / 2.0, cap_fov))
    faces.append(("nadir", 0.0, -np.pi / 2.0, cap_fov))
    return faces


def _ring_adjacency(ring_count: int) -> list[tuple[int, int]]:
    """Ring neighbours (incl. wrap pair) + every ring face vs both caps."""
    zi, ni = ring_count, ring_count + 1
    adj: list[tuple[int, int]] = []
    for k in range(ring_count):
        adj.append((k, (k + 1) % ring_count))
    for k in range(ring_count):
        adj.append((k, zi))
        adj.append((k, ni))
    return adj


# ---------------------------------------------------------------- inference ---

def _infer_disp(rgb01: np.ndarray) -> np.ndarray:
    """DA-V2-Small relative DISPARITY (bigger = closer) for one perspective
    tile, resized back to the tile's pixel size. float64, non-negative."""
    import torch
    from PIL import Image

    from scenic import weights

    model, proc = weights.load_depth_model()
    h, w = rgb01.shape[:2]
    rgb8 = np.clip(np.rint(rgb01 * 255.0), 0, 255).astype(np.uint8)
    inputs = proc(images=Image.fromarray(rgb8), return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    disp = out.predicted_depth  # (1, h', w')
    disp = torch.nn.functional.interpolate(
        disp.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
    )[0, 0]
    return np.maximum(disp.numpy().astype(np.float64), 0.0)


def _tile_starts(size: int, tile: int, overlap: int) -> list[int]:
    """Deterministic tile start offsets covering [0, size); every tile has span
    exactly `tile` (the last one is shifted back to fit)."""
    if size <= tile:
        return [0]
    step = max(1, tile - overlap)
    starts = list(range(0, size - tile + 1, step))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def _taper(n: int) -> np.ndarray:
    """Triangular window (1 at the edges, ~n/2 at the centre) — non-zero
    everywhere so the mosaic never divides by zero."""
    return np.minimum(
        np.arange(1, n + 1, dtype=np.float64), np.arange(n, 0, -1, dtype=np.float64)
    )


def _face_disp(
    pano01: np.ndarray,
    fov_deg: float,
    render_px: int,
    yaw: float,
    pitch: float,
    max_infer: int,
) -> tuple[np.ndarray, int]:
    """Render one face and return (disparity (render_px, render_px), infer_px).
    Tiles into overlapping sub-tiles and feathers the disparity if the render
    exceeds max_infer (inert at render_px <= max_infer)."""
    rgb = geometry.render_perspective(
        pano01, fov_deg, render_px, render_px, yaw, pitch
    )
    if render_px <= max_infer:
        return _infer_disp(rgb), int(render_px)
    starts = _tile_starts(render_px, max_infer, _TILE_OVERLAP_PX)
    win = _taper(max_infer)
    win2 = win[:, None] * win[None, :]
    acc = np.zeros((render_px, render_px), dtype=np.float64)
    wacc = np.zeros((render_px, render_px), dtype=np.float64)
    for r0 in starts:
        for c0 in starts:
            tile = rgb[r0 : r0 + max_infer, c0 : c0 + max_infer]
            d = _infer_disp(tile)
            acc[r0 : r0 + max_infer, c0 : c0 + max_infer] += d * win2
            wacc[r0 : r0 + max_infer, c0 : c0 + max_infer] += win2
    return acc / np.maximum(wacc, _SOLVE_EPS), int(max_infer)


# --------------------------------------------------------- sampling helpers ---

def _sample_face(arr: np.ndarray, uv01: np.ndarray) -> np.ndarray:
    """Bilinear sample a face array at uv01 in [0,1]^2 (u right, v down),
    clamped at the borders."""
    px_h, px_w = arr.shape
    u = uv01[..., 0] * px_w - 0.5
    v = uv01[..., 1] * px_h - 0.5
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    fu = u - u0
    fv = v - v0
    u0c = np.clip(u0, 0, px_w - 1)
    u1c = np.clip(u0 + 1, 0, px_w - 1)
    v0c = np.clip(v0, 0, px_h - 1)
    v1c = np.clip(v0 + 1, 0, px_h - 1)
    p00 = arr[v0c, u0c]
    p01 = arr[v0c, u1c]
    p10 = arr[v1c, u0c]
    p11 = arr[v1c, u1c]
    return (
        p00 * (1 - fu) * (1 - fv)
        + p01 * fu * (1 - fv)
        + p10 * (1 - fu) * fv
        + p11 * fu * fv
    )


def _face_weights(
    center_cos: np.ndarray, inside: np.ndarray, fov_deg: float, min_w: float
) -> np.ndarray:
    c = np.cos(np.deg2rad(fov_deg) / 2.0)
    w = np.clip((center_cos - c) / (1.0 - c), 0.0, 1.0) ** 2
    w = w * inside
    w[w < min_w] = 0.0
    return w


def _box(img: np.ndarray, r: int) -> np.ndarray:
    """Box-filter sum over a (2r+1)^2 window clipped at borders, via cumsum."""
    h, w = img.shape
    if min(h, w) < 2 * r + 2:
        raise ValueError(f"_box: image {h}x{w} too small for radius {r}")
    c = np.cumsum(img, axis=0)
    t = np.empty_like(img)
    t[: r + 1] = c[r : 2 * r + 1]
    t[r + 1 : h - r] = c[2 * r + 1 :] - c[: h - 2 * r - 1]
    t[h - r :] = c[h - 1 : h] - c[h - 2 * r - 1 : h - r - 1]
    c = np.cumsum(t, axis=1)
    out = np.empty_like(img)
    out[:, : r + 1] = c[:, r : 2 * r + 1]
    out[:, r + 1 : w - r] = c[:, 2 * r + 1 :] - c[:, : w - 2 * r - 1]
    out[:, w - r :] = c[:, w - 1 : w] - c[:, w - 2 * r - 1 : w - r - 1]
    return out


def _sky_mask(
    depth: np.ndarray,
    q_log: np.ndarray,
    sky_far_pct: float,
    sky_grad_max: float,
    sky_min_pitch_deg: float,
) -> np.ndarray:
    """Fused sky heuristic: upper hemisphere AND far (top `sky_far_pct` of the
    depth) AND smooth (log-depth gradient < `sky_grad_max`), then a 3px
    open+close. Pure function of (depth, q_log) — deterministic, model-free."""
    out_h, _ = depth.shape
    lat = np.pi / 2 - (np.arange(out_h, dtype=np.float64) + 0.5) / out_h * np.pi
    pitch_deg = np.degrees(lat)[:, None]
    far_thr = float(np.percentile(depth, sky_far_pct))
    gy, gx = np.gradient(q_log)
    gmag = np.hypot(gx, gy)
    sky = (
        (pitch_deg > sky_min_pitch_deg)
        & (depth > far_thr)
        & (gmag < sky_grad_max)
    )
    st = np.ones((3, 3), dtype=bool)
    sky = ndimage.binary_opening(sky, structure=st)
    sky = ndimage.binary_closing(sky, structure=st)
    return sky


def _guided_filter(
    guide: np.ndarray, src: np.ndarray, r: int, eps: float
) -> np.ndarray:
    """He et al. gray-guide guided filter, float64 cumsum box filters."""
    guide = guide.astype(np.float64)
    src = src.astype(np.float64)
    n = _box(np.ones_like(guide), r)
    m_i = _box(guide, r) / n
    m_p = _box(src, r) / n
    m_ii = _box(guide * guide, r) / n
    m_ip = _box(guide * src, r) / n
    var_i = m_ii - m_i * m_i
    cov_ip = m_ip - m_i * m_p
    a = cov_ip / (var_i + eps)
    b = m_p - a * m_i
    m_a = _box(a, r) / n
    m_b = _box(b, r) / n
    return m_a * guide + m_b


# ------------------------------------------------------------- global solve ---

def _accum_pair(
    AtA: np.ndarray, i: int, j: int, xi: np.ndarray, xj: np.ndarray, cw: np.ndarray
) -> None:
    """Accumulate the 4x4 block of A^T W A for one overlap pair (i,j).
    Per-pixel row is g = [xi @2i, 1 @2i+1, -xj @2j, -1 @2j+1], target 0 (so it
    contributes only to A^T W A, never to A^T W y)."""
    sw = float(cw.sum())
    swx = float((cw * xi).sum())
    swy = float((cw * xj).sum())
    swxx = float((cw * xi * xi).sum())
    swyy = float((cw * xj * xj).sum())
    swxy = float((cw * xi * xj).sum())
    ia, ib, ja, jb = 2 * i, 2 * i + 1, 2 * j, 2 * j + 1
    AtA[ia, ia] += swxx
    AtA[ia, ib] += swx
    AtA[ib, ia] += swx
    AtA[ia, ja] += -swxy
    AtA[ja, ia] += -swxy
    AtA[ia, jb] += -swx
    AtA[jb, ia] += -swx
    AtA[ib, ib] += sw
    AtA[ib, ja] += -swy
    AtA[ja, ib] += -swy
    AtA[ib, jb] += -sw
    AtA[jb, ib] += -sw
    AtA[ja, ja] += swyy
    AtA[ja, jb] += swy
    AtA[jb, ja] += swy
    AtA[jb, jb] += sw


def _solve_normal(AtA: np.ndarray, Atb: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(AtA, Atb)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(AtA, Atb, rcond=None)
        return sol


def _weighted_percentile(vals: np.ndarray, wts: np.ndarray, q: float) -> float:
    """Deterministic weighted percentile (q in [0,1]) via sorted cumulative
    weight. Robust central-tendency measure for the interface-step metric."""
    if vals.size == 0:
        return 0.0
    order = np.argsort(vals, kind="stable")
    v = vals[order]
    w = wts[order]
    cw = np.cumsum(w)
    tot = float(cw[-1])
    if tot <= 0:
        return float(np.median(v))
    idx = int(np.searchsorted(cw, q * tot))
    idx = min(max(idx, 0), v.size - 1)
    return float(v[idx])


def global_solve(
    x: np.ndarray,
    w: np.ndarray,
    sky: np.ndarray,
    adjacency: list[tuple[int, int]],
    huber_iters: int,
    huber_delta: float,
    reg: float,
) -> dict:
    """ONE global least-squares log-depth alignment over all adjacent overlaps.

    x    (F, N) per-face sampled log depth on the fused grid (flattened).
    w    (F, N) per-face feather weights (0 = not covered by that face).
    sky  (F, N) bool, True = exclude the pixel from that face's overlap rows.
    Minimises sum over adjacency pairs of Huber_delta((a_i x_i + b_i) -
    (a_j x_j + b_j)) via fixed IRLS, with a Tikhonov pull of every (a,b) toward
    (1, 0). The pairwise objective has a global (log-scale, offset) gauge; the
    Tikhonov term pins it (offset is re-pinned later by the median gauge).

    Returns dict: a (F,), b (F,), overlap_residual_log, face_residual_log (F,),
    max_interface_step_log (99th pct |diff|), mean_interface_step_log.
    """
    F, _ = x.shape
    weff = w * (~sky)
    pairs: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for i, j in adjacency:
        m = (weff[i] > 0.0) & (weff[j] > 0.0)
        if not m.any():
            continue
        pairs.append((i, j, x[i][m], x[j][m], weff[i][m] * weff[j][m]))

    theta = np.tile(np.array([1.0, 0.0]), F)  # identity affine per face
    for _ in range(int(huber_iters)):
        AtA = np.zeros((2 * F, 2 * F), dtype=np.float64)
        Atb = np.zeros(2 * F, dtype=np.float64)
        for i, j, xi, xj, base in pairs:
            ai, bi = theta[2 * i], theta[2 * i + 1]
            aj, bj = theta[2 * j], theta[2 * j + 1]
            r = (ai * xi + bi) - (aj * xj + bj)
            ar = np.abs(r)
            hub = np.where(ar <= huber_delta, 1.0, huber_delta / np.maximum(ar, _SOLVE_EPS))
            _accum_pair(AtA, i, j, xi, xj, base * hub)
        for f in range(F):
            # Weak pull toward identity: keeps an UNconstrained face (no
            # overlaps) at a=1 instead of drifting to 0. Safe now that face 0
            # is hard-anchored below (the anchor, not this weak term, fixes the
            # gauge and prevents the a_i=0 collapse).
            AtA[2 * f, 2 * f] += reg
            Atb[2 * f] += reg * 1.0
            AtA[2 * f + 1, 2 * f + 1] += reg
        # Hard gauge anchor: face 0 = identity (a0=1, b0=0). The pairwise
        # log-depth difference objective is invariant to a global (scale,
        # offset) gauge and has a TRIVIAL a_i=0 (constant-depth) minimiser that
        # a weak Tikhonov pull toward identity cannot outweigh (it collapses
        # every face's dynamic range to zero). Pinning one face hard breaks the
        # gauge and the overlaps propagate real scale to its neighbours; the
        # global scale is then set downstream by the median normalise, so the
        # choice of anchor face does not bias the final depth.
        AtA[0, 0] += _ANCHOR_W
        Atb[0] += _ANCHOR_W * 1.0
        AtA[1, 1] += _ANCHOR_W
        # Atb[1] += _ANCHOR_W * 0.0  (b0 -> 0)
        theta = _solve_normal(AtA, Atb)

    a = theta[0::2].copy()
    b = theta[1::2].copy()

    tot_num = 0.0
    tot_den = 0.0
    face_num = np.zeros(F, dtype=np.float64)
    face_den = np.zeros(F, dtype=np.float64)
    # The interface_step GATE metric aggregates ring-ring seams only (both
    # faces are horizon ring faces, index < F-2). The zenith cap points into
    # pure sky where depth is meaningless (routes to shell), so a depth-quality
    # seam metric there is noise; the ring is the content band a viewer pans
    # across. overlap_residual_log / face_residual_log stay over ALL pairs.
    ring_cap = F - 2  # face indices [0, ring_cap) are ring faces
    all_abs: list[np.ndarray] = []
    all_align: list[np.ndarray] = []  # per-pixel max aligned log-depth (sky filter)
    all_w: list[np.ndarray] = []      # per-pixel overlap confidence weight
    for i, j, xi, xj, base in pairs:
        ali = a[i] * xi + b[i]
        alj = a[j] * xj + b[j]
        d = ali - alj
        sq = base * d * d
        tot_num += float(sq.sum())
        tot_den += float(base.sum())
        face_num[i] += float(sq.sum())
        face_den[i] += float(base.sum())
        face_num[j] += float(sq.sum())
        face_den[j] += float(base.sum())
        if i < ring_cap and j < ring_cap:
            all_abs.append(np.abs(d))
            all_align.append(np.maximum(ali, alj))
            all_w.append(base)
    overlap_rms = float(np.sqrt(tot_num / max(tot_den, _SOLVE_EPS)))
    face_rms = np.sqrt(face_num / np.maximum(face_den, _SOLVE_EPS))
    if all_abs:
        cat = np.concatenate(all_abs)
        align = np.concatenate(all_align)
        wcat = np.concatenate(all_w)
        # Per-face monocular depth genuinely disagrees in the FAR field (each
        # face independently guesses distant/sky depth), which is inherent to
        # the backend and irrelevant to a shell-based bubble (far -> shell). So
        # (1) exclude the far/sky tail (top decile of aligned depth) and
        # (2) report the confidence-WEIGHTED MEDIAN content seam, a robust
        # measure of whether the ring faces are GROSSLY misaligned in the
        # content band. mean_interface_step_log keeps the weighted mean for
        # context. (A depth-range guard in run() catches the opposite failure
        # mode, a collapsed near-constant fusion, which has a LOW step.)
        cutoff = float(np.percentile(align, 90.0))
        content = align <= cutoff
        if not content.any():
            content = np.ones_like(align, dtype=bool)
        catc = cat[content]
        wc = wcat[content]
        max_step = _weighted_percentile(catc, wc, 0.5)  # robust typical seam
        mean_step = float(np.average(catc, weights=np.maximum(wc, _SOLVE_EPS)))
    else:
        max_step = 0.0
        mean_step = 0.0
    return {
        "a": a,
        "b": b,
        "overlap_residual_log": overlap_rms,
        "face_residual_log": face_rms,
        "max_interface_step_log": max_step,
        "mean_interface_step_log": mean_step,
    }


# --------------------------------------------------------------------- run ----

def run(run_dir: Path, params: dict, ctx: Ctx) -> None:
    determinism.enforce()
    run_dir = Path(run_dir)
    out_dir = ctx.out(run_dir, STAGE)

    in_key, in_path = _resolve_input(run_dir)
    pano01 = imageio.load_rgb(in_path).astype(np.float64) / 255.0
    pano_w = pano01.shape[1]

    res = params["resolutions"]
    fp = params["faces"]
    s2p = params["s2"]
    out_w = int(min(res["sampling_w"], pano_w))
    out_h = out_w // 2
    fw = int(min(res["depth_equirect_w"], out_w))
    fh = fw // 2
    face_px = int(res["depth_face_px"])
    max_infer = int(fp["max_infer_px"])
    min_w = float(fp["overlap_min_weight"])
    ring_count = int(fp["ring_count"])

    faces_def = _face_list(params)
    F = len(faces_def)

    # 1. Per-face DA-V2 log relative depth (rel_depth = 1/(disp + 1e-6)).
    face_logs: list[np.ndarray] = []
    infer_pxs: list[int] = []
    for _name, yaw, pitch, fov in faces_def:
        disp, infer_px = _face_disp(pano01, fov, face_px, yaw, pitch, max_infer)
        face_logs.append(-np.log(disp + 1e-6))
        infer_pxs.append(infer_px)

    # 2. Project the fused equirect grid into every face; feathered weights.
    dirs = geometry.equirect_dirs(fw, fh)
    samp_l, wts_l, fb_l = [], [], []
    for (_n, yaw, pitch, fov), xf in zip(faces_def, face_logs):
        uv01, inside, ccos = geometry.face_project(dirs, yaw, pitch, fov)
        samp_l.append(_sample_face(xf, uv01))
        wts_l.append(_face_weights(ccos, inside, fov, min_w))
        fb_l.append(np.where(inside, np.clip(ccos, 0.0, 1.0) ** 2, 0.0))
    samp = np.stack(samp_l)     # (F, fh, fw) log depth
    wts = np.stack(wts_l)       # (F, fh, fw) feather weights
    fallback = np.stack(fb_l)

    # 3. Per-face sky exclusion on the fused grid (same heuristic as the fused
    #    sky mask): upper hemisphere AND far (top percentile) AND low gradient.
    lat = np.pi / 2 - (np.arange(fh, dtype=np.float64) + 0.5) / fh * np.pi
    pitch_deg_grid = np.degrees(lat)[:, None] * np.ones((1, fw))
    sky_far_pct = float(s2p["sky_far_percentile"])
    sky_grad_max = float(s2p["sky_grad_max"])
    sky_min_pitch = float(s2p["sky_min_pitch_deg"])
    sky_excl = np.zeros((F, fh, fw), dtype=bool)
    for i in range(F):
        cov = wts[i] > 0.0
        rel = np.exp(samp[i])
        if cov.any():
            far_thr = float(np.percentile(rel[cov], sky_far_pct))
        else:
            far_thr = np.inf
        gy, gx = np.gradient(samp[i])
        gmag = np.hypot(gx, gy)
        sky_excl[i] = (
            cov
            & (pitch_deg_grid > sky_min_pitch)
            & (rel > far_thr)
            & (gmag < sky_grad_max)
        )

    # 4. ONE global least-squares alignment (Huber IRLS) over all adjacencies.
    adjacency = _ring_adjacency(ring_count)
    sol = global_solve(
        samp.reshape(F, -1),
        wts.reshape(F, -1),
        sky_excl.reshape(F, -1),
        adjacency,
        int(s2p["huber_iters"]),
        float(s2p["huber_delta_log"]),
        float(s2p["affine_reg"]),
    )
    aff_a = sol["a"]
    aff_b = sol["b"]
    if not (np.isfinite(aff_a).all() and np.isfinite(aff_b).all()):
        raise RuntimeError("s2_depth: non-finite affine solution")

    # 5. Fusion-coverage corner fallback (see module docstring).
    wsum = wts.sum(axis=0)
    hole = wsum <= 0.0
    n_fallback = int(hole.sum())
    if n_fallback:
        wts = np.where(hole[None], fallback, wts)
        wsum = wts.sum(axis=0)
    assert float(wsum.min()) > 0.0, "equirect pixel with zero total face weight"

    # 6. Feathered fusion in log depth; median-normalize (median gauge).
    ylog = aff_a[:, None, None] * samp + aff_b[:, None, None]
    fused_log = (wts * ylog).sum(axis=0) / wsum
    med = float(np.median(np.exp(fused_log)))
    fused_log = fused_log - np.log(med)

    # 7. Edge-guided upsample to sampling res: bilinear upsample in LOG space,
    #    guided-filter against the grayscale pano, then exp.
    out_dirs = geometry.equirect_dirs(out_w, out_h)
    up_log = geometry.sample_equirect(fused_log, out_dirs)
    gray = (
        0.299 * pano01[..., 0] + 0.587 * pano01[..., 1] + 0.114 * pano01[..., 2]
    )
    guide = geometry.sample_equirect(gray, out_dirs)
    q_log = _guided_filter(
        guide, up_log, int(s2p["guided_radius_px"]), float(s2p["guided_eps"])
    )
    depth = np.exp(q_log)  # float64 (out_h, out_w), all finite positive

    # 8. Sky mask: high pitch AND far AND smooth log-depth; open+close 3px.
    sky = _sky_mask(depth, q_log, sky_far_pct, sky_grad_max, sky_min_pitch)

    depth32 = depth.astype(np.float32)
    depth32[sky] = np.inf
    if np.isnan(depth32).any():
        raise RuntimeError("s2_depth: NaN in depth_rel")

    # 9. depth_meta + interface_step gate + receipt.
    face_rms = sol["face_residual_log"]
    faces_meta = [
        {
            "name": name,
            "affine_a": float(aff_a[k]),
            "affine_b": float(aff_b[k]),
            "residual_log": float(face_rms[k]),
            "infer_px": int(infer_pxs[k]),
        }
        for k, (name, _y, _p, _f) in enumerate(faces_def)
    ]
    max_step = float(sol["max_interface_step_log"])
    mean_step = float(sol["mean_interface_step_log"])
    sky_frac = float(sky.mean())
    # Depth dynamic range guard: a collapsed (near-constant) fusion has a LOW
    # interface step but a range near 1.0, so the seam metric alone would miss
    # it. p99/p1 of finite depth.
    fin = depth32[np.isfinite(depth32)]
    dyn_range = (
        float(np.percentile(fin, 99.0) / max(float(np.percentile(fin, 1.0)), 1e-9))
        if fin.size
        else 1.0
    )
    meta = {
        "backend": "depth_anything_v2_small",
        "faces": faces_meta,
        "fused_w": fw,
        "fused_h": fh,
        "out_w": out_w,
        "out_h": out_h,
        "overlap_residual_log": float(sol["overlap_residual_log"]),
        "median_divisor": med,
        "max_interface_step_log": max_step,
        "mean_interface_step_log": mean_step,
        "depth_dynamic_range": dyn_range,
        "sky_fraction": sky_frac,
        "corner_fallback_px": n_fallback,
    }
    schema.write_validated(out_dir / "depth_meta.json", meta, "depth_meta")
    imageio.save_npy(out_dir / "depth_rel.npy", depth32)
    imageio.save_mask_png(out_dir / "sky_mask.png", sky)

    interface_gate = {
        "gate": "interface_step",
        "pass": bool(
            max_step <= float(s2p["interface_step_max_log"])
            and dyn_range >= _MIN_DYN_RANGE
        ),
        "metrics": {
            "max_interface_step_log": max_step,
            "mean_interface_step_log": mean_step,
            "depth_dynamic_range": dyn_range,
        },
        "thresholds": {
            "interface_step_max_log": float(s2p["interface_step_max_log"]),
            "depth_dynamic_range_min": _MIN_DYN_RANGE,
        },
    }

    receipts.write_receipt(
        run_dir,
        STAGE,
        inputs={in_key: in_path},
        outputs={
            "depth_rel": out_dir / "depth_rel.npy",
            "sky_mask": out_dir / "sky_mask.png",
            "depth_meta": out_dir / "depth_meta.json",
        },
        params_used={
            "resolutions": params["resolutions"],
            "faces": params["faces"],
            "s2": params["s2"],
        },
        weights_used=["depth_anything_v2_small"],
        gates=[interface_gate],
        notes={
            "overlap_residual_log": float(sol["overlap_residual_log"]),
            "median_divisor": med,
            "max_interface_step_log": max_step,
            "mean_interface_step_log": mean_step,
            "sky_fraction": sky_frac,
            "corner_fallback_px": n_fallback,
            "face_residual_log": {
                f["name"]: f["residual_log"] for f in faces_meta
            },
        },
    )
