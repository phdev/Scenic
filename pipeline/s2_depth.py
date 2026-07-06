"""S2 depth: cubemap DA-V2-Small -> joint affine log-depth fusion ->
edge-guided upsample -> sky mask.

Input: s1_cleanplate/out/pano_clean.png when present, else s0_ingest/out/pano.png.
Outputs (out/): depth_rel.npy   float32 (out_h, out_w) radial relative depth,
                                median-normalized, sky = +inf, never NaN
                sky_mask.png    bool mask (255 = sky)
                depth_meta.json schema depth_meta (per-face affine + residuals)

Deterministic: CPU single-thread torch (determinism.enforce()), no RNG, no
wall-clock, stride-based subsampling only.

Contract deviation (documented): the spec feather weight
clip((center_cos - cos(fov/2))/(1 - cos(fov/2)), 0, 1)^2 is exactly zero at
cube corners for fov < 109.47 deg (a corner direction is 54.74 deg away from
every face axis, > fov/2 = 50 deg), so a handful of equirect pixels would get
zero total weight and the coverage assert would fail. For those pixels ONLY we
fall back to weight = in_frustum * center_cos^2 (positive for at least one
face for every direction, since 100 deg cube faces cover the sphere). The
count of fallback pixels is recorded in depth_meta as corner_fallback_px.
A tiny Tikhonov pull (1e-3) toward (a=1, b=0) keeps the joint affine solve
well-posed on texture-poor scenes; it is negligible against real overlap rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from scenic import determinism, geometry, imageio, receipts, schema
from scenic.stage import Ctx

STAGE = "s2_depth"

_AFFINE_REG = 1e-3      # Tikhonov pull toward identity affine (see docstring)
_SUBSAMPLE_STRIDE = 2   # deterministic pixel subsample for the lstsq assembly


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


def _face_log_depth(
    pano01: np.ndarray, fov_deg: float, face_px: int, yaw: float, pitch: float
) -> np.ndarray:
    """Render one cube face, run DA-V2-Small, return log relative depth
    (face_px, face_px) float64. DA-V2 outputs relative DISPARITY (bigger =
    closer); rel_depth = 1/(disp + 1e-6)."""
    import torch
    from PIL import Image

    from scenic import weights

    model, proc = weights.load_depth_model()
    rgb = geometry.render_perspective(pano01, fov_deg, face_px, face_px, yaw, pitch)
    rgb8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    inputs = proc(images=Image.fromarray(rgb8), return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    disp = out.predicted_depth  # (1, h', w')
    disp = torch.nn.functional.interpolate(
        disp.unsqueeze(1),
        size=(face_px, face_px),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    disp_np = np.maximum(disp.numpy().astype(np.float64), 0.0)
    return -np.log(disp_np + 1e-6)  # log(1 / (disp + 1e-6))


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
    """Box-filter sum over a (2r+1)^2 window clipped at borders, via cumsum.
    float64 in, float64 out. Requires both dims >= 2r+2."""
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


def _solve_joint_affine(
    xs: np.ndarray, ws: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Joint per-face affine alignment in log depth, face 0 anchored (a=1,b=0).
    xs, ws: (6, N) sampled log depths + weights on the subsampled fused grid.
    Minimizes sum_p sum_{i<j} w_i w_j ((a_i x_i + b_i) - (a_j x_j + b_j))^2
    over the 10 free params via np.linalg.lstsq.
    Returns (a[6], b[6], overlap_rms_log, face_rms_log[6])."""
    rows_a: list[np.ndarray] = []
    rows_rhs: list[np.ndarray] = []
    pairs: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(6):
        for j in range(i + 1, 6):
            m = (ws[i] > 0) & (ws[j] > 0)
            if not m.any():
                continue
            xi = xs[i][m]
            xj = xs[j][m]
            wij = ws[i][m] * ws[j][m]
            s = np.sqrt(wij)
            k = xi.size
            block = np.zeros((k, 10), dtype=np.float64)
            if i == 0:
                # x_0 - (a_j x_j + b_j) = 0  ->  a_j x_j + b_j = x_0
                block[:, 2 * (j - 1)] = s * xj
                block[:, 2 * (j - 1) + 1] = s
                rhs = s * xi
            else:
                block[:, 2 * (i - 1)] = s * xi
                block[:, 2 * (i - 1) + 1] = s
                block[:, 2 * (j - 1)] = -s * xj
                block[:, 2 * (j - 1) + 1] = -s
                rhs = np.zeros(k, dtype=np.float64)
            rows_a.append(block)
            rows_rhs.append(rhs)
            pairs.append((i, j, xi, xj, wij))
    reg = np.zeros((10, 10), dtype=np.float64)
    rhs_reg = np.zeros(10, dtype=np.float64)
    for f in range(5):
        reg[2 * f, 2 * f] = _AFFINE_REG
        rhs_reg[2 * f] = _AFFINE_REG  # pull a -> 1
        reg[2 * f + 1, 2 * f + 1] = _AFFINE_REG  # pull b -> 0
    mat = np.concatenate(rows_a + [reg], axis=0)
    vec = np.concatenate(rows_rhs + [rhs_reg], axis=0)
    sol, *_ = np.linalg.lstsq(mat, vec, rcond=None)
    a = np.concatenate([[1.0], sol[0::2]])
    b = np.concatenate([[0.0], sol[1::2]])

    tot_num = 0.0
    tot_den = 0.0
    face_num = np.zeros(6, dtype=np.float64)
    face_den = np.zeros(6, dtype=np.float64)
    for i, j, xi, xj, wij in pairs:
        d = (a[i] * xi + b[i]) - (a[j] * xj + b[j])
        sq = wij * d * d
        tot_num += float(sq.sum())
        tot_den += float(wij.sum())
        face_num[i] += sq.sum()
        face_den[i] += wij.sum()
        face_num[j] += sq.sum()
        face_den[j] += wij.sum()
    overlap_rms = float(np.sqrt(tot_num / max(tot_den, 1e-12)))
    face_rms = np.sqrt(face_num / np.maximum(face_den, 1e-12))
    return a, b, overlap_rms, face_rms


def run(run_dir: Path, params: dict, ctx: Ctx) -> None:
    determinism.enforce()
    run_dir = Path(run_dir)
    out_dir = ctx.out(run_dir, STAGE)

    in_key, in_path = _resolve_input(run_dir)
    pano01 = imageio.load_rgb(in_path).astype(np.float64) / 255.0
    pano_w = pano01.shape[1]

    res = params["resolutions"]
    cube = params["cubemap"]
    s2p = params["s2"]
    out_w = int(min(res["sampling_w"], pano_w))
    out_h = out_w // 2
    fw = int(min(res["depth_equirect_w"], out_w))
    fh = fw // 2
    fov = float(cube["fov_deg"])
    face_px = int(res["depth_face_px"])
    min_w = float(cube["overlap_min_weight"])

    # 1. Per-face DA-V2 log relative depth.
    face_logs = [
        _face_log_depth(pano01, fov, face_px, yaw, pitch)
        for _, yaw, pitch in geometry.CUBE_FACES
    ]

    # 2. Project the fused equirect grid into every face; feathered weights.
    dirs = geometry.equirect_dirs(fw, fh)
    samp_l, wts_l, fb_l = [], [], []
    for (_, yaw, pitch), xf in zip(geometry.CUBE_FACES, face_logs):
        uv01, inside, ccos = geometry.face_project(dirs, yaw, pitch, fov)
        samp_l.append(_sample_face(xf, uv01))
        wts_l.append(_face_weights(ccos, inside, fov, min_w))
        fb_l.append(np.where(inside, np.clip(ccos, 0.0, 1.0) ** 2, 0.0))
    samp = np.stack(samp_l)  # (6, fh, fw)
    wts = np.stack(wts_l)
    fallback = np.stack(fb_l)

    # Cube-corner coverage fallback (see module docstring).
    wsum = wts.sum(axis=0)
    hole = wsum <= 0.0
    n_fallback = int(hole.sum())
    if n_fallback:
        wts = np.where(hole[None], fallback, wts)
        wsum = wts.sum(axis=0)
    assert float(wsum.min()) > 0.0, "equirect pixel with zero total face weight"

    # 3. Joint affine alignment on a deterministic stride-2 pixel subsample.
    ws_sub = wts[:, ::_SUBSAMPLE_STRIDE, ::_SUBSAMPLE_STRIDE].reshape(6, -1)
    xs_sub = samp[:, ::_SUBSAMPLE_STRIDE, ::_SUBSAMPLE_STRIDE].reshape(6, -1)
    aff_a, aff_b, overlap_rms, face_rms = _solve_joint_affine(xs_sub, ws_sub)
    if not (np.isfinite(aff_a).all() and np.isfinite(aff_b).all()):
        raise RuntimeError("s2_depth: non-finite affine solution")

    # 4. Feathered fusion in log depth; median-normalize.
    ylog = aff_a[:, None, None] * samp + aff_b[:, None, None]
    fused_log = (wts * ylog).sum(axis=0) / wsum
    med = float(np.median(np.exp(fused_log)))
    fused_log = fused_log - np.log(med)

    # 5. Edge-guided upsample to sampling res: bilinear upsample in LOG space,
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

    # 6. Sky mask: high pitch AND far AND smooth log-depth; open+close 3px.
    lat = np.pi / 2 - (np.arange(out_h, dtype=np.float64) + 0.5) / out_h * np.pi
    pitch_deg = np.degrees(lat)[:, None]
    far_thr = float(np.percentile(depth, float(s2p["sky_far_percentile"])))
    gy, gx = np.gradient(q_log)
    gmag = np.hypot(gx, gy)
    sky = (
        (pitch_deg > float(s2p["sky_min_pitch_deg"]))
        & (depth > far_thr)
        & (gmag < float(s2p["sky_grad_max"]))
    )
    st = np.ones((3, 3), dtype=bool)
    sky = ndimage.binary_opening(sky, structure=st)
    sky = ndimage.binary_closing(sky, structure=st)

    depth32 = depth.astype(np.float32)
    depth32[sky] = np.inf
    if np.isnan(depth32).any():
        raise RuntimeError("s2_depth: NaN in depth_rel")

    # 7. Artifacts + receipt.
    faces_meta = [
        {
            "name": name,
            "affine_a": float(aff_a[k]),
            "affine_b": float(aff_b[k]),
            "residual_log": float(face_rms[k]),
        }
        for k, (name, _, _) in enumerate(geometry.CUBE_FACES)
    ]
    meta = {
        "backend": "depth_anything_v2_small",
        "faces": faces_meta,
        "fused_w": fw,
        "fused_h": fh,
        "out_w": out_w,
        "out_h": out_h,
        "overlap_residual_log": overlap_rms,
        "median_divisor": med,
        "corner_fallback_px": n_fallback,
        "sky_fraction": float(sky.mean()),
    }
    schema.write_validated(out_dir / "depth_meta.json", meta, "depth_meta")
    imageio.save_npy(out_dir / "depth_rel.npy", depth32)
    imageio.save_mask_png(out_dir / "sky_mask.png", sky)

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
            "cubemap": params["cubemap"],
            "s2": params["s2"],
        },
        weights_used=["depth_anything_v2_small"],
        notes={
            "overlap_residual_log": overlap_rms,
            "median_divisor": med,
            "face_residual_log": {
                f["name"]: f["residual_log"] for f in faces_meta
            },
            "corner_fallback_px": n_fallback,
        },
    )
