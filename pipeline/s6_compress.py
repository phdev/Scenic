"""s6_compress: prune / merge / cap enforcement + .sog packaging.

Reads s4_place/out/splats.ply and applies, in order:
  1. opacity floor        drop sigmoid(opacity_logit) < s6.opacity_floor
  2. kNN isolation prune  cKDTree on non-shell xyz; drop non-shell splats whose
                          distance to the k-th neighbor exceeds
                          isolation_factor * own splat radius (shell
                          sparse and is excluded from both tree and pruning)
  3. duplicate merge      drop each BG splat whose nearest FG splat is closer
                          than merge_dist_factor * exp(max fg log_scale)
  4. cap enforcement      while count > splat_cap and retries remain, re-run
                          s4 placement with stride_multiplier *= 1.5 (this
                          rewrites the s4 outputs + receipt — documented,
                          logged behavior) and redo steps 1-3

Then writes out/scene.ply (scenic layout), out/scene_std.ply (standard 3DGS
layout, no scenic extra props), and out/scene.sog via the pinned
splat-transform node tool. The .sog is a zip; it is re-normalized (entries
sorted by name, fixed DOS epoch date, deflate level 6, zeroed attrs) so the
bytes are deterministic. Counts per step land in out/compress.json.

Pure numpy + scipy; no torch, no RNG.
"""
from __future__ import annotations

import importlib
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from scenic import plyio, receipts, schema
from scenic.stage import Ctx

STAGE = "s6_compress"
SOG_TOOL = "splat-transform@2.7.1+ziprenorm"

# standard 3DGS float layout = scenic float layout minus layer/origin_stage
_STD_FLOAT_PROPS = [
    "x", "y", "z", "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]


# ------------------------------------------------------------ prune / merge


def _opacity_floor(s: plyio.SplatData, floor: float) -> plyio.SplatData:
    """Step 1: drop splats with sigmoid(opacity_logit) < floor."""
    op = plyio.logit_to_opacity(s.opacity_logit.astype(np.float64))
    return s.take(np.flatnonzero(op >= floor))


def _isolation_prune(s: plyio.SplatData, k: int, factor: float) -> plyio.SplatData:
    """Step 2: kNN isolation prune on non-shell splats only.

    d_k = distance to the k-th nearest non-shell neighbor; drop non-shell
    splats with d_k > factor * own_radius, where own_radius = exp(max
    log_scale) of the splat itself. The criterion must be relative to the
    splat's own footprint, NOT a global median: grid spacing grows
    proportionally with depth (stride x angular pixel size x depth), so a
    global median — dominated by dense near-field ground splats — mass-prunes
    everything past mid-distance (observed: 30% of a bubble scene removed,
    leaving a void band at the horizon). A splat whose k-th neighbor sits
    beyond `factor` times its own radius is a genuinely isolated speckle at
    any depth. Shell splats are excluded from both the tree and the pruning
    (the shell is intentionally sparse). Deterministic: single-threaded
    query, stable index order.
    """
    nonshell = np.flatnonzero(s.layer != plyio.LAYER_SHELL)
    if nonshell.size <= 1:
        return s
    pts = s.xyz[nonshell].astype(np.float64)
    kk = min(int(k), nonshell.size - 1)  # adapt when fewer points than k+1
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=kk + 1, workers=1)  # col 0 = self (dist 0)
    d_k = dists[:, kk]
    own_radius = np.exp(
        s.log_scales[nonshell].max(axis=1).astype(np.float64)
    )
    drop_local = d_k > factor * own_radius
    keep = np.ones(len(s), dtype=bool)
    keep[nonshell[drop_local]] = False
    return s.take(np.flatnonzero(keep))


def _merge_duplicates(s: plyio.SplatData, merge_dist_factor: float) -> plyio.SplatData:
    """Step 3: drop each BG splat whose nearest FG splat is closer than
    merge_dist_factor * exp(max fg log_scale) of that neighbor."""
    fg = np.flatnonzero(s.layer == plyio.LAYER_FG)
    bg = np.flatnonzero(s.layer == plyio.LAYER_BG)
    if fg.size == 0 or bg.size == 0:
        return s
    tree = cKDTree(s.xyz[fg].astype(np.float64))
    dists, nn = tree.query(s.xyz[bg].astype(np.float64), k=1, workers=1)
    fg_max_scale = np.exp(s.log_scales[fg][nn].max(axis=1).astype(np.float64))
    drop_local = dists < merge_dist_factor * fg_max_scale
    keep = np.ones(len(s), dtype=bool)
    keep[bg[drop_local]] = False
    return s.take(np.flatnonzero(keep))


# ------------------------------------------------------------ .sog packaging


def write_std_ply(path: Path | str, s: plyio.SplatData) -> None:
    """Standard 3DGS binary-little-endian PLY: the scenic float layout
    without the scenic-only uchar props (layer, origin_stage)."""
    n = len(s)
    q = plyio.canonical_quat(s.quat_wxyz.astype(np.float32))
    cols = np.concatenate(
        [s.xyz, s.normals, s.f_dc, s.opacity_logit[:, None], s.log_scales, q],
        axis=1,
    ).astype("<f4")
    if not np.all(np.isfinite(cols.astype(np.float64))):
        raise ValueError("non-finite values in std PLY columns")
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {p}" for p in _STD_FLOAT_PROPS]
    header += ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(np.ascontiguousarray(cols).tobytes())


def _renorm_zip(path: Path) -> None:
    """Rewrite a zip for byte-determinism: entries sorted by name, DOS-epoch
    date_time, ZIP_DEFLATED level 6, fixed external attrs, fixed creator.

    Note: CPython's zipfile force-sets external_attr to 0o600 << 16 whenever
    it is 0 (`_open_to_write`), so a literal zero is unrepresentable; we pin
    that same constant explicitly — still a fixed value, still deterministic.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as zin:
        names = sorted(zin.namelist())
        datas = {n: zin.read(n) for n in names}
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w") as zout:
        for name in names:
            zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o600 << 16  # stdlib floor; see docstring
            zi.create_system = 3
            zout.writestr(zi, datas[name],
                          compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    tmp.replace(path)


def ply_to_sog(ply_path: Path, sog_path: Path, repo_root: Path) -> None:
    """Run the pinned splat-transform tool, then normalize the zip."""
    tool = (Path(repo_root) / "tools" / "node" / "node_modules" / ".bin"
            / "splat-transform")
    if not tool.exists():
        raise FileNotFoundError(
            f"splat-transform not installed at {tool} (run `make setup`)")
    proc = subprocess.run(
        [str(tool), "-w", "-q", "--max-workers", "0",
         str(ply_path), str(sog_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not Path(sog_path).exists():
        raise RuntimeError(
            f"splat-transform failed (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    _renorm_zip(Path(sog_path))


# ------------------------------------------------------------------- stage


def run(run_dir: Path, params: dict, ctx: Ctx) -> None:
    run_dir = Path(run_dir)
    out = ctx.out(run_dir, STAGE)
    p6 = params["s6"]
    cap = int(params["splat_cap"])
    splats_path = run_dir / "s4_place" / "out" / "splats.ply"
    if not splats_path.exists():
        raise FileNotFoundError(f"missing s4 output {splats_path}")

    def steps() -> tuple[plyio.SplatData, dict]:
        raw = plyio.read_splats(splats_path)
        counts: dict = {"in_count": len(raw)}
        s = _opacity_floor(raw, float(p6["opacity_floor"]))
        counts["after_opacity_floor"] = len(s)
        s = _isolation_prune(s, int(p6["knn_k"]), float(p6["isolation_factor"]))
        counts["after_isolation_prune"] = len(s)
        s = _merge_duplicates(s, float(p6["merge_dist_factor"]))
        counts["after_merge"] = len(s)
        return s, counts

    s, counts = steps()

    # step 4: cap enforcement via s4 re-placement with a coarser stride
    stride_retries: list[dict] = []
    multiplier = 1.0
    while len(s) > cap and len(stride_retries) < int(p6["max_stride_retries"]):
        multiplier *= 1.5
        count_before = len(s)
        s4_place = importlib.import_module("pipeline.s4_place")
        s4_place.run(run_dir, params, ctx, stride_multiplier=multiplier)
        s, counts = steps()
        stride_retries.append({
            "multiplier": multiplier,
            "count_before": count_before,
            "count_after": len(s),
        })

    # step 5: scenic master ply
    scene_ply = out / "scene.ply"
    plyio.write_splats(scene_ply, s)

    # step 6: standard 3DGS ply -> .sog (zip re-normalized for determinism)
    std_ply = out / "scene_std.ply"
    write_std_ply(std_ply, s)
    sog_path = out / "scene.sog"
    ply_to_sog(std_ply, sog_path, ctx.repo_root)

    compress = {
        **counts,
        "final_count": len(s),
        "stride_retries": stride_retries,
        "sog_bytes": int(sog_path.stat().st_size),
        "sog_tool": SOG_TOOL,
    }
    schema.write_validated(out / "compress.json", compress, "compress")

    receipts.write_receipt(
        run_dir,
        STAGE,
        inputs={"splats": splats_path},
        outputs={
            "scene": scene_ply,
            "scene_std": std_ply,
            "sog": sog_path,
            "compress": out / "compress.json",
        },
        params_used={"s6": p6, "splat_cap": cap},
        weights_used=[],
        notes={
            "counts": {
                "in_count": counts["in_count"],
                "after_opacity_floor": counts["after_opacity_floor"],
                "after_isolation_prune": counts["after_isolation_prune"],
                "after_merge": counts["after_merge"],
                "final_count": len(s),
            },
            "stride_retry_count": len(stride_retries),
        },
    )
