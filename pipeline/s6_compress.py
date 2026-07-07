"""s6_compress: two-profile prune / merge / cap enforcement + .sog packaging.

Reads s4_place/out/splats.ply and, for EACH ship profile in
`params.s6.profiles` (review, quest), applies the same pipeline in order:

  1. opacity floor        drop sigmoid(opacity_logit) < s6.opacity_floor
  2. kNN isolation prune  cKDTree on non-shell xyz; drop non-shell splats whose
                          distance to the k-th neighbor exceeds
                          isolation_factor * own splat radius (shell
                          sparse and is excluded from both tree and pruning)
  3. duplicate merge      drop each BG splat whose nearest FG splat is closer
                          than merge_dist_factor * exp(max fg log_scale)
  4. cap enforcement      while count > PROFILE cap and retries remain, re-run
                          s4 placement with stride_multiplier *= 1.5 (this
                          rewrites the s4 outputs + receipt — documented,
                          logged behavior) and redo steps 1-3

Profiles run in a FIXED order: every non-primary profile first (sorted), then
`primary_profile` LAST, so the final on-disk s4 placement is deterministic and
equals the primary (quest) coarsening. Each profile starts from a clean natural
s4 baseline (multiplier 1.0) — if a prior profile coarsened s4 on disk, s4 is
re-run at multiplier 1.0 first so profiles stay independent; s4 is deterministic
so this reproduces the natural placement exactly.

Outputs per profile: scene_<profile>.ply (scenic layout), scene_<profile>_std.ply
(standard 3DGS layout, no scenic extra props), scene_<profile>.sog (the pinned
splat-transform tool output, zip re-normalized for determinism). PRIMARY aliases
scene.ply / scene_std.ply / scene.sog are byte copies of the primary_profile
files (s7 gates load scene.ply — the shipped asset). Counts + byte sizes per
profile land in out/compress.json.

Pure numpy + scipy; no torch, no RNG.
"""
from __future__ import annotations

import importlib
import shutil
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
    profiles_cfg = p6["profiles"]
    primary = str(p6["primary_profile"])
    viewer = str(p6["viewer_profile"])
    max_retries = int(p6["max_stride_retries"])
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

    def rerun_s4(multiplier: float) -> None:
        # importlib each time so a test monkeypatch of sys.modules is honored.
        s4_place = importlib.import_module("pipeline.s4_place")
        s4_place.run(run_dir, params, ctx, stride_multiplier=multiplier)

    # Fixed profile order: every non-primary profile first (sorted for
    # determinism), then the primary profile LAST so the final on-disk s4
    # placement is the primary (quest) coarsening. For {review, quest} this is
    # exactly [review, quest].
    order = sorted(k for k in profiles_cfg if k != primary) + [primary]

    # s4's natural placement (multiplier 1.0) is on disk when s6 begins.
    disk_multiplier = 1.0
    results: dict[str, tuple[plyio.SplatData, dict, list]] = {}

    for name in order:
        cap = int(profiles_cfg[name]["cap"])
        # Clean natural baseline: if a prior profile coarsened s4, reset it so
        # profiles are independent. s4 is deterministic -> reproduces natural.
        if disk_multiplier != 1.0:
            rerun_s4(1.0)
            disk_multiplier = 1.0
        multiplier = 1.0
        s, counts = steps()
        stride_retries: list[dict] = []
        while len(s) > cap and len(stride_retries) < max_retries:
            multiplier *= 1.5
            count_before = len(s)
            rerun_s4(multiplier)
            disk_multiplier = multiplier
            s, counts = steps()
            stride_retries.append({
                "multiplier": multiplier,
                "count_before": count_before,
                "count_after": len(s),
            })
        results[name] = (s, counts, stride_retries)

    # Per-profile artifacts + compress.profiles entries.
    profiles_json: dict[str, dict] = {}
    outputs: dict[str, Path] = {}
    for name in order:
        s, counts, stride_retries = results[name]
        cfg = profiles_cfg[name]
        scene_ply = out / f"scene_{name}.ply"
        std_ply = out / f"scene_{name}_std.ply"
        sog_path = out / f"scene_{name}.sog"
        plyio.write_splats(scene_ply, s)
        write_std_ply(std_ply, s)
        ply_to_sog(std_ply, sog_path, ctx.repo_root)
        profiles_json[name] = {
            "in_count": counts["in_count"],
            "after_opacity_floor": counts["after_opacity_floor"],
            "after_isolation_prune": counts["after_isolation_prune"],
            "after_merge": counts["after_merge"],
            "final_count": len(s),
            "stride_retries": stride_retries,
            "ply_bytes": int(scene_ply.stat().st_size),
            "sog_bytes": int(sog_path.stat().st_size),
            "target": int(cfg["target"]),
            "cap": int(cfg["cap"]),
            "sog_max_mb": float(cfg["sog_max_mb"]),
        }
        outputs[f"scene_{name}"] = scene_ply
        outputs[f"scene_{name}_std"] = std_ply
        outputs[f"scene_{name}_sog"] = sog_path

    # PRIMARY aliases: byte copies of the primary profile's files.
    scene_alias = out / "scene.ply"
    std_alias = out / "scene_std.ply"
    sog_alias = out / "scene.sog"
    shutil.copyfile(out / f"scene_{primary}.ply", scene_alias)
    shutil.copyfile(out / f"scene_{primary}_std.ply", std_alias)
    shutil.copyfile(out / f"scene_{primary}.sog", sog_alias)
    outputs["scene"] = scene_alias
    outputs["scene_std"] = std_alias
    outputs["sog"] = sog_alias

    compress = {
        "sog_tool": SOG_TOOL,
        "primary_profile": primary,
        "viewer_profile": viewer,
        "profiles": profiles_json,
    }
    schema.write_validated(out / "compress.json", compress, "compress")
    outputs["compress"] = out / "compress.json"

    receipts.write_receipt(
        run_dir,
        STAGE,
        inputs={"splats": splats_path},
        outputs=outputs,
        params_used={"s6": p6},
        weights_used=[],
        notes={
            "primary_profile": primary,
            "viewer_profile": viewer,
            "profiles": {
                name: {
                    "final_count": profiles_json[name]["final_count"],
                    "sog_bytes": profiles_json[name]["sog_bytes"],
                    "stride_retry_count": len(profiles_json[name]["stride_retries"]),
                }
                for name in order
            },
        },
    )
