"""Pipeline harness: runs stages in registry order against one run dir.

    uv run python -m scenic.run --pano fixtures/test.jpg --out runs/a
    uv run python -m scenic.run --pano P --out D --only s2_depth
"""
from __future__ import annotations

# Determinism env vars must be set before torch import (stages import torch).
from scenic import determinism  # noqa: E402

determinism.set_env()

import argparse
import shutil
import sys
from pathlib import Path

from scenic import hashing, manifest, params as params_mod, schema
from scenic.stage import Ctx

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_pipeline(
    pano: Path, out: Path, params_path: Path, only: str | None = None
) -> dict | None:
    from pipeline.registry import STAGES, get_stage

    determinism.enforce()
    determinism.block_network()  # no network at stage runtime, enforced
    p = params_mod.load(params_path)
    determinism.set_seed(p.get("seed", 0))

    out.mkdir(parents=True, exist_ok=True)
    snap = out / "params.snapshot.yaml"
    params_bytes = Path(params_path).read_bytes()
    if only and snap.exists() and snap.read_bytes() != params_bytes:
        # A single-stage re-run under different params would leave a stale
        # mixed chain (manifest.build refuses those); fail loudly instead.
        raise SystemExit(
            f"--only {only}: params differ from {snap}; run the full "
            "pipeline, or restore the params that produced this run"
        )
    snap.write_bytes(params_bytes)

    sidecar = pano.with_suffix(pano.suffix + ".license.json")
    ctx = Ctx(
        repo_root=REPO_ROOT,
        pano_path=pano,
        sidecar_path=sidecar,
        params_path=snap,
        weights_dir=REPO_ROOT / "weights",
    )
    names = [n for n, _ in STAGES]
    todo = [only] if only else names
    for name in todo:
        if name not in names:
            raise SystemExit(f"unknown stage {name}; known: {names}")
        stage = get_stage(name)
        # Clear prior state so the receipt provably comes from THIS
        # invocation and out/ holds only files this execution wrote.
        rec = out / name / "receipt.json"
        if rec.exists():
            rec.unlink()
        stage_out = out / name / "out"
        if stage_out.exists():
            shutil.rmtree(stage_out)
        print(f"[scenic] {name} ...", flush=True)
        stage.run(out, p, ctx)
        if not rec.exists():
            raise RuntimeError(f"stage {name} did not write a receipt")
    if only:
        # Any pre-existing manifest now aggregates a stale mix; remove it.
        # manifest.build (full run / accept) re-derives and checks coherence.
        stale_manifest = out / "manifest.json"
        if stale_manifest.exists():
            stale_manifest.unlink()
        return None
    m = manifest.build(out)
    h = manifest.manifest_hash(out)
    print(f"[scenic] manifest {h}  shippable={m['shippable']}")
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pano", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--params", type=Path, default=REPO_ROOT / "params.yaml")
    ap.add_argument("--only", default=None)
    args = ap.parse_args(argv)
    run_pipeline(args.pano, args.out, args.params, args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
