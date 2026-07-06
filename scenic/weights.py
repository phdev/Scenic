"""Weight registry: hash-pinned local weights, license-gated loading.
No network at pipeline runtime — weights are pre-fetched by
tools/fetch_weights.py and verified here on every load."""
from __future__ import annotations

import functools
from pathlib import Path

from scenic import hashing, schema

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = REPO_ROOT / "weights"
ALLOWED_LICENSES = {"Apache-2.0", "MIT", "BSD-3-Clause"}


@functools.lru_cache(maxsize=1)
def load_pins() -> dict:
    pins = schema.read_validated(WEIGHTS_DIR / "pins.json", "pins")
    for key, pin in pins.items():
        if pin["license"] not in ALLOWED_LICENSES:
            raise RuntimeError(
                f"weight {key} license {pin['license']!r} not in {ALLOWED_LICENSES}"
            )
    return pins


def local_dir(key: str, verify: bool = True) -> Path:
    pin = load_pins()[key]
    d = WEIGHTS_DIR / key
    if verify:
        for rel, want in sorted(pin["files"].items()):
            p = d / rel
            if not p.exists():
                raise RuntimeError(
                    f"weight file missing: {p} — run `make fetch-weights`"
                )
            got = hashing.sha256_file(p)
            if got != want:
                raise RuntimeError(f"hash mismatch for {p}: {got} != pinned {want}")
    return d


@functools.lru_cache(maxsize=1)
def load_depth_model():
    from scenic import determinism

    determinism.enforce()
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    d = local_dir("depth_anything_v2_small")
    proc = AutoImageProcessor.from_pretrained(d, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(
        d, local_files_only=True, torch_dtype=torch.float32
    )
    model.eval().to("cpu")
    return model, proc


@functools.lru_cache(maxsize=1)
def load_person_detector():
    from scenic import determinism

    determinism.enforce()
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    d = local_dir("rtdetr_r18")
    proc = AutoImageProcessor.from_pretrained(d, local_files_only=True)
    model = AutoModelForObjectDetection.from_pretrained(
        d, local_files_only=True, torch_dtype=torch.float32
    )
    model.eval().to("cpu")
    return model, proc


def person_label_id(model) -> int:
    for i, name in model.config.id2label.items():
        if name.lower() == "person":
            return int(i)
    raise RuntimeError("no 'person' label in detector config")
