"""Determinism enforcement. enforce() MUST run before torch is imported by
setting thread env vars; scenic.run does this at process start."""
from __future__ import annotations

import hashlib
import os

import numpy as np

_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
}

_enforced = False


def set_env() -> None:
    for k, v in _ENV.items():
        os.environ[k] = v


def enforce() -> None:
    """Idempotent. Env + torch single-thread deterministic CPU."""
    global _enforced
    set_env()
    if _enforced:
        return
    import torch  # after env vars

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    _enforced = True


_GLOBAL_SEED = 0


def set_seed(seed: int) -> None:
    global _GLOBAL_SEED
    _GLOBAL_SEED = int(seed)


def rng(tag: str) -> np.random.Generator:
    """Independent deterministic stream per (params seed, tag)."""
    h = hashlib.sha256(f"{_GLOBAL_SEED}:{tag}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))
