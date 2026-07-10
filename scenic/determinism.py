"""Determinism enforcement.

ORDERING CONSTRAINT: the thread-pinning env vars in _ENV must be in
os.environ BEFORE numpy loads — BLAS backends (OpenBLAS/MKL/veclib) read
them once at library init, which happens on `import numpy`. This module
therefore applies _ENV at module level, ABOVE its own numpy import, so
importing scenic.determinism first pins threads for everything imported
after it. set_env() remains as the idempotent explicit call. enforce() MUST
still run before torch is imported; scenic.run does both at process start.

block_network() is the complementary RUNTIME network guard (audit hook on
socket.connect) to the static import scan in tools/check_no_network.py;
the harness wires it into scenic.run.
"""
from __future__ import annotations

import hashlib
import os
import sys

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

# Applied at import time, BEFORE the numpy import below: setting these after
# a BLAS is already loaded is a silent no-op (the original bug — numpy was
# imported at module top, so set_env() never reached its thread pools).
for _k, _v in _ENV.items():
    os.environ[_k] = _v

import numpy as np  # noqa: E402  — must come after the _ENV loop above

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


_network_blocked = False


def block_network() -> None:
    """Runtime no-network guard: raise RuntimeError on any socket connect.

    Installs a sys.addaudithook firing on the "socket.connect" audit event,
    so DYNAMIC network use invisible to the static import scan
    (tools/check_no_network.py) dies before a packet leaves. Idempotent:
    audit hooks are process-permanent BY DESIGN (they cannot be removed), so
    only one hook is ever installed per process. Subprocesses
    (splat-transform, git) get fresh interpreters and are unaffected.
    """
    global _network_blocked
    if _network_blocked:
        return

    def _hook(event: str, args: tuple) -> None:
        if event == "socket.connect":
            raise RuntimeError(
                "network blocked by scenic.determinism.block_network():"
                f" socket.connect to {args[1]!r}"
            )

    sys.addaudithook(_hook)
    _network_blocked = True


_GLOBAL_SEED = 0


def set_seed(seed: int) -> None:
    global _GLOBAL_SEED
    _GLOBAL_SEED = int(seed)


def rng(tag: str) -> np.random.Generator:
    """Independent deterministic stream per (params seed, tag)."""
    h = hashlib.sha256(f"{_GLOBAL_SEED}:{tag}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))
