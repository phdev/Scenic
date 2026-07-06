"""3DGS PLY read/write (binary little-endian) with scenic extra properties.

Layout per vertex (all little-endian):
  float32: x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2
           rot_0 rot_1 rot_2 rot_3
  uchar:   layer origin_stage

opacity = logit; scale_i = ln(meters); rot = quat wxyz, unit, w >= 0.
f_dc = (rgb01 - 0.5) / SH_C0.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814

LAYER_FG = 0
LAYER_BG = 1
LAYER_SHELL = 2

_FLOAT_PROPS = [
    "x", "y", "z", "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]
_UCHAR_PROPS = ["layer", "origin_stage"]


@dataclass
class SplatData:
    xyz: np.ndarray            # (n,3) f32
    normals: np.ndarray        # (n,3) f32
    f_dc: np.ndarray           # (n,3) f32
    opacity_logit: np.ndarray  # (n,)  f32
    log_scales: np.ndarray     # (n,3) f32
    quat_wxyz: np.ndarray      # (n,4) f32 unit, w>=0
    layer: np.ndarray          # (n,)  u8
    origin_stage: np.ndarray   # (n,)  u8

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def take(self, idx: np.ndarray) -> "SplatData":
        return SplatData(
            self.xyz[idx], self.normals[idx], self.f_dc[idx],
            self.opacity_logit[idx], self.log_scales[idx],
            self.quat_wxyz[idx], self.layer[idx], self.origin_stage[idx],
        )

    @staticmethod
    def concat(parts: list["SplatData"]) -> "SplatData":
        return SplatData(*[
            np.concatenate([getattr(p, f) for p in parts], axis=0)
            for f in ("xyz", "normals", "f_dc", "opacity_logit",
                      "log_scales", "quat_wxyz", "layer", "origin_stage")
        ])


def _dtype() -> np.dtype:
    fields = [(n, "<f4") for n in _FLOAT_PROPS] + [(n, "u1") for n in _UCHAR_PROPS]
    return np.dtype(fields)


def canonical_quat(q: np.ndarray) -> np.ndarray:
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    flip = q[..., 0] < 0
    q = np.where(flip[..., None], -q, q)
    return q


def write_splats(path: Path | str, s: SplatData) -> None:
    n = len(s)
    for name, arr, shape in [
        ("xyz", s.xyz, (n, 3)), ("normals", s.normals, (n, 3)),
        ("f_dc", s.f_dc, (n, 3)), ("opacity_logit", s.opacity_logit, (n,)),
        ("log_scales", s.log_scales, (n, 3)), ("quat_wxyz", s.quat_wxyz, (n, 4)),
        ("layer", s.layer, (n,)), ("origin_stage", s.origin_stage, (n,)),
    ]:
        if tuple(arr.shape) != shape:
            raise ValueError(f"{name} shape {arr.shape} != {shape}")
        if not np.all(np.isfinite(arr.astype(np.float64))):
            raise ValueError(f"{name} contains non-finite values")
    rec = np.empty(n, dtype=_dtype())
    q = canonical_quat(s.quat_wxyz.astype(np.float32))
    cols = np.concatenate(
        [s.xyz, s.normals, s.f_dc, s.opacity_logit[:, None], s.log_scales, q],
        axis=1,
    ).astype("<f4")
    for i, name in enumerate(_FLOAT_PROPS):
        rec[name] = cols[:, i]
    rec["layer"] = s.layer.astype("u1")
    rec["origin_stage"] = s.origin_stage.astype("u1")

    header_lines = ["ply", "format binary_little_endian 1.0",
                    f"element vertex {n}"]
    header_lines += [f"property float {p}" for p in _FLOAT_PROPS]
    header_lines += [f"property uchar {p}" for p in _UCHAR_PROPS]
    header_lines += ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header_lines) + "\n").encode("ascii"))
        f.write(rec.tobytes())


def read_splats(path: Path | str) -> SplatData:
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            c = f.read(1)
            if not c:
                raise ValueError("truncated PLY header")
            header += c
        lines = header.decode("ascii").splitlines()
        n = None
        props: list[tuple[str, str]] = []
        for ln in lines:
            t = ln.split()
            if t[:2] == ["element", "vertex"]:
                n = int(t[2])
            elif t and t[0] == "property":
                props.append((t[2], t[1]))
        if n is None:
            raise ValueError("no vertex element")
        expect = [(p, "float") for p in _FLOAT_PROPS] + [
            (p, "uchar") for p in _UCHAR_PROPS
        ]
        if props != expect:
            raise ValueError(f"unexpected PLY properties: {props}")
        rec = np.frombuffer(f.read(n * _dtype().itemsize), dtype=_dtype(), count=n)
    cols = np.stack([rec[p] for p in _FLOAT_PROPS], axis=1).astype(np.float32)
    return SplatData(
        xyz=cols[:, 0:3], normals=cols[:, 3:6], f_dc=cols[:, 6:9],
        opacity_logit=cols[:, 9], log_scales=cols[:, 10:13],
        quat_wxyz=cols[:, 13:17],
        layer=np.asarray(rec["layer"], dtype=np.uint8),
        origin_stage=np.asarray(rec["origin_stage"], dtype=np.uint8),
    )


def rgb01_to_dc(rgb01: np.ndarray) -> np.ndarray:
    return (rgb01 - 0.5) / SH_C0


def dc_to_rgb01(f_dc: np.ndarray) -> np.ndarray:
    return np.clip(f_dc * SH_C0 + 0.5, 0.0, 1.0)


def opacity_to_logit(a: np.ndarray) -> np.ndarray:
    a = np.clip(a, 1e-6, 1 - 1e-6)
    return np.log(a / (1 - a))


def logit_to_opacity(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
