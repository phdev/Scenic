"""Procedural equirect pano fixture generator.

Writes deterministic synthetic 2:1 equirect panos + CC0 license sidecars into
fixtures/:

  uv run python tools/make_fixtures.py

Scene (photo-plausible, no people, no text/watermark anywhere):
  - smooth sky gradient (bright blue zenith -> pale horizon) + soft sun disc
  - two distant mountain-silhouette ridges around the horizon (dark, varied
    heights, haze-faded at the base)
  - mid-distance rolling colored terrain bands (band boundary rolls with lon)
  - ground plane textured in WORLD space: true perspective, camera
    camera_height_m_default above ground, ray/ground intersection at
    d = h / sin(-pitch) for pitch < 0, world-space checker + value noise
  - a few distant box structures (>8 m) for parallax interest
  - nadir band (pitch < -70 deg) is smooth ground (texture amplitude tapered)

Determinism: all randomness via scenic.determinism.rng(tag) seeded from the
params.yaml seed; geometry via scenic.geometry.equirect_lonlat; JPEG saved
with quality=92, subsampling=0, no EXIF -> byte-identical reruns under the
pinned Pillow. The scene spec is drawn once (resolution independent), so
test.jpg and ci_tiny.jpg depict the same scene at different sizes.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scenic import determinism, geometry, params, schema  # noqa: E402

FIXTURES_DIR = REPO_ROOT / "fixtures"

#: (file name, equirect width). Height is always width // 2 (2:1).
PANOS: tuple[tuple[str, int], ...] = (("test.jpg", 1536), ("ci_tiny.jpg", 512))

JPEG_QUALITY = 92
SUPERSAMPLE = 2  # render at 2x then box-filter: soft, camera-plausible edges

NOISE_TABLE_N = 128  # value-noise lattice period

# Palette (float RGB in [0, 1]).
SKY_ZENITH = np.array([0.24, 0.44, 0.86])
SKY_HORIZON = np.array([0.80, 0.86, 0.93])
SUN_COLOR = np.array([1.0, 0.97, 0.86])
RIDGE_FAR = np.array([0.55, 0.60, 0.71])
RIDGE_NEAR = np.array([0.33, 0.36, 0.42])
GROUND_GRASS = np.array([0.37, 0.43, 0.25])
GROUND_SOIL = np.array([0.47, 0.40, 0.29])
GROUND_HAZE = np.array([0.77, 0.81, 0.87])
BAND_COLORS = np.array(
    [
        [0.42, 0.46, 0.22],  # olive
        [0.62, 0.52, 0.30],  # ochre
        [0.35, 0.48, 0.30],  # sage
        [0.58, 0.48, 0.34],  # tan
        [0.30, 0.42, 0.26],  # moss
    ]
)


def _smoothstep(x: np.ndarray | float) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


@dataclasses.dataclass(frozen=True)
class Harmonics:
    """Sum of integer-frequency sinusoids -> [-1, 1], wrap-continuous in lon."""

    k: np.ndarray
    amp: np.ndarray  # normalized: sum(amp) == 1
    phase: np.ndarray

    @staticmethod
    def draw(r: np.random.Generator, n: int) -> "Harmonics":
        k = np.arange(1, n + 1, dtype=np.float64)
        amp = r.uniform(0.35, 1.0, n) / k
        amp = amp / amp.sum()
        phase = r.uniform(0.0, 2.0 * np.pi, n)
        return Harmonics(k=k, amp=amp, phase=phase)

    def __call__(self, lon: np.ndarray) -> np.ndarray:
        acc = np.zeros_like(lon, dtype=np.float64)
        for k, a, p in zip(self.k, self.amp, self.phase):
            acc += a * np.sin(k * lon + p)
        return acc


@dataclasses.dataclass(frozen=True)
class Box:
    center_xz: tuple[float, float]
    half_xz: tuple[float, float]
    height: float
    color: np.ndarray


@dataclasses.dataclass(frozen=True)
class SceneSpec:
    cam_h: float
    noise_table: np.ndarray  # (N, N) in [0, 1)
    ridge_far: Harmonics
    ridge_near: Harmonics
    band_roll: Harmonics
    sun_lon: float
    sun_lat: float
    boxes: tuple[Box, ...]


def build_scene(cam_h: float) -> SceneSpec:
    """Draw the resolution-independent scene spec. Fixed rng tag order."""
    table = determinism.rng("fixtures:noise_table").random(
        (NOISE_TABLE_N, NOISE_TABLE_N)
    )
    r_ridge = determinism.rng("fixtures:ridges")
    ridge_far = Harmonics.draw(r_ridge, 6)
    ridge_near = Harmonics.draw(r_ridge, 10)
    band_roll = Harmonics.draw(determinism.rng("fixtures:bands"), 5)

    r_box = determinism.rng("fixtures:boxes")
    boxes = []
    n_boxes = 6
    lons = r_box.uniform(-np.pi, np.pi, n_boxes)
    dists = r_box.uniform(12.0, 45.0, n_boxes)  # all "distant" (> 8 m)
    # Wide, low shed-like proportions: at splat-render resolution a tall
    # narrow box reads as a humanoid blob and trips the RT-DETR people gate
    # (observed: person score 0.54 on a ~2x5m box at 20m). Keep every box
    # clearly wider than tall.
    half_x = r_box.uniform(4.0, 7.0, n_boxes)
    half_z = r_box.uniform(4.0, 7.0, n_boxes)
    heights = r_box.uniform(1.5, 2.5, n_boxes)
    tints = r_box.uniform(-0.5, 0.5, (n_boxes, 3))
    for i in range(n_boxes):
        color = np.clip(
            np.array([0.52, 0.48, 0.44]) + 0.18 * tints[i], 0.15, 0.85
        )
        boxes.append(
            Box(
                center_xz=(
                    float(np.sin(lons[i]) * dists[i]),
                    float(np.cos(lons[i]) * dists[i]),
                ),
                half_xz=(float(half_x[i]), float(half_z[i])),
                height=float(heights[i]),
                color=color,
            )
        )
    return SceneSpec(
        cam_h=cam_h,
        noise_table=table,
        ridge_far=ridge_far,
        ridge_near=ridge_near,
        band_roll=band_roll,
        sun_lon=-2.1,
        sun_lat=0.52,  # ~30 deg up: inside the upper-45% sky band
        boxes=tuple(boxes),
    )


def _value_noise(table: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear value noise in [0, 1) on a periodic NxN lattice (smoothstep)."""
    n = table.shape[0]
    i0 = np.floor(x).astype(np.int64)
    j0 = np.floor(y).astype(np.int64)
    fx = _smoothstep(x - i0)
    fy = _smoothstep(y - j0)
    i0 %= n
    j0 %= n
    i1 = (i0 + 1) % n
    j1 = (j0 + 1) % n
    a = table[j0, i0]
    b = table[j0, i1]
    c = table[j1, i0]
    d = table[j1, i1]
    return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy


def _intersect_aabb(
    dirs: np.ndarray, bmin: np.ndarray, bmax: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray/AABB slab test, rays from the origin.

    Returns (t_near, entry_axis, hit). Deterministic argmax tie-break.
    """
    safe = np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    inv = 1.0 / safe
    t0 = bmin * inv
    t1 = bmax * inv
    t_small = np.minimum(t0, t1)
    t_big = np.maximum(t0, t1)
    t_near = t_small.max(axis=-1)
    axis = t_small.argmax(axis=-1)
    t_far = t_big.min(axis=-1)
    hit = (t_far > t_near) & (t_near > 1e-6)
    return t_near, axis, hit


def render_pano(spec: SceneSpec, width: int) -> np.ndarray:
    """Render one equirect pano -> (width//2, width, 3) uint8."""
    if width % 2 != 0:
        raise ValueError("width must be even for a 2:1 equirect")
    out_w, out_h = width, width // 2
    rw, rh = out_w * SUPERSAMPLE, out_h * SUPERSAMPLE

    lon, lat = geometry.equirect_lonlat(rw, rh)
    lon1d = lon[0]  # lon varies along axis 1 only
    dirs = geometry.lonlat_to_dirs(lon, lat)
    table = spec.noise_table

    # --- sky: smooth gradient, pale at the horizon, blue at the zenith ---
    t_sky = _smoothstep(np.clip(lat / (np.pi / 2), 0.0, 1.0) ** 0.7)
    img = SKY_HORIZON + (SKY_ZENITH - SKY_HORIZON) * t_sky[..., None]

    # --- soft sun disc + halo (occluded by everything drawn later) ---
    sun_dir = geometry.lonlat_to_dirs(
        np.array(spec.sun_lon), np.array(spec.sun_lat)
    )
    cos_sun = np.clip(dirs @ sun_dir, -1.0, 1.0)
    ang = np.arccos(cos_sun)
    glow = np.clip(
        0.9 * np.exp(-((ang / 0.03) ** 2)) + 0.35 * np.exp(-((ang / 0.14) ** 2)),
        0.0,
        1.0,
    )
    img = img + glow[..., None] * (SUN_COLOR - img)

    # --- mountain silhouettes: two ridges, haze-faded bases, soft AA edge ---
    soft = 2.0 * geometry.angular_pixel_size(rh)
    prof_far = 0.5 + 0.5 * spec.ridge_far(lon1d)
    h_far_1d = 0.035 + 0.085 * prof_far**1.2
    prof_near = 0.5 + 0.5 * spec.ridge_near(lon1d)
    h_near_1d = 0.005 + 0.16 * prof_near**1.6  # sharper peaks, varied heights
    ridge_tex = 0.95 + 0.10 * _value_noise(
        table,
        lon * (3 * NOISE_TABLE_N / (2 * np.pi)),  # integer table periods: wraps
        lat * 41.0,
    )
    for h_1d, base_col in ((h_far_1d, RIDGE_FAR), (h_near_1d, RIDGE_NEAR)):
        h = np.broadcast_to(h_1d[None, :], lat.shape)
        cov = _smoothstep((h - lat) / soft + 0.5)
        base_haze = 0.45 * (1.0 - np.clip(lat / np.maximum(h, 1e-6), 0.0, 1.0))
        col = base_col * ridge_tex[..., None]
        col = col + base_haze[..., None] * (SKY_HORIZON - col)
        img = img + cov[..., None] * (col - img)

    # --- ground plane: true perspective, world-space texture ---
    below = lat < 0
    sin_dep = np.where(below, np.sin(-lat), 1.0)
    dist = np.where(below, spec.cam_h / np.maximum(sin_dep, 1e-9), np.inf)
    d_c = np.minimum(dist, 4000.0)
    px = dirs[..., 0] * d_c
    pz = dirs[..., 2] * d_c

    # near-field: grass/soil patches + world-space checker + noise octaves
    n_soil = _value_noise(table, px * 0.13 + 31.7, pz * 0.13 + 11.9)
    n1 = _value_noise(table, px * 0.45 + 3.1, pz * 0.45 + 77.2)
    n2 = _value_noise(table, px * 1.7 + 59.0, pz * 1.7 + 23.4)
    checker = ((np.floor(px / 2.0) + np.floor(pz / 2.0)) % 2.0) * 2.0 - 1.0
    pitch_deg = np.degrees(lat)
    # taper texture toward the nadir so pitch < -70 deg stays smooth ground
    amp = 0.25 + 0.75 * _smoothstep((pitch_deg + 75.0) / 15.0)
    near_col = GROUND_GRASS + (GROUND_SOIL - GROUND_GRASS) * _smoothstep(
        1.4 * n_soil - 0.2
    )[..., None]
    tex = 0.65 * n1 + 0.35 * n2 - 0.5
    near_col = near_col * (1.0 + (0.16 * tex + 0.05 * checker) * amp)[..., None]

    # mid-distance rolling colored bands (boundaries roll with lon)
    roll_1d = spec.band_roll(lon1d)
    s = 1.7 * np.log(np.maximum(d_c, 1.0)) + 1.1 * np.broadcast_to(
        roll_1d[None, :], d_c.shape
    )
    s_floor = np.floor(s)
    i0 = s_floor.astype(np.int64) % len(BAND_COLORS)
    i1 = (i0 + 1) % len(BAND_COLORS)
    f = _smoothstep(s - s_floor)[..., None]
    band = BAND_COLORS[i0] * (1.0 - f) + BAND_COLORS[i1] * f
    band = band * (0.90 + 0.20 * _value_noise(table, px * 0.06 + 7.7, pz * 0.06 + 41.3))[
        ..., None
    ]

    w_mid = _smoothstep((d_c - 22.0) / 45.0)[..., None]
    ground = near_col + (band - near_col) * w_mid
    haze = _smoothstep((d_c - 120.0) / 1400.0)[..., None]
    ground = ground + (GROUND_HAZE - ground) * haze
    img = np.where(below[..., None], ground, img)

    # --- distant box structures (true 3D AABBs on the ground plane) ---
    scene_depth = np.where(below, dist, np.inf)
    for box in spec.boxes:
        cx, cz = box.center_xz
        hx, hz = box.half_xz
        bmin = np.array([cx - hx, -spec.cam_h, cz - hz])
        bmax = np.array([cx + hx, -spec.cam_h + box.height, cz + hz])
        t_near, axis, hit = _intersect_aabb(dirs, bmin, bmax)
        vis = hit & (t_near < scene_depth)
        if not vis.any():
            continue
        d_comp = np.take_along_axis(dirs, axis[..., None], axis=-1)[..., 0]
        n_sign = -np.sign(d_comp)
        n_dot_l = n_sign * sun_dir[axis]  # face normal . sun dir
        shade = 0.45 + 0.55 * np.maximum(n_dot_l, 0.0)
        col = box.color * shade[..., None]
        img = np.where(vis[..., None], col, img)
        scene_depth = np.where(vis, t_near, scene_depth)

    # --- 2x2 box-filter down to target size, quantize ---
    img = img.reshape(out_h, SUPERSAMPLE, out_w, SUPERSAMPLE, 3).mean(axis=(1, 3))
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def save_pano(arr: np.ndarray, path: Path) -> None:
    """Deterministic JPEG bytes for the pinned Pillow: q92, 4:4:4, no EXIF."""
    Image.fromarray(arr).save(
        path, format="JPEG", quality=JPEG_QUALITY, subsampling=0
    )


def write_sidecar(pano_path: Path, cam_h: float) -> Path:
    sidecar = {
        "source": "synthetic — tools/make_fixtures.py",
        "license_id": "CC0-1.0",
        "scope_note": "procedurally generated, no third-party content",
        "camera_height_m": cam_h,
    }
    out = pano_path.parent / f"{pano_path.name}.license.json"
    schema.write_validated(out, sidecar, "license_sidecar")
    return out


def generate_all(out_dir: Path) -> list[Path]:
    """Generate every fixture pano + sidecar into out_dir. Deterministic."""
    p = params.load(REPO_ROOT / "params.yaml")
    determinism.set_seed(int(p.get("seed", 0)))
    cam_h = float(p.get("camera_height_m_default", 1.6))
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = build_scene(cam_h)
    written: list[Path] = []
    for name, width in PANOS:
        pano_path = out_dir / name
        save_pano(render_pano(spec, width), pano_path)
        written.append(pano_path)
        written.append(write_sidecar(pano_path, cam_h))
    return written


def main() -> None:
    for path in generate_all(FIXTURES_DIR):
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
