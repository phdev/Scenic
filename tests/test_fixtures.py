"""Tests for tools/make_fixtures.py — deterministic synthetic equirect panos."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "make_fixtures", REPO_ROOT / "tools" / "make_fixtures.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


make_fixtures = _load_tool()

from scenic import schema  # noqa: E402  (repo root importable via the tool)

EXPECTED_PANOS = {"test.jpg": 1536, "ci_tiny.jpg": 512}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory) -> tuple[Path, Path]:
    """Run the generator twice into separate dirs (the determinism oracle)."""
    a = tmp_path_factory.mktemp("fixtures_a")
    b = tmp_path_factory.mktemp("fixtures_b")
    make_fixtures.generate_all(a)
    make_fixtures.generate_all(b)
    return a, b


def test_pano_manifest_pinned():
    assert dict(make_fixtures.PANOS) == EXPECTED_PANOS


def test_two_runs_byte_identical(two_runs):
    a, b = two_runs
    names = [n for n in EXPECTED_PANOS] + [
        f"{n}.license.json" for n in EXPECTED_PANOS
    ]
    for name in names:
        assert (a / name).is_file(), f"missing {name}"
        assert _sha256(a / name) == _sha256(b / name), f"{name} not deterministic"


def test_panos_are_2to1_rgb_jpeg(two_runs):
    a, _ = two_runs
    for name, width in EXPECTED_PANOS.items():
        with Image.open(a / name) as im:
            assert im.format == "JPEG"
            assert im.mode == "RGB"
            assert im.size == (width, width // 2)  # 2:1 equirect
            assert im.size[0] == 2 * im.size[1]
            assert len(im.info.get("exif", b"")) == 0  # no EXIF


def test_license_sidecars_valid(two_runs):
    a, _ = two_runs
    for name in EXPECTED_PANOS:
        obj = json.loads((a / f"{name}.license.json").read_text())
        schema.validate(obj, "license_sidecar")  # raises on violation
        assert obj["license_id"] == "CC0-1.0"
        assert obj["camera_height_m"] == pytest.approx(1.6)
        assert "tools/make_fixtures.py" in obj["source"]
        assert obj["scope_note"]


def test_nadir_band_is_smooth_ground(two_runs):
    """pitch < -70 deg must be smooth textured ground: no text, no marks."""
    a, _ = two_runs
    for name in EXPECTED_PANOS:
        arr = np.asarray(Image.open(a / name)).astype(np.float64)
        h = arr.shape[0]
        start = int(np.ceil(h * (90.0 + 70.0) / 180.0))  # rows below -70 deg
        band = arr[start:]
        assert band.shape[0] > 0
        gy = np.abs(np.diff(band, axis=0))
        gx = np.abs(np.diff(band, axis=1))
        assert gy.mean() < 1.0 and gx.mean() < 1.0
        assert gy.max() < 24 and gx.max() < 24


def test_sky_is_smooth_blue_gradient(two_runs):
    a, _ = two_runs
    for name in EXPECTED_PANOS:
        arr = np.asarray(Image.open(a / name)).astype(np.float64)
        top = arr[: arr.shape[0] // 4]  # well inside the upper-45% sky band
        r, g, b = top[..., 0].mean(), top[..., 1].mean(), top[..., 2].mean()
        assert b > r + 40 and b > g + 30, "sky should read blue"
        # smooth gradient: adjacent-row change stays tiny outside the sun disc
        row_means = top.mean(axis=(1, 2))
        assert np.abs(np.diff(row_means)).max() < 2.0


def test_scene_has_structure(two_runs):
    """Not a flat card: mountains, bands, boxes and ground produce variance."""
    a, _ = two_runs
    for name in EXPECTED_PANOS:
        arr = np.asarray(Image.open(a / name)).astype(np.float64)
        assert arr.std() > 25.0


def test_lon_seam_wraps(two_runs):
    """Scene must be continuous across the +/-pi longitude seam."""
    a, _ = two_runs
    for name in EXPECTED_PANOS:
        arr = np.asarray(Image.open(a / name)).astype(np.int64)
        seam = np.abs(arr[:, 0] - arr[:, -1])
        assert seam.mean() < 8.0 and seam.max() < 40


def test_ground_horizon_split(two_runs):
    """Below-horizon pixels read as terrain (green/brown), not sky blue."""
    a, _ = two_runs
    for name in EXPECTED_PANOS:
        arr = np.asarray(Image.open(a / name)).astype(np.float64)
        h = arr.shape[0]
        ground = arr[int(h * 0.65) :]  # pitch < -27 deg: pure ground
        r, g, b = ground[..., 0].mean(), ground[..., 1].mean(), ground[..., 2].mean()
        assert g > b, "ground should not read as sky"
