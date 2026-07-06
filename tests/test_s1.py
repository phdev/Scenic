"""Tests for pipeline/s1_cleanplate.py (stage-local, fake s0 out dirs).

The `edited_run` fixture executes the real stage once on a clone-stamped
edit of a tiny smooth pano; that run loads the ACTUAL pinned RT-DETR person
detector from ./weights (~80MB, CPU-only) and must report 0 hits — that is
intentional (it is the detector re-run gate under test). The containment-fail
test monkeypatches the detector to avoid a second slow re-run: containment is
independent of detection.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline import s1_cleanplate
from scenic import determinism, imageio, params as params_mod, receipts, schema
from scenic.stage import Ctx

REPO_ROOT = Path(__file__).resolve().parent.parent

MASK_ROWS = slice(40, 60)
MASK_COLS = slice(100, 140)


# ---------------------------------------------------------------- fixtures


def _gradient_pano(w: int = 256, h: int = 128) -> np.ndarray:
    """Tiny smooth-gradient equirect pano: no people (verified in test_s0)."""
    xs = np.linspace(40.0, 200.0, w)[None, :]
    ys = np.linspace(30.0, 180.0, h)[:, None]
    r = np.broadcast_to(xs, (h, w))
    g = np.broadcast_to(ys, (h, w))
    b = (r + g) / 2.0
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _person_mask(h: int = 128, w: int = 256) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[MASK_ROWS, MASK_COLS] = True
    return mask


def _fake_s0(root: Path, hits: int, mask: np.ndarray | None = None):
    """Source pano + fake s0_ingest/out. Returns (run_dir, pano_path)."""
    rgb = _gradient_pano()
    pano = root / "pano.png"
    imageio.save_png(pano, rgb)
    run_dir = root / "run"
    out = run_dir / "s0_ingest" / "out"
    out.mkdir(parents=True)
    # s1 passthrough must be a byte-copy, so plant the exact source bytes
    (out / "pano.png").write_bytes(pano.read_bytes())
    if mask is None:
        mask = np.zeros(rgb.shape[:2], dtype=bool)
    imageio.save_mask_png(out / "person_mask.png", mask)
    if hits == 0:
        boxes = {"views": [], "total_hits": 0}
    else:
        boxes = {
            "views": [
                {
                    "name": "yaw000",
                    "yaw_deg": 0.0,
                    "pitch_deg": 0.0,
                    "fov_deg": 90.0,
                    "boxes": [
                        {"xyxy": [100.0, 100.0, 200.0, 300.0], "score": 0.9}
                    ]
                    * hits,
                }
            ],
            "total_hits": hits,
        }
    schema.write_validated(out / "person_boxes.json", boxes, "person_boxes")
    return run_dir, pano


def _edit_path(pano: Path) -> Path:
    return Path(str(pano) + ".cleanplate.png")


def _clone_stamp(orig: np.ndarray) -> np.ndarray:
    """An edit strictly INSIDE the mask: clone pixels from 40 columns left.
    Keeps the image a smooth gradient (no detector bait) while the gradient
    slope guarantees per-pixel deltas well above DIFF_THRESH."""
    edited = orig.copy()
    edited[45:55, 110:130] = orig[45:55, 70:90]
    assert np.abs(edited.astype(int) - orig.astype(int)).max() > s1_cleanplate.DIFF_THRESH
    return edited


def _ctx(pano: Path) -> Ctx:
    return Ctx(
        repo_root=REPO_ROOT,
        pano_path=pano,
        sidecar_path=pano.with_suffix(pano.suffix + ".license.json"),
        params_path=REPO_ROOT / "params.yaml",
        weights_dir=REPO_ROOT / "weights",
    )


@pytest.fixture(scope="module")
def params() -> dict:
    p = params_mod.load(REPO_ROOT / "params.yaml")
    determinism.set_seed(int(p.get("seed", 0)))
    return p


@pytest.fixture(scope="module")
def pass_run(tmp_path_factory, params):
    """Passthrough: fake s0 with 0 hits, run the real stage once."""
    root = tmp_path_factory.mktemp("s1_pass")
    run_dir, pano = _fake_s0(root, hits=0)
    s1_cleanplate.run(run_dir, params, _ctx(pano))
    return {
        "run_dir": run_dir,
        "out": run_dir / "s1_cleanplate" / "out",
        "pano_bytes": pano.read_bytes(),
    }


@pytest.fixture(scope="module")
def edited_run(tmp_path_factory, params):
    """Edited re-entry: fake s0 with 1 hit + an in-mask edit present.

    NOTE: runs the REAL pinned RT-DETR detector (10 forwards, CPU) on the
    edited pano — a smooth gradient, so hits_after must be 0.
    """
    root = tmp_path_factory.mktemp("s1_edited")
    run_dir, pano = _fake_s0(root, hits=1, mask=_person_mask())
    orig = imageio.load_rgb(run_dir / "s0_ingest" / "out" / "pano.png")
    edited = _clone_stamp(orig)
    imageio.save_png(_edit_path(pano), edited)
    s1_cleanplate.run(run_dir, params, _ctx(pano))
    return {
        "run_dir": run_dir,
        "out": run_dir / "s1_cleanplate" / "out",
        "edited": edited,
    }


# ------------------------------------------------------ passthrough (0 hits)


def test_passthrough_pano_byte_identical(pass_run):
    assert (pass_run["out"] / "pano_clean.png").read_bytes() == pass_run["pano_bytes"]


def test_passthrough_cleanplate_json(pass_run):
    cp = schema.read_validated(pass_run["out"] / "cleanplate.json", "cleanplate")
    assert cp["mode"] == "passthrough"
    assert cp["detector_hits_after"] == 0
    assert cp["containment_ok"] is True


def test_passthrough_receipt_and_gates(pass_run):
    rec = receipts.read_receipt(pass_run["run_dir"], "s1_cleanplate")
    assert rec["stage"] == "s1_cleanplate"
    assert set(rec["outputs"]) == {"pano_clean", "cleanplate"}
    # detector did NOT re-run: no weights recorded
    assert rec["weights"] == []
    gates = {g["gate"]: g for g in rec["gates"]}
    assert set(gates) == {"cleanplate_detector", "cleanplate_containment"}
    assert gates["cleanplate_detector"]["pass"] is True
    assert gates["cleanplate_detector"]["metrics"]["hits"] == 0
    assert gates["cleanplate_detector"]["metrics"]["note"] == "carried from s0"
    assert gates["cleanplate_containment"]["pass"] is True
    # no absolute paths recorded
    for art in list(rec["inputs"].values()) + list(rec["outputs"].values()):
        assert not art["path"].startswith("/")


# ------------------------------------------------- halt (persons, no edit)


def test_halt_writes_package_and_raises(tmp_path, params):
    run_dir, pano = _fake_s0(tmp_path, hits=1, mask=_person_mask())
    with pytest.raises(SystemExit, match="cleanplate required"):
        s1_cleanplate.run(run_dir, params, _ctx(pano))
    out = run_dir / "s1_cleanplate" / "out"
    # package emitted for the human editor
    assert (out / "package" / "pano.png").read_bytes() == pano.read_bytes()
    overlay = imageio.load_rgb(out / "package" / "overlay.png")
    orig = imageio.load_rgb(pano)
    # inside the mask: 50% red tint (red up, green down)
    assert overlay[50, 120, 0] > orig[50, 120, 0]
    assert overlay[50, 120, 1] < orig[50, 120, 1]
    # dilated boundary ring: solid red
    assert tuple(overlay[38, 120]) == (255, 0, 0)
    # far outside: untouched
    assert (overlay[5, 5] == orig[5, 5]).all()
    # pipeline halted unshippable: NO receipt, NO clean plate
    assert not (run_dir / "s1_cleanplate" / "receipt.json").exists()
    assert not (out / "pano_clean.png").exists()


def test_halt_message_names_both_paths(tmp_path, params):
    run_dir, pano = _fake_s0(tmp_path, hits=1, mask=_person_mask())
    with pytest.raises(SystemExit) as ei:
        s1_cleanplate.run(run_dir, params, _ctx(pano))
    msg = str(ei.value)
    assert str(_edit_path(pano)) in msg
    assert "re-run" in msg


# ------------------------------------------- edited re-entry (gates run)


def test_edited_outputs(edited_run):
    cp = schema.read_validated(edited_run["out"] / "cleanplate.json", "cleanplate")
    assert cp["mode"] == "edited"
    # REAL RT-DETR re-ran on the edited pano inside the fixture: 0 hits
    assert cp["detector_hits_after"] == 0
    assert cp["containment_ok"] is True
    clean = imageio.load_rgb(edited_run["out"] / "pano_clean.png")
    assert (clean == edited_run["edited"]).all()


def test_edited_receipt_and_gates(edited_run, params):
    rec = receipts.read_receipt(edited_run["run_dir"], "s1_cleanplate")
    gates = {g["gate"]: g for g in rec["gates"]}
    assert gates["cleanplate_detector"]["pass"] is True
    assert gates["cleanplate_detector"]["metrics"]["hits"] == 0
    assert gates["cleanplate_containment"]["pass"] is True
    assert gates["cleanplate_containment"]["metrics"]["diff_px"] > 0
    assert gates["cleanplate_containment"]["metrics"]["diff_px_outside_mask"] == 0
    # detector re-ran -> weights recorded; params = the s0 subsection read
    assert [w["key"] for w in rec["weights"]] == ["rtdetr_r18"]
    assert rec["params_used"] == {"s0": params["s0"]}
    assert rec["notes"]["detector_hits_before"] == 1
    # the human edit is an input, recorded relative (external/), never absolute
    assert rec["inputs"]["cleanplate_edit"]["path"].startswith("external/")
    for art in list(rec["inputs"].values()) + list(rec["outputs"].values()):
        assert not art["path"].startswith("/")


def test_edited_containment_fail_records_verdict(tmp_path, params, monkeypatch):
    """An out-of-mask edit fails containment: the verdict is RECORDED, the
    receipt is written, and nothing raises (unshippable, not fatal).
    Detector is stubbed (containment is independent of detection)."""
    monkeypatch.setattr(s1_cleanplate, "_detect_hits", lambda *a, **k: 0)
    run_dir, pano = _fake_s0(tmp_path, hits=1, mask=_person_mask())
    orig = imageio.load_rgb(run_dir / "s0_ingest" / "out" / "pano.png")
    edited = _clone_stamp(orig)
    edited[5:10, 5:10] = [250, 10, 10]  # far outside mask + 12px dilation
    imageio.save_png(_edit_path(pano), edited)

    s1_cleanplate.run(run_dir, params, _ctx(pano))  # must NOT raise

    rec = receipts.read_receipt(run_dir, "s1_cleanplate")
    gates = {g["gate"]: g for g in rec["gates"]}
    assert gates["cleanplate_containment"]["pass"] is False
    assert gates["cleanplate_containment"]["metrics"]["diff_px_outside_mask"] > 0
    assert gates["cleanplate_detector"]["pass"] is True
    cp = schema.read_validated(
        run_dir / "s1_cleanplate" / "out" / "cleanplate.json", "cleanplate"
    )
    assert cp["mode"] == "edited"
    assert cp["containment_ok"] is False
    # clean plate still written (humans iterate from it), weights recorded
    assert (run_dir / "s1_cleanplate" / "out" / "pano_clean.png").exists()
    assert [w["key"] for w in rec["weights"]] == ["rtdetr_r18"]


def test_edited_in_mask_diff_is_contained(tmp_path, params, monkeypatch):
    """Containment unit path without the detector: in-mask edits pass."""
    monkeypatch.setattr(s1_cleanplate, "_detect_hits", lambda *a, **k: 0)
    run_dir, pano = _fake_s0(tmp_path, hits=1, mask=_person_mask())
    orig = imageio.load_rgb(run_dir / "s0_ingest" / "out" / "pano.png")
    imageio.save_png(_edit_path(pano), _clone_stamp(orig))
    s1_cleanplate.run(run_dir, params, _ctx(pano))
    rec = receipts.read_receipt(run_dir, "s1_cleanplate")
    gates = {g["gate"]: g for g in rec["gates"]}
    assert gates["cleanplate_containment"]["pass"] is True


def test_edited_shape_mismatch_raises(tmp_path, params):
    run_dir, pano = _fake_s0(tmp_path, hits=1, mask=_person_mask())
    bad = _gradient_pano(w=128, h=64)
    imageio.save_png(_edit_path(pano), bad)
    with pytest.raises(ValueError, match="does not match"):
        s1_cleanplate.run(run_dir, params, _ctx(pano))


def test_missing_s0_artifacts_raise(tmp_path, params):
    run_dir = tmp_path / "run"
    (run_dir / "s0_ingest" / "out").mkdir(parents=True)
    pano = tmp_path / "pano.png"
    imageio.save_png(pano, _gradient_pano())
    with pytest.raises(FileNotFoundError, match="missing s0 artifact"):
        s1_cleanplate.run(run_dir, params, _ctx(pano))
