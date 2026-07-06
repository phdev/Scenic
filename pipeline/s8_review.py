"""s8_review: static self-contained review page (this run vs last accepted).

Reads s6_compress outputs (scene.ply / scene.sog / compress.json) and the
five s7_gates verdicts, obtains 4 standard views (center pose, yaw
0/90/180/270, pitch 0, s7.render_fov_deg, s7.render_px) — reusing s7's
render PNGs byte-for-byte when recognizable in s7_gates/out/renders/, else
rendering scene.ply through scenic.rasterizer — and emits:

  out/index.html        fully self-contained static page: header (run params
                        hash, splat counts from compress.json, sog size),
                        gate verdict table (five gates), side-by-side image
                        grid (this run vs the accepted baseline or a
                        placeholder), SuperSplat footer note. All images are
                        base64 data URIs, CSS is inline; no network, no
                        timestamps, no absolute paths (byte-identical across
                        identical runs in different run dirs).
  out/thumbs/<yaw>.png  the 4 view PNGs
  out/review.json       {page, poses, compared_to_accepted} (schema review)

'Last accepted' = <run_dir>/../_accepted (a sibling run dir promoted by
external tooling). Comparison requires its s8_review/out/review.json plus all
four thumbs; anything less -> compared_to_accepted=false + placeholders.

Runs after s7 and before the manifest build: reads only prior stages' out/
dirs, never manifest.json. Pure numpy; no torch, no RNG.
"""
from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path

import numpy as np

from scenic import hashing, imageio, plyio, rasterizer, receipts, schema
from scenic.stage import Ctx

STAGE = "s8_review"

# the five ship gates, iterated in sorted order everywhere
GATES = ("budgets", "hole", "jitter", "people", "stereo")

# standard review views: center pose, four compass yaws (degrees), pitch 0
YAWS_DEG = (0, 90, 180, 270)

SUPERSPLAT_NOTE = (
    "scene.sog can be opened in SuperSplat (https://superspl.at) "
    "&mdash; drag the file into the editor."
)

_CSS = [
    "body{font-family:monospace;background:#141414;color:#ddd;margin:24px}",
    "h1,h2{font-weight:normal}",
    "table{border-collapse:collapse;margin:12px 0}",
    "th,td{border:1px solid #444;padding:4px 10px;text-align:left;"
    "vertical-align:top}",
    ".pass{color:#4c4}",
    ".fail{color:#e55}",
    "img{display:block;width:256px;height:256px;image-rendering:pixelated}",
    ".ph{width:256px;height:256px;display:flex;align-items:center;"
    "justify-content:center;border:1px dashed #555;color:#777}",
    "footer{margin-top:24px;color:#999}",
]


# ------------------------------------------------------------------ helpers


def _find_s7_render(renders_dir: Path, yaw_deg: int) -> Path | None:
    """Recognize an s7 render of the standard center-pose view for yaw_deg.

    Fixed candidate-name order -> deterministic. Unrecognized naming simply
    falls back to rendering (never wrong, only slower)."""
    if not renders_dir.is_dir():
        return None
    for name in (
        f"center_yaw{yaw_deg:03d}.png",
        f"center_yaw{yaw_deg}.png",
        f"yaw{yaw_deg:03d}.png",
        f"yaw{yaw_deg}.png",
        f"yaw_{yaw_deg}.png",
    ):
        p = renders_dir / name
        if p.exists():
            return p
    return None


def _load_accepted(run_dir: Path) -> tuple[dict[int, bytes], bool]:
    """Thumbs of the last accepted run (runs/_accepted sibling), if complete."""
    acc_out = run_dir.parent / "_accepted" / STAGE / "out"
    review_path = acc_out / "review.json"
    thumb_paths = {y: acc_out / "thumbs" / f"{y}.png" for y in YAWS_DEG}
    if not review_path.exists() or not all(
        thumb_paths[y].exists() for y in YAWS_DEG
    ):
        return {}, False
    # present but schema-invalid = a real error, not a soft skip
    schema.read_validated(review_path, "review")
    return {y: thumb_paths[y].read_bytes() for y in YAWS_DEG}, True


def _fmt_val(v) -> str:
    """Deterministic text for a gate metric/threshold value."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (int, str)):
        return str(v)
    return json.dumps(v, sort_keys=True, separators=(",", ":"))


def _kv_text(d: dict) -> str:
    if not d:
        return "-"
    return ", ".join(f"{k}={_fmt_val(d[k])}" for k in sorted(d))


def _data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _build_html(
    params_hash: str,
    compress: dict,
    sog_bytes: int,
    verdicts: dict[str, dict],
    this_png: dict[int, bytes],
    accepted_png: dict[int, bytes],
    compared: bool,
    px: int,
    fov_deg: float,
) -> str:
    e = html.escape
    L: list[str] = []
    L.append("<!DOCTYPE html>")
    L.append('<html lang="en">')
    L.append("<head>")
    L.append('<meta charset="utf-8">')
    L.append("<title>Scenic review</title>")
    L.append("<style>")
    L.extend(_CSS)
    L.append("</style>")
    L.append("</head>")
    L.append("<body>")
    L.append("<h1>Scenic review</h1>")

    # -- header ---------------------------------------------------------
    sog_mb = sog_bytes / (1024.0 * 1024.0)
    header_rows = [
        ("params hash", params_hash),
        ("splats in", str(int(compress["in_count"]))),
        ("splats final", str(int(compress["final_count"]))),
        ("scene.sog size", f"{sog_bytes} bytes ({sog_mb:.2f} MB)"),
        ("compared to accepted", "yes" if compared else "no"),
    ]
    L.append('<table class="kv">')
    for k, v in header_rows:
        L.append(f"<tr><th>{e(k)}</th><td>{e(v)}</td></tr>")
    L.append("</table>")

    # -- gate verdict table ----------------------------------------------
    L.append("<h2>Gate verdicts</h2>")
    L.append("<table>")
    L.append(
        "<tr><th>gate</th><th>verdict</th><th>metrics</th>"
        "<th>thresholds</th></tr>"
    )
    for g in sorted(verdicts):
        v = verdicts[g]
        cls, word = ("pass", "PASS") if v["pass"] else ("fail", "FAIL")
        L.append(
            f'<tr><td>{e(g)}</td><td class="{cls}">{word}</td>'
            f'<td>{e(_kv_text(v["metrics"]))}</td>'
            f'<td>{e(_kv_text(v["thresholds"]))}</td></tr>'
        )
    L.append("</table>")

    # -- side-by-side view grid -------------------------------------------
    L.append("<h2>Views: this run vs accepted</h2>")
    L.append(f"<p>center pose, fov {fov_deg:g}&deg;, {px}x{px}px</p>")
    L.append('<table class="views">')
    L.append("<tr><th>yaw</th><th>this run</th><th>accepted</th></tr>")
    for yaw in YAWS_DEG:
        this_cell = (
            f'<img src="{_data_uri(this_png[yaw])}" alt="this run yaw {yaw}">'
        )
        if compared:
            acc_cell = (
                f'<img src="{_data_uri(accepted_png[yaw])}" '
                f'alt="accepted yaw {yaw}">'
            )
        else:
            acc_cell = '<div class="ph">no accepted baseline</div>'
        L.append(
            f"<tr><td>{yaw}&deg;</td><td>{this_cell}</td>"
            f"<td>{acc_cell}</td></tr>"
        )
    L.append("</table>")

    L.append(f"<footer><p>{SUPERSPLAT_NOTE}</p></footer>")
    L.append("</body>")
    L.append("</html>")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------- stage


def run(run_dir: Path, params: dict, ctx: Ctx) -> None:
    run_dir = Path(run_dir)
    out = ctx.out(run_dir, STAGE)
    p7 = params["s7"]
    px = int(p7["render_px"])
    fov_deg = float(p7["render_fov_deg"])
    full_params_hash = hashing.sha256_json(params)

    # -- s6 inputs ---------------------------------------------------------
    s6_out = run_dir / "s6_compress" / "out"
    scene_ply = s6_out / "scene.ply"
    scene_sog = s6_out / "scene.sog"
    compress_path = s6_out / "compress.json"
    for pth in (scene_ply, scene_sog, compress_path):
        if not pth.exists():
            raise FileNotFoundError(f"missing s6 output {pth}")
    compress = schema.read_validated(compress_path, "compress")
    sog_bytes = int(scene_sog.stat().st_size)

    # -- s7 verdicts (all five, schema-validated) ---------------------------
    vdir = run_dir / "s7_gates" / "out" / "verdicts"
    verdicts: dict[str, dict] = {}
    for g in GATES:
        vpath = vdir / f"{g}.json"
        if not vpath.exists():
            raise FileNotFoundError(f"missing s7 verdict {vpath}")
        v = schema.read_validated(vpath, "gate_verdict")
        if v["gate"] != g:
            raise ValueError(f"{vpath} declares gate {v['gate']!r}, expected {g!r}")
        verdicts[g] = v

    # -- 4 standard views: reuse s7 renders byte-for-byte, else render ------
    thumbs_dir = out / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = run_dir / "s7_gates" / "out" / "renders"
    splats = None  # lazy: only read the ply if we actually render
    this_png: dict[int, bytes] = {}
    poses: list[dict] = []
    for yaw in YAWS_DEG:
        thumb = thumbs_dir / f"{yaw}.png"
        src = _find_s7_render(renders_dir, yaw)
        if src is not None:
            data = src.read_bytes()
            thumb.write_bytes(data)
            source = "s7_renders"
        else:
            if splats is None:
                splats = plyio.read_splats(scene_ply)
            res = rasterizer.render(
                splats,
                rasterizer.Camera(
                    pos=np.zeros(3, dtype=np.float64),
                    yaw=math.radians(float(yaw)),
                    pitch=0.0,
                ),
                px,
                px,
                fov_deg,
            )
            imageio.save_png(thumb, res["rgb"])
            data = thumb.read_bytes()
            source = "rendered"
        this_png[yaw] = data
        poses.append(
            {
                "yaw_deg": yaw,
                "pitch_deg": 0,
                "fov_deg": fov_deg,
                "px": px,
                "source": source,
            }
        )

    # -- last accepted baseline ---------------------------------------------
    accepted_png, compared = _load_accepted(run_dir)

    # -- artifacts -----------------------------------------------------------
    index_path = out / "index.html"
    index_path.write_text(
        _build_html(
            full_params_hash, compress, sog_bytes, verdicts,
            this_png, accepted_png, compared, px, fov_deg,
        )
    )

    review = {
        "page": "index.html",
        "poses": poses,
        "compared_to_accepted": compared,
    }
    review_path = out / "review.json"
    schema.write_validated(review_path, review, "review")

    inputs: dict[str, Path] = {"scene": scene_ply}
    for g in GATES:
        inputs[f"verdict_{g}"] = vdir / f"{g}.json"
    outputs: dict[str, Path] = {"index": index_path, "review": review_path}
    for yaw in YAWS_DEG:
        outputs[f"thumb_{yaw}"] = thumbs_dir / f"{yaw}.png"

    receipts.write_receipt(
        run_dir,
        STAGE,
        inputs=inputs,
        outputs=outputs,
        params_used={"s7": {"render_px": px, "render_fov_deg": fov_deg}},
        weights_used=[],
        notes={
            "compared_to_accepted": compared,
            "render_sources": {str(p["yaw_deg"]): p["source"] for p in poses},
            "sog_bytes": sog_bytes,
            "full_params_hash": full_params_hash,
        },
    )
