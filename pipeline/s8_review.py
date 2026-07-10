"""s8_review: static self-contained review page (this run vs last accepted).

v2 QUALITY PASS deltas (authoritative, see docs/CONTRACTS.md):

- Reads the two-profile `compress.json` (`profiles.review` + `profiles.quest`).
  The header surfaces the quest final_count/sog AND the review final_count/sog.
- Reads ALL SIX s7 verdicts (budgets, fidelity_at_origin, hole, jitter,
  people, stereo), sorted.
- Layer toggle gallery: surfaces the s7 layer renders
  `center_yaw{NNN}_layer_{fg,bg,shell}.png` as an inline base64 gallery with a
  small JS fg/bg/shell toggle (no network). Missing renders -> placeholder.
- Keeps the 4-view this-vs-accepted grid; the 4 thumbs are ALWAYS rendered
  from scene.ply via the rasterizer. s7's renders dir is never reused for
  thumbs: no current s7 output matches the standard view names (hole writes
  the magenta-dyed `center_yawNNN_magenta.png`, people writes
  `center_yawNNN_normal.png`), so any `center_yawNNN.png` there is stale
  debris that must not become the shipped thumbnail.
- SOG quant: reports per-profile ply_bytes vs sog_bytes ratio from
  compress.json. A python `.sog` decoder does not exist in-repo (splat-transform
  is a node subprocess whose WebP round-trip is not determinism-guaranteed and
  the harness must stay pure/deterministic), so the pixel-level .ply-vs-.sog
  SSIM is skipped: `review.json.sog_ssim = null` with reason
  `decode_unavailable`. The byte ratios + note are shown on the page. This is
  explicitly acceptable per CONTRACTS.
- Copies `scene_review.sog` (viewer_profile) into the s8 out dir and the page's
  inline-viewer note references it. SuperSplat footer retained.

Outputs:
  out/index.html         fully self-contained static page (base64 PNGs, inline
                         CSS + a tiny inline JS toggle; no network, no
                         timestamps, no absolute paths -> byte-identical across
                         identical runs in different run dirs).
  out/thumbs/<yaw>.png   the 4 view PNGs.
  out/scene_review.sog   byte copy of s6's viewer-profile sog.
  out/review.json        {page, poses, compared_to_accepted, layers, sog_ssim,
                         profiles} (schema review).

'Last accepted' = <run_dir>/../_accepted (a sibling run dir promoted by
external tooling). Comparison requires its s8_review/out/review.json plus all
four thumbs; anything less -> compared_to_accepted=false + placeholders.

Receipt inputs record EVERY file whose bytes end up embedded in the hashed
index.html: the s6 scene/sog/compress.json, all six verdicts, each EXISTING
s7 layer render (key `layer_<yaw>_<layer>`, run-relative path so the manifest
can match it against s7's recorded output hash), and — when
compared_to_accepted — the five accepted-baseline files (keys
`accepted_review` + `accepted_thumb_<yaw>`; they live outside the run dir so
they hash as external/<name>). Promoting a new baseline therefore shows up as
a recorded input change, never as unattributable output divergence.

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

# the six ship gates, iterated in sorted order everywhere
GATES = ("budgets", "fidelity_at_origin", "hole", "jitter", "people", "stereo")

# standard review views: center pose, four compass yaws (degrees), pitch 0
YAWS_DEG = (0, 90, 180, 270)

# layer forensics order (matches s7 render suffixes), fixed for determinism
LAYERS = ("fg", "bg", "shell")

SUPERSPLAT_NOTE = (
    "scene_review.sog can be opened in SuperSplat (https://superspl.at) "
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
    # layer toggle gallery
    ".gal{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0}",
    ".gal .cell{border:1px solid #333;padding:6px;text-align:center}",
    ".gal img{width:160px;height:160px}",
    ".gal .ph{width:160px;height:160px}",
    ".gal .layer{display:none}",
    '#gallery[data-layer="fg"] .layer-fg{display:block}',
    '#gallery[data-layer="bg"] .layer-bg{display:block}',
    '#gallery[data-layer="shell"] .layer-shell{display:block}',
    ".toggle button{font-family:monospace;background:#222;color:#ddd;"
    "border:1px solid #555;padding:4px 12px;margin-right:6px;cursor:pointer}",
]

_TOGGLE_JS = (
    "function setLayer(l){"
    "document.getElementById('gallery').setAttribute('data-layer',l);}"
)


# ------------------------------------------------------------------ helpers


def _layer_render_name(yaw_deg: int, layer: str) -> str:
    return f"center_yaw{yaw_deg:03d}_layer_{layer}.png"


def _accepted_out(run_dir: Path) -> Path:
    """The last-accepted baseline's s8 out dir (runs/_accepted sibling)."""
    return run_dir.parent / "_accepted" / STAGE / "out"


def _load_accepted(run_dir: Path) -> tuple[dict[int, bytes], bool]:
    """Thumbs of the last accepted run (runs/_accepted sibling), if complete."""
    acc_out = _accepted_out(run_dir)
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


def _ratio(ply_bytes: int, sog_bytes: int) -> float | None:
    """ply/sog compression ratio; None if sog byte count is unusable."""
    if sog_bytes <= 0:
        return None
    return ply_bytes / sog_bytes


def _fmt_ratio(r: float | None) -> str:
    return "n/a" if r is None else f"{r:.2f}x"


# ------------------------------------------------------------------- HTML


def _build_html(
    params_hash: str,
    profiles: dict,
    primary_profile: str,
    viewer_profile: str,
    verdicts: dict[str, dict],
    this_png: dict[int, bytes],
    accepted_png: dict[int, bytes],
    compared: bool,
    px: int,
    fov_deg: float,
    layer_gallery: list[tuple[int, dict[str, str | None]]],
    sog_ssim: float | None,
    sog_ssim_reason: str,
    viewer_sog_name: str,
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
    def _sog_mb(prof: str) -> float:
        return int(profiles[prof]["sog_bytes"]) / (1024.0 * 1024.0)

    quest_final = int(profiles["quest"]["final_count"])
    review_final = int(profiles["review"]["final_count"])
    quest_sog = int(profiles["quest"]["sog_bytes"])
    review_sog = int(profiles["review"]["sog_bytes"])
    header_rows = [
        ("params hash", params_hash),
        ("primary profile", primary_profile),
        ("viewer profile", viewer_profile),
        ("quest splats final", str(quest_final)),
        ("quest sog size", f"{quest_sog} bytes ({_sog_mb('quest'):.2f} MB)"),
        ("review splats final", str(review_final)),
        ("review sog size", f"{review_sog} bytes ({_sog_mb('review'):.2f} MB)"),
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

    # -- SOG compression (ply vs sog bytes) -------------------------------
    L.append("<h2>SOG compression (ply vs sog bytes)</h2>")
    L.append("<table>")
    L.append(
        "<tr><th>profile</th><th>ply bytes</th><th>sog bytes</th>"
        "<th>SOG byte ratio (ply/sog)</th></tr>"
    )
    for prof in sorted(profiles):
        pb = int(profiles[prof]["ply_bytes"])
        sb = int(profiles[prof]["sog_bytes"])
        r = _fmt_ratio(_ratio(pb, sb))
        L.append(
            f"<tr><td>{e(prof)}</td><td>{pb}</td><td>{sb}</td>"
            f"<td>{e(r)}</td></tr>"
        )
    L.append("</table>")
    if sog_ssim is None:
        L.append(
            f"<p>sog SSIM (.ply vs .sog origin render): decode unavailable "
            f"&mdash; {e(sog_ssim_reason)}. Byte ratios shown above.</p>"
        )
    else:
        L.append(
            f"<p>sog SSIM (.ply vs .sog origin render): {repr(sog_ssim)}.</p>"
        )

    # -- layer toggle gallery --------------------------------------------
    L.append("<h2>Layer toggle (fg / bg / shell)</h2>")
    L.append('<div class="toggle">')
    for layer in LAYERS:
        L.append(
            f'<button type="button" data-layer="{layer}" '
            f"onclick=\"setLayer('{layer}')\">{layer}</button>"
        )
    L.append("</div>")
    L.append('<div class="gal" id="gallery" data-layer="fg">')
    for yaw, uris in layer_gallery:
        L.append('<div class="cell">')
        L.append(f"<div>yaw {yaw}&deg;</div>")
        for layer in LAYERS:
            uri = uris.get(layer)
            if uri is None:
                L.append(
                    f'<div class="layer layer-{layer} ph">'
                    f"{layer} missing</div>"
                )
            else:
                L.append(
                    f'<img class="layer layer-{layer}" src="{uri}" '
                    f'alt="yaw {yaw} {layer}">'
                )
        L.append("</div>")
    L.append("</div>")
    L.append(f"<script>{_TOGGLE_JS}</script>")

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

    # -- inline viewer note ----------------------------------------------
    L.append(
        f"<p>Inline viewer loads <code>{e(viewer_sog_name)}</code> "
        f"(viewer_profile: {e(viewer_profile)}).</p>"
    )

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

    # -- s6 inputs (two-profile) -------------------------------------------
    s6_out = run_dir / "s6_compress" / "out"
    scene_ply = s6_out / "scene.ply"           # primary alias (quest) for 4-view
    review_sog = s6_out / "scene_review.sog"   # viewer_profile sog
    compress_path = s6_out / "compress.json"
    for pth in (scene_ply, review_sog, compress_path):
        if not pth.exists():
            raise FileNotFoundError(f"missing s6 output {pth}")
    compress = schema.read_validated(compress_path, "compress")
    profiles = compress["profiles"]
    for prof in ("review", "quest"):
        if prof not in profiles:
            raise KeyError(f"compress.json profiles missing {prof!r}")
    primary_profile = str(compress["primary_profile"])
    viewer_profile = str(compress["viewer_profile"])

    # -- s7 verdicts (all six, schema-validated) ---------------------------
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

    renders_dir = run_dir / "s7_gates" / "out" / "renders"

    # -- 4 standard views: ALWAYS rendered from scene.ply (never reused from
    #    s7's renders dir — anything matching the standard view names there is
    #    stale debris and must not become the shipped thumbnail)
    thumbs_dir = out / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    splats = plyio.read_splats(scene_ply)
    this_png: dict[int, bytes] = {}
    poses: list[dict] = []
    for yaw in YAWS_DEG:
        thumb = thumbs_dir / f"{yaw}.png"
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
        this_png[yaw] = thumb.read_bytes()
        poses.append(
            {
                "yaw_deg": yaw,
                "pitch_deg": 0,
                "fov_deg": fov_deg,
                "px": px,
                # kept for schema stability; always "rendered" now
                "source": "rendered",
            }
        )

    # -- layer toggle gallery (surface s7 layer renders; placeholder if not) -
    # every layer render actually read gets recorded as a receipt input: its
    # bytes are embedded in the hashed index.html
    layer_gallery: list[tuple[int, dict[str, str | None]]] = []
    layer_filenames: list[str] = []
    layer_inputs: dict[str, Path] = {}
    for yaw in YAWS_DEG:
        uris: dict[str, str | None] = {}
        for layer in LAYERS:
            fname = _layer_render_name(yaw, layer)
            lpath = renders_dir / fname
            if lpath.exists():
                uris[layer] = _data_uri(lpath.read_bytes())
                layer_filenames.append(fname)
                layer_inputs[f"layer_{yaw}_{layer}"] = lpath
            else:
                uris[layer] = None
        layer_gallery.append((yaw, uris))
    layer_filenames = sorted(layer_filenames)

    # -- SOG quant: byte ratios come from compress.json; pixel SSIM skipped --
    # No in-repo python .sog decoder; splat-transform (node) round-trip is not
    # determinism-guaranteed, so sog_ssim = null with an explicit reason.
    sog_ssim: float | None = None
    sog_ssim_reason = (
        "no in-repo python .sog decoder (deterministic round-trip unavailable)"
    )

    # -- copy the viewer-profile sog into the s8 out dir --------------------
    viewer_sog_name = "scene_review.sog"
    out_sog = out / viewer_sog_name
    out_sog.write_bytes(review_sog.read_bytes())

    # -- last accepted baseline ---------------------------------------------
    accepted_png, compared = _load_accepted(run_dir)

    # -- artifacts -----------------------------------------------------------
    index_path = out / "index.html"
    index_path.write_text(
        _build_html(
            full_params_hash,
            profiles,
            primary_profile,
            viewer_profile,
            verdicts,
            this_png,
            accepted_png,
            compared,
            px,
            fov_deg,
            layer_gallery,
            sog_ssim,
            sog_ssim_reason,
            viewer_sog_name,
        )
    )

    review = {
        "page": "index.html",
        "poses": poses,
        "compared_to_accepted": compared,
        "layers": layer_filenames,
        "sog_ssim": sog_ssim,
        "sog_ssim_reason": None if sog_ssim is not None else sog_ssim_reason,
        "profiles": {
            prof: {
                "final_count": int(profiles[prof]["final_count"]),
                "sog_bytes": int(profiles[prof]["sog_bytes"]),
            }
            for prof in sorted(profiles)
        },
    }
    review_path = out / "review.json"
    schema.write_validated(review_path, review, "review")

    inputs: dict[str, Path] = {
        "scene": scene_ply,
        "scene_review_sog": review_sog,
        "compress": compress_path,
    }
    for g in GATES:
        inputs[f"verdict_{g}"] = vdir / f"{g}.json"
    inputs.update(layer_inputs)
    if compared:
        # the baseline bytes are embedded in index.html, so a baseline
        # promotion must show up as a recorded input change (they sit outside
        # the run dir -> hashed as external/<name>)
        acc_out = _accepted_out(run_dir)
        inputs["accepted_review"] = acc_out / "review.json"
        for yaw in YAWS_DEG:
            inputs[f"accepted_thumb_{yaw}"] = acc_out / "thumbs" / f"{yaw}.png"
    outputs: dict[str, Path] = {
        "index": index_path,
        "review": review_path,
        "viewer_sog": out_sog,
    }
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
            "primary_profile": primary_profile,
            "viewer_profile": viewer_profile,
            "sog_ssim": sog_ssim,
            "layer_render_count": len(layer_filenames),
            "profiles": {
                prof: {
                    "final_count": int(profiles[prof]["final_count"]),
                    "sog_bytes": int(profiles[prof]["sog_bytes"]),
                    "ply_bytes": int(profiles[prof]["ply_bytes"]),
                }
                for prof in sorted(profiles)
            },
            "full_params_hash": full_params_hash,
        },
    )
