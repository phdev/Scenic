"""s7_gates: the falsifiability layer.

Loads s6_compress/out/scene.ply ONCE (the shipped quest asset) and runs the
six gate modules (gates/{hole,jitter,stereo,people,budgets,fidelity}.py) over
the shared head-box pose/view matrix. Each verdict is schema-validated
(gate_verdict) and written to out/verdicts/<gate>.json; diagnostic +
representative renders (center pose, 4 yaws, normal via the people gate,
magenta via the hole gate, worst fidelity tile) land in out/renders/.

Layer forensics (v2 quality pass, params.s7.layers): for the center pose and
the 4-yaw ring, the primary scene is re-rendered three extra times from the
origin with only-fg / only-bg / only-shell splats, saved as
center_yaw{NNN}_layer_{fg,bg,shell}.png. The receipt notes carry per-layer
splat counts and per-layer equirect solid-angle coverage.

Gates never abort the pipeline: failures are `pass: false` verdicts in the
receipt's gates list. Hard errors (missing s6 artifacts, schema violations)
raise.

Owner: this module owns pipeline/s7_gates.py, gates/*.py and
tests/test_s7.py only.
"""
from __future__ import annotations

from pathlib import Path

from scenic import metrics, plyio, receipts, schema
from scenic.stage import Ctx

from gates import (
    GATE_ORDER,
    LAYER_ITEMS,
    YAWS_DEG,
    budgets,
    fidelity,
    hole,
    jitter,
    layer_direction_mask,
    people,
    render_layer_view_and_save,
    stereo,
)

STAGE = "s7_gates"


def run(run_dir: Path, params: dict, ctx: Ctx) -> None:
    run_dir = Path(run_dir)
    out = ctx.out(run_dir, STAGE)

    s6_out = run_dir / "s6_compress" / "out"
    scene_path = s6_out / "scene.ply"
    compress_path = s6_out / "compress.json"
    sog_path = s6_out / "scene.sog"
    for p in (scene_path, compress_path, sog_path):
        if not p.exists():
            raise FileNotFoundError(f"{STAGE}: missing s6 artifact {p}")

    splats = plyio.read_splats(scene_path)

    verdicts_dir = out / "verdicts"
    verdicts_dir.mkdir(parents=True, exist_ok=True)
    (out / "renders").mkdir(parents=True, exist_ok=True)

    runners = {
        "hole": lambda: hole.run_gate(splats, params, out),
        "jitter": lambda: jitter.run_gate(splats, params, out),
        "stereo": lambda: stereo.run_gate(splats, params, out),
        "people": lambda: people.run_gate(splats, params, out),
        "budgets": lambda: budgets.run_gate(splats, params, out,
                                            run_dir=run_dir),
        "fidelity_at_origin": lambda: fidelity.run_gate(
            splats, params, out, run_dir=run_dir
        ),
    }
    verdicts: list[dict] = []
    for gate_name in GATE_ORDER:
        verdict = runners[gate_name]()
        schema.write_validated(
            verdicts_dir / f"{gate_name}.json", verdict, "gate_verdict"
        )
        verdicts.append(verdict)

    # --- layer forensics: origin fg/bg/shell renders + coverage notes -------
    layer_counts: dict[str, int] = {}
    layer_solid_angle: dict[str, float] = {}
    if bool(params["s7"].get("layers", False)):
        for layer_value, layer_name in LAYER_ITEMS:
            for yaw in YAWS_DEG:
                render_layer_view_and_save(
                    splats, params, out, layer_value, layer_name, yaw
                )
            sub_xyz = splats.xyz[splats.layer == layer_value]
            layer_counts[layer_name] = int(sub_xyz.shape[0])
            layer_solid_angle[layer_name] = float(
                metrics.solid_angle_fraction(layer_direction_mask(sub_xyz))
            )

    render_names = sorted(p.name for p in (out / "renders").glob("*.png"))
    receipts.write_receipt(
        run_dir,
        STAGE,
        inputs={"scene": scene_path, "compress": compress_path},
        outputs={
            f"verdict_{g}": verdicts_dir / f"{g}.json" for g in GATE_ORDER
        },
        params_used={
            "head_box": params["head_box"],
            "s7": params["s7"],
            "splat_cap": params["splat_cap"],
            "splat_target": params["splat_target"],
            "sog_max_mb": params["sog_max_mb"],
        },
        weights_used=["rtdetr_r18"],
        gates=verdicts,
        notes={
            "renders": render_names,
            "layer_counts": layer_counts,
            "layer_solid_angle": layer_solid_angle,
            "all_pass": bool(all(v["pass"] for v in verdicts)),
        },
    )
