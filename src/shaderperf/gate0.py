"""Gate #0 certification run: measurement-validity experiment.

Establishes whether the headline criterion (paired bootstrap CI lower bound
>= 1.10 on the p50 speedup) can certify known effects on this hardware,
per docs/project-v0.md of the planning repo:

  (a) >= 4/5 known-optimization positive controls (pre-measured 15-20% true
      effects) are certified;
  (b) zero false certifications on null comparisons (unmodified shader vs
      itself as two separate pipelines), checked in both directions;
  (c) sentinel drift stays inside the pre-declared band: every sentinel
      block's p50 within +/-10% of the median sentinel p50 of the run.

Additionally each positive control must be quantization-equivalent to its
baseline: max abs pixel diff <= 1/255 and <= 1% of pixels differing
(pre-declared; the ballast-style patches shift colors by ~1e-5, which can
flip rounding on quantization-boundary pixels).

One invocation = one certification run in a fresh process (cold start).
The full gate requires >= 3 passing runs across >= 2 distinct UTC dates.
Usage:

    uv run python -m shaderperf.gate0 --label day1-evening

Run 1's sentinel p50 rose ~19% over ~50s at full load; a 90s full-load
preheat in run 2 instead produced 5-14% block CV and 19.14% drift. This M5
has no full-load steady state, so timed samples are duty-cycled to limit heat.

Certification requires AC power and an otherwise idle machine. Runs 4-6 on
battery showed sentinel blocks at 3.9/3.6 ms versus ~2.4 ms on AC, with
20-40% block CV versus 2-5% on AC; run 5 also coincided with tooling load.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .harness import GpuContext, ShaderTimer
from .power import parse_power_source
from .protocol import interleaved_compare, pixel_equivalence
from .stats import paired_speedup_ci, summarize

ROOT = Path(__file__).parent.parent.parent

CONTROLS = [
    ("raymarch_sdf", "realistic: early loop termination"),
    ("mandelbrot_orbit", "injected: trig-identity ballast removal"),
    ("fbm_warp", "injected: sub-quantization noise octaves removal"),
    ("voronoi_cells", "injected: trig-identity ballast removal"),
    ("plasma_layers", "injected: trig-identity ballast removal"),
]
SENTINEL = ROOT / "shaders/dev/fbm_smoke.wgsl"

THRESHOLD = 1.10
DRIFT_BAND = 0.10
PIXEL_MAX_DIFF = 1
PIXEL_MAX_FRACTION = 0.01


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"


def sentinel_block(
    timer: ShaderTimer, batch: int, n: int = 10, duty: float = 1.0
) -> dict:
    if not 0.0 < duty <= 1.0:
        raise ValueError("duty must be in the interval (0, 1]")

    samples = []
    for _ in range(n):
        wall_ns = timer.render_once(batch).wall_ns
        samples.append(wall_ns / batch)
        if duty < 1.0:
            time.sleep(wall_ns / 1e9 * (1.0 / duty - 1.0))
    return {
        "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_ns_per_pass": samples,
        "summary": summarize(samples),
        "thermal": _cmd(["pmset", "-g", "therm"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duty", type=float, default=0.3)
    ap.add_argument("--label", type=str, required=True)
    ap.add_argument("--allow-battery", action="store_true")
    args = ap.parse_args()
    if not 0.0 < args.duty <= 1.0:
        ap.error("--duty must be in the interval (0, 1]")

    power_start = parse_power_source(_cmd(["pmset", "-g", "batt"]))
    if power_start["on_ac"] is not True:
        if not args.allow_battery:
            print(
                "error: Gate #0 requires AC power; connect AC power and retry "
                "(--allow-battery is diagnostics only)",
                file=sys.stderr,
            )
            raise SystemExit(2)
        print(
            "WARNING: AC POWER PRECONDITION NOT MET; diagnostic run will not "
            "count toward Gate #0",
            file=sys.stderr,
        )

    t_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ctx = GpuContext()
    rng = np.random.default_rng(args.seed)
    sentinel = ShaderTimer(SENTINEL, context=ctx)
    sentinel.set_uniforms(1.0)

    # Randomize shader-block order per run.
    order = list(range(len(CONTROLS)))
    rng.shuffle(order)

    cold_block = sentinel_block(
        sentinel, args.batch, duty=args.duty
    )  # cold-start record
    sentinels = [sentinel_block(sentinel, args.batch, duty=args.duty)]
    shader_results = {}

    for idx in order:
        name, kind = CONTROLS[idx]
        base_path = ROOT / f"shaders/gate0/{name}.wgsl"
        opt_path = ROOT / f"shaders/gate0/{name}_opt.wgsl"

        base_a = ShaderTimer(base_path, context=ctx)
        base_b = ShaderTimer(base_path, context=ctx)  # second pipeline, same source
        patched = ShaderTimer(opt_path, context=ctx)

        null_cmp = interleaved_compare(
            base_a,
            base_b,
            args.pairs,
            args.batch,
            args.warmup,
            rng,
            duty=args.duty,
        )
        null_rev_ci = paired_speedup_ci(
            null_cmp["b_wall_ns_per_pass"], null_cmp["a_wall_ns_per_pass"]
        )
        pos_cmp = interleaved_compare(
            base_a,
            patched,
            args.pairs,
            args.batch,
            args.warmup,
            rng,
            duty=args.duty,
        )
        pixels = pixel_equivalence(base_a, patched)

        null_certified = (
            null_cmp["speedup_ci"]["ci_low"] >= THRESHOLD
            or null_rev_ci["ci_low"] >= THRESHOLD
        )
        pos_certified = pos_cmp["speedup_ci"]["ci_low"] >= THRESHOLD
        pixels_ok = (
            pixels["max_abs_diff"] <= PIXEL_MAX_DIFF
            and pixels["differing_pixels"] / pixels["total_pixels"]
            <= PIXEL_MAX_FRACTION
        )

        shader_results[name] = {
            "kind": kind,
            "baseline_sha256": base_a.shader_hash,
            "patched_sha256": patched.shader_hash,
            "null": null_cmp,
            "null_reverse_ci": null_rev_ci,
            "null_falsely_certified": null_certified,
            "positive": pos_cmp,
            "positive_certified": pos_certified,
            "pixels": pixels,
            "pixels_ok": pixels_ok,
        }
        ci = pos_cmp["speedup_ci"]
        nci = null_cmp["speedup_ci"]
        print(
            f"{name:18s} null={nci['speedup_point']:.4f}x "
            f"[{nci['ci_low']:.4f},{nci['ci_high']:.4f}] "
            f"false_cert={null_certified}  "
            f"pos={ci['speedup_point']:.4f}x "
            f"[{ci['ci_low']:.4f},{ci['ci_high']:.4f}] "
            f"cert={pos_certified}  pixels_ok={pixels_ok}"
        )
        sentinels.append(sentinel_block(sentinel, args.batch, duty=args.duty))

    # Criteria evaluation.
    p50s = np.array([s["summary"]["p50_ns"] for s in sentinels])
    drift_dev = np.abs(p50s / np.median(p50s) - 1.0)
    crit_a_count = sum(r["positive_certified"] for r in shader_results.values())
    crit_a = crit_a_count >= 4
    crit_b = not any(r["null_falsely_certified"] for r in shader_results.values())
    crit_c = bool(drift_dev.max() <= DRIFT_BAND)
    pixels_all_ok = all(r["pixels_ok"] for r in shader_results.values())
    passed = crit_a and crit_b and crit_c and pixels_all_ok
    power_end = parse_power_source(_cmd(["pmset", "-g", "batt"]))
    power_compliant = (
        power_start["on_ac"] is True and power_end["on_ac"] is True
    )
    counts_toward_gate = passed and power_compliant

    report = {
        "meta": ctx.meta()
        | {
            "label": args.label,
            "process_start_utc": t_start,
            "pairs": args.pairs,
            "batch": args.batch,
            "warmup": args.warmup,
            "seed": args.seed,
            "duty": args.duty,
            "block_order": [CONTROLS[i][0] for i in order],
            "threshold": THRESHOLD,
            "drift_band": DRIFT_BAND,
            "pixel_criteria": {
                "max_abs_diff": PIXEL_MAX_DIFF,
                "max_fraction": PIXEL_MAX_FRACTION,
            },
            "thermal": _cmd(["pmset", "-g", "therm"]),
            "power": power_end["raw"],
            "power_start": power_start,
            "power_end": power_end,
            "power_compliant": power_compliant,
        },
        "cold_start_block": cold_block,
        "sentinel_blocks": sentinels,
        "shaders": shader_results,
        "criteria": {
            "a_positives_certified": f"{crit_a_count}/5",
            "a_pass": crit_a,
            "b_zero_false_null_certifications": crit_b,
            "c_max_sentinel_drift": float(drift_dev.max()),
            "c_pass": crit_c,
            "pixels_all_ok": pixels_all_ok,
            "power_compliant": power_compliant,
            "run_passed": passed,
            "counts_toward_gate": counts_toward_gate,
        },
    }

    out = ROOT / "results" / "gate0" / f"{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"\nsentinel p50 drift vs median: max {drift_dev.max():.2%} "
          f"(band {DRIFT_BAND:.0%})")
    print(f"(a) positives certified: {crit_a_count}/5 -> {'PASS' if crit_a else 'FAIL'}")
    print(f"(b) zero false null certifications -> {'PASS' if crit_b else 'FAIL'}")
    print(f"(c) drift within band -> {'PASS' if crit_c else 'FAIL'}")
    print(f"pixel equivalence all ok -> {'PASS' if pixels_all_ok else 'FAIL'}")
    print(f"RUN VERDICT: {'PASS' if passed else 'FAIL'}")
    print(f"COUNTS TOWARD GATE: {'YES' if counts_toward_gate else 'NO'}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
