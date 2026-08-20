"""CLI: paired comparison of two shaders with randomized interleaving.

This is the gate #0 measurement mechanism: both shaders share one process and
one device, and samples are taken in randomized interleaved order so that
drift (thermal, power state, background load) hits both conditions equally.
Cross-process comparison is known-invalid on this setup: a null comparison
across two processes showed a false 1.26x "speedup" (2026-08-19).

Usage:
    uv run python -m shaderperf.compare BASELINE.wgsl PATCHED.wgsl \
        --pairs 30 --batch 32 --out results/compare.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .harness import GpuContext, ShaderTimer
from .stats import paired_speedup_ci, summarize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("patched", type=Path)
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--time-uniform", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0, help="interleaving order seed")
    ap.add_argument("--threshold", type=float, default=1.10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ctx = GpuContext()
    timers = {
        "baseline": ShaderTimer(
            args.baseline, width=args.width, height=args.height, context=ctx
        ),
        "patched": ShaderTimer(
            args.patched, width=args.width, height=args.height, context=ctx
        ),
    }
    print(f"adapter: {ctx.meta()['adapter'].get('device', '?')}")

    for t in timers.values():
        t.set_uniforms(args.time_uniform)

    # Warm-up: alternate both conditions so both pipelines are hot.
    for _ in range(args.warmup):
        for t in timers.values():
            t.render_once(args.batch)

    # Randomized interleaving: each pair runs both conditions in random order.
    rng = np.random.default_rng(args.seed)
    wall = {"baseline": [], "patched": []}
    order_log = []
    for _ in range(args.pairs):
        order = ["baseline", "patched"]
        if rng.random() < 0.5:
            order.reverse()
        order_log.append(order[0])
        for name in order:
            s = timers[name].render_once(args.batch)
            wall[name].append(s.wall_ns / args.batch)

    ci = paired_speedup_ci(wall["baseline"], wall["patched"])
    certified = ci["ci_low"] >= args.threshold

    report = {
        "meta": ctx.meta()
        | {
            "baseline": timers["baseline"].environment_meta()["shader_sha256"],
            "patched": timers["patched"].environment_meta()["shader_sha256"],
            "baseline_path": str(args.baseline),
            "patched_path": str(args.patched),
            "pairs": args.pairs,
            "warmup": args.warmup,
            "passes_per_sample": args.batch,
            "resolution": [args.width, args.height],
            "time_uniform": args.time_uniform,
            "interleave_seed": args.seed,
            "threshold": args.threshold,
        },
        "raw": {
            "baseline_wall_ns_per_pass": wall["baseline"],
            "patched_wall_ns_per_pass": wall["patched"],
            "first_in_pair": order_log,
        },
        "summary": {
            "baseline": summarize(wall["baseline"]),
            "patched": summarize(wall["patched"]),
            "speedup_ci": ci,
            "certified": certified,
        },
    }

    b, p = report["summary"]["baseline"], report["summary"]["patched"]
    print(f"baseline: p50={b['p50_ns'] / 1e6:.3f} ms  cv={b['cv']:.3%}  n={b['n']}")
    print(f"patched : p50={p['p50_ns'] / 1e6:.3f} ms  cv={p['cv']:.3%}  n={p['n']}")
    print(
        f"speedup : {ci['speedup_point']:.4f}x  "
        f"CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]  "
        f"certified@{args.threshold:.2f}: {certified}"
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
