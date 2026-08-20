"""CLI: measure one shader's GPU time distribution.

Usage:
    uv run python -m shaderperf.measure shaders/dev/fbm_smoke.wgsl \
        --warmup 30 --repeats 100 --out results/smoke.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .harness import ShaderTimer
from .stats import summarize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shader", type=Path)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--time-uniform", type=float, default=1.0)
    ap.add_argument(
        "--batch",
        type=int,
        default=1,
        help="render passes per timed sample; per-pass time = sample / batch",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    timer = ShaderTimer(args.shader, width=args.width, height=args.height)
    print(f"adapter: {timer.environment_meta()['adapter']}")
    print(f"timestamp queries available: {timer.has_timestamps}")

    result = timer.measure(
        warmup=args.warmup,
        repeats=args.repeats,
        time_value=args.time_uniform,
        n_passes=args.batch,
    )

    gpu_ns = [s.gpu_ns / args.batch for s in result.samples if s.gpu_ns is not None]
    wall_ns = [s.wall_ns / args.batch for s in result.samples]

    report = {
        "meta": result.meta,
        "raw": {
            "gpu_ns": gpu_ns,
            "wall_ns": wall_ns,
            "warmup_gpu_ns": [
                s.gpu_ns for s in result.warmup_samples if s.gpu_ns is not None
            ],
            "warmup_wall_ns": [s.wall_ns for s in result.warmup_samples],
        },
        "summary": {
            "gpu": summarize(gpu_ns) if gpu_ns else None,
            "wall": summarize(wall_ns),
        },
    }

    def fmt(s: dict | None, label: str) -> str:
        if not s:
            return f"  {label}: n/a"
        cv = f"{s['cv']:.3%}" if s["cv"] is not None else "n/a"
        return (
            f"  {label}: p50={s['p50_ns'] / 1e6:.3f} ms  "
            f"p95={s['p95_ns'] / 1e6:.3f} ms  cv={cv}  n={s['n']}"
        )

    print(fmt(report["summary"]["gpu"], "gpu "))
    print(fmt(report["summary"]["wall"], "wall"))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
