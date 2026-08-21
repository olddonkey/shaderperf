"""List and benchmark WGSL control-flow region ablations.

Usage:
    uv run python -m shaderperf.ablate SHADER.wgsl --list
    uv run python -m shaderperf.ablate SHADER.wgsl --out region-costs.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from .regions import Region, ablate, find_regions, region_contains_control_exit


def _region_rows(src: str, regions: list[Region]) -> list[dict]:
    return [
        asdict(region)
        | {"control_exit": region_contains_control_exit(src, region)}
        for region in regions
    ]


def print_region_table(src: str, regions: list[Region]) -> None:
    print(
        f"{'id':28s} {'kind':8s} {'line':>5s}  "
        f"{'snippet':60s} {'control_exit':>12s}"
    )
    print(
        f"{'-' * 28} {'-' * 8} {'-' * 5}  "
        f"{'-' * 60} {'-' * 12}"
    )
    for region in regions:
        exit_hint = "yes" if region_contains_control_exit(src, region) else "no"
        print(
            f"{region.id:28s} {region.kind:8s} {region.line:5d}  "
            f"{region.snippet:60s} {exit_hint:>12s}"
        )
    print(f"\n{len(regions)} region(s)")


def _fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value / 1e6:.3f}"


def _fmt_share(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def print_cost_table(results: list[dict], nonredundancy: dict) -> None:
    print(
        f"{'id':28s} {'kind':8s} {'line':>5s} {'valid':>5s} "
        f"{'full ms':>10s} {'variant ms':>10s} {'share':>9s}  share CI"
    )
    print(
        f"{'-' * 28} {'-' * 8} {'-' * 5} {'-' * 5} "
        f"{'-' * 10} {'-' * 10} {'-' * 9}  {'-' * 21}"
    )
    for result in results:
        if result["valid"]:
            share_ci = (
                f"[{_fmt_share(result['share_low'])}, "
                f"{_fmt_share(result['share_high'])}]"
            )
        else:
            share_ci = "-"
        print(
            f"{result['id']:28s} {result['kind']:8s} {result['line']:5d} "
            f"{str(result['valid']):>5s} "
            f"{_fmt_ms(result.get('p50_full')):>10s} "
            f"{_fmt_ms(result.get('p50_variant')):>10s} "
            f"{_fmt_share(result.get('cost_share')):>9s}  {share_ci}"
        )

    verdict = "PASS" if nonredundancy["nonredundancy_ok"] else "FAIL"
    max_share = _fmt_share(nonredundancy["max_share"])
    print(
        "\nnon-redundancy precheck: "
        f"{verdict} (valid regions={nonredundancy['valid_regions']}, "
        f"max share={max_share}; require >=3 and <80%)"
    )


def run_ablation(args: argparse.Namespace, src: str, regions: list[Region]) -> dict:
    # Keep GPU imports below the --list exit so parse-only use never creates a
    # context (and does not need to initialize any measurement machinery).
    import numpy as np

    from .harness import GpuContext, ShaderTimer
    from .protocol import interleaved_compare

    utc_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ctx = GpuContext()
    rng = np.random.default_rng(args.seed)
    full_timer = ShaderTimer(args.shader, context=ctx)
    results: list[dict] = []

    for region in regions:
        variant_source = ablate(src, region)
        variant_hash = hashlib.sha256(variant_source.encode()).hexdigest()
        result = {
            "id": region.id,
            "function": region.function,
            "kind": region.kind,
            "ordinal": region.ordinal,
            "line": region.line,
            "body_start": region.body_start,
            "body_end": region.body_end,
            "variant_sha256": variant_hash,
        }
        try:
            variant_timer = ShaderTimer(
                args.shader,
                context=ctx,
                source_override=variant_source,
            )
        except Exception as exc:
            result.update(valid=False, error=str(exc))
            results.append(result)
            continue

        comparison = interleaved_compare(
            full_timer,
            variant_timer,
            args.pairs,
            args.batch,
            args.warmup,
            rng,
            duty=args.duty,
        )
        p50_full = comparison["a_summary"]["p50_ns"]
        p50_variant = comparison["b_summary"]["p50_ns"]
        speedup_ci = comparison["speedup_ci"]
        result.update(
            valid=True,
            p50_full=p50_full,
            p50_variant=p50_variant,
            speedup_ci=speedup_ci,
            cost_share=1.0 - p50_variant / p50_full,
            share_low=1.0 - 1.0 / speedup_ci["ci_low"],
            share_high=1.0 - 1.0 / speedup_ci["ci_high"],
        )
        results.append(result)

    shares = [r["cost_share"] for r in results if r["valid"]]
    nonredundancy = {
        "valid_regions": len(shares),
        "max_share": max(shares) if shares else None,
        "nonredundancy_ok": len(shares) >= 3 and max(shares) < 0.80,
    }
    return {
        "meta": ctx.meta()
        | {
            "shader_path": str(args.shader),
            "shader_sha256": full_timer.shader_hash,
            "pairs": args.pairs,
            "batch": args.batch,
            "warmup": args.warmup,
            "duty": args.duty,
            "seed": args.seed,
            "utc_start": utc_start,
        },
        "regions": _region_rows(src, regions),
        "results": results,
        "nonredundancy": nonredundancy,
        "nonredundancy_ok": nonredundancy["nonredundancy_ok"],
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shader", type=Path, help="WGSL shader path")
    parser.add_argument("--list", action="store_true", help="list regions only")
    parser.add_argument("--pairs", type=_positive_int, default=12)
    parser.add_argument("--batch", type=_positive_int, default=32)
    parser.add_argument("--warmup", type=_nonnegative_int, default=3)
    parser.add_argument("--duty", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 < args.duty <= 1.0:
        parser.error("--duty must be in the interval (0, 1]")

    src = args.shader.read_text()
    regions = find_regions(src)
    if args.list:
        print_region_table(src, regions)
        return 0

    report = run_ablation(args, src, regions)
    print_cost_table(report["results"], report["nonredundancy"])
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
