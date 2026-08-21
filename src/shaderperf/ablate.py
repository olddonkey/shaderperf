"""List and benchmark WGSL control-flow region ablations.

Usage:
    uv run python -m shaderperf.ablate SHADER.wgsl --list
    uv run python -m shaderperf.ablate SHADER.wgsl --out region-costs.json
    uv run python -m shaderperf.ablate SHADER.wgsl --compare OTHER.wgsl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .regions import Region, ablate, find_regions, region_contains_control_exit


COLLAPSE_SHARE = 0.80


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


def _classification(cost_share: float, collapse_share: float) -> dict:
    collapse = cost_share >= collapse_share
    return {"collapse": collapse, "informative": not collapse}


def classify_results(
    results: list[dict], collapse_share: float = COLLAPSE_SHARE
) -> dict:
    """Summarize valid collapse and informative region results."""
    valid = [result for result in results if result.get("valid", False)]
    collapse = [
        result
        for result in valid
        if result["cost_share"] >= collapse_share
    ]
    informative = [
        result
        for result in valid
        if result["cost_share"] < collapse_share
    ]
    informative_shares = [result["cost_share"] for result in informative]
    return {
        "valid_regions": len(valid),
        "collapse_regions": len(collapse),
        "informative_regions": len(informative),
        "max_informative_share": (
            max(informative_shares) if informative_shares else None
        ),
        "nonredundancy_ok": len(informative) >= 3,
    }


def region_structure_mismatch(
    base_regions: list[Region], other_regions: list[Region]
) -> str | None:
    """Describe the first mismatch in two ordered region structures."""
    for index, (base, other) in enumerate(
        zip(base_regions, other_regions), start=1
    ):
        base_key = (base.function, base.kind, base.ordinal, base.id)
        other_key = (other.function, other.kind, other.ordinal, other.id)
        if base_key != other_key:
            return (
                f"first mismatch at region {index}: "
                f"base={base.id!r} "
                f"({base.function}, {base.kind}, ordinal {base.ordinal}), "
                f"other={other.id!r} "
                f"({other.function}, {other.kind}, ordinal {other.ordinal})"
            )

    if len(base_regions) != len(other_regions):
        index = min(len(base_regions), len(other_regions)) + 1
        base_id = (
            repr(base_regions[index - 1].id)
            if index <= len(base_regions)
            else "<missing>"
        )
        other_id = (
            repr(other_regions[index - 1].id)
            if index <= len(other_regions)
            else "<missing>"
        )
        return (
            f"region counts differ (base={len(base_regions)}, "
            f"other={len(other_regions)}); first mismatch at region {index}: "
            f"base={base_id}, other={other_id}"
        )
    return None


def regions_structurally_match(
    base_regions: list[Region], other_regions: list[Region]
) -> bool:
    return region_structure_mismatch(base_regions, other_regions) is None


def _result_class(result: dict) -> str:
    if not result.get("valid", False):
        return "invalid"
    return "collapse" if result["collapse"] else "informative"


def print_cost_table(results: list[dict], nonredundancy: dict) -> None:
    print(
        f"{'id':28s} {'kind':8s} {'line':>5s} {'class':>11s} "
        f"{'full ms':>10s} {'variant ms':>10s} {'share':>9s}  share CI"
    )
    print(
        f"{'-' * 28} {'-' * 8} {'-' * 5} {'-' * 11} "
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
            f"{_result_class(result):>11s} "
            f"{_fmt_ms(result.get('p50_full')):>10s} "
            f"{_fmt_ms(result.get('p50_variant')):>10s} "
            f"{_fmt_share(result.get('cost_share')):>9s}  {share_ci}"
        )

    verdict = "PASS" if nonredundancy["nonredundancy_ok"] else "FAIL"
    max_share = _fmt_share(nonredundancy["max_informative_share"])
    print(
        "\nnon-redundancy precheck: "
        f"{verdict} (informative regions="
        f"{nonredundancy['informative_regions']}, "
        f"collapse regions={nonredundancy['collapse_regions']}, "
        f"max informative share={max_share}; require >=3 informative)"
    )


def _measured_side(comparison: dict) -> dict:
    p50_full = comparison["a_summary"]["p50_ns"]
    p50_variant = comparison["b_summary"]["p50_ns"]
    speedup_ci = comparison["speedup_ci"]
    cost_share = 1.0 - p50_variant / p50_full
    return {
        "valid": True,
        "p50_full": p50_full,
        "p50_variant": p50_variant,
        "speedup_ci": speedup_ci,
        "cost_share": cost_share,
        "share_low": 1.0 - 1.0 / speedup_ci["ci_low"],
        "share_high": 1.0 - 1.0 / speedup_ci["ci_high"],
        "full_samples": comparison["a_wall_ns_per_pass"],
        "variant_samples": comparison["b_wall_ns_per_pass"],
        "first_in_pair": comparison["first_in_pair"],
    } | _classification(cost_share, COLLAPSE_SHARE)


def print_comparison_table(results: list[dict]) -> None:
    print(
        f"{'id':28s} {'kind':8s} {'class(base)':>12s} "
        f"{'share base':>10s} {'share other':>11s} {'delta':>9s}  "
        f"{'delta CI':>21s} {'distinguishable':>15s}"
    )
    print(
        f"{'-' * 28} {'-' * 8} {'-' * 12} "
        f"{'-' * 10} {'-' * 11} {'-' * 9}  "
        f"{'-' * 21} {'-' * 15}"
    )
    for result in results:
        base = result["base"]
        other = result["other"]
        difference = result.get("share_difference")
        if difference is None:
            delta = "-"
            delta_ci = "-"
            distinguishable = "-"
        else:
            delta = _fmt_share(difference["delta_point"])
            delta_ci = (
                f"[{_fmt_share(difference['ci_low'])}, "
                f"{_fmt_share(difference['ci_high'])}]"
            )
            distinguishable = (
                "yes" if difference["distinguishable_from_zero"] else "no"
            )
        print(
            f"{result['id']:28s} {result['kind']:8s} "
            f"{_result_class(base):>12s} "
            f"{_fmt_share(base.get('cost_share')):>10s} "
            f"{_fmt_share(other.get('cost_share')):>11s} "
            f"{delta:>9s}  {delta_ci:>21s} {distinguishable:>15s}"
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
        result.update(_classification(result["cost_share"], COLLAPSE_SHARE))
        results.append(result)

    nonredundancy = classify_results(results, COLLAPSE_SHARE)
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
            "collapse_share": COLLAPSE_SHARE,
            "utc_start": utc_start,
        },
        "regions": _region_rows(src, regions),
        "results": results,
        "nonredundancy": nonredundancy,
        "nonredundancy_ok": nonredundancy["nonredundancy_ok"],
    }


def run_comparison(
    args: argparse.Namespace,
    base_src: str,
    base_regions: list[Region],
    other_src: str,
    other_regions: list[Region],
) -> dict:
    # Structural validation happens before this function, so a mismatch exits
    # without importing or constructing any GPU measurement machinery.
    import numpy as np

    from .harness import GpuContext, ShaderTimer
    from .protocol import interleaved_compare
    from .stats import bootstrap_share_difference

    utc_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ctx = GpuContext()
    rng = np.random.default_rng(args.seed)
    results: list[dict] = []

    for base_region, other_region in zip(base_regions, other_regions):
        base_variant_source = ablate(base_src, base_region)
        other_variant_source = ablate(other_src, other_region)
        result = {
            "id": base_region.id,
            "function": base_region.function,
            "kind": base_region.kind,
            "ordinal": base_region.ordinal,
            "base_line": base_region.line,
            "other_line": other_region.line,
            "base_variant_sha256": hashlib.sha256(
                base_variant_source.encode()
            ).hexdigest(),
            "other_variant_sha256": hashlib.sha256(
                other_variant_source.encode()
            ).hexdigest(),
        }

        base_timers = None
        other_timers = None
        try:
            base_timers = (
                ShaderTimer(args.shader, context=ctx),
                ShaderTimer(
                    args.shader,
                    context=ctx,
                    source_override=base_variant_source,
                ),
            )
        except Exception as exc:
            result["base"] = {"valid": False, "error": str(exc)}
        try:
            other_timers = (
                ShaderTimer(args.compare, context=ctx),
                ShaderTimer(
                    args.compare,
                    context=ctx,
                    source_override=other_variant_source,
                ),
            )
        except Exception as exc:
            result["other"] = {"valid": False, "error": str(exc)}

        if base_timers is not None:
            base_comparison = interleaved_compare(
                *base_timers,
                args.pairs,
                args.batch,
                args.warmup,
                rng,
                duty=args.duty,
            )
            result["base"] = _measured_side(base_comparison)
        if other_timers is not None:
            other_comparison = interleaved_compare(
                *other_timers,
                args.pairs,
                args.batch,
                args.warmup,
                rng,
                duty=args.duty,
            )
            result["other"] = _measured_side(other_comparison)

        result["comparable"] = bool(
            result["base"]["valid"] and result["other"]["valid"]
        )
        if result["comparable"]:
            result["share_difference"] = bootstrap_share_difference(
                result["base"]["full_samples"],
                result["base"]["variant_samples"],
                result["other"]["full_samples"],
                result["other"]["variant_samples"],
                seed=args.seed,
            )
        results.append(result)

    return {
        "meta": ctx.meta()
        | {
            "mode": "region_matched_comparison",
            "base_shader_path": str(args.shader),
            "base_shader_sha256": hashlib.sha256(base_src.encode()).hexdigest(),
            "other_shader_path": str(args.compare),
            "other_shader_sha256": hashlib.sha256(other_src.encode()).hexdigest(),
            "pairs": args.pairs,
            "batch": args.batch,
            "warmup": args.warmup,
            "duty": args.duty,
            "seed": args.seed,
            "collapse_share": COLLAPSE_SHARE,
            "utc_start": utc_start,
        },
        "base_regions": _region_rows(base_src, base_regions),
        "other_regions": _region_rows(other_src, other_regions),
        "results": results,
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="list regions only")
    mode.add_argument(
        "--compare",
        type=Path,
        metavar="OTHER.wgsl",
        help="compare structurally matched regions against another shader",
    )
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

    if args.compare is not None:
        other_src = args.compare.read_text()
        other_regions = find_regions(other_src)
        mismatch = region_structure_mismatch(regions, other_regions)
        if mismatch is not None:
            print(
                "error: --compare requires structurally matched regions; "
                f"{mismatch}",
                file=sys.stderr,
            )
            return 2
        report = run_comparison(
            args, src, regions, other_src, other_regions
        )
        print_comparison_table(report["results"])
    else:
        report = run_ablation(args, src, regions)
        print_cost_table(report["results"], report["nonredundancy"])
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
