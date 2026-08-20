"""CLI for ShaderPerf perceptual-equivalence verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .verifier import verify


def _comma_separated_floats(value: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "sample times must be comma-separated floats"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one sample time is required")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref", type=Path, help="reference WGSL shader")
    parser.add_argument("test", type=Path, help="test WGSL shader")
    parser.add_argument(
        "--sample-times",
        type=_comma_separated_floats,
        default=[1.0],
        help="comma-separated sample times (default: 1.0)",
    )
    parser.add_argument("--tc-start", type=float, default=1.0)
    parser.add_argument("--tc-step", type=float, default=0.05)
    parser.add_argument(
        "--tc-count",
        type=int,
        default=8,
        help="number of temporal frames; 0 disables the check (default: 8)",
    )
    parser.add_argument("--tau-flip", type=float, required=True)
    parser.add_argument("--tau-tc", type=float, default=None)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.tc_count < 0:
        parser.error("--tc-count must be non-negative")
    if args.tc_count > 0 and args.tau_tc is None:
        parser.error("--tau-tc is required unless --tc-count is 0")

    tc_times = [
        args.tc_start + index * args.tc_step for index in range(args.tc_count)
    ]
    report = verify(
        args.ref,
        args.test,
        sample_times=args.sample_times,
        tc_times=tc_times,
        tau_flip=args.tau_flip,
        tau_tc=args.tau_tc,
        width=args.width,
        height=args.height,
    )

    for sample in report["samples"]:
        numerical = "PASS" if sample["numerical_test"]["passed"] else "FAIL"
        print(
            f"sample time={sample['time']:.9g} "
            f"flip_mean={sample['flip_mean']:.9g} numerical={numerical}"
        )

    tc_status = "PASS" if report["tc_pass"] else "FAIL"
    tc_numerical = "PASS" if report["tc_numerical_pass"] else "FAIL"
    print(
        f"tc_max_delta={report['tc_max_delta']:.9g} "
        f"tc_pass={tc_status} numerical={tc_numerical}"
    )
    print(f"VERDICT: {'PASS' if report['verdict'] else 'FAIL'}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    return 0 if report["verdict"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
