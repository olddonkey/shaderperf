"""Commit, reveal, and inspect ShaderPerf sealed samples."""

from __future__ import annotations

import argparse
import json
import sys

from .sealed import (
    DEFAULT_N,
    DEFAULT_T_RANGE,
    commit_task,
    load_sealed_times,
    reveal_task,
    verify_commitment,
)


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    commit = subparsers.add_parser("commit", help="create a sealed commitment")
    commit.add_argument("--task", required=True, help="evaluation task ID")
    commit.add_argument("--n", type=int, default=DEFAULT_N)
    commit.add_argument(
        "--t-range",
        type=float,
        nargs=2,
        metavar=("START", "END"),
        default=DEFAULT_T_RANGE,
    )
    commit.add_argument("--force", action="store_true")

    reveal = subparsers.add_parser("reveal", help="publish a task's secret")
    reveal.add_argument("--task", required=True, help="evaluation task ID")

    check = subparsers.add_parser("check", help="check a revealed seed")
    check.add_argument("--task", required=True, help="evaluation task ID")
    check.add_argument("--seed", required=True, help="revealed secret seed")

    times = subparsers.add_parser("times", help="regenerate local sealed times")
    times.add_argument("--task", required=True, help="evaluation task ID")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        if args.command == "commit":
            record = commit_task(
                args.task,
                n=args.n,
                t_range=tuple(args.t_range),
                force=args.force,
            )
            _print_json(record)
            return 0
        if args.command == "reveal":
            _print_json(reveal_task(args.task))
            return 0
        if args.command == "check":
            matches = verify_commitment(args.task, args.seed)
            print("OK" if matches else "MISMATCH")
            return 0 if matches else 1

        print(
            "WARNING: sealed times must never be exposed to an agent.",
            file=sys.stderr,
        )
        _print_json(load_sealed_times(args.task))
        return 0
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
