"""Fit and publish the frozen spatial perceptual threshold.

Pre-declared rule (verbatim):
candidate thresholds are the midpoints between adjacent distinct FLIP values of labeled pairs; tau_FLIP = the candidate maximizing balanced accuracy (mean of true-acceptance rate on acceptable-labeled and true-rejection rate on unacceptable-labeled); ties broken toward the SMALLER threshold (conservative).

``tau_tc`` is deliberately not selected here.  It requires temporal sequences
and will be calibrated by a later unit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


VALID_LABELS = {"acceptable", "unacceptable"}
THRESHOLD_RULE = (
    "candidate thresholds are the midpoints between adjacent distinct FLIP "
    "values of labeled pairs; tau_FLIP = the candidate maximizing balanced "
    "accuracy (mean of true-acceptance rate on acceptable-labeled and "
    "true-rejection rate on unacceptable-labeled); ties broken toward the "
    "SMALLER threshold (conservative)."
)


def _confusion(rows: list[dict], threshold: float) -> dict:
    true_acceptance = sum(
        row["label"] == "acceptable" and row["flip_mean"] <= threshold
        for row in rows
    )
    false_rejection = sum(
        row["label"] == "acceptable" and row["flip_mean"] > threshold
        for row in rows
    )
    true_rejection = sum(
        row["label"] == "unacceptable" and row["flip_mean"] > threshold
        for row in rows
    )
    false_acceptance = sum(
        row["label"] == "unacceptable" and row["flip_mean"] <= threshold
        for row in rows
    )
    return {
        "true_acceptance": true_acceptance,
        "false_rejection": false_rejection,
        "true_rejection": true_rejection,
        "false_acceptance": false_acceptance,
    }


def _balanced_accuracy(confusion: dict) -> float:
    acceptable_total = confusion["true_acceptance"] + confusion["false_rejection"]
    unacceptable_total = confusion["true_rejection"] + confusion["false_acceptance"]
    if acceptable_total == 0 or unacceptable_total == 0:
        raise ValueError("calibration requires both acceptable and unacceptable labels")
    true_acceptance_rate = confusion["true_acceptance"] / acceptable_total
    true_rejection_rate = confusion["true_rejection"] / unacceptable_total
    return (true_acceptance_rate + true_rejection_rate) / 2.0


def choose_threshold(labeled_pairs: list[dict]) -> dict:
    """Apply the pre-declared threshold rule to labeled FLIP observations."""

    rows = []
    for index, pair in enumerate(labeled_pairs):
        label = pair.get("label")
        if label not in VALID_LABELS:
            raise ValueError(f"invalid label at index {index}: {label!r}")
        value = float(pair["flip_mean"])
        if not math.isfinite(value):
            raise ValueError(f"non-finite FLIP value at index {index}")
        rows.append(
            {
                "pair_id": str(pair.get("pair_id", f"row-{index}")),
                "flip_mean": value,
                "label": label,
            }
        )

    if not rows:
        raise ValueError("no labeled calibration pairs")
    distinct_values = sorted({row["flip_mean"] for row in rows})
    if len(distinct_values) < 2:
        raise ValueError("at least two distinct FLIP values are required")

    candidates = [
        (left + right) / 2.0
        for left, right in zip(distinct_values, distinct_values[1:])
    ]
    candidate_rows = []
    for threshold in candidates:
        confusion = _confusion(rows, threshold)
        candidate_rows.append(
            {
                "threshold": threshold,
                "balanced_accuracy": _balanced_accuracy(confusion),
                "confusion": confusion,
            }
        )

    # max preserves the first occurrence on a tie; candidates are ascending,
    # which implements the conservative smaller-threshold tie break.
    chosen = max(candidate_rows, key=lambda item: item["balanced_accuracy"])
    threshold = chosen["threshold"]
    curve = sorted(
        (
            {
                "pair_id": row["pair_id"],
                "flip_mean": row["flip_mean"],
                "label": row["label"],
            }
            for row in rows
        ),
        key=lambda row: (row["flip_mean"], row["label"], row["pair_id"]),
    )
    inversions = [
        row
        for row in curve
        if (
            row["label"] == "unacceptable" and row["flip_mean"] < threshold
        )
        or (row["label"] == "acceptable" and row["flip_mean"] > threshold)
    ]
    return {
        "threshold_rule": THRESHOLD_RULE,
        "tau_FLIP": threshold,
        "balanced_accuracy": chosen["balanced_accuracy"],
        "confusion": chosen["confusion"],
        "calibration_curve": curve,
        "candidate_thresholds": candidate_rows,
        "inversions": inversions,
        "tau_tc": None,
        "tau_tc_status": (
            "not calibrated here; temporal sequences are required by a later unit"
        ),
    }


def load_labeled_pairs(directory: Path) -> list[dict]:
    manifest = json.loads((directory / "manifest.json").read_text())
    labels = json.loads((directory / "labels.json").read_text())
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("manifest.json must contain a pairs list")

    joined = []
    for pair in pairs:
        pair_id = pair["pair_id"]
        label_record = labels.get(pair_id)
        if label_record is None:
            continue
        joined.append(
            {
                "pair_id": pair_id,
                "flip_mean": pair["flip_mean"],
                "label": label_record["label"],
            }
        )
    return joined


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("results/calibration"),
        help="directory containing manifest.json and labels.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = load_labeled_pairs(args.dir)
        report = choose_threshold(rows)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    output_path = args.dir / "calibration_report.json"
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    confusion = report["confusion"]
    print(f"labeled pairs: {len(rows)}")
    print(f"tau_FLIP: {report['tau_FLIP']:.9g}")
    print(f"balanced accuracy: {report['balanced_accuracy']:.6f}")
    print(
        "confusion: "
        f"true_acceptance={confusion['true_acceptance']} "
        f"false_rejection={confusion['false_rejection']} "
        f"true_rejection={confusion['true_rejection']} "
        f"false_acceptance={confusion['false_acceptance']}"
    )
    print(f"inversions: {len(report['inversions'])}")
    print(f"tau_tc: {report['tau_tc_status']}")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
