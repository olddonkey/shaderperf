"""Aggregate Gate #0 certification runs into a Markdown summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .power import parse_power_source


ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT / "results" / "gate0"
SUMMARY_PATH = RESULTS_DIR / "SUMMARY.md"


def _pass(value: object) -> str:
    return "PASS" if value is True else "FAIL"


def _load_runs() -> list[dict]:
    runs = []
    for path in RESULTS_DIR.glob("*.json"):
        try:
            with path.open() as stream:
                data = json.load(stream)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            print(f"warning: skipping {path}: {exc}", file=sys.stderr)
            continue
        if not isinstance(data, dict) or "criteria" not in data:
            print(f'warning: skipping {path}: missing "criteria"', file=sys.stderr)
            continue
        data["_path"] = path
        runs.append(data)
    return sorted(
        runs,
        key=lambda run: str(run.get("meta", {}).get("process_start_utc", "")),
    )


def run_power_compliant(run: dict) -> bool | None:
    """Return a run's recorded or legacy-derived power compliance."""
    meta = run.get("meta", {})
    if "power_compliant" in meta:
        value = meta["power_compliant"]
        return value if isinstance(value, bool) else None

    raw = meta.get("power")
    if not isinstance(raw, str):
        return None
    return parse_power_source(raw)["on_ac"]


def _power_label(run: dict) -> str:
    meta = run.get("meta", {})
    parsed_records = [
        meta[key]
        for key in ("power_start", "power_end")
        if isinstance(meta.get(key), dict)
    ]
    if parsed_records:
        sources = [record.get("on_ac") for record in parsed_records]
        if False in sources:
            return "battery"
        if sources and all(source is True for source in sources):
            return "AC"
        return "n/a"

    raw = meta.get("power")
    if isinstance(raw, str):
        source = parse_power_source(raw)["on_ac"]
        if source is True:
            return "AC"
        if source is False:
            return "battery"
    if meta.get("power_compliant") is True:
        return "AC"
    return "n/a"


def overall_verdict(runs: list[dict]) -> tuple[str, list[dict]]:
    """Return the gate verdict and the compliant passing runs it counts."""
    compliant_passing = [
        run
        for run in runs
        if run.get("criteria", {}).get("run_passed") is True
        and run_power_compliant(run) is True
    ]
    passing_dates = {
        start.split("T", 1)[0]
        for run in compliant_passing
        if (start := run.get("meta", {}).get("process_start_utc"))
    }
    if len(compliant_passing) >= 3 and len(passing_dates) >= 2:
        return "GATE #0: PASS", compliant_passing

    missing = []
    if len(compliant_passing) < 3:
        missing.append(
            f"need {3 - len(compliant_passing)} more compliant passing run(s)"
        )
    if len(passing_dates) < 2:
        missing.append(
            "need compliant passing runs across "
            f"{2 - len(passing_dates)} more distinct UTC date(s)"
        )

    excluded = [
        run
        for run in runs
        if run.get("criteria", {}).get("run_passed") is True
        and run_power_compliant(run) is not True
    ]
    if excluded:
        battery_count = sum(_power_label(run) == "battery" for run in excluded)
        unknown_count = len(excluded) - battery_count
        reasons = []
        if battery_count:
            reasons.append("battery power")
        if unknown_count:
            reasons.append("power n/a")
        run_word = "run" if len(excluded) == 1 else "runs"
        missing.append(
            f"{len(excluded)} passing {run_word} excluded: {', '.join(reasons)}"
        )

    return f"GATE #0: PENDING ({'; '.join(missing)})", compliant_passing


def _markdown(runs: list[dict]) -> str:
    headers = [
        "label",
        "process_start_utc",
        "duty",
        "power",
        "criteria a",
        "criteria b",
        "criteria c",
        "pixels_all_ok",
        "run_passed",
        "counts",
    ]
    rows = []
    for run in runs:
        meta = run.get("meta", {})
        criteria = run["criteria"]
        drift = criteria.get("c_max_sentinel_drift")
        drift_text = f"{drift:.2%}" if isinstance(drift, (int, float)) else "n/a"
        duty = meta.get("duty", "n/a")
        rows.append(
            [
                str(meta.get("label", run["_path"].stem)),
                str(meta.get("process_start_utc", "n/a")),
                str(duty),
                _power_label(run),
                f'{criteria.get("a_positives_certified", "n/a")} '
                f'({_pass(criteria.get("a_pass"))})',
                _pass(criteria.get("b_zero_false_null_certifications")),
                f'{drift_text} ({_pass(criteria.get("c_pass"))})',
                str(criteria.get("pixels_all_ok", "n/a")),
                str(criteria.get("run_passed", "n/a")),
                str(
                    criteria.get("run_passed") is True
                    and run_power_compliant(run) is True
                ),
            ]
        )

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)

    verdict, _ = overall_verdict(runs)

    return "\n".join(lines + ["", verdict, ""])


def main() -> None:
    summary = _markdown(_load_runs())
    SUMMARY_PATH.write_text(summary)
    print(summary, end="")


if __name__ == "__main__":
    main()
