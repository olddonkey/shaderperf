"""Aggregate Gate #0 certification runs into a Markdown summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path


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


def _markdown(runs: list[dict]) -> str:
    headers = [
        "label",
        "process_start_utc",
        "duty",
        "criteria a",
        "criteria b",
        "criteria c",
        "pixels_all_ok",
        "run_passed",
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
                f'{criteria.get("a_positives_certified", "n/a")} '
                f'({_pass(criteria.get("a_pass"))})',
                _pass(criteria.get("b_zero_false_null_certifications")),
                f'{drift_text} ({_pass(criteria.get("c_pass"))})',
                str(criteria.get("pixels_all_ok", "n/a")),
                str(criteria.get("run_passed", "n/a")),
            ]
        )

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)

    passing = [run for run in runs if run["criteria"].get("run_passed") is True]
    passing_dates = {
        start.split("T", 1)[0]
        for run in passing
        if (start := run.get("meta", {}).get("process_start_utc"))
    }
    if len(passing) >= 3 and len(passing_dates) >= 2:
        verdict = "GATE #0: PASS"
    else:
        missing = []
        if len(passing) < 3:
            missing.append(f"need {3 - len(passing)} more passing run(s)")
        if len(passing_dates) < 2:
            missing.append(
                "need passing runs across "
                f"{2 - len(passing_dates)} more distinct UTC date(s)"
            )
        verdict = f"GATE #0: PENDING ({'; '.join(missing)})"

    return "\n".join(lines + ["", verdict, ""])


def main() -> None:
    summary = _markdown(_load_runs())
    SUMMARY_PATH.write_text(summary)
    print(summary, end="")


if __name__ == "__main__":
    main()
