"""Power-source parsing shared by Gate #0 runner and reporting."""

from __future__ import annotations


def parse_power_source(pmset_batt_output: str) -> dict:
    """Parse the active power source from ``pmset -g batt`` output."""
    for line in pmset_batt_output.splitlines():
        if "Now drawing from" not in line:
            continue
        if "AC Power" in line:
            return {"on_ac": True, "raw": pmset_batt_output}
        if "Battery Power" in line:
            return {"on_ac": False, "raw": pmset_batt_output}
    return {"on_ac": None, "raw": pmset_batt_output}
