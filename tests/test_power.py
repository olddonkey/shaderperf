from __future__ import annotations

import unittest

from shaderperf.gate0_report import overall_verdict, run_power_compliant
from shaderperf.power import parse_power_source


AC_SAMPLE = """Now drawing from 'AC Power'
 -InternalBattery-0 ... 80%; AC attached; not charging present: true"""
BATTERY_SAMPLE = """Now drawing from 'Battery Power'
 -InternalBattery-0 ... 71%; discharging; 3:17 remaining present: true"""


class ParsePowerSourceTests(unittest.TestCase):
    def test_ac_power(self):
        self.assertEqual(
            parse_power_source(AC_SAMPLE), {"on_ac": True, "raw": AC_SAMPLE}
        )

    def test_battery_power(self):
        self.assertEqual(
            parse_power_source(BATTERY_SAMPLE),
            {"on_ac": False, "raw": BATTERY_SAMPLE},
        )

    def test_unavailable(self):
        raw = "unavailable: boom"
        self.assertEqual(parse_power_source(raw), {"on_ac": None, "raw": raw})


class RunPowerCompliantTests(unittest.TestCase):
    def test_explicit_false(self):
        self.assertIs(
            run_power_compliant({"meta": {"power_compliant": False}}), False
        )

    def test_legacy_ac(self):
        self.assertIs(run_power_compliant({"meta": {"power": AC_SAMPLE}}), True)

    def test_legacy_battery(self):
        self.assertIs(
            run_power_compliant({"meta": {"power": BATTERY_SAMPLE}}), False
        )

    def test_missing_power(self):
        self.assertIsNone(run_power_compliant({"meta": {}}))


def _run(
    label: str,
    start: str,
    *,
    passed: bool = True,
    power_compliant: bool = True,
) -> dict:
    return {
        "meta": {
            "label": label,
            "process_start_utc": start,
            "power_compliant": power_compliant,
        },
        "criteria": {"run_passed": passed},
    }


class OverallVerdictTests(unittest.TestCase):
    def test_three_compliant_passes_across_two_dates_pass(self):
        runs = [
            _run("run-1", "2026-08-20T01:00:00Z"),
            _run("run-2", "2026-08-20T02:00:00Z"),
            _run("run-3", "2026-08-21T01:00:00Z"),
        ]

        verdict, counted = overall_verdict(runs)

        self.assertEqual(verdict, "GATE #0: PASS")
        self.assertEqual(counted, runs)

    def test_non_compliant_pass_is_excluded(self):
        runs = [
            _run("run-1", "2026-08-20T01:00:00Z"),
            _run("run-2", "2026-08-21T01:00:00Z"),
            _run(
                "run-3",
                "2026-08-21T02:00:00Z",
                power_compliant=False,
            ),
        ]

        verdict, counted = overall_verdict(runs)

        self.assertIn("GATE #0: PENDING", verdict)
        self.assertIn("need 1 more compliant passing run(s)", verdict)
        self.assertIn("1 passing run excluded", verdict)
        self.assertEqual(counted, runs[:2])

    def test_three_compliant_passes_on_one_date_are_pending(self):
        runs = [
            _run("run-1", "2026-08-20T01:00:00Z"),
            _run("run-2", "2026-08-20T02:00:00Z"),
            _run("run-3", "2026-08-20T03:00:00Z"),
        ]

        verdict, counted = overall_verdict(runs)

        self.assertIn("GATE #0: PENDING", verdict)
        self.assertIn("1 more distinct UTC date(s)", verdict)
        self.assertEqual(counted, runs)


if __name__ == "__main__":
    unittest.main()
