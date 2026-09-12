import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from shaderperf.llm import (
    BUDGET_USD,
    MockChat,
    add_run_usage,
    check_run_allowed,
    empty_ledger,
)
from shaderperf.trajectory import (
    extract_last_wgsl,
    format_feedback,
    run_trajectory,
)


def successful_evaluation(flip=0.012345678, speedup=1.25):
    return {
        "compile_ok": True,
        "compile_error": None,
        "numerical": {
            "has_nan": False,
            "has_inf": False,
            "min": 0.0,
            "max": 1.0,
            "in_range": True,
            "passed": True,
        },
        "visible_flips": [
            {"time": 1.0, "flip_mean": flip, "passed": True},
            {"time": 3.7, "flip_mean": flip + 0.01, "passed": True},
        ],
        "speedup_ci": {
            "speedup_point": speedup,
            "ci_low": speedup - 0.1,
            "ci_high": speedup + 0.1,
        },
    }


class FakeEvaluator:
    def __init__(self, results=None, before_call=None):
        self.results = list(results or [successful_evaluation()])
        self.before_call = before_call
        self.calls = []

    def evaluate(self, source):
        if self.before_call is not None:
            self.before_call(len(self.calls))
        self.calls.append(source)
        if not self.results:
            raise AssertionError("unexpected evaluator call")
        return self.results.pop(0)


class TrajectoryTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def run_case(self, **overrides):
        defaults = {
            "chat": MockChat(["```wgsl\nfirst\n```"]),
            "evaluator": FakeEvaluator(),
            "task_path": self.root / "task.wgsl",
            "task_source": "original",
            "label": "test-run",
            "out_dir": self.root / "trajectories",
            "max_attempts": 1,
            "max_total_tokens": 10_000,
            "max_output_tokens": 1_000,
            "visible_times": [1.0, 3.7],
            "tau_flip": 0.134861432,
            "env_meta": {"fake": True},
        }
        defaults.update(overrides)
        return run_trajectory(**defaults)


class FencedBlockTests(TrajectoryTestCase):
    def test_last_fenced_block_wins(self):
        text = "before\n```wgsl\nold\n```\nmiddle\n```wgsl\nnew\n```"

        self.assertEqual(extract_last_wgsl(text), "new")

    def test_missing_block_costs_an_attempt_and_feedback_explains_it(self):
        evaluator = FakeEvaluator()

        result = self.run_case(
            chat=MockChat(["I need more information"]), evaluator=evaluator
        )

        self.assertEqual(result["stop_reason"], "max_attempts")
        self.assertEqual(len(result["attempts"]), 1)
        self.assertFalse(result["attempts"][0]["compile_ok"])
        self.assertIsNone(result["attempts"][0]["submission_sha256"])
        self.assertEqual(evaluator.calls, [])
        self.assertIn("no ```wgsl fenced block", result["messages"][-1]["content"])
        self.assertIn("consumed one attempt", result["messages"][-1]["content"])


class StopConditionTests(TrajectoryTestCase):
    def test_stops_on_max_attempts(self):
        evaluator = FakeEvaluator(
            [successful_evaluation(), successful_evaluation()]
        )

        result = self.run_case(
            chat=MockChat(
                ["```wgsl\none\n```", "```wgsl\ntwo\n```"]
            ),
            evaluator=evaluator,
            max_attempts=2,
        )

        self.assertEqual(result["stop_reason"], "max_attempts")
        self.assertEqual(evaluator.calls, ["one", "two"])

    def test_stops_on_total_token_budget_after_recording_attempt(self):
        evaluator = FakeEvaluator()

        result = self.run_case(
            chat=MockChat(["```wgsl\none\n```"]),
            evaluator=evaluator,
            max_attempts=4,
            max_total_tokens=1,
        )

        self.assertEqual(result["stop_reason"], "token_budget")
        self.assertEqual(len(result["attempts"]), 1)
        self.assertGreaterEqual(result["usage"]["total_tokens"], 1)
        self.assertEqual(evaluator.calls, ["one"])

    def test_final_stops_only_after_valid_submission(self):
        evaluator = FakeEvaluator()

        result = self.run_case(
            chat=MockChat(["```wgsl\nvalid\n```", "FINAL"]),
            evaluator=evaluator,
            max_attempts=4,
        )

        self.assertEqual(result["stop_reason"], "final")
        self.assertEqual(evaluator.calls, ["valid"])
        self.assertEqual(len(result["attempts"]), 2)
        self.assertTrue(result["attempts"][-1]["final_requested"])

    def test_final_before_valid_submission_costs_an_attempt(self):
        result = self.run_case(
            chat=MockChat(["FINAL"]), evaluator=FakeEvaluator(), max_attempts=1
        )

        self.assertEqual(result["stop_reason"], "max_attempts")
        self.assertFalse(result["attempts"][0]["final_requested"])

    def test_real_run_stops_after_crossing_dollar_cap(self):
        ledger = empty_ledger()
        ledger_path = self.root / "agent" / "ledger.json"
        prices = {
            "gpt-5.6-sol": {
                "input_per_mtok": 1.0,
                "output_per_mtok": 1.0,
            }
        }

        result = self.run_case(
            chat=MockChat(["```wgsl\none\n```"]),
            evaluator=FakeEvaluator(),
            max_attempts=4,
            real_run=True,
            prices=prices,
            ledger=ledger,
            ledger_path=ledger_path,
            budget_usd=0.000001,
        )

        self.assertEqual(result["stop_reason"], "dollar_budget")
        self.assertEqual(len(result["attempts"]), 1)
        self.assertGreaterEqual(ledger["spent_usd"], 0.000001)
        self.assertTrue(ledger_path.is_file())


class FeedbackTests(unittest.TestCase):
    def test_feedback_has_visible_values_and_speedup_only(self):
        feedback = format_feedback(
            successful_evaluation(),
            tau_flip=0.134861432,
            attempts_remaining=3,
            tokens_remaining=900,
        )

        self.assertIn("FLIP=0.012345678", feedback)
        self.assertIn("Speedup vs original: 1.250000x", feedback)
        self.assertIn("CI [1.150000, 1.350000]", feedback)
        lowered = feedback.lower()
        self.assertNotIn("sealed", lowered)
        self.assertNotIn("tau_tc", lowered)
        self.assertNotIn("region", lowered)


class LedgerTests(unittest.TestCase):
    def test_adding_runs_accumulates_usage_and_cost(self):
        ledger = empty_ledger()

        add_run_usage(
            ledger, input_tokens=100, output_tokens=20, cost_usd=0.25, label="a"
        )
        add_run_usage(
            ledger, input_tokens=30, output_tokens=7, cost_usd=0.10, label="b"
        )

        self.assertEqual(ledger["input_tokens"], 130)
        self.assertEqual(ledger["output_tokens"], 27)
        self.assertAlmostEqual(ledger["spent_usd"], 0.35)
        self.assertEqual(len(ledger["runs"]), 2)

    def test_budget_check_refuses_at_cap(self):
        ledger = empty_ledger()
        ledger["spent_usd"] = BUDGET_USD
        prices = {"model": {"input_per_mtok": 1.0, "output_per_mtok": 1.0}}

        with self.assertRaisesRegex(RuntimeError, "spend cap reached"):
            check_run_allowed(ledger, prices, "model")

    def test_budget_check_refuses_zero_prices_without_override(self):
        ledger = empty_ledger()
        prices = {"model": {"input_per_mtok": 0.0, "output_per_mtok": 0.0}}

        with self.assertRaisesRegex(RuntimeError, "cannot be enforced"):
            check_run_allowed(ledger, prices, "model")
        check_run_allowed(ledger, prices, "model", allow_unpriced=True)


class TrajectoryLogTests(TrajectoryTestCase):
    def test_schema_hashes_and_incremental_partial_file(self):
        partial_path = self.root / "trajectories" / "test-run.partial.json"

        def verify_previous_partial(call_index):
            if call_index == 1:
                snapshot = json.loads(partial_path.read_text(encoding="utf-8"))
                self.assertEqual(len(snapshot["attempts"]), 1)
                self.assertEqual(snapshot["attempts"][0]["submission_source"], "one")

        evaluator = FakeEvaluator(
            [successful_evaluation(), successful_evaluation()],
            before_call=verify_previous_partial,
        )
        result = self.run_case(
            chat=MockChat(
                ["```wgsl\none\n```", "```wgsl\ntwo\n```"]
            ),
            evaluator=evaluator,
            max_attempts=2,
        )

        final_path = self.root / "trajectories" / "test-run.json"
        self.assertTrue(partial_path.is_file())
        self.assertTrue(final_path.is_file())
        persisted = json.loads(final_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["schema_version"], 1)
        self.assertEqual(persisted["stop_reason"], "max_attempts")
        self.assertEqual(persisted["env_meta"], {"fake": True})
        self.assertEqual(
            persisted["attempts"][0]["submission_sha256"],
            hashlib.sha256(b"one").hexdigest(),
        )
        self.assertEqual(persisted, result)


if __name__ == "__main__":
    unittest.main()
