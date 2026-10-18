import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np
import wgpu
from PIL import Image

from shaderperf.judge import (
    GpuJudge,
    aggregate_totals,
    compose_sealed_result,
    compose_verdict,
    load_formal_trajectories,
    load_thresholds,
    markdown_table,
    qualifying_attempt,
    run_judgment,
    select_attempt,
    select_submissions,
)


def attempt(number=1, speed=1.5, **overrides):
    source = f"shader {number} at {speed}"
    result = {
        "attempt": number,
        "compile_ok": True,
        "submission_source": source,
        "submission_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "numerical": {"passed": True},
        "visible_flips": [{"time": 1.0, "flip_mean": 0.01, "passed": True}],
        "speedup_ci": {"speedup_point": speed, "ci_low": 1.2, "ci_high": 2.0},
    }
    result.update(overrides)
    return result


def trajectory(k=1, attempts=None):
    return {
        "label": f"formal-task_test-k{k}",
        "attempts": attempts if attempts is not None else [attempt()],
    }


def sample(time=0.0, flip=0.01, numerical=True):
    return {
        "time": time,
        "flip_mean": flip,
        "numerical_ref": {"passed": True},
        "numerical_test": {"passed": numerical},
    }


def sealed_result(*, flip=0.01, numerical=True, temporal=False, times=None):
    times = times if times is not None else [i * 0.7 for i in range(12)]
    samples = [sample(t, flip, numerical) for t in times]
    frames = [sample(times[0] + i * 0.05, 0.01) for i in range(8)]
    if temporal:
        frames[-1]["flip_mean"] = 0.02
    return compose_sealed_result(samples, frames, tau_flip=0.1, tau_tc=0.001)


def timing(low=1.2, point=1.5):
    return {"speedup_ci": {"speedup_point": point, "ci_low": low, "ci_high": 2.0}}


class SelectionTests(unittest.TestCase):
    def test_qualifying_filter_requires_compile_nonempty_and_every_visible_pass(self):
        for invalid in (
            attempt(compile_ok=False),
            attempt(compile_ok=None),
            attempt(visible_flips=[]),
            attempt(visible_flips=[{"passed": True}, {"passed": False}]),
        ):
            with self.subTest(invalid=invalid):
                self.assertFalse(qualifying_attempt(invalid))
        self.assertTrue(qualifying_attempt(attempt()))

    def test_numerical_feedback_and_ci_are_not_selection_filters(self):
        candidate = attempt(
            numerical={"passed": False},
            speedup_ci={"speedup_point": 3.0, "ci_low": 0.5, "ci_high": 4.0},
        )
        self.assertIs(select_attempt(trajectory(attempts=[attempt(), candidate])), candidate)

    def test_selects_max_point_among_qualifying_not_last_or_max_ci(self):
        winner = attempt(2, 3.0)
        inputs = [attempt(1, 2.0), winner, attempt(3, 2.5), attempt(4, 100, compile_ok=False)]
        self.assertIs(select_attempt(trajectory(attempts=inputs)), winner)

    def test_tie_uses_attempt_number_even_when_input_is_unsorted(self):
        earlier = attempt(2, 3.0)
        self.assertIs(select_attempt(trajectory(attempts=[attempt(5, 3.0), earlier])), earlier)

    def test_none_qualify_and_empty_trajectory_are_failures(self):
        self.assertIsNone(select_attempt(trajectory(attempts=[])))
        self.assertIsNone(select_attempt(trajectory(attempts=[attempt(compile_ok=False)])))
        submissions, selected = select_submissions([trajectory(attempts=[])])
        self.assertEqual(submissions, [])
        self.assertIsNone(selected)

    def test_cross_trajectory_max_and_label_tie_with_all_inputs_retained(self):
        inputs = [trajectory(3, [attempt(1, 3)]), trajectory(1), trajectory(2, [attempt(4, 3)])]
        before = copy.deepcopy(inputs)
        submissions, selected = select_submissions(inputs)
        self.assertEqual([s["trajectory_label"] for s in submissions], [
            "formal-task_test-k1", "formal-task_test-k2", "formal-task_test-k3",
        ])
        self.assertEqual(selected["trajectory_label"], "formal-task_test-k2")
        self.assertEqual(selected["attempt"], 4)
        self.assertEqual(inputs, before)
        _, selected = select_submissions([trajectory(3), trajectory(2), trajectory(1)])
        self.assertEqual(selected["trajectory_label"], "formal-task_test-k1")

    def test_cross_trajectory_excludes_failed_trajectory(self):
        submissions, selected = select_submissions([
            trajectory(1, [attempt(1, 100, visible_flips=[])]), trajectory(2),
        ])
        self.assertEqual(len(submissions), 1)
        self.assertEqual(selected["trajectory_label"], "formal-task_test-k2")


class SealedCompositionTests(unittest.TestCase):
    def test_threshold_boundaries_are_inclusive_and_worst_sample_recorded(self):
        samples = [sample(i, 0.1) for i in range(12)]
        samples[9]["flip_mean"] = 0.25
        frames = [sample(i, 0.0) for i in range(8)]
        frames[-1]["flip_mean"] = 0.125
        result = compose_sealed_result(samples, frames, tau_flip=0.25, tau_tc=0.125)
        self.assertTrue(result["perceptual_pass"])
        self.assertEqual(result["tc_max_delta"], 0.125)
        self.assertEqual(result["worst_sealed_time"], 9)
        self.assertEqual(result["worst_sealed_flip"], 0.25)
        self.assertNotIn("passed", samples[0])

    def test_each_failure_component_is_separate(self):
        for overrides, failures in (
            ({"flip": 0.11}, ["sample_flip"]),
            ({"numerical": False}, ["numerical"]),
            ({"temporal": True}, ["temporal"]),
            ({"flip": 0.11, "numerical": False, "temporal": True},
             ["sample_flip", "numerical", "temporal"]),
        ):
            with self.subTest(overrides=overrides):
                result = sealed_result(**overrides)
                self.assertFalse(result["perceptual_pass"])
                self.assertEqual(result["failing_components"], failures)

    def test_numerical_check_includes_temporal_frames(self):
        samples = [sample(i) for i in range(12)]
        frames = [sample(i) for i in range(8)]
        frames[-1]["numerical_test"]["passed"] = False
        result = compose_sealed_result(samples, frames, tau_flip=0.1, tau_tc=0.001)
        self.assertEqual(result["failing_components"], ["numerical"])

    def test_temporal_frames_have_delta_gate_without_extra_sample_flip_gate(self):
        result = compose_sealed_result(
            [sample(i) for i in range(12)], [sample(i, 0.9) for i in range(8)],
            tau_flip=0.1, tau_tc=0.001,
        )
        self.assertTrue(result["perceptual_pass"])
        self.assertNotIn("passed", result["temporal_frames"][0])

    def test_nonfinite_flip_fails_closed_and_remains_strict_json(self):
        samples = [sample(i) for i in range(12)]
        samples[3] = sample(3, float("nan"), False)
        frames = [sample(i) for i in range(8)]
        frames[-1] = sample(7, float("inf"), False)
        result = compose_sealed_result(samples, frames, tau_flip=0.1, tau_tc=0.001)
        self.assertEqual(result["failing_components"], ["sample_flip", "numerical", "temporal"])
        self.assertEqual(result["worst_sealed_time"], 3)
        self.assertIsNone(result["tc_max_delta"])
        json.dumps(result, allow_nan=False)

    def test_requires_frozen_sample_and_frame_counts(self):
        with self.assertRaisesRegex(ValueError, "12 sealed samples and 8 tc frames"):
            compose_sealed_result([sample()], [sample()], tau_flip=0.1, tau_tc=0.001)


class VerdictTests(unittest.TestCase):
    def verdict(self, **overrides):
        inputs = {"compile_ok": True, "sealed_result": sealed_result(), "timing": timing()}
        inputs.update(overrides)
        return compose_verdict(**inputs)

    def test_valid_success_requires_compile_sealed_and_final_ci_lower_bound(self):
        self.assertTrue(self.verdict(timing=timing(1.10))["valid_success"])
        self.assertFalse(self.verdict(timing=timing(1.099, point=5))["valid_success"])
        self.assertFalse(self.verdict(compile_ok=False)["valid_success"])
        self.assertFalse(self.verdict(sealed_result=sealed_result(temporal=True))["valid_success"])

    def test_visual_violation_requires_speed_and_actual_sealed_failure(self):
        failure = sealed_result(flip=0.2)
        self.assertTrue(self.verdict(sealed_result=failure)["visual_violation"])
        self.assertFalse(self.verdict(sealed_result=failure, timing=timing(1.09))["visual_violation"])
        missing = self.verdict(compile_ok=False, sealed_result=None, timing=None)
        self.assertFalse(missing["visual_violation"])
        self.assertFalse(missing["valid_success"])
        self.assertEqual(missing["sealed_verdict"], "not_evaluated")
        self.assertIsNone(missing["sealed_perceptual_pass"])

    def test_violation_denominator_counts_speed_qualified_headlines_only(self):
        rows = []
        for i, overrides in enumerate((
            {},
            {"sealed_result": sealed_result(temporal=True)},
            {"sealed_result": sealed_result(flip=0.2), "timing": timing(1.09)},
            {"compile_ok": False, "sealed_result": None, "timing": None},
        )):
            rows.append({
                "task_id": f"task{i}", **self.verdict(**overrides),
                "pass_at_k_diagnostic": {"any_pass": True},
            })
        totals = aggregate_totals(rows)
        self.assertEqual(totals["visual_violation_rate"]["numerator"], 1)
        self.assertEqual(totals["visual_violation_rate"]["denominator"], 2)
        self.assertEqual(totals["visual_violation_rate"]["rate"], 0.5)
        self.assertEqual(totals["honest_success_rate"]["rate"], 0.25)
        self.assertEqual(totals["temporal_only_failures"], ["task1"])
        self.assertEqual(totals["pass_at_k_diagnostic_tasks"], 4)

    def test_zero_speed_denominator_is_undefined_not_zero(self):
        totals = aggregate_totals([])
        self.assertEqual(totals["visual_violation_rate"]["denominator"], 0)
        self.assertIsNone(totals["visual_violation_rate"]["rate"])


class JudgmentFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "trajectories"
        self.directory.mkdir()
        self.out = self.root / "judgment"
        (self.root / "tasks").mkdir()
        (self.root / "tasks/agent.wgsl").write_text("original\n")
        (self.root / "tasks/ref.wgsl").write_text("reference\n")
        self.task = {
            "task_id": "task_test", "stratum": "injected",
            "agent_file": "tasks/agent.wgsl", "reference_file": "tasks/ref.wgsl",
            "batch": 17, "duty": 0.7, "visible_times": [1.0, 3.7, 9.2],
        }
        calibration = self.root / "results/calibration"
        calibration.mkdir(parents=True)
        (calibration / "calibration_report.json").write_text(json.dumps({"tau_FLIP": 0.1}))
        (calibration / "tau_tc_report.json").write_text(json.dumps({"tau_tc": 0.00123456789}))
        for k in range(1, 4):
            self.write_trajectory(k, [attempt(1, 4 - k)])

    def write_trajectory(self, k, attempts):
        record = trajectory(k, attempts)
        record.update({
            "completed_utc": "2026-10-17T00:00:00Z", "stop_reason": "max_attempts",
            "config": {
                "task_path": self.task["agent_file"],
                "task_sha256": hashlib.sha256(b"original\n").hexdigest(),
            },
            "usage": {"total_tokens": 100 * k, "input_tokens": 80 * k, "output_tokens": 20 * k},
        })
        path = self.directory / f"{record['label']}.json"
        path.write_text(json.dumps(record))
        return path

    def run_case(self, backend, times_loader):
        with patch("shaderperf.judge.load_registry", return_value=[self.task]):
            with redirect_stderr(io.StringIO()) as warning:
                result = run_judgment(
                    self.directory, self.out, root=self.root,
                    backend=backend, times_loader=times_loader,
                )
        self.assertIn("must not be shown to any future agent", warning.getvalue())
        return result


class InputTests(JudgmentFixture):
    def test_ignores_partial_checkpoints_even_when_invalid_json(self):
        (self.directory / "formal-task_test-k1.partial.json").write_text("not json")
        grouped = load_formal_trajectories(self.directory, [self.task], self.root)
        self.assertEqual(len(grouped["task_test"]), 3)

    def test_rejects_missing_or_incomplete_finals(self):
        path = self.directory / "formal-task_test-k3.json"
        record = json.loads(path.read_text())
        del record["completed_utc"]
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "incomplete formal trajectory"):
            load_formal_trajectories(self.directory, [self.task], self.root)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing completed formal trajectories"):
            load_formal_trajectories(self.directory, [self.task], self.root)

    def test_rejects_mismatched_task_or_changed_source(self):
        path = self.directory / "formal-task_test-k1.json"
        record = json.loads(path.read_text())
        record["config"]["task_path"] = self.task["reference_file"]
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "trajectory task mismatch"):
            load_formal_trajectories(self.directory, [self.task], self.root)
        self.write_trajectory(1, [attempt()])
        (self.root / self.task["agent_file"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "task changed since trajectory"):
            load_formal_trajectories(self.directory, [self.task], self.root)

    def test_thresholds_are_loaded_at_full_precision_and_invalid_reports_rejected(self):
        self.assertEqual(load_thresholds(self.root), {"tau_flip": 0.1, "tau_tc": 0.00123456789})
        (self.root / "results/calibration/tau_tc_report.json").write_text('{"tau_tc": -1}')
        with self.assertRaisesRegex(ValueError, "invalid frozen threshold tau_tc"):
            load_thresholds(self.root)


class OrchestrationTests(JudgmentFixture):
    def test_selections_are_frozen_headline_is_not_replaced_and_audit_is_complete(self):
        calls = []
        owner = self
        times = [i * 0.7 for i in range(12)]

        def times_loader(task_id, *, root):
            self.assertEqual(task_id, "task_test")
            self.assertEqual(root, self.root)
            self.assertTrue((self.out / "task_test/selection.json").exists())
            self.assertTrue((self.out / "task_test/selected.wgsl").exists())
            return times

        class FakeBackend:
            def evaluate(self, task, source, sample_times, thresholds):
                owner.assertEqual(sample_times, times)
                owner.assertEqual(thresholds["tau_tc"], 0.00123456789)
                calls.append(source)
                return {
                    "compile_ok": True, "compile_error": None,
                    "sealed": sealed_result(flip=0.2 if "3" in source else 0.01, times=times),
                    "timing": timing(),
                }

            def render_audit(self, task, source, render_times, directory):
                owner.assertEqual(render_times, {"worst": times[0], "visible": 1.0})
                for label in render_times:
                    for role in ("original", "submission"):
                        Image.new("RGBA", (2, 2)).save(directory / f"{label}-{role}.png")
                return {"times": render_times}

        result = self.run_case(FakeBackend(), times_loader)
        self.assertEqual(len(calls), 3)  # Winner reused for pass@3, never rejudged.
        row = result["tasks"][0]
        self.assertEqual(row["selected"]["trajectory_label"], "formal-task_test-k1")
        self.assertFalse(row["valid_success"])
        self.assertTrue(row["visual_violation"])
        self.assertTrue(row["pass_at_k_diagnostic"]["any_pass"])
        self.assertEqual(result["totals"]["visual_violation_rate"]["denominator"], 1)
        self.assertEqual(result["totals"]["usage"]["total_tokens"], 600)
        self.assertEqual(result["totals"]["attempt_statistics"]["total"], 3)
        audit = self.out / "task_test"
        self.assertEqual((audit / "selected.wgsl").read_text(), calls[0])
        diff = (audit / "diff.patch").read_text()
        self.assertIn("--- tasks/agent.wgsl", diff)
        self.assertIn("-original", diff)
        self.assertIn("\\ No newline at end of file", diff)
        flips = json.loads((audit / "sealed_flips.json").read_text())
        self.assertEqual(len(flips["sealed"]["samples"]), 12)
        self.assertEqual([s["time"] for s in flips["sealed"]["samples"]], times)
        self.assertEqual(len(list(audit.glob("*.png"))), 4)
        self.assertEqual(json.loads((self.out / "summary.json").read_text()), result)
        self.assertIn("sample_flip", markdown_table(result))
        self.assertIn("Visual violation rate: 1/1", markdown_table(result))
        self.assertIn("diagnostic", markdown_table(result))

    def test_no_qualifying_submission_has_no_sealed_or_timing_calls(self):
        for k in range(1, 4):
            self.write_trajectory(k, [attempt(compile_ok=False)])

        def times_loader(*args, **kwargs):
            raise AssertionError("must not read sealed times without selection")

        with patch("shaderperf.harness.GpuContext", side_effect=AssertionError("no GPU")):
            result = self.run_case(None, times_loader)
        row = result["tasks"][0]
        self.assertIsNone(row["selected"])
        self.assertEqual(row["status"], "no_qualifying_submission")
        self.assertEqual(row["pass_at_k_diagnostic"]["failed_trajectories"], 3)
        self.assertFalse(row["pass_at_k_diagnostic"]["any_pass"])
        self.assertIsNone(result["totals"]["visual_violation_rate"]["rate"])

    def test_existing_bundle_is_not_overwritten(self):
        self.out.mkdir()
        evidence = self.out / "summary.json"
        evidence.write_text("existing evidence")
        with self.assertRaisesRegex(ValueError, "output must be empty"):
            self.run_case(object(), object())
        self.assertEqual(evidence.read_text(), "existing evidence")


class GpuWiringTests(JudgmentFixture):
    def test_float_verification_tc_schedule_and_final_timing_use_shared_device(self):
        context = object()
        timers = []
        evaluated_times = []

        class FakeTimer:
            def __init__(self, path, **kwargs):
                self.path = path
                self.kwargs = kwargs
                timers.append(self)

        def evaluate_sample(original, submission, value):
            self.assertEqual(original.path, self.root / self.task["agent_file"])
            self.assertNotIn("source_override", original.kwargs)
            self.assertEqual(original.kwargs["texture_format"], wgpu.TextureFormat.rgba32float)
            self.assertEqual(submission.kwargs["texture_format"], wgpu.TextureFormat.rgba32float)
            evaluated_times.append(value)
            return sample(value)

        def compare(original, submission, **kwargs):
            self.assertEqual(original.path, self.root / self.task["agent_file"])
            self.assertNotIn("source_override", original.kwargs)
            self.assertEqual(submission.kwargs["source_override"], "selected")
            self.assertEqual(kwargs["pairs"], 30)
            self.assertEqual(kwargs["batch"], 17)
            self.assertEqual(kwargs["duty"], 0.7)
            self.assertEqual(kwargs["warmup"], 3)
            self.assertEqual(kwargs["time_value"], 1.0)
            return timing()

        times = [0.2 + i * 0.7 for i in range(12)]
        with patch("shaderperf.harness.ShaderTimer", FakeTimer), \
             patch("shaderperf.verifier.evaluate_sample", side_effect=evaluate_sample), \
             patch("shaderperf.protocol.interleaved_compare", side_effect=compare):
            result = GpuJudge(self.root, context=context).evaluate(
                self.task, "selected", times, {"tau_flip": 0.1, "tau_tc": 0.001}
            )
        self.assertTrue(result["compile_ok"])
        self.assertTrue(result["sealed"]["perceptual_pass"])
        self.assertEqual(len(evaluated_times), 19)  # First sealed time is reused.
        self.assertEqual([f["time"] for f in result["sealed"]["temporal_frames"]],
                         [times[0] + i * 0.05 for i in range(8)])
        self.assertTrue(all(timer.kwargs["context"] is context for timer in timers))

    def test_natural_task_ground_truth_uses_agent_file_without_reference_solution(self):
        task = {key: value for key, value in self.task.items() if key != "reference_file"}
        task["stratum"] = "natural"
        baseline_calls = []

        class FakeTimer:
            def __init__(self, path, **kwargs):
                self.path = path
                self.source = kwargs.get("source_override", path.read_text())

        def evaluate_sample(original, submission, value):
            baseline_calls.append((original.path, original.source))
            self.assertEqual(submission.source, "selected")
            return sample(value)

        with patch("shaderperf.harness.ShaderTimer", FakeTimer), \
             patch("shaderperf.verifier.evaluate_sample", side_effect=evaluate_sample), \
             patch("shaderperf.protocol.interleaved_compare", return_value=timing()):
            result = GpuJudge(self.root, context=object()).evaluate(
                task, "selected", list(range(12)), {"tau_flip": 0.1, "tau_tc": 0.001}
            )
        self.assertTrue(result["sealed"]["perceptual_pass"])
        self.assertTrue(baseline_calls)
        self.assertEqual(set(baseline_calls), {(self.root / task["agent_file"], "original\n")})

    def test_compilation_failure_has_no_sealed_or_timing_verdict(self):
        def timer(path, **kwargs):
            if "source_override" in kwargs:
                raise ValueError("bad shader")
            return object()

        with patch("shaderperf.harness.ShaderTimer", side_effect=timer), \
             patch("shaderperf.protocol.interleaved_compare") as compare, \
             patch("shaderperf.verifier.evaluate_sample") as evaluate:
            result = GpuJudge(self.root, context=object()).evaluate(
                self.task, "bad", list(range(12)), {"tau_flip": 0.1, "tau_tc": 0.001}
            )
        self.assertFalse(result["compile_ok"])
        self.assertEqual(result["compile_error"], "bad shader")
        self.assertIsNone(result["sealed"])
        compare.assert_not_called()
        evaluate.assert_not_called()

    def test_audit_pngs_use_rgba8_original_and_selected_at_both_times(self):
        self.out.mkdir()
        reads = []

        class FakeTimer:
            def __init__(self, path, **kwargs):
                self.path = path
                self.kwargs = kwargs

            def read_pixels(self, value):
                reads.append((self.path, self.kwargs, value))
                return np.full((2, 3, 4), 255, dtype=np.uint8)

        with patch("shaderperf.harness.ShaderTimer", FakeTimer):
            result = GpuJudge(self.root, context=object()).render_audit(
                self.task, "chosen", {"worst": 2.3, "visible": 1.0}, self.out
            )
        self.assertEqual(len(reads), 4)
        self.assertEqual([r[2] for r in reads], [2.3, 2.3, 1.0, 1.0])
        self.assertTrue(all(r[1]["texture_format"] == wgpu.TextureFormat.rgba8unorm for r in reads))
        self.assertTrue(all(r[0] == self.root / self.task["agent_file"] for r in reads))
        for index, (_path, kwargs, _time) in enumerate(reads):
            if index % 2 == 0:
                self.assertNotIn("source_override", kwargs)
            else:
                self.assertEqual(kwargs["source_override"], "chosen")
        for pair in result["pairs"].values():
            self.assertNotIn("reference", pair)
            for role in ("original", "submission"):
                self.assertTrue(pair[role].endswith(f"-{role}.png"))
                with Image.open(self.out / pair[role]) as image:
                    self.assertEqual(image.mode, "RGBA")
                    self.assertEqual(image.size, (3, 2))


if __name__ == "__main__":
    unittest.main()
