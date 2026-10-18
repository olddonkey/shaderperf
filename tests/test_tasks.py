"""CPU-only checks for task ports; WGSL compilation/rendering is maintainer-gated."""

from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

from shaderperf.tasks_registry import load_registry


ROOT = Path(__file__).resolve().parents[1]
PARAMS_LINE = "struct Params { resolution: vec2<f32>, time: f32, _pad: f32, };"
EXPECTED_TASKS = {
    "task_tunnel.wgsl",
    "task_dots3.wgsl",
    "task_cherry.wgsl",
    "task_codebasealpha.wgsl",
    "task_gears_backup.wgsl",
    "task_dkmandel.wgsl",
    "task_dkburning.wgsl",
    "task_lobes.wgsl",
    "task_menger.wgsl",
    "task_goursat.wgsl",
    "task_devchatter.wgsl",
}
# Deliberately crude, line-based heuristic, not WGSL type inference: a line
# mentioning i32/u32 is exempt. Keep integer remainder explicitly typed on
# its own line; otherwise even a named integer operand can be flagged here.
FLOAT_REMAINDER = re.compile(r"[0-9a-z_)\]]\s*%\s*[0-9a-z_(]")


def _without_comments(source: str) -> str:
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.DOTALL)


class TaskPortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shaders = {
            path.name: path.read_text(encoding="utf-8")
            for path in sorted((ROOT / "tasks").glob("task_*.wgsl"))
        }

    def test_expected_task_files_exist(self):
        self.assertTrue(EXPECTED_TASKS.issubset(self.shaders))

    def test_exact_house_params_struct(self):
        for name, source in self.shaders.items():
            with self.subTest(shader=name):
                self.assertIn(PARAMS_LINE, _without_comments(source).splitlines())

    def test_vertex_and_fragment_entry_points(self):
        for name, source in self.shaders.items():
            with self.subTest(shader=name):
                code = _without_comments(source)
                self.assertRegex(code, r"@vertex\s+fn\s+vs_main\s*\(")
                self.assertRegex(
                    code,
                    r"@fragment\s+fn\s+fs_main\(@builtin\(position\)\s+"
                    r"frag_pos:\s*vec4<f32>\)\s*->\s*@location\(0\)\s+vec4<f32>",
                )

    def test_only_house_uniform_binding(self):
        for name, source in self.shaders.items():
            with self.subTest(shader=name):
                code = _without_comments(source)
                self.assertEqual(re.findall(r"@group\s*\(\s*(\d+)\s*\)", code), ["0"])
                self.assertEqual(re.findall(r"@binding\s*\(\s*(\d+)\s*\)", code), ["0"])
                self.assertIn("@group(0) @binding(0) var<uniform> params: Params;", code)

    def test_no_textures_or_glsl_interface_leftovers(self):
        for name, source in self.shaders.items():
            with self.subTest(shader=name):
                self.assertNotRegex(_without_comments(source), r"\btexture\w*\b")
                self.assertNotRegex(source, r"\b(?:iTime|iResolution|fragColor)\b")

    def test_no_float_remainder_heuristic(self):
        for name, source in self.shaders.items():
            for number, line in enumerate(_without_comments(source).splitlines(), 1):
                if "i32" in line or "u32" in line:
                    continue
                with self.subTest(shader=name, line=number):
                    self.assertIsNone(FLOAT_REMAINDER.search(line), line)

    def test_final_rgb_clamp_and_opaque_alpha(self):
        for name, source in self.shaders.items():
            with self.subTest(shader=name):
                self.assertRegex(
                    _without_comments(source),
                    r"return\s+vec4<f32>\(\s*clamp\(col,\s*vec3<f32>\(0\.0\),\s*"
                    r"vec3<f32>\(1\.0\)\),\s*1\.0\s*\);\s*\}\s*$",
                )

    def test_registry_loads(self):
        self.assertEqual(len(load_registry()), 4)


class TaskRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = load_registry(ROOT)

    def test_exact_task_assignments(self):
        expected = {
            "task_tunnel": ("natural", 4, 0.3),
            "task_gears_backup": ("natural", 4, 0.3),
            "task_cherry": ("injected", 72, 0.7),
            "task_codebasealpha": ("injected", 384, 0.7),
        }
        self.assertEqual(
            {t["task_id"]: (t["stratum"], t["batch"], t["duty"]) for t in self.tasks},
            expected,
        )
        for task in self.tasks:
            name = task["task_id"]
            injected = task["stratum"] == "injected"
            with self.subTest(task=name):
                self.assertEqual(
                    task["agent_file"], f"tasks/{name}{'_injected' if injected else ''}.wgsl"
                )
                self.assertEqual(
                    task["reference_file"], f"tasks/{name}{'' if injected else '_ref'}.wgsl"
                )
                self.assertEqual(task["visible_times"], [1.0, 3.7, 9.2])
                self.assertIsInstance(task["headroom"], dict)
                self.assertGreaterEqual(task["headroom"]["ci_low"], 1.15)
                self.assertEqual(task["headroom"]["batch"], task["batch"])
                self.assertEqual(task["headroom"]["duty"], task["duty"])
                self.assertEqual(task["sealed_task"], name)

    def test_every_referenced_file_exists(self):
        for task in self.tasks:
            paths = task["never_shown"] | {task["agent_file"], task["reference_file"]}
            if "injection_notes" in task:
                paths.add(task["injection_notes"])
            for path in paths:
                with self.subTest(task=task["task_id"], file=path):
                    self.assertTrue((ROOT / path).is_file())

    def test_source_metadata_and_local_license_evidence(self):
        fields = {
            "title", "author", "url", "license", "license_evidence",
            "popularity", "contamination_flag",
        }
        for task in self.tasks:
            with self.subTest(task=task["task_id"]):
                self.assertEqual(set(task["source"]), fields)
                for value in task["source"].values():
                    self.assertIsInstance(value, str)
                    self.assertTrue(value)
                evidence = task["source"]["license_evidence"].split(":", 1)[0]
                self.assertTrue((ROOT / evidence).is_file())

    def test_injected_agents_are_distinct_and_have_no_hint_markers(self):
        for task in self.tasks:
            if task["stratum"] == "injected":
                with self.subTest(task=task["task_id"]):
                    self.assertNotEqual(task["agent_file"], task["reference_file"])
                    source = (ROOT / task["agent_file"]).read_text(encoding="utf-8")
                    self.assertNotIn("INJECT", source.upper())
                    self.assertNotIn("never shown", source.lower())
                    notes = (ROOT / task["injection_notes"]).read_text(encoding="utf-8")
                    self.assertIn("INJECT-BEGIN", notes)
                    self.assertIn("INJECT-END", notes)

    def test_natural_references_marked_and_all_variants_keep_provenance(self):
        for task in self.tasks:
            original = (ROOT / f"tasks/{task['task_id']}.wgsl").read_text(encoding="utf-8")
            provenance = original.split("\nstruct Params", 1)[0]
            path = task["reference_file"] if task["stratum"] == "natural" else task["agent_file"]
            source = (ROOT / path).read_text(encoding="utf-8")
            with self.subTest(task=task["task_id"]):
                self.assertTrue(source.startswith(provenance))
                if task["stratum"] == "natural":
                    self.assertTrue(path.endswith("_ref.wgsl"))
                    self.assertIn("REFERENCE SOLUTION — never shown to the agent", source)

    def test_never_shown_contains_references_and_notes_and_excludes_agents(self):
        hidden = set()
        for task in self.tasks:
            with self.subTest(task=task["task_id"]):
                expected = {task["reference_file"]}
                if task["stratum"] == "injected":
                    expected.add(task["injection_notes"])
                self.assertIsInstance(task["never_shown"], set)
                self.assertEqual(task["never_shown"], expected)
                hidden.update(task["never_shown"])
        self.assertFalse(hidden & {t["agent_file"] for t in self.tasks})

    def test_default_root_is_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.chdir(temporary):
            self.assertEqual(load_registry(), self.tasks)

    def test_loads_return_independent_records(self):
        records = load_registry(ROOT)
        records[0]["never_shown"].clear()
        records[0]["source"]["title"] = "changed"
        self.assertEqual(load_registry(ROOT), self.tasks)

    def test_gears_is_registered_and_dots3_remains_a_rejected_candidate(self):
        original = (ROOT / "tasks/task_gears_backup.wgsl").read_text(encoding="utf-8")
        reference = (ROOT / "tasks/task_gears_backup_ref.wgsl").read_text(encoding="utf-8")
        self.assertTrue(reference.startswith(original.split("\nstruct Params", 1)[0]))
        self.assertIn("REFERENCE SOLUTION — never shown to the agent", reference)
        registered = {t["task_id"] for t in self.tasks}
        self.assertIn("task_gears_backup", registered)
        self.assertNotIn("task_dots3", registered)
        for filename in ("task_dots3.wgsl", "task_dots3_ref.wgsl"):
            with self.subTest(file=filename):
                self.assertTrue((ROOT / "tasks" / filename).is_file())
        dots3_reference = (ROOT / "tasks/task_dots3_ref.wgsl").read_text(encoding="utf-8")
        self.assertIn("REFERENCE SOLUTION — never shown to the agent", dots3_reference)


class TaskRegistryValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.records = json.loads((ROOT / "tasks/registry.json").read_text(encoding="utf-8"))
        for task in self.records:
            for name in {task["agent_file"], task["reference_file"], *task["never_shown"]}:
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")

    def write_registry(self, records=None):
        (self.root / "tasks/registry.json").write_text(
            json.dumps(self.records if records is None else records), encoding="utf-8"
        )

    def test_rejects_invalid_strata(self):
        for stratum in ("unknown", None, 0):
            with self.subTest(stratum=stratum):
                self.records[0]["stratum"] = stratum
                self.write_registry()
                with self.assertRaisesRegex(ValueError, "invalid stratum"):
                    load_registry(self.root)

    def test_rejects_nonpositive_and_noninteger_batch(self):
        for batch in (0, -1, 1.5, True, "48", None):
            with self.subTest(batch=batch):
                self.records[0]["batch"] = batch
                self.write_registry()
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    load_registry(self.root)

    def test_rejects_invalid_or_missing_duty(self):
        for duty in (0, -0.1, 1.01, True, "0.7", None, float("nan"), float("inf")):
            with self.subTest(duty=duty):
                self.records[0]["duty"] = duty
                self.write_registry()
                with self.assertRaisesRegex(ValueError, "duty must be a number"):
                    load_registry(self.root)
        del self.records[0]["duty"]
        self.write_registry()
        with self.assertRaisesRegex(ValueError, "duty must be a number"):
            load_registry(self.root)

    def test_accepts_duty_endpoints_within_range(self):
        for duty in (0.001, 0.3, 0.7, 1, 1.0):
            with self.subTest(duty=duty):
                self.records[0]["duty"] = duty
                self.write_registry()
                self.assertEqual(load_registry(self.root)[0]["duty"], duty)

    def test_rejects_missing_referenced_files(self):
        for key in ("agent_file", "reference_file", "injection_notes"):
            records = copy.deepcopy(self.records)
            records[2][key] = "tasks/missing.wgsl"
            with self.subTest(key=key):
                self.write_registry(records)
                with self.assertRaisesRegex(ValueError, "missing or outside"):
                    load_registry(self.root)

    def test_rejects_absolute_and_escaping_paths(self):
        # Both candidates exist, so the rejection must be about the root boundary.
        outside = ROOT / "tasks/task_tunnel.wgsl"
        for path in (str(outside), os.path.relpath(outside, self.root)):
            with self.subTest(path=path):
                self.records[0]["agent_file"] = path
                self.write_registry()
                with self.assertRaises(ValueError):
                    load_registry(self.root)

    def test_rejects_injected_agent_equal_to_reference(self):
        self.records[2]["agent_file"] = self.records[2]["reference_file"]
        self.write_registry()
        with self.assertRaisesRegex(ValueError, "different files"):
            load_registry(self.root)

    def test_rejects_visible_material_from_another_tasks_hidden_set(self):
        self.records[0]["agent_file"] = self.records[1]["reference_file"]
        self.write_registry()
        with self.assertRaisesRegex(ValueError, "overlap never_shown"):
            load_registry(self.root)

    def test_rejects_unhidden_reference_and_notes(self):
        for key in ("reference_file", "injection_notes"):
            records = copy.deepcopy(self.records)
            records[2]["never_shown"].remove(records[2][key])
            with self.subTest(key=key):
                self.write_registry(records)
                with self.assertRaisesRegex(ValueError, "must be never_shown"):
                    load_registry(self.root)

    def test_rejects_invalid_records_and_duplicate_ids(self):
        for records in ({}, [], [None], [self.records[0], self.records[0]]):
            with self.subTest(records=records):
                self.write_registry(records)
                with self.assertRaises(ValueError):
                    load_registry(self.root)


if __name__ == "__main__":
    unittest.main()
