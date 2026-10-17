"""CPU-only checks for task ports; WGSL compilation/rendering is maintainer-gated."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARAMS_LINE = "struct Params { resolution: vec2<f32>, time: f32, _pad: f32, };"
EXPECTED_TASKS = {
    "task_tunnel.wgsl",
    "task_dots3.wgsl",
    "task_cherry.wgsl",
    "task_codebasealpha.wgsl",
    "task_gears_backup.wgsl",
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

    @unittest.skip("Task registry comes with the next unit.")
    def test_registry_placeholder(self):
        pass


if __name__ == "__main__":
    unittest.main()
