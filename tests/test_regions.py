from __future__ import annotations

import contextlib
import io
import re
import unittest
from pathlib import Path

from shaderperf.ablate import main as ablate_main
from shaderperf.regions import (
    ablate,
    find_regions,
    region_contains_control_exit,
    strip_comments,
)


ROOT = Path(__file__).resolve().parents[1]
RAYMARCH = ROOT / "shaders/gate0/raymarch_sdf.wgsl"
VORONOI = ROOT / "shaders/gate0/voronoi_cells.wgsl"


def _braces_balanced(src: str) -> bool:
    depth = 0
    for char in strip_comments(src):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


class InlineRegionTests(unittest.TestCase):
    def test_for_with_nested_if_else_has_expected_regions(self):
        src = """\
fn shade() {
    for (var i = 0; i < 4; i = i + 1) {
        if (i == 2) {
            work_a();
        } else {
            work_b();
        }
    }
}
"""

        regions = find_regions(src)

        self.assertEqual([r.kind for r in regions], ["for", "if", "else"])
        self.assertEqual([r.function for r in regions], ["shade"] * 3)
        self.assertEqual([r.ordinal for r in regions], [1, 2, 3])
        self.assertEqual(
            [r.id for r in regions],
            ["shade:for#1", "shade:if#2", "shade:else#3"],
        )

    def test_else_if_is_one_region_kind(self):
        src = """\
fn shade() {
    if (first()) { one(); }
    else if (second()) { two(); }
    else { three(); }
}
"""

        regions = find_regions(src)

        self.assertEqual([r.kind for r in regions], ["if", "else_if", "else"])
        self.assertEqual(sum(r.kind == "if" for r in regions), 1)

    def test_comment_braces_and_keywords_are_ignored(self):
        src = """\
fn shade() {
    // if (x) { fake(); }
    /* for (;;) { also_fake(); } and a stray } */
    while (active()) {
        do_work(); // }
    }
}
"""

        regions = find_regions(src)

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].kind, "while")

    def test_ablation_only_replaces_selected_body(self):
        src = """\
fn shade() {
    for (var i = 0; i < 2; i = i + 1) {
        if (ready()) { work(); }
    }
}
"""
        first, second = find_regions(src)

        first_variant = ablate(src, first)
        second_variant = ablate(src, second)

        self.assertNotEqual(first_variant, src)
        self.assertNotEqual(second_variant, src)
        self.assertNotEqual(first_variant, second_variant)
        for region, variant in (
            (first, first_variant),
            (second, second_variant),
        ):
            self.assertEqual(
                variant,
                src[: region.body_start] + " " + src[region.body_end :],
            )
            self.assertEqual(src[region.body_start - 1], "{")
            self.assertEqual(src[region.body_end], "}")
            self.assertTrue(_braces_balanced(variant))

    def test_empty_and_comment_only_bodies_are_skipped(self):
        src = """\
fn shade() {
    if (a()) { }
    while (b()) { /* no executable text */ }
    loop { work(); }
}
"""

        regions = find_regions(src)

        self.assertEqual([r.kind for r in regions], ["loop"])
        self.assertEqual(regions[0].ordinal, 1)

    def test_control_exit_hint(self):
        src = """\
fn shade() {
    loop { if (done()) { break; } }
    while (active()) { keep_working(); }
}
"""
        loop_region, if_region, while_region = find_regions(src)

        self.assertTrue(region_contains_control_exit(src, loop_region))
        self.assertTrue(region_contains_control_exit(src, if_region))
        self.assertFalse(region_contains_control_exit(src, while_region))


class FixtureRegionTests(unittest.TestCase):
    def _assert_fixture_offsets(self, path: Path, minimum: int):
        src = path.read_text()
        regions = find_regions(src)
        function_names = set(re.findall(r"\bfn\s+([A-Za-z_]\w*)\s*\(", src))

        self.assertGreaterEqual(len(regions), minimum)
        for region in regions:
            self.assertIn(region.function, function_names)
            self.assertEqual(src[region.body_start - 1], "{")
            self.assertEqual(src[region.body_end], "}")
            self.assertTrue(src[region.body_start : region.body_end].strip())
        return src, regions

    def test_raymarch_fixture(self):
        _, regions = self._assert_fixture_offsets(RAYMARCH, 5)

        self.assertTrue(
            any(r.function == "fs_main" and r.kind == "for" for r in regions)
        )

    def test_voronoi_fixture(self):
        _, regions = self._assert_fixture_offsets(VORONOI, 4)

        self.assertGreaterEqual(sum(r.kind == "for" for r in regions), 2)
        self.assertTrue(any(r.kind == "if" for r in regions))
        self.assertTrue(any(r.kind in {"else", "else_if"} for r in regions))

    def test_list_cli_for_both_fixtures(self):
        for path in (RAYMARCH, VORONOI):
            with self.subTest(shader=path.name), contextlib.redirect_stdout(
                io.StringIO()
            ) as stdout:
                exit_code = ablate_main([str(path), "--list"])

            self.assertEqual(exit_code, 0)
            table = stdout.getvalue()
            self.assertIn("snippet", table)
            self.assertRegex(table, r"\n\d+ region\(s\)\n$")
            self.assertNotIn("\n0 region(s)\n", table)


if __name__ == "__main__":
    unittest.main()
