from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shaderperf.calibrate import choose_threshold
from shaderperf.calibration import (
    SHADER_DIR,
    VARIANTS,
    apply_constant_light,
    apply_posterize,
    apply_variant,
    pair_plan,
    remove_safe_block,
    rewrite_knob,
)
from shaderperf.label_tool import (
    _page,
    left_is_reference,
    load_labels,
    update_label_atomic,
    write_labels_atomic,
)


class VariantTransformTests(unittest.TestCase):
    def test_knob_rewrite_replaces_exact_const_line_only(self):
        source = (
            "const CLOUD_OCTAVES: i32 = 6;\n"
            "const OTHER: i32 = 9;\n"
            "let mention = CLOUD_OCTAVES;\n"
        )

        transformed = rewrite_knob(source, "CLOUD_OCTAVES", 2)

        self.assertEqual(
            transformed,
            "const CLOUD_OCTAVES: i32 = 2;\n"
            "const OTHER: i32 = 9;\n"
            "let mention = CLOUD_OCTAVES;\n",
        )

    def test_knob_rewrite_rejects_missing_or_duplicate_lines(self):
        with self.assertRaises(ValueError):
            rewrite_knob("const OTHER: i32 = 6;\n", "CLOUD_OCTAVES", 2)
        with self.assertRaises(ValueError):
            rewrite_knob(
                "const CLOUD_OCTAVES: i32 = 6;\n" * 2,
                "CLOUD_OCTAVES",
                2,
            )

    def test_safe_block_deletion_removes_only_marked_lines(self):
        source = (
            "before\n"
            "    // SAFE-BEGIN\n"
            "    let ballast = sin(1.0);\n"
            "    col = col + vec3<f32>(ballast * 0.0000001);\n"
            "    // SAFE-END\n"
            "after\n"
        )

        self.assertEqual(remove_safe_block(source), "before\nafter\n")

    def test_posterize_substitution_replaces_only_site(self):
        source = (
            "var col = vec3<f32>(0.5);\n"
            "    // POSTERIZE-SITE\n"
            "return vec4<f32>(col, 1.0);\n"
        )

        transformed = apply_posterize(source, 8.0)

        self.assertEqual(
            transformed,
            "var col = vec3<f32>(0.5);\n"
            "    col = floor(col * 8.0) / 8.0;\n"
            "return vec4<f32>(col, 1.0);\n",
        )

    def test_constant_light_substitution_replaces_only_site(self):
        source = "var diffuse = 0.2;\n    // FLATLIGHT-SITE\nreturn diffuse;\n"

        transformed = apply_constant_light(source)

        self.assertEqual(
            transformed,
            "var diffuse = 0.2;\n    diffuse = 0.65;\nreturn diffuse;\n",
        )

    def test_declared_variants_touch_only_their_expected_regions(self):
        for shader, shader_spec in VARIANTS.items():
            source = (SHADER_DIR / f"{shader}.wgsl").read_text()
            for variant in shader_spec.variants:
                with self.subTest(shader=shader, variant=variant.variant_id):
                    transformed = apply_variant(source, variant)
                    self.assertNotEqual(transformed, source)
                    if variant.transform == "safe_ballast_removal":
                        self.assertEqual(transformed, remove_safe_block(source))
                        self.assertNotIn("// SAFE-BEGIN", transformed)
                        self.assertNotIn("// SAFE-END", transformed)
                    else:
                        source_lines = source.splitlines()
                        transformed_lines = transformed.splitlines()
                        self.assertEqual(len(source_lines), len(transformed_lines))
                        differences = [
                            index
                            for index, (before, after) in enumerate(
                                zip(source_lines, transformed_lines)
                            )
                            if before != after
                        ]
                        self.assertEqual(len(differences), 1)

    def test_pair_plan_size_safe_count_and_seed_determinism(self):
        first = pair_plan(19)
        second = pair_plan(19)
        other_seed = pair_plan(20)

        first_ids = [pair["pair_id"] for pair in first]
        self.assertEqual(first_ids, [pair["pair_id"] for pair in second])
        self.assertNotEqual(first_ids, [pair["pair_id"] for pair in other_seed])
        self.assertEqual(len(first), 50)
        self.assertEqual(
            sum(pair["expected"] == "acceptable" for pair in first), 7
        )
        self.assertEqual(len(first_ids), len(set(first_ids)))


class ThresholdRuleTests(unittest.TestCase):
    def test_perfect_separation_has_unit_balanced_accuracy(self):
        rows = [
            {"pair_id": "a", "flip_mean": 0.01, "label": "acceptable"},
            {"pair_id": "b", "flip_mean": 0.02, "label": "acceptable"},
            {"pair_id": "c", "flip_mean": 0.20, "label": "unacceptable"},
            {"pair_id": "d", "flip_mean": 0.30, "label": "unacceptable"},
        ]

        report = choose_threshold(rows)

        self.assertGreater(report["tau_FLIP"], 0.02)
        self.assertLess(report["tau_FLIP"], 0.20)
        self.assertEqual(report["balanced_accuracy"], 1.0)
        self.assertEqual(report["inversions"], [])

    def test_overlap_reports_inversions(self):
        rows = [
            {"pair_id": "a", "flip_mean": 0.01, "label": "acceptable"},
            {"pair_id": "b", "flip_mean": 0.30, "label": "acceptable"},
            {"pair_id": "c", "flip_mean": 0.10, "label": "unacceptable"},
            {"pair_id": "d", "flip_mean": 0.20, "label": "unacceptable"},
        ]

        report = choose_threshold(rows)

        self.assertTrue(report["inversions"])

    def test_threshold_rule_is_deterministic_and_ties_choose_smaller(self):
        rows = [
            {"pair_id": "d", "flip_mean": 0.4, "label": "unacceptable"},
            {"pair_id": "a", "flip_mean": 0.1, "label": "acceptable"},
            {"pair_id": "c", "flip_mean": 0.3, "label": "acceptable"},
            {"pair_id": "b", "flip_mean": 0.2, "label": "unacceptable"},
        ]

        first = choose_threshold(rows)
        second = choose_threshold(list(reversed(rows)))

        self.assertEqual(first, second)
        best = max(
            row["balanced_accuracy"] for row in first["candidate_thresholds"]
        )
        tied = [
            row["threshold"]
            for row in first["candidate_thresholds"]
            if row["balanced_accuracy"] == best
        ]
        self.assertEqual(first["tau_FLIP"], min(tied))


class LabelStoreTests(unittest.TestCase):
    def test_atomic_write_round_trip_and_update_preserves_existing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "labels.json"
            existing = {
                "0123456789": {
                    "label": "acceptable",
                    "utc": "2026-09-12T12:00:00Z",
                }
            }
            write_labels_atomic(path, existing)

            self.assertEqual(load_labels(path), existing)
            updated = update_label_atomic(
                path,
                "abcdef0123",
                "unacceptable",
                "2026-09-12T12:01:00Z",
            )

            self.assertEqual(load_labels(path), updated)
            self.assertEqual(updated["0123456789"], existing["0123456789"])
            self.assertEqual(updated["abcdef0123"]["label"], "unacceptable")
            self.assertEqual(list(path.parent.glob(".labels.json.*.tmp")), [])

    def test_side_randomization_is_deterministic(self):
        self.assertEqual(
            left_is_reference("0123456789"), left_is_reference("0123456789")
        )

    def test_page_exposes_only_blinded_image_sides(self):
        page = _page("0123456789", index=0, total=50, labeled=0).decode("utf-8")

        self.assertIn("/image/0123456789/left", page)
        self.assertIn("/image/0123456789/right", page)
        for hidden_value in (
            "_ref.png",
            "_var.png",
            "known_safe",
            "graded_degradation",
            "variant_id",
            "severity",
            "expected",
        ):
            self.assertNotIn(hidden_value, page)


if __name__ == "__main__":
    unittest.main()
