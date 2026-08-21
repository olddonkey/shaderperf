from __future__ import annotations

import unittest

from shaderperf.stats import bootstrap_share_difference


class BootstrapShareDifferenceTests(unittest.TestCase):
    def test_identical_pairs_contain_zero(self):
        result = bootstrap_share_difference(
            [100.0] * 12,
            [90.0] * 12,
            [100.0] * 12,
            [90.0] * 12,
            n_boot=1_000,
            seed=7,
        )

        self.assertAlmostEqual(result["delta_point"], 0.0)
        self.assertLessEqual(result["ci_low"], 0.0)
        self.assertGreaterEqual(result["ci_high"], 0.0)
        self.assertFalse(result["distinguishable_from_zero"])

    def test_clearly_different_pairs_are_distinguishable(self):
        result = bootstrap_share_difference(
            [100.0] * 12,
            [80.0] * 12,
            [100.0] * 12,
            [100.0] * 12,
            n_boot=1_000,
            seed=11,
        )

        self.assertAlmostEqual(result["delta_point"], 0.2)
        self.assertGreater(result["ci_low"], 0.0)
        self.assertTrue(result["distinguishable_from_zero"])

    def test_fixed_seed_is_deterministic(self):
        args = (
            [98.0, 100.0, 101.0, 102.0, 105.0],
            [78.0, 79.0, 82.0, 83.0, 86.0],
            [97.0, 99.0, 100.0, 103.0, 104.0],
            [94.0, 97.0, 99.0, 101.0, 102.0],
        )

        first = bootstrap_share_difference(*args, n_boot=500, seed=23)
        second = bootstrap_share_difference(*args, n_boot=500, seed=23)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
