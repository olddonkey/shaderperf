import json
import unittest

import numpy as np

from shaderperf.verifier import flip_mean, numerical_check, quantize_ldr


class FlipMeanTests(unittest.TestCase):
    def test_identical_images_have_zero_error(self):
        image = np.full((32, 32, 3), 0.25, dtype=np.float32)
        self.assertEqual(flip_mean(image, image.copy()), 0.0)

    def test_shifted_image_has_positive_error(self):
        reference = np.full((32, 32, 3), 0.25, dtype=np.float32)
        shifted = reference + np.float32(0.2)
        self.assertGreater(flip_mean(reference, shifted), 0.0)


class QuantizeLdrTests(unittest.TestCase):
    def test_shape_range_and_idempotence(self):
        image = np.array(
            [
                [[0.1, 0.2, 0.3, 1.0], [0.4, 0.5, 0.6, 1.0]],
                [[0.7, 0.8, 0.9, 1.0], [0.0, 1.0, 0.501, 1.0]],
            ],
            dtype=np.float32,
        )
        quantized = quantize_ldr(image)

        self.assertEqual(quantized.shape, (2, 2, 3))
        self.assertEqual(quantized.dtype, np.float32)
        self.assertGreaterEqual(float(quantized.min()), 0.0)
        self.assertLessEqual(float(quantized.max()), 1.0)
        np.testing.assert_array_equal(quantize_ldr(quantized), quantized)

    def test_values_outside_range_are_clipped(self):
        image = np.array([[[-0.5, 0.5, 1.5, 1.0]]], dtype=np.float32)
        quantized = quantize_ldr(image)

        self.assertEqual(float(quantized[0, 0, 0]), 0.0)
        self.assertEqual(float(quantized[0, 0, 2]), 1.0)


class NumericalCheckTests(unittest.TestCase):
    def setUp(self):
        self.clean = np.full((2, 2, 4), 0.5, dtype=np.float32)
        self.clean[..., 3] = 1.0

    def test_clean_image_passes(self):
        result = numerical_check(self.clean)

        self.assertFalse(result["has_nan"])
        self.assertFalse(result["has_inf"])
        self.assertTrue(result["in_range"])
        self.assertTrue(result["passed"])

    def test_nan_fails_with_nan_flag(self):
        image = self.clean.copy()
        image[0, 0, 0] = np.nan
        result = numerical_check(image)

        self.assertTrue(result["has_nan"])
        self.assertFalse(result["has_inf"])
        self.assertIsNone(result["min"])
        self.assertIsNone(result["max"])
        self.assertFalse(result["passed"])

    def test_inf_fails_with_inf_flag(self):
        image = self.clean.copy()
        image[0, 0, 1] = np.inf
        result = numerical_check(image)

        self.assertFalse(result["has_nan"])
        self.assertTrue(result["has_inf"])
        self.assertIsInstance(result["min"], float)
        self.assertIsNone(result["max"])
        self.assertFalse(result["passed"])

    def test_non_finite_result_is_strict_json_serializable(self):
        image = self.clean.copy()
        image[0, 0, 0] = np.nan

        encoded = json.dumps(numerical_check(image), allow_nan=False)

        self.assertIn('"min": null', encoded)
        self.assertIn('"max": null', encoded)

    def test_out_of_range_value_fails(self):
        image = self.clean.copy()
        image[0, 0, 2] = 1.5
        result = numerical_check(image)

        self.assertFalse(result["has_nan"])
        self.assertFalse(result["has_inf"])
        self.assertFalse(result["in_range"])
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
