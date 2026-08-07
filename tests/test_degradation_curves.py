import unittest
import math
from nightmarenet.evaluation.degradation_curves import calculate_degradation_curves


class TestDegradationCurvesUnit(unittest.TestCase):
    def test_calculate_degradation_curves_basic(self):
        data = {
            "model_a": {
                "noise": {
                    "strengths": [0.1, 0.5, 0.9],
                    "accuracies": [0.9, 0.7, 0.4],
                }
            }
        }
        curves = calculate_degradation_curves(data)
        self.assertIn("model_a", curves)
        self.assertEqual(len(curves["model_a"]), 3)
        self.assertEqual(curves["model_a"][0]["strength"], 0.1)
        self.assertEqual(curves["model_a"][0]["robustness"], 0.9)

    def test_empty_and_single_point(self):
        empty_curves = calculate_degradation_curves({})
        self.assertEqual(empty_curves, {})

        single = {
            "model_b": {
                "blur": {
                    "strengths": [0.0],
                    "accuracies": [1.0],
                }
            }
        }
        res = calculate_degradation_curves(single)
        self.assertEqual(len(res["model_b"]), 1)

    def test_flat_and_extreme_values(self):
        flat = {
            "model_c": {
                "flat_dist": {
                    "strengths": [1e-5, 1e5],
                    "accuracies": [0.5, 0.5],
                }
            }
        }
        res = calculate_degradation_curves(flat)
        self.assertEqual(res["model_c"][0]["robustness"], 0.5)
        self.assertEqual(res["model_c"][1]["robustness"], 0.5)

    def test_nan_values(self):
        nan_data = {
            "model_d": {
                "nan_dist": {
                    "strengths": [0.1],
                    "accuracies": [float("nan")],
                }
            }
        }
        res = calculate_degradation_curves(nan_data)
        self.assertTrue(math.isnan(res["model_d"][0]["robustness"]))


if __name__ == "__main__":
    unittest.main()
