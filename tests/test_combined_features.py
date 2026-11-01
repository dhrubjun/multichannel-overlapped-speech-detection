
import unittest
import numpy as np

from src.features.combined import extract_combined_features


class TestCombinedFeatures(unittest.TestCase):

    def test_shape_and_alignment(self):
        rng = np.random.default_rng(42)

        audio = (
            rng.standard_normal((8, 32000))
            * 0.01
        ).astype(np.float32)

        features, times = extract_combined_features(audio)

        self.assertEqual(features.shape, (200, 2136))
        self.assertEqual(times.shape, (200,))
        self.assertTrue(np.isfinite(features).all())

        self.assertAlmostEqual(times[0], 0.005)
        self.assertAlmostEqual(times[-1], 1.995)

    def test_invalid_audio(self):
        audio = np.zeros((2, 16000), dtype=np.float32)

        with self.assertRaises(ValueError):
            extract_combined_features(audio)


if __name__ == "__main__":
    unittest.main()