
import unittest
import numpy as np

from src.data.frame_labels import (
    generate_valid_mask,
)


class TestValidMask(unittest.TestCase):

    def test_no_uncertain_intervals(self):
        times = (np.arange(100) + 0.5) * 0.01

        mask = generate_valid_mask([], times)

        self.assertTrue(np.all(mask))

    def test_vocal_sound_interval(self):
        times = (np.arange(200) + 0.5) * 0.01

        mask = generate_valid_mask(
            [(0.5, 1.0)],
            times
        )

        self.assertTrue(np.all(mask[:50]))
        self.assertTrue(np.all(~mask[50:100]))
        self.assertTrue(np.all(mask[100:]))

    def test_invalid_uncertainty_interval(self):
        times = (np.arange(100) + 0.5) * 0.01

        with self.assertRaises(ValueError):
            generate_valid_mask(
                [(1.0, 0.5)],
                times
            )


if __name__ == "__main__":
    unittest.main()