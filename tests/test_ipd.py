
import unittest
import numpy as np

from src.features.ipd import extract_ipd
from src.features.logmel import extract_logmel


class TestIPD(unittest.TestCase):

    def setUp(self):
        self.fs = 16000
        self.n_samples = self.fs
        self.t = np.arange(self.n_samples) / self.fs

    def test_shape_and_timestamps(self):
        audio = np.zeros(
            (8, self.n_samples), dtype=np.float32
        )

        features, times = extract_ipd(audio)

        self.assertEqual(features.shape, (100, 4, 257, 2))
        self.assertEqual(times.shape, (100,))

        expected_times = (
            np.arange(100) + 0.5
        ) * 0.01

        np.testing.assert_allclose(
            times, expected_times, atol=1e-9
        )

    def test_identical_microphones(self):
        signal = np.sin(
            2 * np.pi * 1000 * self.t
        ).astype(np.float32)

        audio = np.tile(signal, (8, 1))
        features, _ = extract_ipd(audio)

        # 1000 Hz corresponds to FFT bin 32.
        cosine = features[10:90, :, 32, 0]
        sine = features[10:90, :, 32, 1]

        np.testing.assert_allclose(
            cosine, 1.0, atol=1e-4
        )
        np.testing.assert_allclose(
            sine, 0.0, atol=1e-4
        )

    def test_known_delay(self):
        frequency = 1000
        delay_samples = 2

        reference = np.sin(
            2 * np.pi * frequency * self.t
        ).astype(np.float32)

        # Delayed version of the same tone.
        delayed = np.zeros_like(reference)
        delayed[delay_samples:] = reference[:-delay_samples]

        audio = np.tile(reference, (8, 1))
        audio[4] = delayed

        features, _ = extract_ipd(audio)

        # For pair 01-05, positive delay gives
        # positive phase under X1 * conj(X5).
        expected_phase = (
            2 * np.pi * frequency
            * delay_samples / self.fs
        )

        measured_cos = features[10:90, 0, 32, 0].mean()
        measured_sin = features[10:90, 0, 32, 1].mean()

        self.assertAlmostEqual(
            measured_cos,
            np.cos(expected_phase),
            delta=0.02,
        )
        self.assertAlmostEqual(
            measured_sin,
            np.sin(expected_phase),
            delta=0.02,
        )

    def test_silence(self):
        audio = np.zeros(
            (8, self.n_samples), dtype=np.float32
        )

        features, _ = extract_ipd(audio)

        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.all(features == 0))

    def test_alignment_with_logmel(self):
        rng = np.random.default_rng(42)

        audio = rng.standard_normal(
            (8, self.n_samples)
        ).astype(np.float32)

        logmel, logmel_times = extract_logmel(audio[0])
        ipd, ipd_times = extract_ipd(audio)

        self.assertEqual(logmel.shape[0], ipd.shape[0])

        np.testing.assert_allclose(
            logmel_times, ipd_times, atol=1e-9
        )

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            extract_ipd(
                np.zeros(self.fs, dtype=np.float32)
            )

        with self.assertRaises(ValueError):
            extract_ipd(
                np.zeros((8, self.fs), dtype=np.float32),
                sample_rate=8000,
            )


if __name__ == "__main__":
    unittest.main()