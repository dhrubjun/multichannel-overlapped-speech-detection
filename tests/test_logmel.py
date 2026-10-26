
import unittest
import numpy as np

from src.features.logmel import (
    extract_logmel,
    make_mel_filterbank,
)


class TestLogMel(unittest.TestCase):

    def setUp(self):
        self.fs = 16000
        self.t = np.arange(self.fs) / self.fs

    def test_output_shape_and_timestamps(self):
        audio = np.sin(
            2 * np.pi * 440 * self.t
        ).astype(np.float32)

        features, times = extract_logmel(audio)

        self.assertEqual(features.shape, (100, 80))
        self.assertEqual(times.shape, (100,))
        self.assertTrue(
            np.allclose(
                times,
                (np.arange(100) + 0.5) * 0.01,
                atol=1e-9,
            )
        )

    def test_silence(self):
        audio = np.zeros(self.fs, dtype=np.float32)
        features, _ = extract_logmel(audio)

        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(
            np.allclose(features, np.log(1e-10))
        )

    def test_frequency_localization(self):
        audio = np.sin(
            2 * np.pi * 440 * self.t
        ).astype(np.float32)

        features, _ = extract_logmel(audio)
        filters = make_mel_filterbank()

        frequencies = np.fft.rfftfreq(
            512, d=1 / self.fs
        )

        band = np.argmax(features[10:90].mean(axis=0))
        frequency = frequencies[
            np.argmax(filters[band])
        ]

        self.assertLess(abs(frequency - 440), 100)

    def test_invalid_sample_rate(self):
        with self.assertRaises(ValueError):
            extract_logmel(
                np.zeros(self.fs, dtype=np.float32),
                sample_rate=8000,
            )

    def test_nonfinite_audio(self):
        audio = np.zeros(self.fs, dtype=np.float32)
        audio[100] = np.nan

        with self.assertRaises(ValueError):
            extract_logmel(audio)

    def test_explicit_frame_count(self):
        audio = np.zeros(self.fs, dtype=np.float32)

        features, times = extract_logmel(
            audio, n_frames=100
        )

        self.assertEqual(features.shape, (100, 80))
        self.assertAlmostEqual(times[0], 0.005)
        self.assertAlmostEqual(times[-1], 0.995)


if __name__ == "__main__":
    unittest.main()