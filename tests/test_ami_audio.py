
import tempfile
import unittest
import wave
from pathlib import Path

from src.data.ami_audio import validate_array_audio


class TestAMIAudio(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audio_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_wav(
        self,
        mic,
        sample_rate=16000,
        frames=1600,
        channels=1,
        sample_width=2,
    ):
        path = self.audio_dir / f"TEST.Array1-{mic:02d}.wav"

        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(sample_width)
            wav.setframerate(sample_rate)

            # Write silent PCM samples
            wav.writeframes(
                bytes(frames * channels * sample_width)
            )

        return path

    def create_all_mics(self):
        for mic in range(1, 9):
            self.create_wav(mic)

    def test_valid_eight_microphones(self):
        self.create_all_mics()

        result = validate_array_audio(
            meeting="TEST",
            audio_dir=self.audio_dir,
        )

        self.assertEqual(result["microphones"], 8)
        self.assertEqual(result["sample_rate"], 16000)
        self.assertEqual(result["frames"], 1600)
        self.assertAlmostEqual(result["duration"], 0.1)

    def test_missing_microphone(self):
        for mic in range(1, 8):
            self.create_wav(mic)

        with self.assertRaises(FileNotFoundError):
            validate_array_audio(
                meeting="TEST",
                audio_dir=self.audio_dir,
            )

    def test_sample_rate_mismatch(self):
        self.create_all_mics()

        # Replace microphone 4 with a different sample rate
        self.create_wav(4, sample_rate=8000)

        with self.assertRaises(ValueError):
            validate_array_audio(
                meeting="TEST",
                audio_dir=self.audio_dir,
            )

    def test_frame_count_mismatch(self):
        self.create_all_mics()

        self.create_wav(5, frames=1500)

        with self.assertRaises(ValueError):
            validate_array_audio(
                meeting="TEST",
                audio_dir=self.audio_dir,
            )

    def test_stereo_microphone(self):
        self.create_all_mics()

        self.create_wav(3, channels=2)

        with self.assertRaises(ValueError):
            validate_array_audio(
                meeting="TEST",
                audio_dir=self.audio_dir,
            )

    def test_sample_width_mismatch(self):
        self.create_all_mics()

        self.create_wav(6, sample_width=1)

        with self.assertRaises(ValueError):
            validate_array_audio(
                meeting="TEST",
                audio_dir=self.audio_dir,
            )


if __name__ == "__main__":
    unittest.main()