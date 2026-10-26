
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data.speech_dataset import SpeechSequenceDataset


class TestSpeechSequenceDataset(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        (self.root / "outputs/features").mkdir(parents=True)
        (self.root / "outputs/labels").mkdir(parents=True)

        self.create_meeting("TEST_A", 450, value=2.0)
        self.create_meeting("TEST_B", 230, value=5.0)

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_meeting(self, meeting, n_frames, value):
        features = np.full(
            (n_frames, 80), value, dtype=np.float32
        )
        labels = (np.arange(n_frames) % 3).astype(np.int64)
        valid_mask = np.ones(n_frames, dtype=bool)

        # Include an uncertain annotation.
        valid_mask[10] = False

        frame_times = (
            np.arange(n_frames) + 0.5
        ) * 0.01

        np.savez_compressed(
            self.root / "outputs/features"
            / f"{meeting}_logmel_mic01.npz",
            features=features,
            frame_times=frame_times,
        )

        np.savez_compressed(
            self.root / "outputs/labels"
            / f"{meeting}_frame_labels.npz",
            labels=labels,
            valid_mask=valid_mask,
            frame_times=frame_times,
        )

    def make_dataset(self, meetings, stride=200, stats=None):
        return SpeechSequenceDataset(
            meetings=meetings,
            project_root=self.root,
            sequence_length=200,
            stride=stride,
            normalization_stats=stats,
        )

    def test_sequence_shapes(self):
        dataset = self.make_dataset(["TEST_A"])
        sample = dataset[0]

        self.assertEqual(sample["features"].shape, (200, 80))
        self.assertEqual(sample["labels"].shape, (200,))
        self.assertEqual(sample["valid_mask"].shape, (200,))

        self.assertEqual(sample["features"].dtype, torch.float32)
        self.assertEqual(sample["labels"].dtype, torch.int64)
        self.assertEqual(sample["valid_mask"].dtype, torch.bool)

    def test_training_overlap(self):
        dataset = self.make_dataset(["TEST_A"], stride=100)

        starts = [start for _, start in dataset.windows]

        self.assertEqual(starts, [0, 100, 200, 300, 400])

    def test_evaluation_coverage(self):
        dataset = self.make_dataset(["TEST_A"], stride=200)

        coverage = np.zeros(450, dtype=np.int32)

        for _, start in dataset.windows:
            end = min(start + 200, 450)
            coverage[start:end] += 1

        self.assertTrue(np.all(coverage == 1))

    def test_final_padding(self):
        dataset = self.make_dataset(["TEST_A"], stride=200)
        last = dataset[-1]

        self.assertEqual(last["start_frame"], 400)
        self.assertEqual(last["features"].shape, (200, 80))

        # 50 real frames, followed by 150 padded frames.
        self.assertTrue(
            torch.all(last["features"][50:] == 0)
        )
        self.assertTrue(
            torch.all(last["labels"][50:] == -100)
        )
        self.assertFalse(
            last["valid_mask"][50:].any().item()
        )

    def test_uncertain_labels_are_masked(self):
        dataset = self.make_dataset(["TEST_A"])
        sample = dataset[0]

        self.assertFalse(sample["valid_mask"][10].item())
        self.assertTrue(sample["valid_mask"][9].item())
        self.assertTrue(sample["valid_mask"][11].item())

    def test_meeting_boundaries(self):
        dataset = self.make_dataset(
            ["TEST_A", "TEST_B"], stride=200
        )

        self.assertEqual(len(dataset), 5)

        meetings = [dataset[i]["meeting"] for i in range(5)]
        self.assertEqual(
            meetings,
            ["TEST_A", "TEST_A", "TEST_A",
             "TEST_B", "TEST_B"],
        )

        # A sequence must contain frames from one meeting only.
        self.assertTrue(
            torch.all(dataset[0]["features"] == 2.0)
        )
        self.assertTrue(
            torch.all(dataset[3]["features"] == 5.0)
        )

    def test_normalization(self):
        stats_path = self.root / "stats.npz"

        np.savez_compressed(
            stats_path,
            mean=np.full(80, 2.0, dtype=np.float32),
            std=np.full(80, 2.0, dtype=np.float32),
        )

        dataset = self.make_dataset(
            ["TEST_B"], stats=stats_path
        )

        sample = dataset[0]

        # (5 - 2) / 2 = 1.5
        self.assertTrue(
            torch.allclose(
                sample["features"],
                torch.full((200, 80), 1.5),
            )
        )

    def test_invalid_stride(self):
        with self.assertRaises(ValueError):
            self.make_dataset(["TEST_A"], stride=250)


if __name__ == "__main__":
    unittest.main()