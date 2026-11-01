
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.m2_dataset import M2SequenceDataset


class TestM2Dataset(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        feature_dir = self.root / "outputs" / "features"
        label_dir = self.root / "outputs" / "labels"

        feature_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)

        self.meeting = "TEST001"
        self.n_frames = 253

        rng = np.random.default_rng(42)

        features = np.zeros(
            (self.n_frames, 2136),
            dtype=np.float32,
        )

        features[:, :80] = rng.normal(
            loc=5.0,
            scale=2.0,
            size=(self.n_frames, 80),
        ).astype(np.float32)

        features[:, 80:100] = 0.5

        np.save(
            feature_dir / f"{self.meeting}_m2_combined.npy",
            features,
        )

        labels = (
            np.arange(self.n_frames) % 3
        ).astype(np.int64)

        valid_mask = np.ones(
            self.n_frames,
            dtype=bool,
        )

        valid_mask[10:15] = False

        frame_times = (
            np.arange(self.n_frames, dtype=np.float64)
            + 0.5
        ) * 0.01

        np.savez_compressed(
            label_dir / f"{self.meeting}_frame_labels.npz",
            labels=labels,
            valid_mask=valid_mask,
            frame_times=frame_times,
        )

        np.savez_compressed(
            feature_dir / "stats.npz",
            mean=np.full(80, 5.0, dtype=np.float32),
            std=np.full(80, 2.0, dtype=np.float32),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_dataset(self, stride=200):
        return M2SequenceDataset(
            meetings=[self.meeting],
            project_root=self.root,
            sequence_length=200,
            stride=stride,
            normalization_stats=(
                "outputs/features/stats.npz"
            ),
        )

    def test_memory_mapping_and_normalization(self):
        dataset = self.make_dataset()

        self.assertIsInstance(
            dataset.recordings[0]["features"],
            np.memmap,
        )

        sample = dataset[0]
        x = sample["features"].numpy()

        self.assertEqual(x.shape, (200, 2136))
        self.assertTrue(np.isfinite(x).all())

        raw = dataset.recordings[0]["features"]

        expected = (raw[:200, :80] - 5.0) / 2.0

        np.testing.assert_allclose(
            x[:, :80],
            expected,
            atol=1e-6,
            rtol=0,
        )

        np.testing.assert_array_equal(
            x[:, 80:],
            raw[:200, 80:],
        )

        self.assertTrue(
            np.all(x[:, 100:] == 0)
        )

    def test_last_sequence_padding(self):
        dataset = self.make_dataset(stride=200)

        self.assertEqual(len(dataset), 2)

        sample = dataset[-1]

        x = sample["features"].numpy()
        y = sample["labels"].numpy()
        mask = sample["valid_mask"].numpy()

        # 253 total frames, last sequence begins at 200.
        self.assertEqual(
            int(sample["start_frame"]),
            200,
        )

        np.testing.assert_array_equal(
            y[:53],
            np.arange(200, 253) % 3,
        )

        self.assertTrue(np.all(y[53:] == -100))
        self.assertTrue(np.all(~mask[53:]))
        self.assertTrue(np.all(x[53:] == 0))

    def test_invalid_frames_preserved(self):
        dataset = self.make_dataset()

        sample = dataset[0]

        mask = sample["valid_mask"].numpy()

        self.assertTrue(np.all(~mask[10:15]))
        self.assertTrue(np.all(mask[:10]))
        self.assertTrue(np.all(mask[15:200]))

    def test_pytorch_batch(self):
        dataset = self.make_dataset(stride=100)

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(loader))

        self.assertEqual(
            tuple(batch["features"].shape),
            (2, 200, 2136),
        )

        self.assertEqual(
            tuple(batch["labels"].shape),
            (2, 200),
        )

        self.assertEqual(
            tuple(batch["valid_mask"].shape),
            (2, 200),
        )

        self.assertEqual(
            batch["features"].dtype,
            torch.float32,
        )

        self.assertTrue(
            torch.isfinite(batch["features"]).all()
        )


if __name__ == "__main__":
    unittest.main()