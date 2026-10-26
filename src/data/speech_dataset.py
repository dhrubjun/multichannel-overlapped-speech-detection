
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SpeechSequenceDataset(Dataset):
    def __init__(
        self,
        meetings,
        project_root=".",
        sequence_length=200,
        stride=100,
        normalization_stats=None,
    ):
        self.sequence_length = sequence_length
        self.stride = stride
        self.recordings = []
        self.windows = []

        if sequence_length <= 0 or stride <= 0:
            raise ValueError("Sequence length and stride must be positive")

        root = Path(project_root)

        self.mean = None
        self.std = None

        if normalization_stats is not None:
            stats_path = Path(normalization_stats)

            with np.load(stats_path) as stats:
                self.mean = stats["mean"].astype(np.float32)
                self.std = stats["std"].astype(np.float32)

            if self.mean.shape != (80,) or self.std.shape != (80,):
                raise ValueError("Expected 80 normalization values")

            if not np.isfinite(self.mean).all():
                raise ValueError("Non-finite normalization means")

            if not np.isfinite(self.std).all():
                raise ValueError("Non-finite normalization std")

            if np.any(self.std <= 1e-8):
                raise ValueError("Invalid normalization std")

        for meeting in meetings:
            feature_file = (
                root / "outputs" / "features"
                / f"{meeting}_logmel_mic01.npz"
            )
            label_file = (
                root / "outputs" / "labels"
                / f"{meeting}_frame_labels.npz"
            )

            with np.load(feature_file) as data:
                features = data["features"].astype(np.float32)
                feature_times = data["frame_times"]

            with np.load(label_file) as data:
                labels = data["labels"].astype(np.int64)
                valid_mask = data["valid_mask"].astype(bool)
                label_times = data["frame_times"]

            if features.shape != (len(labels), 80):
                raise ValueError(f"{meeting}: feature shape mismatch")

            if len(valid_mask) != len(labels):
                raise ValueError(f"{meeting}: mask length mismatch")

            if not np.allclose(
                feature_times, label_times, rtol=0, atol=1e-9
            ):
                raise ValueError(f"{meeting}: timestamp mismatch")

            if not np.isfinite(features).all():
                raise ValueError(f"{meeting}: non-finite features")

            recording_index = len(self.recordings)

            self.recordings.append({
                "meeting": meeting,
                "features": features,
                "labels": labels,
                "valid_mask": valid_mask,
            })

            n_frames = len(labels)

            # Training: overlapping sequences are permitted.
            # Evaluation: a stride equal to sequence length gives
            # nonoverlapping sequences, with padding at the end.
            if stride > sequence_length:
                raise ValueError(
                    "Stride must not exceed sequence length"
                )

            starts = list(range(0, n_frames, stride))

            for start in starts:
                self.windows.append((recording_index, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        recording_index, start = self.windows[index]
        recording = self.recordings[recording_index]

        end = start + self.sequence_length

        x = recording["features"][start:end]
        y = recording["labels"][start:end]
        mask = recording["valid_mask"][start:end]

        # Apply the fixed training-set normalization.
        if self.mean is not None:
            x = (x - self.mean) / self.std

        actual_length = len(y)

        if actual_length < self.sequence_length:
            pad = self.sequence_length - actual_length
            x = np.pad(x, ((0, pad), (0, 0)))
            y = np.pad(y, (0, pad), constant_values=-100)
            mask = np.pad(mask, (0, pad), constant_values=False)

        return {
            "features": torch.from_numpy(
                np.ascontiguousarray(x)
            ),
            "labels": torch.from_numpy(
                np.ascontiguousarray(y)
            ),
            "valid_mask": torch.from_numpy(
                np.ascontiguousarray(mask)
            ),
            "meeting": recording["meeting"],
            "start_frame": start,
        }