
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class M2SequenceDataset(Dataset):
    """
    Memory-mapped M2 log-Mel + IPD sequence dataset.

    M2 features:
        Columns 0:80    -> log-Mel
        Columns 80:2136 -> masked IPD

    Only the log-Mel columns are standardized.
    """

    def __init__(
        self,
        meetings,
        project_root=".",
        sequence_length=200,
        stride=100,
        normalization_stats=None,
    ):
        if sequence_length <= 0:
            raise ValueError("Sequence length must be positive")

        if stride <= 0 or stride > sequence_length:
            raise ValueError("Invalid sequence stride")

        self.sequence_length = sequence_length
        self.stride = stride
        self.recordings = []
        self.windows = []

        root = Path(project_root)

        self.mean = None
        self.std = None

        if normalization_stats is not None:
            stats_path = Path(normalization_stats)

            if not stats_path.is_absolute():
                stats_path = root / stats_path

            with np.load(stats_path) as stats:
                self.mean = stats["mean"].astype(np.float32)
                self.std = stats["std"].astype(np.float32)

            if self.mean.shape != (80,):
                raise ValueError("Expected 80 normalization means")

            if self.std.shape != (80,):
                raise ValueError("Expected 80 normalization stds")

            if not np.isfinite(self.mean).all():
                raise ValueError("Non-finite normalization means")

            if not np.isfinite(self.std).all():
                raise ValueError("Non-finite normalization stds")

            if np.any(self.std <= 1e-8):
                raise ValueError("Invalid normalization stds")

        for meeting in meetings:
            feature_path = (
                root / "outputs" / "features"
                / f"{meeting}_m2_combined.npy"
            )

            label_path = (
                root / "outputs" / "labels"
                / f"{meeting}_frame_labels.npz"
            )

            # Memory mapping avoids loading the full
            # feature matrix into RAM.
            features = np.load(
                feature_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            with np.load(label_path) as data:
                labels = data["labels"].astype(np.int64)
                valid_mask = data["valid_mask"].astype(bool)
                frame_times = data["frame_times"]

            if features.ndim != 2 or features.shape[1] != 2136:
                raise ValueError(
                    f"{meeting}: expected 2136 features"
                )

            n_frames = features.shape[0]

            if not (
                len(labels)
                == len(valid_mask)
                == len(frame_times)
                == n_frames
            ):
                raise ValueError(
                    f"{meeting}: frame-count mismatch"
                )

            if features.dtype != np.float32:
                raise ValueError(
                    f"{meeting}: expected float32 features"
                )

            if not np.all(
                (np.diff(frame_times) > 0)
            ):
                raise ValueError(
                    f"{meeting}: invalid frame timestamps"
                )

            expected_times = (
                np.arange(n_frames, dtype=np.float64) + 0.5
            ) * 0.01

            if not np.allclose(
                frame_times,
                expected_times,
                rtol=0,
                atol=1e-9,
            ):
                raise ValueError(
                    f"{meeting}: timestamp mismatch"
                )

            recording_index = len(self.recordings)

            self.recordings.append({
                "meeting": meeting,
                "features": features,
                "labels": labels,
                "valid_mask": valid_mask,
            })

            # Match M1 sequence-generation behaviour.
            for start in range(0, n_frames, stride):
                self.windows.append(
                    (recording_index, start)
                )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        recording_index, start = self.windows[index]

        recording = self.recordings[recording_index]

        end = start + self.sequence_length

        # Read only the required sequence from disk.
        # Copy to a writable array for normalization.
        x = np.array(
            recording["features"][start:end],
            dtype=np.float32,
            copy=True,
        )

        y = recording["labels"][start:end].copy()
        mask = recording["valid_mask"][start:end].copy()

        # Normalize only the 80 spectral features.
        # Masked IPD zeros are preserved exactly.
        if self.mean is not None:
            x[:, :80] = (
                x[:, :80] - self.mean
            ) / self.std

        actual_length = len(y)

        # Pad the final sequence when necessary.
        if actual_length < self.sequence_length:
            pad = self.sequence_length - actual_length

            x = np.pad(
                x,
                ((0, pad), (0, 0)),
                mode="constant",
            )

            y = np.pad(
                y,
                (0, pad),
                constant_values=-100,
            )

            mask = np.pad(
                mask,
                (0, pad),
                constant_values=False,
            )

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