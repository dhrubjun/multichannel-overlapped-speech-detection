"""Memory-mapped M1 dataset for the official-group AMI ES/IS split.

Keep this alongside the legacy speech_dataset.py, without changing it.
"""
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class M1OfficialDataset(Dataset):
    def __init__(
        self,
        meetings,
        project_root=".",
        sequence_length=200,
        stride=100,
        normalization_stats=None,
    ):
        if not meetings:
            raise ValueError("At least one meeting is required")
        if sequence_length <= 0 or not (0 < stride <= sequence_length):
            raise ValueError("Expected 0 < stride <= sequence_length")
        if len(set(meetings)) != len(meetings):
            raise ValueError("Duplicate meeting IDs")

        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        root = Path(project_root)
        self.recordings = []
        # Cumulative per-meeting window counts, avoids one tuple per window.
        self.cumulative_windows = [0]

        self.mean = self.std = None
        if normalization_stats is not None:
            stats_path = Path(normalization_stats)
            if not stats_path.is_absolute():
                stats_path = root / stats_path
            with np.load(stats_path, allow_pickle=False) as stats:
                self.mean = stats["mean"].astype(np.float32)
                self.std = stats["std"].astype(np.float32)
                if "train_meetings" in stats.files:
                    stat_train = set(stats["train_meetings"].tolist())
                    if not stat_train:
                        raise ValueError("Normalization train_meetings is empty")
            if self.mean.shape != (80,) or self.std.shape != (80,):
                raise ValueError("Expected 80 normalization values")
            if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
                raise ValueError("Nonfinite normalization values")
            if np.any(self.std <= 1e-8):
                raise ValueError("Invalid normalization standard deviations")

        for meeting in meetings:
            feature_path = (root / "outputs" / "features_m1_official_npy"
                            / f"{meeting}_logmel_mic01.npy")
            label_path = (root / "outputs" / "labels"
                          / f"{meeting}_frame_labels.npz")
            features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
            if not isinstance(features, np.memmap):
                raise ValueError(f"{meeting}: feature array is not memory-mapped")
            if features.ndim != 2 or features.shape[1] != 80 or features.dtype != np.float32:
                raise ValueError(f"{meeting}: expected float32 [frames, 80]")
            with np.load(label_path, allow_pickle=False) as label_file:
                labels = label_file["labels"].astype(np.int64)
                valid_mask = label_file["valid_mask"].astype(np.bool_)
                times = label_file["frame_times"]
                hop = float(label_file["hop_seconds"])
            n_frames = features.shape[0]
            if not (n_frames == len(labels) == len(valid_mask) == len(times)):
                raise ValueError(f"{meeting}: feature/label frame-count mismatch")
            if not np.isclose(hop, 0.01, atol=1e-10, rtol=0):
                raise ValueError(f"{meeting}: incorrect hop")
            if not np.array_equal(np.unique(labels), np.intersect1d(np.unique(labels), [0, 1, 2])):
                raise ValueError(f"{meeting}: labels outside {{0,1,2}}")
            if not np.allclose(times, (np.arange(n_frames) + 0.5) * 0.01,
                               atol=1e-9, rtol=0):
                raise ValueError(f"{meeting}: label timestamps mismatch")
            if not valid_mask.any():
                raise ValueError(f"{meeting}: no valid frames")
            # Explicit final tail window, padded with invalid labels.
            n_windows = (n_frames - 1) // self.stride + 1
            self.recordings.append({
                "meeting": meeting, "features": features,
                "labels": labels, "valid_mask": valid_mask,
                "frames": n_frames,
            })
            self.cumulative_windows.append(self.cumulative_windows[-1] + n_windows)

    def __len__(self):
        return self.cumulative_windows[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        recording_idx = bisect_right(self.cumulative_windows, index) - 1
        recording = self.recordings[recording_idx]
        local_idx = index - self.cumulative_windows[recording_idx]
        start = local_idx * self.stride
        end = min(start + self.sequence_length, recording["frames"])

        # Explicit copy: memmap slices are read-only and normalization is in-place.
        x = np.array(recording["features"][start:end], dtype=np.float32, copy=True)
        y = recording["labels"][start:end].copy()
        mask = recording["valid_mask"][start:end].copy()

        if self.mean is not None:
            x -= self.mean
            x /= self.std

        n = end - start
        if n < self.sequence_length:
            pad = self.sequence_length - n
            x = np.pad(x, ((0, pad), (0, 0)), mode="constant")
            y = np.pad(y, (0, pad), constant_values=-100)
            mask = np.pad(mask, (0, pad), constant_values=False)

        return {
            "features": torch.from_numpy(np.ascontiguousarray(x)),
            "labels": torch.from_numpy(np.ascontiguousarray(y)),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(mask)),
            "meeting": recording["meeting"],
            "start_frame": start,
        }
