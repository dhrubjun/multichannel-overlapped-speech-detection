
from pathlib import Path

import numpy as np
import torch


def count_training_classes(
    meetings,
    labels_dir="outputs/labels",
    num_classes=3,
):
    """
    Count unique valid frames from original AMI labels.

    Returns counts of shape (num_classes,).
    """
    if not meetings:
        raise ValueError("At least one training meeting is required")

    counts = np.zeros(num_classes, dtype=np.int64)

    for meeting in meetings:
        path = Path(labels_dir) / f"{meeting}_frame_labels.npz"

        if not path.is_file():
            raise FileNotFoundError(path)

        with np.load(path) as data:
            labels = data["labels"]
            valid_mask = data["valid_mask"]

        if labels.shape != valid_mask.shape:
            raise ValueError(f"Shape mismatch in {meeting}")

        if valid_mask.dtype != np.bool_:
            raise ValueError(f"Invalid mask dtype in {meeting}")

        usable_labels = labels[
            valid_mask & (labels != -100)
        ]

        if np.any(
            (usable_labels < 0) | (usable_labels >= num_classes)
        ):
            raise ValueError(f"Invalid labels in {meeting}")

        counts += np.bincount(
            usable_labels.astype(np.int64),
            minlength=num_classes,
        )

    return counts


def compute_class_weights(
    counts,
    strategy="sqrt_inverse",
):
    """
    Calculate class weights from training class counts.

    Weights are normalized relative to Class 0.

    Supported strategies:
        none
        sqrt_inverse
        inverse
    """
    counts = np.asarray(counts, dtype=np.float64)

    if counts.ndim != 1 or len(counts) != 3:
        raise ValueError("Expected counts for exactly 3 classes")

    if not np.all(np.isfinite(counts)) or np.any(counts <= 0):
        raise ValueError("All classes must have positive counts")

    if strategy == "none":
        weights = np.ones(3, dtype=np.float64)

    elif strategy == "sqrt_inverse":
        weights = np.sqrt(counts[0] / counts)

    elif strategy == "inverse":
        weights = counts[0] / counts

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return torch.tensor(weights, dtype=torch.float32)