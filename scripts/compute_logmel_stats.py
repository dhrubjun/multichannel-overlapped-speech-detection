
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_MEETINGS = (
    "ES2008a",
    "ES2008b",
    "ES2008c",
)

N_MELS = 80


def main():
    feature_dir = PROJECT_ROOT / "outputs" / "features"
    label_dir = PROJECT_ROOT / "outputs" / "labels"

    total_count = 0
    total_sum = np.zeros(N_MELS, dtype=np.float64)
    total_sum_sq = np.zeros(N_MELS, dtype=np.float64)

    print("TRAIN-ONLY LOG-MEL NORMALIZATION")
    print("=" * 50)

    for meeting in TRAIN_MEETINGS:
        feature_path = (
            feature_dir / f"{meeting}_logmel_mic01.npz"
        )
        label_path = (
            label_dir / f"{meeting}_frame_labels.npz"
        )

        with np.load(feature_path) as data:
            features = data["features"]
            feature_times = data["frame_times"]

        with np.load(label_path) as data:
            mask = data["valid_mask"]
            label_times = data["frame_times"]

        if features.ndim != 2 or features.shape[1] != N_MELS:
            raise ValueError(f"{meeting}: invalid feature shape")

        if mask.dtype != np.bool_ or len(mask) != len(features):
            raise ValueError(f"{meeting}: invalid mask")

        if not np.allclose(
            feature_times, label_times, rtol=0, atol=1e-9
        ):
            raise ValueError(f"{meeting}: timestamp mismatch")

        if not np.isfinite(features).all():
            raise ValueError(f"{meeting}: non-finite features")

        x = features[mask].astype(np.float64)

        if len(x) == 0:
            raise ValueError(f"{meeting}: no valid frames")

        total_count += len(x)
        total_sum += x.sum(axis=0)
        total_sum_sq += np.square(x).sum(axis=0)

        print(f"{meeting}: {len(x):,} valid frames")

    mean = total_sum / total_count
    variance = np.maximum(
        total_sum_sq / total_count - mean**2, 0.0
    )
    std = np.sqrt(variance)

    if not np.isfinite(mean).all():
        raise ValueError("Invalid means")

    if not np.isfinite(std).all() or np.any(std <= 1e-8):
        raise ValueError("Invalid standard deviations")

    output_path = feature_dir / "logmel_train_stats.npz"

    np.savez_compressed(
        output_path,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        count=np.int64(total_count),
        train_meetings=np.array(TRAIN_MEETINGS),
    )

    print(f"\nTotal valid frames: {total_count:,}")
    print(f"Mean range: {mean.min():.6f}, {mean.max():.6f}")
    print(f"Std range: {std.min():.6f}, {std.max():.6f}")
    print(f"Saved: {output_path}")

    if total_count != 525629:
        raise ValueError(
            f"Unexpected training frame count: {total_count}"
        )

    print("TRAINING STATISTICS VALIDATION PASSED")


if __name__ == "__main__":
    main()