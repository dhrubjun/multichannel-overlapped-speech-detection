
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.speech_dataset import SpeechSequenceDataset
from src.models.tcn import SpeechTCN


CHECKPOINT = (
    "outputs/training/"
    "m1_full_sqrt_weighted_seed42/best_model.pt"
)


def remove_short_runs(predicted, min_frames):
    """Remove predicted overlap runs shorter than min_frames."""
    filtered = predicted.copy()

    changes = np.diff(
        np.pad(predicted.astype(np.int8), (1, 1))
    )

    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    for start, end in zip(starts, ends):
        if end - start < min_frames:
            filtered[start:end] = False

    return filtered


def calculate_metrics(predicted, actual):
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return precision, recall, f1, tp, fp


def main():
    with np.load(
        "outputs/labels/ES2003a_frame_labels.npz"
    ) as data:
        labels = data["labels"]
        valid = data["valid_mask"]

    dataset = SpeechSequenceDataset(
        meetings=["ES2003a"],
        sequence_length=200,
        stride=200,
        normalization_stats=(
            "outputs/features/logmel_train_stats.npz"
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model = SpeechTCN()
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    # Preserve original frame positions and time gaps.
    scores = np.full(
        len(labels), np.nan, dtype=np.float32
    )

    with torch.no_grad():
        for batch in loader:
            probabilities = torch.softmax(
                model(batch["features"]), dim=-1
            )[..., 2].numpy()

            for j, start in enumerate(
                batch["start_frame"].tolist()
            ):
                end = min(
                    start + probabilities.shape[1],
                    len(labels),
                )

                scores[start:end] = probabilities[
                    j, :end - start
                ]

    usable = valid & (labels != -100)

    assert usable.sum() == 111472
    assert np.isfinite(scores[usable]).all()

    actual = usable & (labels == 2)

    
    # Inspect the longest completely false overlap regions.
    threshold = 0.75
    predicted = (scores >= threshold) & usable

    changes = np.diff(
        np.pad(predicted.astype(np.int8), (1, 1))
    )

    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    false_runs = []

    for start, end in zip(starts, ends):
        # Only include predicted regions with no true overlap.
        if np.any(actual[start:end]):
            continue

        duration_frames = end - start

        single_frames = int(
            np.sum(labels[start:end] == 1)
        )

        silence_frames = int(
            np.sum(labels[start:end] == 0)
        )

        false_runs.append((
            duration_frames,
            start,
            end,
            single_frames,
            silence_frames,
        ))

    false_runs.sort(reverse=True)

    print("\nLONGEST COMPLETELY FALSE OVERLAP REGIONS")
    print("=" * 75)

    print("Total pure false-positive runs:", len(false_runs))

    for duration, start, end, single, silence in false_runs[:15]:
        print(
            f"Start: {start * 0.01:8.2f}s | "
            f"End: {end * 0.01:8.2f}s | "
            f"Duration: {duration * 10:5d}ms | "
            f"Single: {single:3d} | "
            f"No speech: {silence:3d}"
        )


    print("OVERLAP DURATION FILTER EXPERIMENT")
    print("=" * 80)
    print("Valid frames:", usable.sum())
    print("Actual overlap frames:", actual.sum())
    print()

    print(
        f"{'Threshold':>10} {'Min ms':>8} "
        f"{'Precision':>11} {'Recall':>11} "
        f"{'F1':>9} {'TP':>7} {'FP':>7}"
    )

    for threshold in (0.65, 0.70, 0.75, 0.80):
        predicted = (scores >= threshold) & usable

        for min_frames in (1, 5, 10, 20, 30):
            filtered = remove_short_runs(
                predicted, min_frames
            )

            p, r, f1, tp, fp = calculate_metrics(
                filtered, actual
            )

            print(
                f"{threshold:10.2f} "
                f"{min_frames * 10:8d} "
                f"{p:11.4f} "
                f"{r:11.4f} "
                f"{f1:9.4f} "
                f"{tp:7d} {fp:7d}"
            )

    
    # Analyze false overlap predictions around
    # independent 200-frame sequence boundaries.

    threshold = 0.75

    predicted_overlap = (scores >= threshold) & usable

    # Focus on the dominant error:
    # actual single-speaker -> predicted overlap.
    actual_single = usable & (labels == 1)

    false_overlap = predicted_overlap & actual_single

    # Each evaluation sequence contains 200 frames.
    frame_position = np.arange(len(labels)) % 200

    print("\nTCN SEQUENCE BOUNDARY ERROR ANALYSIS")
    print("=" * 75)

    for edge_frames in (10, 30, 60):

        near_edge = (
            (frame_position < edge_frames)
            | (frame_position >= 200 - edge_frames)
        )

        interior = ~near_edge

        near_single = int(
            np.sum(actual_single & near_edge)
        )

        interior_single = int(
            np.sum(actual_single & interior)
        )

        near_fp = int(
            np.sum(false_overlap & near_edge)
        )

        interior_fp = int(
            np.sum(false_overlap & interior)
        )

        near_rate = near_fp / near_single
        interior_rate = interior_fp / interior_single

        print(
            f"\nWithin {edge_frames * 10} ms "
            f"of sequence boundaries:"
        )

        print(
            f"  Single frames near boundary: {near_single}"
        )
        print(
            f"  False overlap near boundary: {near_fp}"
        )
        print(
            f"  False-positive rate near: {near_rate:.4f}"
        )

        print(
            f"  Single frames in interior: {interior_single}"
        )
        print(
            f"  False overlap in interior: {interior_fp}"
        )
        print(
            f"  False-positive rate interior: {interior_rate:.4f}"
        )

        print(
            f"  Near/interior rate ratio: "
            f"{near_rate / interior_rate:.3f}"
            if interior_rate > 0
            else "  Near/interior rate ratio: undefined"
        )



if __name__ == "__main__":
    main()