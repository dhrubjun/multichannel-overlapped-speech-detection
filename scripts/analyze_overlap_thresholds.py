
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.speech_dataset import SpeechSequenceDataset
from src.models.tcn import SpeechTCN


CHECKPOINT = (
    "outputs/training/"
    "m1_full_sqrt_weighted_seed42/best_model.pt"
)


def main():
    device = "cpu"

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
        map_location=device,
        weights_only=True,
    )

    model = SpeechTCN().to(device)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    scores = []
    references = []
    argmax_predictions = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["labels"]
            mask = batch["valid_mask"]

            usable = mask & (labels != -100)

            logits = model(features)
            probabilities = torch.softmax(logits, dim=-1)

            scores.append(
                probabilities[..., 2][usable].cpu().numpy()
            )

            references.append(
                (labels[usable] == 2).numpy()
            )

            argmax_predictions.append(
                probabilities.argmax(dim=-1)[usable]
                .cpu().numpy()
            )

    scores = np.concatenate(scores)
    references = np.concatenate(references)
    argmax_predictions = np.concatenate(
        argmax_predictions
    )

    assert len(scores) == 111472
    assert references.sum() == 2218

    print("OVERLAP THRESHOLD ANALYSIS")
    print("=" * 75)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Valid frames:", len(scores))
    print("Actual overlap frames:", references.sum())

    print(
        f"\n{'Threshold':>10} "
        f"{'Precision':>12} "
        f"{'Recall':>12} "
        f"{'F1':>12} "
        f"{'Predicted':>12}"
    )

    results = []

    for threshold in np.arange(0.05, 1.0, 0.05):
        predicted = scores >= threshold

        tp = int(np.sum(predicted & references))
        fp = int(np.sum(predicted & ~references))
        fn = int(np.sum(~predicted & references))

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        results.append(
            (threshold, precision, recall, f1, tp + fp)
        )

        print(
            f"{threshold:10.2f} "
            f"{precision:12.4f} "
            f"{recall:12.4f} "
            f"{f1:12.4f} "
            f"{tp + fp:12d}"
        )

    best = max(results, key=lambda row: row[3])

    print("\nBEST THRESHOLD ON DEVELOPMENT SET")
    print(f"Threshold: {best[0]:.2f}")
    print(f"Precision: {best[1]:.4f}")
    print(f"Recall: {best[2]:.4f}")
    print(f"F1: {best[3]:.4f}")

    predicted = argmax_predictions == 2
    tp = int(np.sum(predicted & references))
    fp = int(np.sum(predicted & ~references))
    fn = int(np.sum(~predicted & references))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    print("\nORIGINAL THREE-CLASS ARGMAX")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")


if __name__ == "__main__":
    main()