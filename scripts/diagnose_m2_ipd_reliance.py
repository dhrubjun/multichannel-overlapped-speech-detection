
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.m2_dataset import M2SequenceDataset
from src.models.tcn import SpeechTCN
from src.training.evaluation import binary_average_precision


CHECKPOINT = (
    "outputs/training/m2_weighted_v1/best_model.pt"
)

torch.manual_seed(42)

dataset = M2SequenceDataset(
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
    weights_only=False,
)

model = SpeechTCN(input_features=2136)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def evaluate(condition):
    scores = []
    targets = []
    predictions = []
    original = []
    changed_probability = []

    rng = np.random.default_rng(42)

    with torch.inference_mode():
        for batch in loader:
            x = batch["features"].clone()
            y = batch["labels"]
            valid = batch["valid_mask"] & (y != -100)

            if condition == "zero_ipd":
                x[:, :, 80:] = 0.0

            elif condition == "shuffle_ipd":
                # Shuffle valid IPD frames within each
                # sequence while preserving log-Mel.
                # Do not move padded or invalid frames.
                for b in range(x.shape[0]):
                    positions = torch.where(valid[b])[0]

                    if len(positions) > 1:
                        permutation = rng.permutation(
                            len(positions)
                        )

                        original_ipd = x[
                            b, positions, 80:
                        ].clone()

                        x[b, positions, 80:] = (
                            original_ipd[permutation]
                        )

            elif condition != "original":
                raise ValueError(condition)

            logits = model(x)
            probabilities = torch.softmax(logits, dim=-1)

            score = probabilities[:, :, 2]
            pred = probabilities.argmax(dim=-1)

            scores.append(
                score[valid].cpu().numpy()
            )
            targets.append(
                (y[valid] == 2).cpu().numpy()
            )
            predictions.append(
                pred[valid].cpu().numpy()
            )

    scores = np.concatenate(scores)
    targets = np.concatenate(targets)
    predictions = np.concatenate(predictions)

    ap = binary_average_precision(scores, targets)

    positive = predictions == 2

    tp = int(np.sum(positive & targets))
    fp = int(np.sum(positive & ~targets))
    fn = int(np.sum(~positive & targets))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "ap": ap,
        "f1": f1,
        "mean_overlap_score": float(scores[targets].mean()),
        "mean_nonoverlap_score": float(scores[~targets].mean()),
        "scores": scores,
        "predictions": predictions,
    }


print("\nM2: SPATIAL FEATURE RELIANCE DIAGNOSTIC")
print("=" * 70)

results = {}

for condition in ("original", "zero_ipd", "shuffle_ipd"):
    result = evaluate(condition)
    results[condition] = result

    print(f"\nCondition: {condition}")
    print(f"Overlap AP: {result['ap']:.6f}")
    print(f"Argmax overlap F1: {result['f1']:.6f}")
    print(
        "Mean overlap score, true overlap:",
        f"{result['mean_overlap_score']:.6f}",
    )
    print(
        "Mean overlap score, non-overlap:",
        f"{result['mean_nonoverlap_score']:.6f}",
    )

# Confirm that original inference reproduces M2-W.
np.testing.assert_allclose(
    results["original"]["ap"],
    0.13034696126549508,
    rtol=0,
    atol=1e-6,
)

baseline = results["original"]

print("\nSENSITIVITY RELATIVE TO ORIGINAL")
print("=" * 70)

for condition in ("zero_ipd", "shuffle_ipd"):
    result = results[condition]

    changed = np.mean(
        baseline["predictions"] != result["predictions"]
    ) * 100

    mean_abs_change = np.mean(
        np.abs(
            baseline["scores"] - result["scores"]
        )
    )

    print(f"\n{condition}")
    print(f"Change in overlap AP: {result['ap'] - baseline['ap']:+.6f}")
    print(f"Changed argmax predictions: {changed:.2f}%")
    print(f"Mean absolute score change: {mean_abs_change:.6f}")

print("\nIPD RELIANCE DIAGNOSTIC COMPLETED")