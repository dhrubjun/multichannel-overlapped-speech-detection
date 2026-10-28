
import numpy as np
import torch
import torch.nn.functional as F


def binary_average_precision(scores, targets):
    """
    Average precision for one binary class.
    Handles tied scores at the same threshold.
    Returns 0.0 if no positive reference frames exist.
    """
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=bool)

    if scores.ndim != 1 or targets.shape != scores.shape:
        raise ValueError("Scores and targets must be matching 1D arrays")

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite prediction scores")

    positives = int(targets.sum())

    if positives == 0:
        return 0.0

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_targets = targets[order]

    tp = np.cumsum(sorted_targets)
    positions = np.arange(1, len(scores) + 1)

    # Evaluate once at each distinct score threshold.
    last_in_group = np.r_[
        sorted_scores[1:] != sorted_scores[:-1],
        True,
    ]

    precision = tp[last_in_group] / positions[last_in_group]
    recall = tp[last_in_group] / positives

    recall_increments = np.diff(
        np.r_[0.0, recall]
    )

    return float(np.sum(precision * recall_increments))


def evaluate_model(model, dataloader, device="cpu"):
    """
    Evaluate on nonoverlapping sequences.

    Returns global frame-level metrics across all valid
    development frames. No optimizer updates.
    """
    model.eval()

    confusion = np.zeros((3, 3), dtype=np.int64)

    all_overlap_scores = []
    all_overlap_targets = []

    total_loss = 0.0
    total_valid = 0

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["valid_mask"].to(device)

            usable = mask & (labels != -100)
            n_valid = int(usable.sum().item())

            if n_valid == 0:
                continue

            logits = model(features)

            valid_logits = logits[usable]
            valid_labels = labels[usable]

            # Unweighted loss, summed over valid frames.
            loss_sum = F.cross_entropy(
                valid_logits,
                valid_labels,
                reduction="sum",
            )

            total_loss += float(loss_sum.item())
            total_valid += n_valid

            probabilities = torch.softmax(
                valid_logits, dim=-1
            )

            predictions = probabilities.argmax(dim=-1)

            y_true = valid_labels.cpu().numpy()
            y_pred = predictions.cpu().numpy()

            np.add.at(
                confusion,
                (y_true, y_pred),
                1,
            )

            all_overlap_scores.append(
                probabilities[:, 2].cpu().numpy()
            )
            all_overlap_targets.append(
                y_true == 2
            )

    if total_valid == 0:
        raise RuntimeError("No valid evaluation frames")

    per_class_f1 = []

    for cls in range(3):
        tp = int(confusion[cls, cls])
        fp = int(confusion[:, cls].sum()) - tp
        fn = int(confusion[cls, :].sum()) - tp

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        per_class_f1.append(f1)

        if cls == 2:
            overlap_precision = precision
            overlap_recall = recall
            overlap_f1 = f1

    overlap_scores = np.concatenate(all_overlap_scores)
    overlap_targets = np.concatenate(all_overlap_targets)

    overlap_ap = binary_average_precision(
        overlap_scores,
        overlap_targets,
    )

    return {
        "loss": total_loss / total_valid,
        "valid_frames": total_valid,
        "overlap_precision": overlap_precision,
        "overlap_recall": overlap_recall,
        "overlap_f1": overlap_f1,
        "overlap_ap": overlap_ap,
        "macro_f1": float(np.mean(per_class_f1)),
        "confusion_matrix": confusion,
    }