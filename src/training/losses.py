
import torch
import torch.nn.functional as F


def masked_cross_entropy(
    logits,
    labels,
    valid_mask,
    class_weights=None,
    reduction="mean",
):
    """
    Cross-entropy over valid speech-state frames.

    logits:      (batch, time, 3)
    labels:      (batch, time)
    valid_mask:  (batch, time), Boolean
    class_weights: optional tensor of shape (3,)
    reduction: "mean" or "sum"
    """
    if logits.ndim != 3 or logits.shape[-1] != 3:
        raise ValueError("Expected logits of shape (B, T, 3)")

    if labels.shape != logits.shape[:2]:
        raise ValueError("Label shape mismatch")

    if valid_mask.shape != labels.shape:
        raise ValueError("Mask shape mismatch")

    if valid_mask.dtype != torch.bool:
        raise ValueError("Validity mask must be Boolean")

    if reduction not in ("mean", "sum"):
        raise ValueError("Reduction must be 'mean' or 'sum'")

    usable = valid_mask & (labels != -100)

    if not usable.any().item():
        raise ValueError("Batch has no valid labeled frames")

    # Intel XPU compatibility:
    # Avoid Boolean indexing of GPU tensors.
    if logits.device.type == "xpu":
        targets = labels.masked_fill(~usable, -100)

        # Validate class labels on CPU.
        valid_targets = targets.detach().cpu()
        valid_targets = valid_targets[valid_targets != -100]

        if not torch.all(
            (valid_targets >= 0) & (valid_targets <= 2)
        ).item():
            raise ValueError("Invalid class label")

        return F.cross_entropy(
            logits.transpose(1, 2),
            targets,
            weight=class_weights,
            ignore_index=-100,
            reduction=reduction,
        )

    selected_logits = logits[usable]
    selected_labels = labels[usable]

    if not torch.all(
        (selected_labels >= 0) & (selected_labels <= 2)
    ).item():
        raise ValueError("Invalid class label")

    per_frame_loss = F.cross_entropy(
        selected_logits,
        selected_labels,
        weight=class_weights,
        reduction="none",
    )

    if reduction == "sum":
        return per_frame_loss.sum()

    if class_weights is None:
        return per_frame_loss.mean()

    # For weighted loss, match PyTorch's weighted-mean
    # convention: divide by sum of selected class weights.
    denominator = class_weights[selected_labels].sum()

    return per_frame_loss.sum() / denominator