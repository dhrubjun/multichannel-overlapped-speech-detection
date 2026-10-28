
import torch

from src.training.losses import masked_cross_entropy


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device="cpu",
    max_grad_norm=1.0,
    class_weights=None,
):
    """
    Train for one epoch.

    Supports unweighted and class-weighted cross-entropy.

    Returns:
        Weighted or unweighted average training loss,
        valid frame occurrences,
        number of processed batches.
    """
    model.train()

    if class_weights is not None:
        class_weights = class_weights.to(device)

    total_loss = 0.0
    total_denominator = 0.0
    total_valid = 0
    batches_processed = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        usable = valid_mask & (labels != -100)
        n_valid = int(usable.sum().item())

        if n_valid == 0:
            continue

        optimizer.zero_grad(set_to_none=True)

        logits = model(features)

        loss = masked_cross_entropy(
            logits,
            labels,
            valid_mask,
            class_weights=class_weights,
            reduction="mean",
        )

        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_grad_norm,
            error_if_nonfinite=True,
        )

        optimizer.step()

        # Aggregate epoch loss using the correct denominator.
        if class_weights is None:
            denominator = float(n_valid)
        else:
            denominator = float(
                class_weights[labels[usable]].sum().item()
            )

        total_loss += loss.item() * denominator
        total_denominator += denominator

        total_valid += n_valid
        batches_processed += 1

    if total_valid == 0:
        raise RuntimeError("No valid training frames found")

    return {
        "loss": total_loss / total_denominator,
        "valid_frame_occurrences": total_valid,
        "batches": batches_processed,
    }