
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.speech_dataset import SpeechSequenceDataset
from src.models.tcn import SpeechTCN
from src.training.engine import train_one_epoch
from src.training.evaluation import evaluate_model
from src.training.early_stopping import EarlyStopping

from src.training.class_weights import (
    count_training_classes,
    compute_class_weights,
)


TRAIN_MEETINGS = ("ES2008a", "ES2008b", "ES2008c")
DEV_MEETINGS = ("ES2003a",)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(args):
    """Create training and development DataLoaders."""

    stats_path = "outputs/features/logmel_train_stats.npz"

    train_dataset = SpeechSequenceDataset(
        meetings=list(TRAIN_MEETINGS),
        sequence_length=200,
        stride=100,
        normalization_stats=stats_path,
    )

    dev_dataset = SpeechSequenceDataset(
        meetings=list(DEV_MEETINGS),
        sequence_length=200,
        stride=200,
        normalization_stats=stats_path,
    )

    # Small datasets for pipeline testing only.
    if args.smoke_test:
        train_dataset = Subset(
            train_dataset,
            range(min(32, len(train_dataset))),
        )

        dev_dataset = Subset(
            dev_dataset,
            range(min(64, len(dev_dataset))),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, dev_loader


def main():

    parser = argparse.ArgumentParser(
        description="Train M1 log-Mel + TCN baseline"
    )

    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    # Early stopping configuration
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Stop after this many epochs without AP improvement",
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum development AP improvement",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/training/m1_logmel_tcn",
    )

    parser.add_argument(
        "--class-weighting",
        choices=["none", "sqrt_inverse"],
        default="none",
        help="Training loss class-weighting strategy",
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Validate arguments
    # --------------------------------------------------

    if args.epochs < 1:
        parser.error("Epochs must be positive")

    if args.batch_size < 1:
        parser.error("Batch size must be positive")

    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("Learning rate must be positive and finite")

    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Weight decay must be nonnegative and finite")

    if args.patience < 1:
        parser.error("Patience must be at least 1")

    if not np.isfinite(args.min_delta) or args.min_delta < 0:
        parser.error("min_delta must be nonnegative and finite")

    # --------------------------------------------------
    # Device selection
    # --------------------------------------------------

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA was requested but is unavailable"
        )

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------

    set_seed(args.seed)

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    output_dir = Path(args.output_dir)

    # Prevent accidental overwriting of a saved run.
    existing_files = (
        "config.json",
        "history.json",
        "best_model.pt",
        "summary.json",
    )

    if any(
        (output_dir / name).exists()
        for name in existing_files
    ):
        parser.error(
            f"Output directory already contains experiment files: "
            f"{output_dir}. Use a new output directory."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_loader, dev_loader = create_dataloaders(args)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = SpeechTCN().to(device)

    n_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


    class_weights = None
    class_counts = count_training_classes(
        TRAIN_MEETINGS
    )

    if args.class_weighting != "none":
        class_weights = compute_class_weights(
            class_counts,
            strategy=args.class_weighting,
        ).to(device)

    print("\nTRAINING CLASS WEIGHTS")

    if class_weights is None:
        print("Unweighted cross-entropy")
    else:
        for cls, weight in enumerate(class_weights):
            print(f"Class {cls}: {weight.item():.6f}")

    # --------------------------------------------------
    # Early stopping
    # --------------------------------------------------

    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
    )

    # --------------------------------------------------
    # Experiment configuration
    # --------------------------------------------------

    config = {
        "model": "SpeechTCN",
        "features": "logmel_mic01",
        "train_meetings": list(TRAIN_MEETINGS),
        "dev_meetings": list(DEV_MEETINGS),
        "sequence_length": 200,
        "train_stride": 100,
        "dev_stride": 200,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "smoke_test": args.smoke_test,
        "checkpoint_metric": "overlap_ap",
        "class_weighting": args.class_weighting,
        "class_counts": class_counts.tolist(),
        "class_weights": (
            class_weights.cpu().tolist()
            if class_weights is not None
            else None
        ),
        "patience": args.patience,
        "min_delta": args.min_delta,
        "parameters": n_parameters,
    }

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=2)

    # --------------------------------------------------
    # Training information
    # --------------------------------------------------

    print("=" * 65, flush=True)
    print("M1: LOG-MEL + TCN TRAINING", flush=True)
    print("=" * 65, flush=True)

    print(f"Device: {device}", flush=True)
    print(f"Smoke test: {args.smoke_test}", flush=True)
    print(f"Train sequences: {len(train_loader.dataset)}", flush=True)
    print(f"Dev sequences: {len(dev_loader.dataset)}", flush=True)
    print(f"Parameters: {n_parameters:,}", flush=True)
    print(f"Maximum epochs: {args.epochs}", flush=True)
    print(f"Early stopping patience: {args.patience}", flush=True)
    print(f"Minimum AP improvement: {args.min_delta}", flush=True)

    history = []

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    for epoch in range(1, args.epochs + 1):

        # Train one epoch
        train_results = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            max_grad_norm=1.0,
            class_weights=class_weights,
        )

        # Evaluate on development data
        dev_results = evaluate_model(
            model=model,
            dataloader=dev_loader,
            device=device,
        )

        # Store metrics
        record = {
            "epoch": epoch,
            "train_loss": train_results["loss"],
            "train_valid_frame_occurrences":
                train_results["valid_frame_occurrences"],
            "dev_loss": dev_results["loss"],
            "dev_valid_frames": dev_results["valid_frames"],
            "overlap_precision": dev_results["overlap_precision"],
            "overlap_recall": dev_results["overlap_recall"],
            "overlap_f1": dev_results["overlap_f1"],
            "overlap_ap": dev_results["overlap_ap"],
            "macro_f1": dev_results["macro_f1"],
            "confusion_matrix":
                dev_results["confusion_matrix"].tolist(),
        }

        history.append(record)

        # Print epoch results
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train loss: {record['train_loss']:.4f} | "
            f"Dev loss: {record['dev_loss']:.4f} | "
            f"Overlap P: {record['overlap_precision']:.4f} | "
            f"Overlap R: {record['overlap_recall']:.4f} | "
            f"Overlap F1: {record['overlap_f1']:.4f} | "
            f"Overlap AP: {record['overlap_ap']:.4f} | "
            f"Macro-F1: {record['macro_f1']:.4f}",
            flush=True,
        )

        # ----------------------------------------------
        # Early stopping and checkpoint selection
        # ----------------------------------------------

        improved, should_stop = early_stopping.update(
            record["overlap_ap"],
            epoch,
        )

        if improved:

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "config": config,
                    "dev_metrics": record,
                },
                output_dir / "best_model.pt",
            )

            print(
                f"  New best checkpoint saved "
                f"(development overlap AP: "
                f"{early_stopping.best_score:.6f})",
                flush=True,
            )

        # ----------------------------------------------
        # Save history after each completed epoch
        # ----------------------------------------------

        with open(
            output_dir / "history.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(history, f, indent=2)

        # ----------------------------------------------
        # Stop if no improvement for patience epochs
        # ----------------------------------------------

        if should_stop:
            print(
                f"\nEarly stopping at epoch {epoch}. "
                f"Best epoch: {early_stopping.best_epoch}",
                flush=True,
            )
            break

    # --------------------------------------------------
    # Final experiment summary
    # --------------------------------------------------

    summary = {
        "completed_at":
            datetime.now(timezone.utc).isoformat(),
        "epochs_completed": len(history),
        "best_epoch": early_stopping.best_epoch,
        "best_overlap_ap": early_stopping.best_score,
        "early_stopped": len(history) < args.epochs,
        "smoke_test": args.smoke_test,
    }

    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------

    print("\nTRAINING COMPLETED", flush=True)
    print(
        f"Epochs completed: {len(history)}",
        flush=True,
    )
    print(
        f"Best epoch: {early_stopping.best_epoch}",
        flush=True,
    )
    print(
        f"Best development overlap AP: "
        f"{early_stopping.best_score:.6f}",
        flush=True,
    )
    print(
        f"Early stopped: {summary['early_stopped']}",
        flush=True,
    )
    print(
        f"Results: {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()