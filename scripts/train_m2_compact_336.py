"""Train compact M2 using official AMI ES/IS 72/12 split."""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from src.models.tcn import SpeechTCN
from src.training.engine import train_one_epoch
from src.training.evaluation import evaluate_model
from src.training.early_stopping import EarlyStopping

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/dataset_audit/official_es_is"
FEATURES = ROOT / "outputs/features/m2_compact_336"
LABELS = ROOT / "outputs/labels"
STATS = AUDIT / "m1_logmel_train_stats.npz"


def read_split(name):
    path = AUDIT / f"{name}_meetings.txt"
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class CompactM2Dataset(Dataset):
    def __init__(self, meetings, stride):
        self.recordings = []
        self.windows = []
        self.sequence_length = 200

        with np.load(STATS, allow_pickle=False) as stats:
            self.mean = stats["mean"].astype(np.float32)
            self.std = stats["std"].astype(np.float32)

        if self.mean.shape != (80,) or self.std.shape != (80,):
            raise ValueError("Invalid Log-Mel normalization shapes")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("Non-finite normalization statistics")
        if np.any(self.std <= 0):
            raise ValueError("Invalid normalization standard deviations")

        for meeting in meetings:
            feature_path = FEATURES / f"{meeting}_m2_336.npy"
            label_path = LABELS / f"{meeting}_frame_labels.npz"

            x = np.load(
                feature_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            with np.load(label_path, allow_pickle=False) as data:
                labels = data["labels"].astype(np.int64)
                mask = data["valid_mask"].astype(bool)
                times = data["frame_times"]

            if x.ndim != 2 or x.shape[1] != 336:
                raise ValueError(f"{meeting}: invalid feature dimensions")

            if x.dtype != np.float32:
                raise ValueError(f"{meeting}: expected float32")

            if len(x) != len(labels) or len(x) != len(mask):
                raise ValueError(f"{meeting}: label alignment mismatch")

            if len(times) != len(x):
                raise ValueError(f"{meeting}: timestamp length mismatch")

            expected = (
                np.arange(len(x), dtype=np.float64) + 0.5
            ) * 0.01

            if not np.allclose(times, expected, rtol=0, atol=1e-9):
                raise ValueError(f"{meeting}: timestamp mismatch")

            index = len(self.recordings)
            self.recordings.append((meeting, x, labels, mask))

            for start in range(0, len(x), stride):
                self.windows.append((index, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        recording_index, start = self.windows[index]

        meeting, features, labels, mask = self.recordings[
            recording_index
        ]

        end = start + self.sequence_length

        x = np.array(
            features[start:end],
            dtype=np.float32,
            copy=True,
        )

        y = labels[start:end].copy()
        valid = mask[start:end].copy()

        x[:, :80] = (x[:, :80] - self.mean) / self.std

        padding = self.sequence_length - len(x)

        if padding:
            x = np.pad(x, ((0, padding), (0, 0)))
            y = np.pad(y, (0, padding), constant_values=-100)
            valid = np.pad(valid, (0, padding), constant_values=False)

        return {
            "features": torch.from_numpy(np.ascontiguousarray(x)),
            "labels": torch.from_numpy(np.ascontiguousarray(y)),
            "valid_mask": torch.from_numpy(
                np.ascontiguousarray(valid)
            ),
            "meeting": meeting,
            "start_frame": start,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if min(
        args.epochs, args.batch_size,
        args.patience, args.cpu_threads
    ) < 1:
        parser.error("Training settings must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)

    train_meetings = read_split("train")
    dev_meetings = read_split("development")
    test_meetings = read_split("test")

    if (
        len(train_meetings),
        len(dev_meetings),
        len(test_meetings)
    ) != (72, 12, 12):
        raise RuntimeError("Unexpected official split sizes")

    if len(set(train_meetings + dev_meetings + test_meetings)) != 96:
        raise RuntimeError("Duplicate meetings across partitions")

    with np.load(STATS, allow_pickle=False) as stats:
        if set(stats["train_meetings"].tolist()) != set(train_meetings):
            raise RuntimeError("Normalization provenance mismatch")
        if int(stats["count"]) != 12_147_428:
            raise RuntimeError("Unexpected training normalization count")

    print("Loading official training and development datasets...")

    train_ds = CompactM2Dataset(train_meetings, stride=100)
    dev_ds = CompactM2Dataset(dev_meetings, stride=200)

    full_train_windows = len(train_ds)
    full_dev_windows = len(dev_ds)

    if args.smoke_test:
        train_ds = Subset(train_ds, range(min(16, len(train_ds))))
        dev_ds = Subset(dev_ds, range(min(16, len(dev_ds))))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = SpeechTCN(input_features=336)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )

    parameters = sum(p.numel() for p in model.parameters())

    if args.output_dir is None:
        mode = "smoke" if args.smoke_test else "full"
        output = (
            ROOT / "outputs/training/m2_compact_336"
            / f"{mode}_seed{args.seed}"
        )
    else:
        output = args.output_dir.resolve()

    output.mkdir(parents=True, exist_ok=True)

    protected = (
        "config.json", "history.json",
        "best_model.pt", "summary.json",
    )

    if any((output / name).exists() for name in protected):
        raise FileExistsError(
            f"Experiment output already exists: {output}"
        )

    epochs = 1 if args.smoke_test else args.epochs

    config = {
        "model": "SpeechTCN",
        "features": "80_logmel_256_ipd",
        "input_features": 336,
        "parameters": parameters,
        "train_meetings": train_meetings,
        "dev_meetings": dev_meetings,
        "test_meetings_used": False,
        "normalization_stats": str(STATS),
        "normalization_train_frames": 12_147_428,
        "train_windows": full_train_windows,
        "dev_windows": full_dev_windows,
        "sequence_length": 200,
        "train_stride": 100,
        "dev_stride": 200,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": args.batch_size,
        "epochs": epochs,
        "patience": args.patience,
        "class_weighting": "none",
        "checkpoint_metric": "pooled_development_overlap_ap",
        "seed": args.seed,
        "device": "cpu",
        "torch_version": torch.__version__,
        "smoke_test": args.smoke_test,
    }

    (output / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    stopping = EarlyStopping(patience=args.patience)
    history = []

    print("\nM2 COMPACT OFFICIAL DATASET TRAINING", flush=True)
    print("=" * 65, flush=True)
    print("Device: CPU", flush=True)
    print("Input dimensions: 336", flush=True)
    print("Parameters:", parameters, flush=True)
    print("Training meetings:", len(train_meetings), flush=True)
    print("Development meetings:", len(dev_meetings), flush=True)
    print("Training sequences:", len(train_ds), flush=True)
    print("Development sequences:", len(dev_ds), flush=True)
    print("Maximum epochs:", epochs, flush=True)
    print("Class weighting: none", flush=True)

    for epoch in range(1, epochs + 1):
        train_result = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device="cpu",
            max_grad_norm=1.0,
            class_weights=None,
        )

        dev_result = evaluate_model(
            model=model,
            dataloader=dev_loader,
            device="cpu",
        )

        record = {
            "epoch": epoch,
            "train_loss": float(train_result["loss"]),
            "dev_loss": float(dev_result["loss"]),
            "dev_valid_frames": int(dev_result["valid_frames"]),
            "overlap_ap": float(dev_result["overlap_ap"]),
            "overlap_f1": float(dev_result["overlap_f1"]),
            "overlap_precision": float(
                dev_result["overlap_precision"]
            ),
            "overlap_recall": float(dev_result["overlap_recall"]),
            "macro_f1": float(dev_result["macro_f1"]),
            "confusion_matrix":
                dev_result["confusion_matrix"].tolist(),
        }

        history.append(record)

        (output / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        improved, should_stop = stopping.update(
            record["overlap_ap"], epoch
        )

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train loss: {record['train_loss']:.4f} | "
            f"Dev loss: {record['dev_loss']:.4f} | "
            f"Overlap AP: {record['overlap_ap']:.4f} | "
            f"Overlap F1: {record['overlap_f1']:.4f} | "
            f"Macro-F1: {record['macro_f1']:.4f}",
            flush=True,
        )

        if improved:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "dev_metrics": record,
            }, output / "best_model.pt")

            print("  New best checkpoint saved", flush=True)

        if should_stop:
            print(
                f"Early stopping at epoch {epoch}",
                flush=True,
            )
            break

    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "smoke_test": args.smoke_test,
        "epochs_completed": len(history),
        "best_epoch": stopping.best_epoch,
        "best_overlap_ap": stopping.best_score,
        "early_stopped": len(history) < epochs,
    }

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nTRAINING COMPLETED")
    print("Best epoch:", stopping.best_epoch)
    print("Best development AP:", stopping.best_score)
    print("Results:", output)


if __name__ == "__main__":
    main()
