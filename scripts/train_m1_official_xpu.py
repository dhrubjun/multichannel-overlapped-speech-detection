"""Train the original SpeechTCN on the 72/12 official AMI train/dev split.

From repository root, with .venv activated:
    python -m scripts.train_m1_official --smoke-test
    python -m scripts.train_m1_official --seed 42 --device auto

This reuses src/models/tcn.py, src/training/{engine,evaluation,early_stopping}.py,
and src/data/m1_official_dataset.py. Test meetings are never loaded.
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.m1_official_dataset import M1OfficialDataset
from src.models.tcn import SpeechTCN
from src.training.early_stopping import EarlyStopping
from src.training.engine import train_one_epoch
from src.training.evaluation import evaluate_model

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs" / "dataset_audit" / "official_es_is"
EXPECTED_TRAIN = 72
EXPECTED_DEV = 12
EXPECTED_TEST = 12
EXPECTED_VALID_TRAIN = 12_147_428
EXPECTED_VALID_DEV = 1_920_947


def read_meetings(split):
    path = AUDIT / f"{split}_meetings.txt"
    ids = [s.strip() for s in path.read_text(encoding="utf-8").splitlines()
           if s.strip() and not s.lstrip().startswith("#")]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return ids


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="M1 SpeechTCN, official 72/12 AMI split")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Train on 2 batches and evaluate on 2 batches for 1 epoch")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size and patience must be positive")
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be positive and finite")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be nonnegative and finite")

    cuda_available = torch.cuda.is_available()
    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
    if args.device == "cuda" and not cuda_available:
        parser.error("CUDA requested but unavailable")
    if args.device == "xpu" and not xpu_available:
        parser.error("XPU requested but unavailable")
    chosen = ("cuda" if cuda_available else "xpu" if xpu_available else "cpu") if args.device == "auto" else args.device
    device = torch.device(chosen)
    set_seed(args.seed)

    splits = {name: read_meetings(name) for name in ("train", "development", "test")}
    if [len(splits[x]) for x in ("train", "development", "test")] != [
        EXPECTED_TRAIN, EXPECTED_DEV, EXPECTED_TEST
    ]:
        raise ValueError("Official split counts are unexpected")
    if (set(splits["train"]) & set(splits["development"]) or
        set(splits["train"]) & set(splits["test"]) or
        set(splits["development"]) & set(splits["test"])):
        raise ValueError("Dataset leakage between splits")

    stats_path = AUDIT / "m1_logmel_train_stats.npz"
    with np.load(stats_path, allow_pickle=False) as stats:
        if int(stats["count"]) != EXPECTED_VALID_TRAIN:
            raise ValueError("Normalization frame count does not match official training split")
        if set(stats["train_meetings"].tolist()) != set(splits["train"]):
            raise ValueError("Normalization provenance differs from official train meetings")

    train_ds = M1OfficialDataset(
        meetings=splits["train"], project_root=ROOT,
        sequence_length=200, stride=100, normalization_stats=stats_path,
    )
    dev_ds = M1OfficialDataset(
        meetings=splits["development"], project_root=ROOT,
        sequence_length=200, stride=200, normalization_stats=stats_path,
    )
    if sum(int(x["valid_mask"].sum()) for x in train_ds.recordings) != EXPECTED_VALID_TRAIN:
        raise ValueError("Training valid frame count mismatch")
    if sum(int(x["valid_mask"].sum()) for x in dev_ds.recordings) != EXPECTED_VALID_DEV:
        raise ValueError("Development valid frame count mismatch")

    if args.smoke_test:
        train_ds = Subset(train_ds, range(min(args.batch_size * 2, len(train_ds))))
        dev_ds = Subset(dev_ds, range(min(args.batch_size * 2, len(dev_ds))))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    model = SpeechTCN(input_features=80).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    stopping = EarlyStopping(patience=args.patience, min_delta=0.0)

    if args.output_dir is None:
        kind = "smoke" if args.smoke_test else "full"
        output = ROOT / "outputs" / "training" / "m1_official" / f"{kind}_seed{args.seed}"
    else:
        output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protected = ("config.json", "history.json", "best_model.pt", "summary.json")
    if any((output / name).exists() for name in protected):
        parser.error(f"Experiment files already exist in {output}. Use --output-dir elsewhere.")

    epochs = 1 if args.smoke_test else args.epochs
    config = {
        "model": "SpeechTCN", "input_features": 80, "parameters": params,
        "train_meetings": splits["train"], "development_meetings": splits["development"],
        "test_meetings_used_for_training": False,
        "stats_path": str(stats_path), "normalization_train_frames": EXPECTED_VALID_TRAIN,
        "sequence_length": 200, "train_stride": 100, "development_stride": 200,
        "batch_size": args.batch_size, "epochs_requested": epochs,
        "learning_rate": args.lr, "weight_decay": args.weight_decay,
        "optimizer": "AdamW", "loss": "unweighted masked cross entropy",
        "checkpoint_metric": "development_overlap_ap", "patience": args.patience,
        "seed": args.seed, "device": str(device), "smoke_test": args.smoke_test,
        "train_windows_used": len(train_ds), "development_windows_used": len(dev_ds),
        "torch_version": torch.__version__,
        "gpu_name": (torch.xpu.get_device_name(0) if device.type == "xpu" else
                     torch.cuda.get_device_name(0) if device.type == "cuda" else None),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("M1 OFFICIAL SPLIT TRAINING", flush=True)
    print(f"Device: {device}; torch: {torch.__version__}; CUDA: {cuda_available}; XPU: {xpu_available}", flush=True)
    print(f"Mode: {'SMOKE (2 train + 2 dev batches)' if args.smoke_test else 'FULL'}", flush=True)
    print(f"Windows: train={len(train_ds):,}, dev={len(dev_ds):,}", flush=True)
    print(f"Model parameters: {params:,}; output={output}", flush=True)
    history = []
    for epoch in range(1, epochs + 1):
        train_result = train_one_epoch(
            model=model, dataloader=train_loader, optimizer=optimizer,
            device=device, max_grad_norm=1.0, class_weights=None,
        )
        dev_result = evaluate_model(model=model, dataloader=dev_loader, device=device)
        record = {
            "epoch": epoch, "train_loss": float(train_result["loss"]),
            "train_valid_frame_occurrences": int(train_result["valid_frame_occurrences"]),
            "train_batches_processed": int(train_result["batches"]),
            "dev_loss": float(dev_result["loss"]),
            "dev_valid_frames": int(dev_result["valid_frames"]),
            "overlap_ap": float(dev_result["overlap_ap"]),
            "overlap_f1": float(dev_result["overlap_f1"]),
            "overlap_precision": float(dev_result["overlap_precision"]),
            "overlap_recall": float(dev_result["overlap_recall"]),
            "macro_f1": float(dev_result["macro_f1"]),
            "confusion_matrix": dev_result["confusion_matrix"].tolist(),
        }
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        improved, should_stop = stopping.update(record["overlap_ap"], epoch)
        print(f"Epoch {epoch:02d}/{epochs:02d} | train loss {record['train_loss']:.4f} | "
              f"dev AP {record['overlap_ap']:.4f} | dev F1 {record['overlap_f1']:.4f} | "
              f"macro F1 {record['macro_f1']:.4f}", flush=True)
        if improved:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "config": config,
                "dev_metrics": record,
            }, output / "best_model.pt")
            print(f"  New best model, development AP={record['overlap_ap']:.6f}", flush=True)
        if should_stop:
            print(f"Early stopping at epoch {epoch}, best epoch {stopping.best_epoch}", flush=True)
            break

    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "smoke_test": args.smoke_test, "epochs_completed": len(history),
        "best_epoch": stopping.best_epoch,
        "best_overlap_ap": stopping.best_score,
        "metrics_are_partial_smoke_test": args.smoke_test,
        "output_dir": str(output),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SMOKE TEST PASSED" if args.smoke_test else "TRAINING COMPLETED", flush=True)
    print(f"Best development AP: {stopping.best_score:.6f}", flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
