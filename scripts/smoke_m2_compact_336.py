"""CPU smoke test for compact 336-feature M2."""

from pathlib import Path
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from src.models.tcn import SpeechTCN
from src.training.engine import train_one_epoch
from src.training.evaluation import evaluate_model

ROOT = Path(__file__).resolve().parents[1]

FEATURES = ROOT / "outputs/features/m2_compact_336"
LABELS = ROOT / "outputs/labels"
AUDIT = ROOT / "outputs/dataset_audit/official_es_is"
STATS = AUDIT / "m1_logmel_train_stats.npz"
OUTPUT = ROOT / "outputs/training/m2_compact_336/smoke_seed42"

TRAIN_MEETING = "ES2008a"
DEV_MEETING = "ES2003a"

SEED = 42
BATCH_SIZE = 8
SEQUENCE_LENGTH = 200


class CompactDataset(Dataset):
    def __init__(self, meeting, stats_path):
        self.meeting = meeting

        feature_path = FEATURES / f"{meeting}_m2_336.npy"
        label_path = LABELS / f"{meeting}_frame_labels.npz"

        self.features = np.load(
            feature_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        with np.load(label_path, allow_pickle=False) as data:
            self.labels = data["labels"].astype(np.int64)
            self.mask = data["valid_mask"].astype(bool)
            times = data["frame_times"]

        with np.load(stats_path, allow_pickle=False) as data:
            self.mean = data["mean"].astype(np.float32)
            self.std = data["std"].astype(np.float32)

        assert self.features.ndim == 2
        assert self.features.shape[1] == 336
        assert self.features.dtype == np.float32
        assert self.mean.shape == self.std.shape == (80,)
        assert np.all(self.std > 0)

        n = len(self.features)
        assert len(self.labels) == len(self.mask) == len(times) == n

        expected = (np.arange(n) + 0.5) * 0.01
        assert np.allclose(times, expected, rtol=0, atol=1e-9)

        # Non-overlapping sequences for the smoke test.
        self.starts = list(range(0, n, SEQUENCE_LENGTH))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        end = start + SEQUENCE_LENGTH

        x = np.array(
            self.features[start:end],
            dtype=np.float32,
            copy=True,
        )
        y = self.labels[start:end].copy()
        mask = self.mask[start:end].copy()

        # Normalize only the 80 Log-Mel features.
        x[:, :80] = (x[:, :80] - self.mean) / self.std

        length = len(y)

        if length < SEQUENCE_LENGTH:
            padding = SEQUENCE_LENGTH - length
            x = np.pad(x, ((0, padding), (0, 0)))
            y = np.pad(y, (0, padding), constant_values=-100)
            mask = np.pad(mask, (0, padding), constant_values=False)

        return {
            "features": torch.from_numpy(np.ascontiguousarray(x)),
            "labels": torch.from_numpy(np.ascontiguousarray(y)),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(mask)),
            "meeting": self.meeting,
            "start_frame": start,
        }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    torch.set_num_threads(min(4, torch.get_num_threads()))

    train_manifest = (
        AUDIT / "train_meetings.txt"
    ).read_text(encoding="utf-8").splitlines()

    dev_manifest = (
        AUDIT / "development_meetings.txt"
    ).read_text(encoding="utf-8").splitlines()

    assert len(train_manifest) == 72
    assert len(dev_manifest) == 12
    assert TRAIN_MEETING in train_manifest
    assert DEV_MEETING in dev_manifest
    assert set(train_manifest).isdisjoint(dev_manifest)

    with np.load(STATS, allow_pickle=False) as stats:
        assert set(stats["train_meetings"].tolist()) == set(train_manifest)
        assert int(stats["count"]) == 12_147_428

    train_dataset = CompactDataset(TRAIN_MEETING, STATS)
    dev_dataset = CompactDataset(DEV_MEETING, STATS)

    train_subset = Subset(train_dataset, range(16))
    dev_subset = Subset(dev_dataset, range(16))

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0,
    )

    dev_loader = DataLoader(
        dev_subset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
    )

    model = SpeechTCN(input_features=336)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )

    print("M2 COMPACT CPU SMOKE TEST")
    print("=" * 60)
    print("Input features: 336")
    print("Training meeting:", TRAIN_MEETING)
    print("Development meeting:", DEV_MEETING)
    print("Training sequences:", len(train_subset))
    print("Development sequences:", len(dev_subset))
    print("Device: CPU")
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    train = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        device="cpu",
        max_grad_norm=1.0,
        class_weights=None,
    )

    dev = evaluate_model(
        model=model,
        dataloader=dev_loader,
        device="cpu",
    )

    result = {
        "smoke_test": True,
        "seed": SEED,
        "input_features": 336,
        "train_meeting": TRAIN_MEETING,
        "dev_meeting": DEV_MEETING,
        "train_loss": float(train["loss"]),
        "dev_loss": float(dev["loss"]),
        "overlap_ap": float(dev["overlap_ap"]),
        "overlap_f1": float(dev["overlap_f1"]),
        "macro_f1": float(dev["macro_f1"]),
        "dev_valid_frames": int(dev["valid_frames"]),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = OUTPUT / "smoke_result.json"

    if report.exists():
        raise FileExistsError(f"Existing report: {report}")

    report.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nRESULTS")
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
