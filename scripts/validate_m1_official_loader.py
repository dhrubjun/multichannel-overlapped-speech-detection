"""Validate the expanded M1 memmap dataset without training a model.

Run: python -m scripts.validate_m1_official_loader
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.m1_official_dataset import M1OfficialDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = PROJECT_ROOT / "outputs" / "dataset_audit" / "official_es_is"


def read_ids(split):
    path = AUDIT_ROOT / f"{split}_meetings.txt"
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicates in {path}")
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    splits = {s: read_ids(s) for s in ("train", "development", "test")}
    assert [len(splits[s]) for s in ("train", "development", "test")] == [72, 12, 12]
    for a in splits:
        for b in splits:
            if a != b and set(splits[a]) & set(splits[b]):
                raise ValueError(f"Dataset leakage between {a} and {b}")

    stats_path = AUDIT_ROOT / "m1_logmel_train_stats.npz"
    with np.load(stats_path, allow_pickle=False) as stats:
        assert int(stats["count"]) == 12147428
        assert set(stats["train_meetings"].tolist()) == set(splits["train"])

    datasets = {}
    for split in ("train", "development", "test"):
        stride = 100 if split == "train" else 200
        ds = M1OfficialDataset(
            meetings=splits[split], project_root=PROJECT_ROOT,
            sequence_length=200, stride=stride,
            normalization_stats=stats_path,
        )
        datasets[split] = ds
        valid = sum(int(r["valid_mask"].sum()) for r in ds.recordings)
        print(f"{split}: {len(ds.recordings)} meetings, {len(ds):,} windows, "
              f"{valid:,} valid frames")

    assert sum(int(r["valid_mask"].sum()) for r in datasets["train"].recordings) == 12147428
    assert sum(int(r["valid_mask"].sum()) for r in datasets["development"].recordings) == 1920947
    assert sum(int(r["valid_mask"].sum()) for r in datasets["test"].recordings) == 2270411

    for split, ds in datasets.items():
        for idx in sorted(set([0, len(ds) // 2, len(ds) - 1])):
            item = ds[idx]
            assert item["features"].shape == (200, 80)
            assert item["labels"].shape == (200,)
            assert item["valid_mask"].shape == (200,)
            assert item["features"].dtype == torch.float32
            assert item["labels"].dtype == torch.int64
            assert item["valid_mask"].dtype == torch.bool
            assert bool(torch.isfinite(item["features"]).all())
            assert bool(torch.all((item["labels"][item["valid_mask"]] >= 0) & (item["labels"][item["valid_mask"]] <= 2)))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        assert batch["features"].shape[1:] == (200, 80)
        assert batch["labels"].shape[1:] == (200,)
        assert batch["valid_mask"].shape[1:] == (200,)
        assert bool(torch.isfinite(batch["features"]).all())
        print(f"{split}: batch {tuple(batch['features'].shape)}, PASS")

    print("VALIDATION PASSED: M1 official memory-mapped loader")


if __name__ == "__main__":
    main()
