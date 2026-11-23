"""Detailed, reproducible M1 evaluation on the 12 official AMI development meetings.

Run from repository root:
  python -m scripts.evaluate_m1_official_development --device xpu

Uses the exact 200-frame, nonoverlapping, padded development sequence convention
from train_m1_official_xpu.py. It never accesses the test meeting audio/features.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.m1_official_dataset import M1OfficialDataset
from src.models.tcn import SpeechTCN
from src.training.evaluation import binary_average_precision

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs" / "dataset_audit" / "official_es_is"
STATS = AUDIT / "m1_logmel_train_stats.npz"
TRAINING_ROOT = ROOT / "outputs" / "training" / "m1_official"
DEFAULT_OUTPUT = ROOT / "outputs" / "evaluation" / "m1_official_development"
SEEDS = (42, 43, 44)
EXPECTED_DEV_FRAMES = 1_920_947
CLASSES = ("no_speech", "single_speaker", "overlap")


def read_split(name):
    path = AUDIT / f"{name}_meetings.txt"
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate meeting in {path}")
    return values


def select_device(name):
    cuda = torch.cuda.is_available()
    xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    if name == "auto":
        name = "cuda" if cuda else "xpu" if xpu else "cpu"
    if name == "cuda" and not cuda:
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "xpu" and not xpu:
        raise RuntimeError("XPU was requested but is unavailable")
    return torch.device(name)


def class_metrics(confusion):
    result = {}
    for k, label in enumerate(CLASSES):
        tp = int(confusion[k, k])
        fp = int(confusion[:, k].sum()) - tp
        fn = int(confusion[k, :].sum()) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        result[label] = {"precision": p, "recall": r, "f1": f1,
                         "support": int(confusion[k, :].sum()),
                         "predicted": int(confusion[:, k].sum())}
    return result


def evaluate_meeting(model, meeting, device, batch_size):
    ds = M1OfficialDataset(meetings=[meeting], project_root=ROOT,
                           sequence_length=200, stride=200,
                           normalization_stats=STATS)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    confusion = np.zeros((3, 3), dtype=np.int64)
    scores = []
    targets = []
    loss_sum = 0.0
    valid = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x = batch["features"].to(device)
            # Intentionally use CPU-side masking to avoid XPU boolean-indexing
            # backend errors, as in the fixed training-time evaluator.
            logits = model(x).cpu()
            labels = batch["labels"]
            usable = batch["valid_mask"] & (labels != -100)
            if not bool(usable.any()):
                continue
            z = logits[usable]
            y = labels[usable]
            if not bool(torch.all((y >= 0) & (y < 3))):
                raise ValueError(f"Invalid labels in {meeting}")
            valid += int(y.numel())
            loss_sum += float(torch.nn.functional.cross_entropy(z, y, reduction="sum"))
            probability = torch.softmax(z, dim=-1)
            pred = probability.argmax(dim=-1)
            y_np = y.numpy()
            p_np = pred.numpy()
            confusion += np.bincount(3*y_np + p_np, minlength=9).reshape(3,3)
            scores.append(probability[:, 2].numpy())
            targets.append(y_np == 2)
    expected = int(ds.recordings[0]["valid_mask"].sum())
    if valid != expected:
        raise RuntimeError(f"{meeting}: counted {valid} valid frames, expected {expected}")
    if not scores:
        raise RuntimeError(f"No valid evaluation scores for {meeting}")
    return {"meeting": meeting, "valid_frames": valid,
            "loss": loss_sum / valid, "loss_sum": loss_sum,
            "confusion": confusion,
            "scores": np.concatenate(scores), "targets": np.concatenate(targets),
            "overlap_ap": binary_average_precision(np.concatenate(scores), np.concatenate(targets))}


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ap-tolerance", type=float, default=0.0005,
                        help="Absolute allowed difference from saved development AP")
    args = parser.parse_args()
    if args.batch_size < 1 or not math.isfinite(args.ap_tolerance) or args.ap_tolerance < 0:
        parser.error("Invalid batch-size or AP tolerance")
    device = select_device(args.device)
    splits = {split: read_split(split) for split in ("train", "development", "test")}
    if tuple(len(splits[s]) for s in ("train", "development", "test")) != (72, 12, 12):
        raise ValueError("Unexpected split sizes")
    if (set(splits["train"]) & set(splits["development"]) or
        set(splits["train"]) & set(splits["test"]) or
        set(splits["development"]) & set(splits["test"])):
        raise ValueError("Meeting leakage across splits")
    with np.load(STATS, allow_pickle=False) as stats:
        if int(stats["count"]) != 12_147_428 or set(stats["train_meetings"].tolist()) != set(splits["train"]):
            raise ValueError("Normalization statistics were not fitted on official training set")
    checkpoints = {seed: TRAINING_ROOT / f"full_xpu_seed{seed}" / "best_model.pt" for seed in SEEDS}
    for seed, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(f"Seed {seed}: checkpoint missing: {path}")
    output = args.output_dir.resolve()
    filenames = ("per_meeting_metrics.csv", "class_metrics.csv", "confusion_matrices.csv",
                 "seed_summary.csv", "three_seed_summary.json", "analysis_notes.txt")
    if any((output / name).exists() for name in filenames):
        raise FileExistsError(f"Evaluation results already present in {output}; choose a new --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    per_meeting = []
    class_rows = []
    confusion_rows = []
    seed_rows = []
    manifest = splits["development"]
    print(f"M1 DETAILED DEVELOPMENT EVALUATION: {len(manifest)} meetings, device={device}", flush=True)
    for seed in SEEDS:
        with torch.serialization.safe_globals(
            [torch.torch_version.TorchVersion]
        ):
            checkpoint = torch.load(
                checkpoints[seed],
                map_location="cpu",
                weights_only=True,
            )
        saved = checkpoint["dev_metrics"]
        config = checkpoint.get("config", {})
        if int(config.get("seed", -1)) != seed or config.get("smoke_test") is not False:
            raise ValueError(f"Seed {seed}: checkpoint config mismatch or smoke-test checkpoint")
        if set(config.get("development_meetings", [])) != set(manifest):
            raise ValueError(f"Seed {seed}: checkpoint development split differs")
        model = SpeechTCN(input_features=80).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        total_conf = np.zeros((3, 3), dtype=np.int64)
        total_loss = 0.0
        pooled_scores, pooled_targets = [], []
        total_valid = 0
        print(f"\nSeed {seed}, checkpoint epoch {checkpoint['epoch']}", flush=True)
        for i, meeting in enumerate(manifest, start=1):
            result = evaluate_meeting(model, meeting, device, args.batch_size)
            cm = result["confusion"]
            total_conf += cm
            total_loss += result["loss_sum"]
            total_valid += result["valid_frames"]
            pooled_scores.append(result["scores"])
            pooled_targets.append(result["targets"])
            metrics = class_metrics(cm)
            per_meeting.append({"seed": seed, "meeting": meeting,
                "valid_frames": result["valid_frames"],
                "overlap_support": metrics["overlap"]["support"],
                "overlap_prevalence": metrics["overlap"]["support"] / result["valid_frames"],
                "overlap_ap": result["overlap_ap"],
                "overlap_precision": metrics["overlap"]["precision"],
                "overlap_recall": metrics["overlap"]["recall"],
                "overlap_f1": metrics["overlap"]["f1"],
                "macro_f1": float(np.mean([m["f1"] for m in metrics.values()]))})
            print(f" {i:02d}/12 {meeting}: AP={result['overlap_ap']:.4f}, "
                  f"overlap F1={metrics['overlap']['f1']:.4f}, "
                  f"valid={result['valid_frames']:,}", flush=True)
            del result
        if total_valid != EXPECTED_DEV_FRAMES:
            raise RuntimeError(f"Seed {seed}: valid-frame count {total_valid} != {EXPECTED_DEV_FRAMES}")
        pooled_ap = binary_average_precision(np.concatenate(pooled_scores), np.concatenate(pooled_targets))
        pooled_metrics = class_metrics(total_conf)
        pooled_f1 = float(np.mean([m["f1"] for m in pooled_metrics.values()]))
        expected_ap = float(saved["overlap_ap"])
        ap_error = abs(pooled_ap - expected_ap)
        if ap_error > args.ap_tolerance:
            raise RuntimeError(f"Seed {seed}: reproduced AP={pooled_ap:.6f} versus "
                               f"training checkpoint={expected_ap:.6f}; discrepancy {ap_error:.6g}. "
                               "Check normalization, evaluation, and checkpoint consistency.")
        seed_rows.append({"seed": seed, "best_epoch": int(checkpoint["epoch"]),
            "valid_frames": total_valid, "overlap_ap": pooled_ap,
            "overlap_precision": pooled_metrics["overlap"]["precision"],
            "overlap_recall": pooled_metrics["overlap"]["recall"],
            "overlap_f1": pooled_metrics["overlap"]["f1"],
            "macro_f1": pooled_f1, "dev_loss": total_loss / total_valid,
            "saved_training_ap": expected_ap, "ap_absolute_error": ap_error})
        for cls in CLASSES:
            class_rows.append({"seed": seed, "class": cls, **pooled_metrics[cls]})
        for i in range(3):
            for j in range(3):
                confusion_rows.append({"seed": seed, "actual": CLASSES[i],
                                       "predicted": CLASSES[j], "frames": int(total_conf[i,j])})
        print(f" POOL: AP={pooled_ap:.6f}, overlap F1={pooled_metrics['overlap']['f1']:.6f}, "
              f"macro F1={pooled_f1:.6f}, saved AP={expected_ap:.6f}, PASS", flush=True)
        del model, pooled_scores, pooled_targets
        if device.type == "xpu":
            torch.xpu.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(output / filenames[0], per_meeting, list(per_meeting[0].keys()))
    write_csv(output / filenames[1], class_rows, list(class_rows[0].keys()))
    write_csv(output / filenames[2], confusion_rows, list(confusion_rows[0].keys()))
    write_csv(output / filenames[3], seed_rows, list(seed_rows[0].keys()))
    aggregate = {}
    for key in ("overlap_ap", "overlap_precision", "overlap_recall", "overlap_f1", "macro_f1"):
        values = np.asarray([row[key] for row in seed_rows], dtype=np.float64)
        aggregate[key] = {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1))}
    summary = {"dataset": "AMI ES/IS official-group development split",
               "device": str(device), "test_set_evaluated": False,
               "seeds": list(SEEDS), "development_meetings": manifest,
               "valid_development_frames": EXPECTED_DEV_FRAMES,
               "classification_rule": "argmax across classes 0,1,2",
               "checkpoint_selection": "highest development overlap AP",
               "pooled_ap_note": "Pooled over all valid development frames, not mean of meeting APs",
               "metrics_mean_and_sample_std": aggregate,
               "per_seed": seed_rows}
    (output / filenames[4]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    notes = ("M1 OFFICIAL DEVELOPMENT EVALUATION\n"
             "All three checkpoints selected by development overlap AP.\n"
             "Scores pooled over every valid frame for pooled overlap AP.\n"
             "Per-meeting AP calculated separately and is not averaged to obtain pooled AP.\n"
             "Class metrics use argmax predictions and zero collar.\n"
             "Invalid and padded frames excluded from all metrics.\n"
             "Development only; test split remains untouched.\n"
             "Performance on development is model-selection performance, not independent test generalization.\n")
    (output / filenames[5]).write_text(notes, encoding="utf-8")
    print("\nTHREE-SEED DEVELOPMENT SUMMARY (mean +/- sample SD)")
    for key, value in aggregate.items():
        print(f"{key}: {value['mean']:.6f} +/- {value['sample_std']:.6f}")
    print(f"EVALUATION PASSED: results in {output}")


if __name__ == "__main__":
    main()
