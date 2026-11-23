"""Resumable M2 extraction for official AMI ES/IS 72/12/12 splits.

Run from the repository root:
    python -m scripts.batch_extract_m2_official --limit 5
    python -m scripts.batch_extract_m2_official

Uses existing scripts.extract_m2_combined, which must have its output directory
configured to D:\\DeepLearning_AMI\\m2_features (or pass --feature-dir to this
script and update the extractor to accept --output-dir).

This script never overwrites complete feature files and checks all existing
outputs before treating them as complete. Metadata is verified as well as the
.npy data using bounded-size chunks (without reading 2,136 dimensions into RAM).
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "outputs" / "dataset_audit" / "official_es_is"
DEFAULT_FEATURE_DIR = Path(r"D:\DeepLearning_AMI\m2_features")
FEATURE_WIDTH = 2136
EXPECTED_SPLIT_SIZES = {"train": 72, "development": 12, "test": 12}
LABEL_DIR = ROOT / "outputs" / "labels"
REPORT = MANIFEST_DIR / "m2_feature_extraction_audit.csv"


def read_manifests():
    splits = {}
    for split, count in EXPECTED_SPLIT_SIZES.items():
        p = MANIFEST_DIR / f"{split}_meetings.txt"
        if not p.exists():
            raise FileNotFoundError(p)
        ids = [s.strip() for s in p.read_text(encoding="utf-8-sig").splitlines()
               if s.strip() and not s.lstrip().startswith("#")]
        if len(ids) != count or len(ids) != len(set(ids)):
            raise ValueError(f"{split}: expected {count} unique meeting IDs; got {len(ids)}")
        splits[split] = ids
    all_ids = [x for group in splits.values() for x in group]
    if len(set(all_ids)) != 96:
        raise ValueError("Overlap between split manifests, or unexpected meeting count")
    return splits


def label_frame_count(meeting):
    p = LABEL_DIR / f"{meeting}_frame_labels.npz"
    if not p.is_file():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=False) as d:
        n = len(d["labels"])
        if n < 1 or len(d["valid_mask"]) != n or len(d["frame_times"]) != n:
            raise ValueError(f"{meeting}: invalid label arrays")
        if not np.isclose(float(d["hop_seconds"]), .01, atol=1e-10, rtol=0):
            raise ValueError(f"{meeting}: unexpected frame hop")
    return n


def validate_output(meeting, feature_dir, expected_frames, chunk_frames=4096):
    feature = feature_dir / f"{meeting}_m2_combined.npy"
    meta = feature_dir / f"{meeting}_m2_combined.json"
    if not feature.exists() or not meta.exists():
        raise FileNotFoundError(f"{meeting}: missing .npy or .json output")
    data = json.loads(meta.read_text(encoding="utf-8"))
    if (data.get("meeting") != meeting or
            int(data.get("n_frames", -1)) != expected_frames or
            int(data.get("n_features", -1)) != FEATURE_WIDTH or
            data.get("smoke_test") is not False or
            data.get("complete_meeting") is not True or
            int(data.get("sample_rate", -1)) != 16000 or
            int(data.get("hop_samples", -1)) != 160 or
            data.get("microphone_pairs") != [[1, 5], [2, 6], [3, 7], [4, 8]] or
            not np.isclose(float(data.get("mask_threshold_db", float("nan"))), -30.0)):
        raise ValueError(f"{meeting}: unexpected metadata: {meta}")
    x = np.load(feature, mmap_mode="r", allow_pickle=False)
    if x.shape != (expected_frames, FEATURE_WIDTH) or x.dtype != np.float32:
        raise ValueError(f"{meeting}: wrong feature shape or dtype: {x.shape}, {x.dtype}")
    for start in range(0, expected_frames, chunk_frames):
        chunk = x[start:start+chunk_frames]
        if not np.isfinite(chunk).all():
            raise ValueError(f"{meeting}: non-finite features at frame {start}")
        spatial = chunk[:, 80:]
        if (spatial.min() < -1.00001) or (spatial.max() > 1.00001):
            raise ValueError(f"{meeting}: IPD outside [-1, 1] at frame {start}")
    size_bytes = feature.stat().st_size
    del x
    return size_bytes


def save_report(rows, report_path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(".partial.csv")
    columns = ["split", "meeting", "frames", "status", "size_gib", "seconds", "message"]
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, report_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--limit", type=int, default=None,
                        help="Validate/extract only the first N meetings")
    parser.add_argument("--chunk-frames", type=int, default=1000)
    parser.add_argument("--min-free-gib", type=float, default=5.0,
                        help="Stop before extraction if drive free space is below this buffer")
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.chunk_frames < 1 or args.min_free_gib < 0:
        parser.error("Invalid chunk size or free space buffer")

    splits = read_manifests()
    meetings = [(sp, m) for sp, ids in splits.items() for m in ids]
    if args.limit:
        meetings = meetings[:args.limit]
    feature_dir = args.feature_dir.resolve()
    feature_dir.mkdir(parents=True, exist_ok=True)
    # The existing single-meeting extractor hardcodes D:\\... . Fail clearly if
    # an alternate path was selected, rather than writing to a surprising place.
    if feature_dir != DEFAULT_FEATURE_DIR.resolve():
        raise ValueError(
            "The existing extractor uses D:\\DeepLearning_AMI\\m2_features. "
            "Change its output_dir to your --feature-dir first, or omit this flag."
        )

    rows = []
    print(f"M2 OFFICIAL BATCH: {len(meetings)} meetings", flush=True)
    print(f"Destination: {feature_dir}", flush=True)
    print(f"Report: {args.report}", flush=True)
    failures = 0
    for idx, (split, meeting) in enumerate(meetings, 1):
        start_time = time.perf_counter()
        print(f"\n[{idx}/{len(meetings)}] {split}: {meeting}", flush=True)
        n = 0
        try:
            n = label_frame_count(meeting)
            feature = feature_dir / f"{meeting}_m2_combined.npy"
            metadata = feature_dir / f"{meeting}_m2_combined.json"
            partial = feature_dir / f"{meeting}_m2_combined.partial.npy"
            if partial.exists():
                raise RuntimeError(
                    f"Found incomplete extraction: {partial}. "
                    "Inspect/remove only this .partial.npy before retrying."
                )
            if feature.exists() or metadata.exists():
                # Intentionally require both files and full structural validation.
                nbytes = validate_output(meeting, feature_dir, n)
                status = "validated_existing"
            else:
                free = shutil.disk_usage(feature_dir).free
                expected_bytes = n * FEATURE_WIDTH * 4 + 1024
                if free < expected_bytes + args.min_free_gib * (1024**3):
                    raise OSError(
                        f"Insufficient disk space: need approx "
                        f"{expected_bytes/1024**3:.2f} GiB plus "
                        f"{args.min_free_gib:g} GiB free margin; "
                        f"available {free/1024**3:.2f} GiB"
                    )
                cmd = [sys.executable, "-m", "scripts.extract_m2_combined",
                       "--meeting", meeting, "--chunk-frames", str(args.chunk_frames)]
                print("Running:", " ".join(cmd), flush=True)
                subprocess.run(cmd, cwd=ROOT, check=True)
                nbytes = validate_output(meeting, feature_dir, n)
                status = "extracted"
            print(f"PASS: {meeting}, {nbytes/1024**3:.3f} GiB, {status}", flush=True)
            message = ""
            size_gib = round(nbytes / (1024**3), 4)
        except Exception as exc:
            failures += 1
            status = "failed"
            size_gib = ""
            message = f"{type(exc).__name__}: {exc}"
            print(f"FAIL: {message}", file=sys.stderr, flush=True)
        rows.append({"split": split, "meeting": meeting, "frames": n,
                     "status": status, "size_gib": size_gib,
                     "seconds": round(time.perf_counter() - start_time, 2),
                     "message": message})
        save_report(rows, args.report)
        if failures:
            break
    print(f"\nCompleted {len(rows)}/{len(meetings)}; failures={failures}", flush=True)
    print(f"Report: {args.report}", flush=True)
    if failures:
        sys.exit(1)
    if not args.limit and len(rows) == 96:
        print("M2 OFFICIAL EXTRACTION AND VALIDATION PASSED: 96/96", flush=True)


if __name__ == "__main__":
    main()
