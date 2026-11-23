"""Generate/check AMI frame labels for the audited official ES/IS split.

Run from repository root:
    python scripts/batch_generate_ami_labels.py
    python scripts/batch_generate_ami_labels.py --force

The script does not use the test set to fit or select any model.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SPLITS = {
    "train": "train_meetings.txt",
    "development": "development_meetings.txt",
    "test": "test_meetings.txt",
}
LABEL_SUFFIX = "_frame_labels.npz"


def validate_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if not {"labels", "valid_mask", "frame_times", "hop_seconds"}.issubset(data.files):
            raise ValueError("Missing required arrays")
        labels = data["labels"]
        valid = data["valid_mask"]
        times = data["frame_times"]
        hop = float(data["hop_seconds"])
        if labels.ndim != 1 or valid.ndim != 1 or times.ndim != 1:
            raise ValueError("Expected 1D labels, mask, and times")
        if len(labels) == 0 or not (len(labels) == len(valid) == len(times)):
            raise ValueError("Array lengths do not match")
        if valid.dtype != np.dtype("bool"):
            raise ValueError("valid_mask must be boolean")
        if not np.isin(labels, [0, 1, 2]).all():
            raise ValueError("Unexpected class label")
        if abs(hop - .01) > 1e-9:
            raise ValueError("Unexpected frame hop")
        if not np.all(np.isfinite(times)):
            raise ValueError("Invalid frame times")
        counts = np.bincount(labels[valid].astype(np.int64), minlength=3)
        return {"frames": int(len(labels)), "valid": int(valid.sum()),
                "excluded": int((~valid).sum()),
                "class_0": int(counts[0]), "class_1": int(counts[1]),
                "class_2": int(counts[2])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", default="outputs/dataset_audit/official_es_is")
    parser.add_argument("--label-dir", default="outputs/labels")
    parser.add_argument("--force", action="store_true", help="Regenerate existing valid label files")
    args = parser.parse_args()
    root = Path.cwd()
    if not (root / "src" / "data" / "frame_labels.py").is_file():
        parser.error("Run this from the repository root")
    manifests = root / args.manifest_dir
    label_dir = root / args.label_dir
    label_dir.mkdir(parents=True, exist_ok=True)
    outputs = root / "outputs" / "dataset_audit" / "official_es_is"
    outputs.mkdir(parents=True, exist_ok=True)
    assignments = []
    for split, fname in SPLITS.items():
        p = manifests / fname
        if not p.is_file():
            parser.error(f"Missing manifest: {p}")
        meetings = [x.strip() for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
        assignments.extend((split, m) for m in meetings)
    names = [m for _, m in assignments]
    if len(names) != len(set(names)):
        parser.error("Duplicate meeting IDs across manifests")
    if len(assignments) != 96:
        parser.error(f"Expected 96 manifest entries; found {len(assignments)}")

    records = []
    failure_count = 0
    for i, (split, meeting) in enumerate(assignments, 1):
        path = label_dir / (meeting + LABEL_SUFFIX)
        status = "existing"
        details = ""
        if args.force or not path.exists():
            status = "generated"
        else:
            try:
                validate_npz(path)
            except Exception as exc:
                print(f"[{i}/96] Regenerating invalid existing file {meeting}: {exc}", flush=True)
                status = "generated"
        if status == "generated":
            print(f"[{i}/96] Generating {split} / {meeting}", flush=True)
            command = [sys.executable, "-m", "scripts.generate_frame_labels", "--meeting", meeting]
            result = subprocess.run(command, cwd=root, text=True, capture_output=True)
            if result.returncode != 0:
                status = "failed"
                details = (result.stdout + "\n" + result.stderr).strip()[-2000:]
            else:
                match = re.search(r"Annotation issues reported:\s*(\d+)", result.stdout)
                if match:
                    details = f"annotation_issues={match.group(1)}"
        stats = {}
        if status != "failed":
            try:
                stats = validate_npz(path)
            except Exception as exc:
                status = "failed"
                details = f"Output validation failed: {exc}"
        if status == "failed":
            failure_count += 1
        records.append({"split": split, "meeting": meeting, "status": status,
                        "notes": details, **stats})
        print(f"[{i}/96] {meeting}: {status}" + (f" | {details}" if details else ""), flush=True)

    csv_path = outputs / "frame_label_audit.csv"
    fields = ["split", "meeting", "status", "notes", "frames", "valid", "excluded", "class_0", "class_1", "class_2"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    summary = {}
    for split in SPLITS:
        subset = [r for r in records if r["split"] == split]
        passed = [r for r in subset if r["status"] != "failed"]
        sums = {field: sum(r[field] for r in passed) for field in
                ("frames", "valid", "excluded", "class_0", "class_1", "class_2")}
        valid_total = sums["valid"]
        summary[split] = {"meetings": len(subset), "passed": len(passed), **sums,
                          "class_percent_valid": {
                              str(c): round(100*sums[f"class_{c}"]/valid_total, 4) if valid_total else None
                              for c in (0, 1, 2)}}
    summary_path = outputs / "frame_label_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nFRAME LABEL AUDIT SUMMARY")
    for split, stats in summary.items():
        print(f"{split}: {stats['passed']}/{stats['meetings']} meetings, "
              f"{stats['valid']:,} valid frames, "
              f"class % {stats['class_percent_valid']}")
    print(f"Audit CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")
    if failure_count:
        print(f"FAILED: {failure_count} meeting(s). Inspect audit CSV before feature extraction.")
        return 1
    print("PASSED: 96/96 label files structurally validated. Review annotation issues separately.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
