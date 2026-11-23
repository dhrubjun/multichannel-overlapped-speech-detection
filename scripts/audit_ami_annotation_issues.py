"""Audit annotation parser issues for the prepared AMI ES/IS split.

Run at repository root:
    python -m scripts.audit_ami_annotation_issues

Read-only: does not regenerate labels or modify source audio/annotations.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.data.ami_annotations import discover_speakers, load_ami_annotations
from src.data.frame_labels import generate_labels, generate_valid_mask


DEFAULT_ROOT = Path(r"C:\Master 1st Sem\Deep Learning\Project\amicorpus")
DEFAULT_MANIFEST_DIR = Path("outputs/dataset_audit/official_es_is")
DEFAULT_LABEL_DIR = Path("outputs/labels")

MANIFESTS = (
    ("train", "train_meetings.txt"),
    ("development", "development_meetings.txt"),
    ("test", "test_meetings.txt"),
)

# Diagnostic categories only; not all imply bad data.
HIGH_ATTENTION = {
    "missing timestamp",
    "invalid timestamp",
    "non-finite timestamp",
    "outside audio duration",
    "end before start",
    "zero-duration interval",
}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ami-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--out-dir", type=Path,
                        default=DEFAULT_MANIFEST_DIR / "annotation_issues")
    args = parser.parse_args()

    words_dir = args.ami_root / "ami_public_manual_1.6.2" / "words"
    if not words_dir.is_dir():
        parser.error(f"Missing annotations directory: {words_dir}")

    splits = {}
    for split, name in MANIFESTS:
        path = args.manifest_dir / name
        if not path.is_file():
            parser.error(f"Missing split manifest: {path}")
        meetings = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
        splits[split] = meetings
    flat = [m for ms in splits.values() for m in ms]
    if len(flat) != len(set(flat)):
        parser.error("Duplicate meeting found across manifests")
    if len(flat) != 96:
        print(f"WARNING: expected 96 meetings from current split, found {len(flat)}", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    issue_rows = []
    meeting_rows = []
    issue_counts = defaultdict(Counter)
    status_counts = Counter()
    error_rows = []

    for split, meetings in splits.items():
        for index, meeting in enumerate(meetings, 1):
            try:
                wav_path = args.ami_root / meeting / "audio" / f"{meeting}.Array1-01.wav"
                with wave.open(str(wav_path), "rb") as w:
                    duration = w.getnframes() / w.getframerate()
                speakers = discover_speakers(meeting, words_dir)
                speaker_intervals, uncertain_intervals, issues = load_ami_annotations(
                    meeting=meeting, words_dir=words_dir, duration=duration, speakers=speakers
                )
                # Reproduce label and validity-mask generation without changing saved files.
                labels, _activity, times = generate_labels(speaker_intervals, duration, hop=.01)
                valid = generate_valid_mask(uncertain_intervals, times)
                label_path = args.label_dir / f"{meeting}_frame_labels.npz"
                with np.load(label_path, allow_pickle=False) as saved:
                    saved_labels = saved["labels"]
                    saved_valid = saved["valid_mask"]
                    if not np.array_equal(saved_labels, labels):
                        raise ValueError("Saved labels differ from parser regeneration")
                    if not np.array_equal(saved_valid, valid):
                        raise ValueError("Saved valid mask differs from parser regeneration")

                counts = Counter(reason for _speaker, _tag, reason, _detail in issues)
                for speaker, tag, reason, detail in issues:
                    # 'positive-duration gap' is explicitly masked and is not equivalent to malformed XML.
                    issue_rows.append(dict(split=split, meeting=meeting, speaker=speaker,
                                           tag=tag, reason=reason, detail=str(detail),
                                           priority="review" if reason in HIGH_ATTENTION else "informational"))
                    issue_counts[split][reason] += 1
                valid_counts = np.bincount(labels[valid].astype(np.int64), minlength=3)
                meeting_rows.append(dict(
                    split=split, meeting=meeting, speakers=len(speakers),
                    duration_seconds=round(duration, 3), issues=len(issues),
                    attention_issues=sum(counts[k] for k in HIGH_ATTENTION),
                    positive_duration_gaps=counts["positive-duration gap"],
                    zero_duration_gaps=counts["zero-duration gap"],
                    valid_frames=int(valid.sum()), excluded_frames=int((~valid).sum()),
                    class0=int(valid_counts[0]), class1=int(valid_counts[1]),
                    class2=int(valid_counts[2]), status="PASS"))
                status_counts[split] += 1
                print(f"[{split:11}] {index:02d}/{len(meetings)} {meeting}: "
                      f"{len(issues)} reported issues, {int((~valid).sum())} masked frames")
            except Exception as exc:
                error_rows.append(dict(split=split, meeting=meeting, error=f"{type(exc).__name__}: {exc}"))
                print(f"FAIL {meeting}: {type(exc).__name__}: {exc}", file=sys.stderr)

    write_csv(args.out_dir / "annotation_issues.csv", issue_rows,
              ["split", "meeting", "speaker", "tag", "reason", "detail", "priority"])
    write_csv(args.out_dir / "meeting_summary.csv", meeting_rows,
              ["split", "meeting", "speakers", "duration_seconds", "issues", "attention_issues",
               "positive_duration_gaps", "zero_duration_gaps", "valid_frames", "excluded_frames",
               "class0", "class1", "class2", "status"])
    write_csv(args.out_dir / "errors.csv", error_rows, ["split", "meeting", "error"])

    result = {
        "meetings_expected": len(flat),
        "meetings_checked": len(meeting_rows),
        "meetings_failed": len(error_rows),
        "issues_total": len(issue_rows),
        "review_issues_total": sum(row["priority"] == "review" for row in issue_rows),
        "issue_counts_by_split": {split: dict(issue_counts[split]) for split in splits},
        "summary_by_split": {
            split: {
                "meetings": status_counts[split],
                "issues": sum(r["issues"] for r in meeting_rows if r["split"] == split),
                "review_issues": sum(r["attention_issues"] for r in meeting_rows if r["split"] == split),
                "excluded_frames": sum(r["excluded_frames"] for r in meeting_rows if r["split"] == split),
            } for split in splits
        },
        "limitations": [
            "Parser issues are not an acoustic/transcript correctness audit.",
            "Missing/invalid timestamps cannot be localized and masked automatically.",
            "Positive-duration gap and vocalsound intervals are excluded by the valid mask.",
            "Known ES2010d recording dropouts need a separate audio-quality review.",
        ],
    }
    (args.out_dir / "annotation_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    print("\nANNOTATION ISSUE AUDIT SUMMARY")
    print(f"Validated: {len(meeting_rows)}/{len(flat)} meetings; failures: {len(error_rows)}")
    print(f"Parser-reported issues: {len(issue_rows)}")
    print(f"Review-priority issues: {result['review_issues_total']}")
    for split, info in result["summary_by_split"].items():
        print(f"{split}: {info}")
    print(f"Reports: {args.out_dir.resolve()}")
    print("NOTE: Structural agreement does not prove acoustic label correctness.")
    if error_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
