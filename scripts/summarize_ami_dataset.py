"""Summarize five AMI meeting labels and annotation issues.

Run from the project root:
    python -m scripts.summarize_ami_dataset

The generated CSV reports are written to outputs/reports/ (gitignored).
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.ami_annotations import discover_speakers, load_ami_annotations
from src.data.ami_audio import validate_array_audio

DEFAULT_SPLITS = {
    "ES2008a": "train",
    "ES2008b": "train",
    "ES2008c": "train",
    "ES2003a": "development",
    "ES2004a": "test",
}

SUMMARY_COLUMNS = [
    "meeting", "split", "speakers", "duration_seconds", "hop_seconds",
    "total_frames", "valid_frames", "excluded_frames",
    "no_speech_frames", "single_speaker_frames", "overlap_frames",
    "valid_no_speech_frames", "valid_single_speaker_frames", "valid_overlap_frames",
    "valid_overlap_percent", "annotation_issue_count",
]
ISSUE_COLUMNS = ["meeting", "split", "speaker", "element", "issue_type", "count"]


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize_meeting(meeting, split, ami_root, labels_dir):
    label_file = labels_dir / f"{meeting}_frame_labels.npz"
    if not label_file.is_file():
        raise FileNotFoundError(f"Missing label file: {label_file}")

    audio_dir = ami_root / meeting / "audio"
    info = validate_array_audio(meeting=meeting, audio_dir=audio_dir, array=1, n_mics=8)
    duration = float(info["duration"])

    with np.load(label_file, allow_pickle=False) as saved:
        labels = saved["labels"]
        valid_mask = saved["valid_mask"]
        activity = saved["speaker_activity"]
        times = saved["frame_times"]
        hop = float(saved["hop_seconds"])

    if not (labels.ndim == valid_mask.ndim == times.ndim == 1 and activity.ndim == 2):
        raise ValueError(f"{meeting}: unexpected array dimensions")
    n = len(labels)
    if n == 0 or len(valid_mask) != n or len(times) != n or activity.shape[1] != n:
        raise ValueError(f"{meeting}: inconsistent frame counts")
    if valid_mask.dtype != np.bool_ or activity.dtype != np.bool_:
        raise ValueError(f"{meeting}: masks/activity must be Boolean")
    if not np.isfinite(hop) or hop <= 0 or n != round(duration / hop):
        raise ValueError(f"{meeting}: frame count does not agree with audio duration")
    if not np.all(np.isin(labels, [0, 1, 2])):
        raise ValueError(f"{meeting}: invalid class values")
    if not np.array_equal(labels, np.minimum(activity.sum(axis=0), 2)):
        raise ValueError(f"{meeting}: labels disagree with speaker activity")
    if not np.allclose(times, (np.arange(n) + 0.5) * hop, rtol=0, atol=1e-7):
        raise ValueError(f"{meeting}: frame timestamps do not match the frame grid")

    words_dir = ami_root / "ami_public_manual_1.6.2" / "words"
    speakers = discover_speakers(meeting=meeting, words_dir=words_dir)
    _, _, issues = load_ami_annotations(
        meeting=meeting, words_dir=words_dir, duration=duration, speakers=speakers
    )
    if activity.shape[0] != len(speakers):
        raise ValueError(f"{meeting}: speaker-activity rows != discovered speakers")

    totals = [int(np.count_nonzero(labels == i)) for i in range(3)]
    valid = [int(np.count_nonzero((labels == i) & valid_mask)) for i in range(3)]
    n_valid = int(valid_mask.sum())

    summary = {
        "meeting": meeting, "split": split, "speakers": len(speakers),
        "duration_seconds": f"{duration:.6f}", "hop_seconds": f"{hop:.6f}",
        "total_frames": n, "valid_frames": n_valid,
        "excluded_frames": int(n - n_valid),
        "no_speech_frames": totals[0], "single_speaker_frames": totals[1],
        "overlap_frames": totals[2],
        "valid_no_speech_frames": valid[0], "valid_single_speaker_frames": valid[1],
        "valid_overlap_frames": valid[2],
        "valid_overlap_percent": f"{100 * valid[2] / n_valid:.3f}" if n_valid else "0.000",
        "annotation_issue_count": len(issues),
    }

    issue_counts = Counter()
    for issue in issues:
        if len(issue) < 3:
            raise ValueError(f"{meeting}: unexpected issue structure {issue!r}")
        speaker, element, issue_type = issue[:3]
        issue_counts[(str(speaker), str(element), str(issue_type))] += 1

    issue_rows = [
        {"meeting": meeting, "split": split, "speaker": speaker,
         "element": element, "issue_type": issue_type, "count": count}
        for (speaker, element, issue_type), count in sorted(issue_counts.items())
    ]
    return summary, issue_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ami-root",
        type=Path,
        default=Path(r"C:\Master 1st Sem\Deep Learning\Project\amicorpus"),
    )
    parser.add_argument("--labels-dir", type=Path, default=Path("outputs/labels"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports"))
    args = parser.parse_args()

    summaries, issue_rows = [], []
    print("AMI DATASET REPORT")
    print("=" * 64)

    for meeting, split in DEFAULT_SPLITS.items():
        summary, issues = summarize_meeting(meeting, split, args.ami_root, args.labels_dir)
        summaries.append(summary)
        issue_rows.extend(issues)
        print(
            f"{meeting:<9} {split:<12} "
            f"valid={summary['valid_frames']:>7,} "
            f"overlap={summary['valid_overlap_frames']:>6,} "
            f"issues={summary['annotation_issue_count']:>3}"
        )

    summary_path = args.output_dir / "ami_dataset_summary.csv"
    issues_path = args.output_dir / "ami_annotation_issues.csv"
    write_csv(summary_path, SUMMARY_COLUMNS, summaries)
    write_csv(issues_path, ISSUE_COLUMNS, issue_rows)
    print("\nReports saved:")
    print(f"  {summary_path}")
    print(f"  {issues_path}")
    print("Note: class 0 means no annotated speech, not confirmed acoustic silence.")


if __name__ == "__main__":
    main()