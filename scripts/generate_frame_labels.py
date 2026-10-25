
from pathlib import Path
import numpy as np

import wave

import argparse

from src.data.ami_annotations import (
    load_ami_annotations,
    discover_speakers,
)
from src.data.frame_labels import (
    generate_labels,
    generate_valid_mask,
)

from src.data.ami_audio import validate_array_audio

WORDS_DIR = Path(
    r"C:\Master 1st Sem\Deep Learning\Project"
    r"\amicorpus\ami_public_manual_1.6.2\words"
)

AUDIO_DIR = Path(
    r"C:\Master 1st Sem\Deep Learning\Project"
    r"\amicorpus\ES2008a\audio"
)

parser = argparse.ArgumentParser(
    description="Generate frame-level labels from AMI annotations"
)

parser.add_argument(
    "--meeting",
    type=str,
    default="ES2008a",
    help="AMI meeting ID, for example ES2008a"
)

args = parser.parse_args()

MEETING = args.meeting
AMI_ROOT = Path(
    r"C:\Master 1st Sem\Deep Learning\Project\amicorpus"
)

AUDIO_DIR = AMI_ROOT / MEETING / "audio"

# Validate all eight Array 1 microphone files
audio_info = validate_array_audio(
    meeting=MEETING,
    audio_dir=AUDIO_DIR,
    array=1,
    n_mics=8,
)

# Obtain the recording duration from validated metadata
DURATION = audio_info["duration"]
HOP = 0.01
SPEAKERS = discover_speakers(
    meeting=MEETING,
    words_dir=WORDS_DIR,
)

# 1. Load validated AMI annotations
speaker_intervals, vocal_intervals, issues = (
    load_ami_annotations(
        meeting=MEETING,
        words_dir=WORDS_DIR,
        duration=DURATION,
        speakers=SPEAKERS,
    )
)

# 2. Generate frame-level labels
labels, activity, frame_times = generate_labels(
    intervals=speaker_intervals,
    duration=DURATION,
    hop=HOP,
)

# 3. Generate uncertainty mask
valid_mask = generate_valid_mask(
    uncertain_intervals=vocal_intervals,
    frame_times=frame_times,
)

# 4. Save outputs locally
output_dir = Path("outputs/labels")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"{MEETING}_frame_labels.npz"

np.savez_compressed(
    output_file,
    labels=labels,
    valid_mask=valid_mask,
    speaker_activity=activity,
    frame_times=frame_times,
    hop_seconds=HOP,
)

# 5. Print validation summary
print("AMI FRAME LABEL GENERATION")
print("=" * 45)

print(f"Meeting: {MEETING}")
print(f"Speakers: {len(speaker_intervals)}")
print(f"Frames: {len(labels)}")
print(f"Frame hop: {HOP * 1000:.0f} ms")
print(f"Duration: {len(labels) * HOP:.3f} s")

print("\nClass distribution:")

for label, name in [
    (0, "No annotated speech"),
    (1, "Single speaker"),
    (2, "Overlap"),
]:
    count = np.sum(labels == label)

    print(
        f"{name}: {count} frames "
        f"({100 * count / len(labels):.2f}%)"
    )

print(f"\nExcluded frames: {(~valid_mask).sum()}")
print(f"Valid frames: {valid_mask.sum()}")

print(f"\nAnnotation issues reported: {len(issues)}")
print(f"Saved: {output_file}")