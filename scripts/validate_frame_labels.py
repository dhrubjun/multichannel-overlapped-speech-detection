
from pathlib import Path
import argparse
import numpy as np

from src.data.ami_audio import validate_array_audio


parser = argparse.ArgumentParser(
    description="Validate AMI frame labels"
)

parser.add_argument(
    "--meeting",
    default="ES2008a",
    help="AMI meeting identifier"
)

args = parser.parse_args()

MEETING = args.meeting

LABEL_FILE = (
    Path("outputs")
    / "labels"
    / f"{MEETING}_frame_labels.npz"
)

AMI_ROOT = Path(
    r"C:\Master 1st Sem\Deep Learning\Project\amicorpus"
)

AUDIO_DIR = AMI_ROOT / MEETING / "audio"

if not LABEL_FILE.is_file():
    raise FileNotFoundError(
        f"Label file not found: {LABEL_FILE}"
    )

# Obtain the exact recording duration from WAV headers
audio_info = validate_array_audio(
    meeting=MEETING,
    audio_dir=AUDIO_DIR,
    array=1,
    n_mics=8,
)

audio_duration = audio_info["duration"]

with np.load(LABEL_FILE) as data:
    labels = data["labels"]
    valid_mask = data["valid_mask"]
    activity = data["speaker_activity"]
    frame_times = data["frame_times"]
    hop = float(data["hop_seconds"])

print("FRAME LABEL VALIDATION")
print("=" * 45)
print(f"Meeting: {MEETING}")
print(f"File: {LABEL_FILE}")
print(f"Audio duration: {audio_duration:.6f} s")
print(f"Frame hop: {hop * 1000:.1f} ms")

print(f"\nLabels shape: {labels.shape}")
print(f"Speaker activity shape: {activity.shape}")
print(f"Valid mask shape: {valid_mask.shape}")
print(f"Frame times shape: {frame_times.shape}")

# Validate number of dimensions
assert labels.ndim == 1
assert valid_mask.ndim == 1
assert activity.ndim == 2
assert frame_times.ndim == 1

n = len(labels)
n_speakers = activity.shape[0]

assert n_speakers > 0

# All arrays must have consistent frame counts
assert len(valid_mask) == n
assert activity.shape[1] == n
assert len(frame_times) == n

# Ensure valid hop size
assert np.isfinite(hop)
assert hop > 0

# Labels must contain only 0, 1, or 2
assert np.all(np.isin(labels, [0, 1, 2]))

# Check label agreement with speaker activity
speaker_count = activity.sum(axis=0)
expected_labels = np.minimum(speaker_count, 2)

assert np.array_equal(labels, expected_labels)

# Check frame timestamps
assert np.all(np.isfinite(frame_times))

if n > 1:
    assert np.all(np.diff(frame_times) > 0)

expected_times = (np.arange(n) + 0.5) * hop

assert np.allclose(
    frame_times,
    expected_times,
    rtol=0,
    atol=1e-7,
)

# Verify the expected frame count from audio duration
expected_n = round(audio_duration / hop)

assert n == expected_n, (
    f"Frame count mismatch: {n} vs {expected_n}"
)

# Rounding to the nearest frame allows at most half a hop error
assert abs(n * hop - audio_duration) <= hop / 2 + 1e-9

# Masks and activity must be Boolean
assert valid_mask.dtype == np.bool_
assert activity.dtype == np.bool_

print("\nAll structural checks passed.")
print(f"Detected speakers: {n_speakers}")

print("\nVALID FRAMES BY CLASS")
print("=" * 45)

for label, name in [
    (0, "No annotated speech"),
    (1, "Single speaker"),
    (2, "Overlap"),
]:
    count = np.sum(
        (labels == label) & valid_mask
    )

    print(f"{name}: {count} valid frames")

print(f"\nTotal valid frames: {valid_mask.sum()}")
print(f"Excluded frames: {(~valid_mask).sum()}")

print("\nFRAME LABEL VALIDATION PASSED")