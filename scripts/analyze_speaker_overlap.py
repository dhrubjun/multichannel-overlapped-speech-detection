
from pathlib import Path
import xml.etree.ElementTree as ET

WORDS_DIR = Path(
    r"C:\Master 1st Sem\Deep Learning\Project"
    r"\amicorpus\ami_public_manual_1.6.2\words"
)

MEETING = "ES2008a"
AUDIO_DURATION = 1043.360


def load_speech_intervals(speaker):
    path = WORDS_DIR / f"{MEETING}.{speaker}.words.xml"
    root = ET.parse(path).getroot()

    intervals = []

    for element in root:
        tag = element.tag.split("}")[-1]

        if tag != "w":
            continue

        if element.get("punc") == "true":
            continue

        try:
            start = float(element.get("starttime"))
            end = float(element.get("endtime"))
        except (TypeError, ValueError):
            continue

        if not (0 <= start < end <= AUDIO_DURATION):
            continue

        intervals.append((start, end))

    return sorted(intervals)


def merge_intervals(intervals):
    """Combine overlapping or touching intervals from one speaker."""
    merged = []

    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (
                previous_start,
                max(previous_end, end)
            )
        else:
            merged.append((start, end))

    return merged


events = []

for speaker in "ABCD":
    intervals = merge_intervals(load_speech_intervals(speaker))

    print(f"Speaker {speaker}: {len(intervals)} merged speech intervals")

    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))

# Sort timestamps, then group all changes at identical times
events.sort()
changes = {}

for time, change in events:
    changes[time] = changes.get(time, 0) + change

durations = {0: 0.0, 1: 0.0, 2: 0.0}
overlap_intervals = []

active_speakers = 0
previous_time = 0.0

for time in sorted(changes):
    duration = time - previous_time

    if active_speakers == 0:
        durations[0] += duration
    elif active_speakers == 1:
        durations[1] += duration
    else:
        durations[2] += duration

        if duration > 0:
            overlap_intervals.append((previous_time, time))

    active_speakers += changes[time]

    if not 0 <= active_speakers <= 4:
        raise ValueError("Invalid active-speaker count")

    previous_time = time

if active_speakers != 0:
    raise ValueError("Unclosed speech intervals")

# Include the final interval after the last spoken word
durations[0] += AUDIO_DURATION - previous_time

print("\nANNOTATION-DERIVED ACTIVITY")
print("=" * 45)

for label, name in [
    (0, "No annotated speaker"),
    (1, "Single-speaker speech"),
    (2, "Overlapping speech")
]:
    duration = durations[label]
    percentage = 100 * duration / AUDIO_DURATION

    print(
        f"{name}: {duration:.2f} s "
        f"({percentage:.2f}%)"
    )

print(f"\nTotal duration: {sum(durations.values()):.3f} s")

print("\nFirst 10 overlap intervals:")

for start, end in overlap_intervals[:10]:
    print(
        f"  {start:.3f} - {end:.3f} s "
        f"(duration: {end-start:.3f} s)"
    )