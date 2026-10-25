
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict

WORDS_DIR = Path(
    r"C:\Master 1st Sem\Deep Learning\Project"
    r"\amicorpus\ami_public_manual_1.6.2\words"
)

MEETING = "ES2008a"
AUDIO_DURATION = 1043.360


def load_intervals(speaker):
    path = WORDS_DIR / f"{MEETING}.{speaker}.words.xml"
    root = ET.parse(path).getroot()

    intervals = []

    for element in root:
        if element.tag.split("}")[-1] != "w":
            continue

        if element.get("punc") == "true":
            continue

        try:
            start = float(element.get("starttime"))
            end = float(element.get("endtime"))
        except (TypeError, ValueError):
            continue

        if 0 <= start < end <= AUDIO_DURATION:
            intervals.append((start, end))

    # Merge intervals from the same speaker
    merged = []

    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            old_start, old_end = merged[-1]
            merged[-1] = (old_start, max(old_end, end))
        else:
            merged.append((start, end))

    return merged


# Each event records a speaker entering or leaving
events = defaultdict(list)

for speaker in "ABCD":
    for start, end in load_intervals(speaker):
        events[start].append((speaker, True))
        events[end].append((speaker, False))


active = set()
previous_time = 0.0
raw_overlaps = []

for time in sorted(events):

    # The active set describes the interval
    # between previous_time and time
    if len(active) >= 2 and time > previous_time:
        raw_overlaps.append(
            (previous_time, time, tuple(sorted(active)))
        )

    # Apply all changes at the same timestamp
    for speaker, is_start in events[time]:
        if is_start:
            active.add(speaker)
        else:
            active.discard(speaker)

    previous_time = time

assert not active, "Some speech intervals were not closed"

# Merge touching overlap intervals into episodes
episodes = []

for start, end, speakers in raw_overlaps:

    if episodes and start <= episodes[-1][1] + 1e-9:
        episodes[-1][1] = max(episodes[-1][1], end)
        episodes[-1][2].update(speakers)
    else:
        episodes.append([start, end, set(speakers)])


total_overlap = sum(
    end - start for start, end, _ in episodes
)

print(f"Number of overlap episodes: {len(episodes)}")
print(f"Total overlap duration: {total_overlap:.3f} s")

# Compare with previous analysis
assert abs(total_overlap - 28.93) < 0.02, (
    "Overlap duration differs from previous analysis"
)

print("\nFirst 15 overlap episodes")
print("=" * 65)

for start, end, speakers in episodes[:15]:
    print(
        f"{start:8.3f} - {end:8.3f} s | "
        f"{end-start:6.3f} s | "
        f"Speakers: {','.join(sorted(speakers))}"
    )

print("\nDuration-based episode statistics")

durations = [end - start for start, end, _ in episodes]

if durations:
    print(f"Shortest: {min(durations):.3f} s")
    print(f"Longest:  {max(durations):.3f} s")
    print(f"Mean:     {sum(durations)/len(durations):.3f} s")