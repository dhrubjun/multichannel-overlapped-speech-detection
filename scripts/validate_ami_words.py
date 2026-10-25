
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET
import math

# Location of AMI word annotations
WORDS_DIR = Path(
    r"C:\Master 1st Sem\Deep Learning\Project"
    r"\amicorpus\ami_public_manual_1.6.2\words"
)

MEETING = "ES2008a"
AUDIO_DURATION = 1043.360

for speaker in ["A", "B", "C", "D"]:

    file_path = WORDS_DIR / f"{MEETING}.{speaker}.words.xml"

    root = ET.parse(file_path).getroot()

    counts = Counter()
    vocal_types = Counter()
    issues = []

    for element in root:

        # Get the XML element type without its namespace
        tag = element.tag.split("}")[-1]

        counts["total_elements"] += 1

        if tag == "vocalsound":
            counts["vocal_sounds"] += 1
            vocal_types[element.get("type", "unknown")] += 1

        elif tag == "w":
            if element.get("punc") == "true":
                counts["punctuation"] += 1
            else:
                counts["spoken_word_entries"] += 1

        # Check timestamps
        start_text = element.get("starttime")
        end_text = element.get("endtime")

        if start_text is None or end_text is None:
            issues.append(f"{tag}: missing timestamp")
            continue

        try:
            start = float(start_text)
            end = float(end_text)
        except ValueError:
            issues.append(f"{tag}: invalid timestamp")
            continue

        if not (
            math.isfinite(start)
            and math.isfinite(end)
            and 0 <= start <= end <= AUDIO_DURATION
        ):
            issues.append(
                f"{tag}: invalid interval {start}-{end}"
            )
            continue

        # Identify spoken words with zero duration
        if tag == "w" and element.get("punc") != "true":

            if end > start:
                counts["positive_duration_words"] += 1
            else:
                counts["zero_duration_words"] += 1

    print(f"\nSpeaker {speaker}")
    print("-" * 40)

    print(f"Total elements: {counts['total_elements']}")
    print(f"Spoken word entries: {counts['spoken_word_entries']}")
    print(f"Positive-duration words: {counts['positive_duration_words']}")
    print(f"Zero-duration words: {counts['zero_duration_words']}")
    print(f"Punctuation entries: {counts['punctuation']}")
    print(f"Vocal sounds: {counts['vocal_sounds']}")
    print(f"Vocal sound types: {dict(vocal_types)}")
    print(f"Timestamp issues: {len(issues)}")

    for issue in issues[:5]:
        print(f"  {issue}")