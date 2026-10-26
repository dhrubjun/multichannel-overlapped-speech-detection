
from pathlib import Path
import math
import xml.etree.ElementTree as ET


def discover_speakers(meeting, words_dir):
    """
    Discover participant IDs from AMI word annotation files.

    Example:
        ES2008a.A.words.xml -> A
        ES2008a.B.words.xml -> B
    """

    words_dir = Path(words_dir)

    pattern = f"{meeting}.*.words.xml"
    files = sorted(words_dir.glob(pattern))

    speakers = []

    for file in files:
        parts = file.name.split(".")

        # Expected: meeting.speaker.words.xml
        if (
            len(parts) == 4
            and parts[0] == meeting
            and parts[2] == "words"
            and parts[3] == "xml"
            and len(parts[1]) == 1
            and parts[1].isalpha()
        ):
            speakers.append(parts[1])

    if not speakers:
        raise FileNotFoundError(
            f"No speaker annotations found for {meeting}"
        )

    return sorted(set(speakers))


def load_ami_annotations(
    meeting,
    words_dir,
    duration,
    speakers="ABCD",
):
    """
    Read AMI word annotations for one meeting.

    Returns:
        speaker_intervals: List of intervals per speaker
        vocal_intervals: Non-lexical vocal-sound intervals
        issues: Annotation entries requiring investigation
    """

    words_dir = Path(words_dir)

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid recording duration")

    speaker_intervals = []
    vocal_intervals = []
    issues = []

    for speaker in speakers:
        file_path = (
            words_dir / f"{meeting}.{speaker}.words.xml"
        )

        if not file_path.is_file():
            raise FileNotFoundError(file_path)

        root = ET.parse(file_path).getroot()
        intervals = []

        for element in root:
            tag = element.tag.split("}")[-1]

            if tag not in {"w", "vocalsound", "gap"}:
                continue

            if tag == "w" and element.get("punc") == "true":
                continue

            start_text = element.get("starttime")
            end_text = element.get("endtime")

            if start_text is None or end_text is None:
                issues.append(
                    (speaker, tag, "missing timestamp",
                     element.text)
                )
                continue

            try:
                start = float(start_text)
                end = float(end_text)
            except ValueError:
                issues.append(
                    (speaker, tag, "invalid timestamp",
                     element.text)
                )
                continue

            if not (math.isfinite(start) and math.isfinite(end)):
                issues.append(
                    (speaker, tag, "non-finite timestamp", element.text)
                )
                continue

            if start < 0 or end > duration:
                issues.append(
                    (speaker, tag, "outside audio duration", element.text)
                )
                continue

            if start > end:
                issues.append(
                    (speaker, tag, "end before start", element.text)
                )
                continue

            if start == end:
                if tag == "gap":
                    issues.append(
                        (speaker, tag, "zero-duration gap",
                         element.text)
                    )
                else:
                    issues.append(
                        (speaker, tag, "zero-duration interval",
                         element.text)
                    )
                continue

            if tag == "w":
                intervals.append((start, end))

            elif tag == "vocalsound":
                vocal_intervals.append((start, end))

            elif tag == "gap":
                issues.append(
                    (speaker, tag, "positive-duration gap", (start, end))
                )
                vocal_intervals.append((start, end))

        speaker_intervals.append(sorted(intervals))

    return speaker_intervals, vocal_intervals, issues