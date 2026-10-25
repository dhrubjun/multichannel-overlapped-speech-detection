
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

WORDS_DIR = Path(
    r"C:\Master 1st Sem\Deep Learning\Project"
    r"\amicorpus\ami_public_manual_1.6.2\words"
)

MEETING = "ES2008a"

for speaker in "ABCD":
    path = WORDS_DIR / f"{MEETING}.{speaker}.words.xml"
    root = ET.parse(path).getroot()

    tag_counts = Counter(
        element.tag.split("}")[-1]
        for element in root
    )

    print(f"\nSpeaker {speaker}")
    print("-" * 45)
    print("All XML element types:")

    for tag, count in sorted(tag_counts.items()):
        print(f"  {tag}: {count}")

    print("\nElements without timestamps:")

    missing = 0

    for element in root:
        if (
            element.get("starttime") is None
            or element.get("endtime") is None
        ):
            missing += 1

            print(f"  Tag: {element.tag.split('}')[-1]}")
            print(f"  Attributes: {element.attrib}")
            print(f"  Text: {element.text!r}")
            print()

    if missing == 0:
        print("  None")