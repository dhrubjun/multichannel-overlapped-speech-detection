
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter

ANNOTATION_DIR = Path(
    r"C:\Master 1st Sem\Deep Learning\Project"
    r"\amicorpus\ami_public_manual_1.6.2"
)

meeting = "ES2008a"

for speaker in ["A", "B", "C", "D"]:
    path = (
        ANNOTATION_DIR
        / "segments"
        / f"{meeting}.{speaker}.segments.xml"
    )

    root = ET.parse(path).getroot()
    elements = list(root)

    print(f"\nSpeaker {speaker}")
    print(f"Total elements: {len(elements)}")

    tag_counts = Counter(element.tag for element in elements)
    print(f"Element types: {dict(tag_counts)}")

    print("\nFirst 5 elements:")

    for element in elements[:5]:
        print(f"  Tag: {element.tag}")
        print(f"  Attributes: {element.attrib}")

        for child in list(element)[:2]:
            print(
                f"  Child: {child.tag}, "
                f"Attributes: {child.attrib}"
            )

        print()