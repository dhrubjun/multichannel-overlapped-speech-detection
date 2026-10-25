
from pathlib import Path
import xml.etree.ElementTree as ET

ANNOTATION_DIR = Path(
    r"C:\Master 1st Sem\Deep Learning\Project"
    r"\amicorpus\ami_public_manual_1.6.2"
)

meeting = "ES2008a"

for speaker in ["A", "B", "C", "D"]:
    path = (
        ANNOTATION_DIR
        / "words"
        / f"{meeting}.{speaker}.words.xml"
    )

    tree = ET.parse(path)
    root = tree.getroot()

    print(f"\nSpeaker {speaker}")
    print(f"Root tag: {root.tag}")
    print(f"Total XML elements: {len(list(root))}")

    print("\nFirst 8 elements:")

    for element in list(root)[:8]:
        print(f"  Tag: {element.tag}")
        print(f"  Attributes: {element.attrib}")
        print(f"  Text: {element.text!r}")
        print()