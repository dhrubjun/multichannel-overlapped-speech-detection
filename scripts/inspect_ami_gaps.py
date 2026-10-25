
from pathlib import Path
import xml.etree.ElementTree as ET

WORDS_DIR = Path(
    r"C:\Master 1st Sem\Deep Learning\Project"
    r"\amicorpus\ami_public_manual_1.6.2\words"
)

MEETING = "ES2008a"

for speaker in "ABCD":
    path = WORDS_DIR / f"{MEETING}.{speaker}.words.xml"
    root = ET.parse(path).getroot()

    for element in root:
        tag = element.tag.split("}")[-1]

        if tag != "gap":
            continue

        print(f"\nSpeaker {speaker}")
        print(f"Attributes: {element.attrib}")
        print(f"Text: {element.text!r}")

        for child in element:
            print(f"Child tag: {child.tag}")
            print(f"Child attributes: {child.attrib}")