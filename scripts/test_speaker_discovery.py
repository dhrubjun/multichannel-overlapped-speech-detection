
from pathlib import Path

from src.data.ami_annotations import discover_speakers

WORDS_DIR = Path(
    r"C:\Master 1st Sem\Deep Learning\Project"
    r"\amicorpus\ami_public_manual_1.6.2\words"
)

meeting = "ES2008a"

speakers = discover_speakers(
    meeting=meeting,
    words_dir=WORDS_DIR,
)

print(f"Meeting: {meeting}")
print(f"Speakers discovered: {speakers}")
print(f"Number of speakers: {len(speakers)}")