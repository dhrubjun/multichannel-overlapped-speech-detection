
from pathlib import Path

from src.data.ami_annotations import load_ami_annotations

WORDS_DIR = Path(
    r"C:\Master 1st Sem\Deep Learning\Project"
    r"\amicorpus\ami_public_manual_1.6.2\words"
)

speaker_intervals, vocal_intervals, issues = (
    load_ami_annotations(
        meeting="ES2008a",
        words_dir=WORDS_DIR,
        duration=1043.360,
    )
)

print("AMI ANNOTATION LOADER VALIDATION")
print("=" * 45)

for speaker, intervals in zip("ABCD", speaker_intervals):
    print(f"Speaker {speaker}: {len(intervals)} valid words")

print(f"\nVocal-sound intervals: {len(vocal_intervals)}")

print("\nReported annotation issues:")
for issue in issues:
    print(issue)

print(f"\nTotal issues: {len(issues)}")