
from pathlib import Path
import argparse

from src.data.ami_audio import validate_array_audio

parser = argparse.ArgumentParser(
    description="Validate AMI multichannel audio"
)

parser.add_argument(
    "--meeting",
    default="ES2008a"
)

args = parser.parse_args()

AMI_ROOT = Path(
    r"C:\Master 1st Sem\Deep Learning\Project\amicorpus"
)

audio_dir = AMI_ROOT / args.meeting / "audio"

result = validate_array_audio(
    meeting=args.meeting,
    audio_dir=audio_dir,
    array=1,
    n_mics=8,
)

print("AMI MULTICHANNEL AUDIO VALIDATION")
print("=" * 45)

for key, value in result.items():
    print(f"{key}: {value}")

print("\nAll eight microphone headers are compatible.")