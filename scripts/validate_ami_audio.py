
from pathlib import Path
import argparse

from src.data.ami_audio import validate_array_audio


AMI_ROOT = Path(
    r"C:\Master 1st Sem\Deep Learning\Project\amicorpus"
)

# Training and development meetings only.
# ES2004a remains held out.
MEETINGS = [
    "ES2008a",
    "ES2008b",
    "ES2008c",
    "ES2003a",
]


def main():
    parser = argparse.ArgumentParser(
        description="Validate AMI Array 1 recordings"
    )
    parser.add_argument(
        "--meeting",
        choices=MEETINGS + ["all"],
        default="all",
    )
    args = parser.parse_args()

    meetings = (
        MEETINGS if args.meeting == "all"
        else [args.meeting]
    )

    print("\nM2.1: AMI MULTICHANNEL AUDIO AUDIT")
    print("=" * 60)

    all_passed = True

    for meeting in meetings:
        print(f"\nMeeting: {meeting}")
        print("-" * 45)

        audio_dir = AMI_ROOT / meeting / "audio"

        try:
            result = validate_array_audio(
                meeting=meeting,
                audio_dir=audio_dir,
                array=1,
                n_mics=8,
            )

            for key, value in result.items():
                print(f"{key}: {value}")

            if result["sample_rate"] != 16000:
                raise ValueError("Expected 16 kHz")

            if result["sample_width_bits"] != 16:
                raise ValueError("Expected 16-bit PCM")

            print("STATUS: PASS")

        except (FileNotFoundError, ValueError, OSError) as error:
            all_passed = False
            print(f"STATUS: FAIL ({error})")

    print("\n" + "=" * 60)

    if all_passed:
        print("M2.1 HEADER AUDIT: PASSED")
    else:
        print("M2.1 HEADER AUDIT: FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()