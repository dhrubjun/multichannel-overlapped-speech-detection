
from pathlib import Path

from scipy.io import wavfile


DATA_ROOT = Path("../amicorpus")

OUTPUT_DIR = Path(
    "outputs/analysis/m1_false_overlap_audio"
)

REGIONS = [
    (570.53, 571.51),
    (742.00, 742.89),
    (826.27, 827.15),
]


def main():
    matches = list(
        DATA_ROOT.rglob("ES2003a.Array1-01.wav")
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one matching microphone file, "
            f"found {len(matches)}. Check DATA_ROOT."
        )

    audio_path = matches[0]

    sample_rate, audio = wavfile.read(audio_path)

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16 kHz audio, got {sample_rate}"
        )

    if audio.ndim != 1:
        raise ValueError("Expected mono microphone audio")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Source:", audio_path)
    print("Sample rate:", sample_rate)
    print()

    for start, end in REGIONS:
        # One second of context on each side.
        clip_start = max(0.0, start - 1.0)
        clip_end = min(
            len(audio) / sample_rate,
            end + 1.0,
        )

        i0 = round(clip_start * sample_rate)
        i1 = round(clip_end * sample_rate)

        filename = (
            f"ES2003a_FP_{start:.2f}_{end:.2f}.wav"
        )

        output_path = OUTPUT_DIR / filename

        wavfile.write(
            output_path,
            sample_rate,
            audio[i0:i1],
        )

        print(
            f"Saved: {output_path} | "
            f"Clip: {clip_start:.2f}-{clip_end:.2f}s"
        )

    print("\nFALSE OVERLAP CLIP EXTRACTION PASSED")


if __name__ == "__main__":
    main()