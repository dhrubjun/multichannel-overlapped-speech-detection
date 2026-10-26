
import argparse
import sys
import wave
from pathlib import Path

import numpy as np

# Allow imports when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ami_audio import validate_array_audio
from src.features.logmel import extract_logmel


DEFAULT_AMI_ROOT = Path(
    r"C:\Master 1st Sem\Deep Learning\Project\amicorpus"
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract aligned single-channel log-Mel features"
    )
    parser.add_argument("--meeting", required=True)
    parser.add_argument(
        "--ami-root",
        type=Path,
        default=DEFAULT_AMI_ROOT,
    )
    args = parser.parse_args()

    meeting = args.meeting
    mic = "01"

    audio_dir = args.ami_root / meeting / "audio"
    audio_path = audio_dir / f"{meeting}.Array1-{mic}.wav"

    label_path = (
        PROJECT_ROOT / "outputs" / "labels"
        / f"{meeting}_frame_labels.npz"
    )

    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    # Reuse the existing M0 audio validation.
    info = validate_array_audio(
        meeting=meeting,
        audio_dir=audio_dir,
        array=1,
        n_mics=8,
    )

    with np.load(label_path) as data:
        labels = data["labels"]
        valid_mask = data["valid_mask"]
        label_times = data["frame_times"]
        hop = float(data["hop_seconds"])

    if not np.isclose(hop, 0.01, rtol=0, atol=1e-10):
        raise ValueError("Expected 10 ms label hop")

    if not (
        len(labels) == len(valid_mask) == len(label_times)
    ):
        raise ValueError("Label arrays have different lengths")

    with wave.open(str(audio_path), "rb") as wav:
        fs = wav.getframerate()

        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("Expected mono 16-bit PCM")

        n_samples = wav.getnframes()

        audio = np.frombuffer(
            wav.readframes(n_samples),
            dtype="<i2",
        ).astype(np.float32) / 32768.0

    if fs != 16000:
        raise ValueError("Expected 16 kHz audio")

    expected_frames = round(n_samples / 160)

    if len(labels) != expected_frames:
        raise ValueError(
            f"Audio/label frame count mismatch: "
            f"{expected_frames} vs {len(labels)}"
        )

    features, feature_times = extract_logmel(
        audio,
        sample_rate=fs,
        n_frames=len(labels),
    )

    if features.shape != (len(labels), 80):
        raise ValueError("Unexpected feature shape")

    if not np.isfinite(features).all():
        raise ValueError("Non-finite features detected")

    if not np.allclose(
        feature_times,
        label_times,
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError("Feature/label timestamps differ")

    output_dir = PROJECT_ROOT / "outputs" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir / f"{meeting}_logmel_mic{mic}.npz"
    )

    np.savez_compressed(
        output_path,
        features=features,
        frame_times=feature_times,
        sample_rate=fs,
        hop_seconds=hop,
        window_seconds=0.025,
        n_fft=512,
        n_mels=80,
        microphone=f"Array1-{mic}",
    )

    print("\nLOG-MEL FEATURE EXTRACTION")
    print("=" * 45)
    print(f"Meeting: {meeting}")
    print(f"Microphone: Array1-{mic}")
    print(f"Audio duration: {n_samples / fs:.3f} s")
    print(f"Validated microphones: {info['microphones']}")
    print(f"Feature shape: {features.shape}")
    print(f"Feature range: {features.min():.4f}, "
          f"{features.max():.4f}")
    print(f"Valid label frames: {valid_mask.sum()}")
    print(
        "Maximum timestamp error: "
        f"{np.max(np.abs(feature_times - label_times)):.3e} s"
    )
    print(f"Saved: {output_path}")
    print("\nFEATURE VALIDATION PASSED")


if __name__ == "__main__":
    main()