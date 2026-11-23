
import argparse
import json
import wave
from pathlib import Path

import numpy as np

from src.features.combined import extract_combined_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AMI_ROOT = Path(
    r"C:\Master 1st Sem\Deep Learning\Project\amicorpus"
)

FS = 16000
HOP = 160
RIGHT_CONTEXT = 280

MEETINGS = (
    "ES2008a",
    "ES2008b",
    "ES2008c",
    "ES2003a",
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--meeting",
        required=True,
        help="AMI meeting ID, for example ES2002a",
    )

    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process only the first N frames for testing",
    )

    args = parser.parse_args()

    if args.chunk_frames < 1:
        parser.error("chunk-frames must be positive")

    if args.max_frames is not None and args.max_frames < 1:
        parser.error("max-frames must be positive")

    meeting = args.meeting

    audio_dir = AMI_ROOT / meeting / "audio"

    label_path = (
        PROJECT_ROOT / "outputs" / "labels"
        / f"{meeting}_frame_labels.npz"
    )

    with np.load(label_path) as data:
        labels = data["labels"]
        valid_mask = data["valid_mask"]
        label_times = data["frame_times"]

    total_frames = len(labels)

    if (
        len(valid_mask) != total_frames
        or len(label_times) != total_frames
    ):
        raise ValueError("Invalid label arrays")

    n_frames = total_frames

    if args.max_frames is not None:
        n_frames = min(
            total_frames,
            args.max_frames,
        )

    smoke_test = n_frames < total_frames

    output_dir = Path(r"D:\DeepLearning_AMI\m2_features")
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_smoke" if smoke_test else ""

    output_path = (
        output_dir
        / f"{meeting}_m2_combined{suffix}.npy"
    )

    metadata_path = output_path.with_suffix(".json")

    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}"
        )

    temp_path = output_path.with_name(
        output_path.stem + ".partial.npy"
    )

    if temp_path.exists():
        raise FileExistsError(
            f"Partial output already exists: {temp_path}"
        )

    wav_files = []

    try:
        for mic in range(1, 9):
            path = (
                audio_dir
                / f"{meeting}.Array1-{mic:02d}.wav"
            )

            wav = wave.open(str(path), "rb")

            if (
                wav.getframerate() != FS
                or wav.getnchannels() != 1
                or wav.getsampwidth() != 2
                or wav.getcomptype() != "NONE"
            ):
                wav.close()
                raise ValueError(
                    f"Invalid WAV format: {path}"
                )

            wav_files.append(wav)

        lengths = [wav.getnframes() for wav in wav_files]

        if len(set(lengths)) != 1:
            raise ValueError(
                "Microphone sample counts differ"
            )

        total_samples = lengths[0]

        if round(total_samples / HOP) != total_frames:
            raise ValueError(
                "Audio and label frame counts differ"
            )

        output = np.lib.format.open_memmap(
            temp_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_frames, 2136),
        )

        print("\nM2.3: CHUNKED FEATURE EXTRACTION")
        print("=" * 60)
        print("Meeting:", meeting)
        print("Frames:", n_frames)
        print("Chunk frames:", args.chunk_frames)
        print("Smoke test:", smoke_test)
        print("Output shape:", output.shape)

        for start in range(
            0, n_frames, args.chunk_frames
        ):
            count = min(
                args.chunk_frames,
                n_frames - start,
            )

            # For interior chunks, read one extra
            # frame before the requested output.
            # This reproduces the preceding 120
            # audio samples required by the STFT.
            context_frames = 1 if start > 0 else 0

            read_start = (
                start - context_frames
            ) * HOP

            last_frame = start + count - 1

            read_end = min(
                total_samples,
                last_frame * HOP + RIGHT_CONTEXT,
            )

            read_count = read_end - read_start

            channels = []

            for wav in wav_files:
                wav.setpos(read_start)
                raw = wav.readframes(read_count)

                signal = np.frombuffer(
                    raw,
                    dtype="<i2",
                ).astype(np.float32) / 32768.0

                if len(signal) != read_count:
                    raise ValueError(
                        "Incomplete WAV read"
                    )

                channels.append(signal)

            audio = np.stack(channels)

            features, times = extract_combined_features(
                audio,
                n_frames=count + context_frames,
                mask_threshold_db=-30.0,
            )

            # Discard the extra context frame.
            features = features[context_frames:]
            times = times[context_frames:]

            absolute_times = (
                read_start / FS + times
            )

            expected_times = label_times[
                start:start + count
            ]

            np.testing.assert_allclose(
                absolute_times,
                expected_times,
                rtol=0,
                atol=1e-9,
            )

            if features.shape != (count, 2136):
                raise ValueError(
                    "Incorrect chunk feature shape"
                )

            if not np.isfinite(features).all():
                raise ValueError(
                    "Non-finite features"
                )

            output[start:start + count] = features

            print(
                f"Frames {start:,} to "
                f"{start + count - 1:,}: PASS",
                flush=True,
            )

        output.flush()
        del output

        # Replace the partial file only after
        # successful extraction and validation.
        temp_path.replace(output_path)

        metadata = {
            "meeting": meeting,
            "n_frames": n_frames,
            "n_features": 2136,
            "sample_rate": FS,
            "hop_samples": HOP,
            "mask_threshold_db": -30.0,
            "microphone_pairs": [
                [1, 5],
                [2, 6],
                [3, 7],
                [4, 8],
            ],
            "feature_order": [
                "logmel_mic01_80",
                "flattened_ipd_2056",
            ],
            "smoke_test": smoke_test,
            "complete_meeting": not smoke_test,
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        print("\nEXTRACTION COMPLETED")
        print("Saved:", output_path)
        print("Metadata:", metadata_path)

    finally:
        for wav in wav_files:
            wav.close()


if __name__ == "__main__":
    main()