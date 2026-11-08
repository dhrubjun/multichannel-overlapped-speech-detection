
"""
M2-D2.2: Microphone delays in annotated single-speaker regions.

Uses ES2008a training labels to select valid class-1 windows.
Analyzes all 28 microphone pairs using the previously
implemented normalized cross-correlation estimator.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.analyze_m2_channel_phase import (
    estimate_pair_delay,
)

MEETING = "ES2008a"
AUDIO_DIR = Path("../amicorpus") / MEETING / "audio"
LABEL_PATH = Path(
    f"outputs/labels/{MEETING}_frame_labels.npz"
)

FS = 16000
WINDOW_SECONDS = 2.0
FRAME_HOP_SECONDS = 0.01
MAX_WINDOWS = 12
MIN_SEPARATION_SECONDS = 20.0

SPEED_OF_SOUND = 343.0
MAX_DIAMETER_M = 0.20

MAX_PHYSICAL_LAG = (
    MAX_DIAMETER_M / SPEED_OF_SOUND * FS
)

PAIRS = list(combinations(range(8), 2))


def select_windows():
    with np.load(LABEL_PATH) as data:
        labels = data["labels"]
        valid = data["valid_mask"]
        times = data["frame_times"]

    single = (labels == 1) & valid

    frames_per_window = int(
        round(WINDOW_SECONDS / FRAME_HOP_SECONDS)
    )

    # Require 100 ms of class-1 speech on either side.
    guard = 10

    # Running sum identifies continuous class-1 regions.
    kernel = np.ones(
        frames_per_window + 2 * guard,
        dtype=np.int32,
    )

    counts = np.convolve(
        single.astype(np.int32),
        kernel,
        mode="valid",
    )

    eligible = np.flatnonzero(
        counts == len(kernel)
    ) + guard

    selected = []
    min_separation_frames = int(
        MIN_SEPARATION_SECONDS / FRAME_HOP_SECONDS
    )

    # Deterministic selection distributed across meeting.
    if len(eligible):
        
        # Select candidates across the full meeting timeline,
        # instead of stopping after the first 12 eligible regions.
        target_frames = np.linspace(
            eligible[0],
            eligible[-1],
            MAX_WINDOWS,
        )

        for target in target_frames:
            # Consider eligible regions closest to this
            # position in the recording.
            candidates = eligible[
                np.argsort(np.abs(eligible - target))
            ]

            for candidate in candidates:
                start_frame = int(candidate)

                if all(
                    abs(start_frame - previous)
                    >= min_separation_frames
                    for previous in selected
                ):
                    selected.append(start_frame)
                    break

        selected.sort()


    if not selected:
        raise RuntimeError(
            "No suitable single-speaker windows found"
        )

    # Frame times are centered at 5, 15, ... ms.
    starts = [
        float(times[i] - FRAME_HOP_SECONDS / 2)
        for i in selected
    ]

    return starts


def load_window(start_seconds):
    start_sample = int(round(start_seconds * FS))
    length_samples = int(round(WINDOW_SECONDS * FS))

    channels = []

    for mic in range(1, 9):
        path = AUDIO_DIR / (
            f"{MEETING}.Array1-{mic:02d}.wav"
        )

        with sf.SoundFile(path) as audio_file:
            if audio_file.samplerate != FS:
                raise ValueError(
                    f"Unexpected sample rate: {path}"
                )

            if audio_file.channels != 1:
                raise ValueError(
                    f"Expected mono recording: {path}"
                )

            audio_file.seek(start_sample)

            signal = audio_file.read(
                frames=length_samples,
                dtype="float32",
                always_2d=False,
            )

        if len(signal) != length_samples:
            raise RuntimeError(
                f"Incomplete audio window: {path}"
            )

        channels.append(signal)

    return np.stack(channels)


def main():
    starts = select_windows()

    print("\nM2-D2.2: SINGLE-SPEAKER DELAY AUDIT")
    print("=" * 72)
    print("Meeting:", MEETING)
    print("Windows selected:", len(starts))
    print("Window duration:", WINDOW_SECONDS, "s")
    print(
        "Maximum physical lag:",
        f"{MAX_PHYSICAL_LAG:.2f} samples"
    )

    all_lags = []
    closure_errors = []

    for index, start in enumerate(starts, 1):
        audio = load_window(start)

        delays = {}
        correlations = {}

        for i, j in PAIRS:
            lag, corr = estimate_pair_delay(
                audio[i],
                audio[j],
            )

            delays[(i, j)] = lag
            correlations[(i, j)] = corr

        all_lags.append(
            [delays[pair] for pair in PAIRS]
        )

        # Check one illustrative three-microphone
        # relationship using the same lag convention.
        closure = (
            delays[(0, 1)]
            + delays[(1, 5)]
            - delays[(0, 5)]
        )

        closure_errors.append(closure)

        print(
            f"\nWindow {index:02d}: "
            f"{start:.2f} to "
            f"{start + WINDOW_SECONDS:.2f} s"
        )

        for i, j in [
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]:
            print(
                f"  Pair {i+1:02d}-{j+1:02d}: "
                f"lag={delays[(i, j)]:+3d}, "
                f"corr={correlations[(i, j)]:.3f}"
            )

        print(
            "  Closure error (01,02,06):",
            f"{closure:+d} samples"
        )

    lag_matrix = np.asarray(all_lags)

    print("\nALL-PAIR SUMMARY")
    print("=" * 72)
    print(
        "Pair    Median lag   Min lag   Max lag  "
        "Outside physical bound"
    )

    for k, (i, j) in enumerate(PAIRS):
        values = lag_matrix[:, k]

        outside = int(
            np.sum(
                np.abs(values) > MAX_PHYSICAL_LAG
            )
        )

        print(
            f"{i+1:02d}-{j+1:02d}"
            f"{np.median(values):>12.1f}"
            f"{np.min(values):>10d}"
            f"{np.max(values):>10d}"
            f"{outside:>17d}"
        )

    closure_errors = np.asarray(closure_errors)

    print("\nCONSISTENCY SUMMARY")
    print("=" * 72)
    print(
        "Median absolute closure error:",
        f"{np.median(np.abs(closure_errors)):.2f}",
        "samples"
    )
    print(
        "Maximum absolute closure error:",
        int(np.max(np.abs(closure_errors))),
        "samples"
    )

    print("\nSINGLE-SPEAKER DELAY AUDIT COMPLETED")


if __name__ == "__main__":
    main()