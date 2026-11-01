
from pathlib import Path
import argparse
import wave

import numpy as np
from scipy.signal import correlate, correlation_lags

from src.data.ami_audio import validate_array_audio


AMI_ROOT = Path(
    r"C:\Master 1st Sem\Deep Learning\Project\amicorpus"
)

MEETINGS = ["ES2008a", "ES2008b", "ES2008c", "ES2003a"]

MIC_PAIRS = [(1, 5), (2, 6), (3, 7), (4, 8)]

# Use short clips rather than reading entire recordings.
CLIP_SECONDS = 1.0
MAX_LAG_MS = 2.0

# Sample different parts of each meeting.
POSITION_FRACTIONS = [0.10, 0.30, 0.50, 0.70, 0.90]


def read_clip(meeting, mic, start_seconds):
    path = (
        AMI_ROOT / meeting / "audio"
        / f"{meeting}.Array1-{mic:02d}.wav"
    )

    with wave.open(str(path), "rb") as wav:
        fs = wav.getframerate()
        start = round(start_seconds * fs)
        count = round(CLIP_SECONDS * fs)

        if start < 0 or start + count > wav.getnframes():
            raise ValueError(f"Invalid clip position: {path}")

        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(f"Expected mono 16-bit PCM: {path}")

        wav.setpos(start)
        raw = wav.readframes(count)

    audio = np.frombuffer(raw, dtype="<i2")
    return audio.astype(np.float64) / 32768.0, fs


def estimate_delay(x1, x2, fs):
    # Remove DC offsets.
    x1 = x1 - np.mean(x1)
    x2 = x2 - np.mean(x2)

    rms1 = np.sqrt(np.mean(x1 ** 2))
    rms2 = np.sqrt(np.mean(x2 ** 2))

    if rms1 < 1e-5 or rms2 < 1e-5:
        return None, None

    # Normalize signal amplitudes.
    x1 = x1 / rms1
    x2 = x2 / rms2

    # FFT-based cross-correlation.
    corr = correlate(x2, x1, mode="full", method="fft")
    lags = correlation_lags(
        len(x2), len(x1), mode="full"
    )

    max_lag = round(MAX_LAG_MS * fs / 1000)
    selected = np.abs(lags) <= max_lag

    corr = corr[selected]
    lags = lags[selected]

    # Compensate for fewer overlapping samples at nonzero lag.
    overlap_counts = len(x1) - np.abs(lags)
    corr = corr / overlap_counts

    best = np.argmax(np.abs(corr))

    lag = int(lags[best])
    peak = float(corr[best])

    return lag, peak


def analyze_meeting(meeting):
    audio_dir = AMI_ROOT / meeting / "audio"

    info = validate_array_audio(
        meeting=meeting,
        audio_dir=audio_dir,
        array=1,
        n_mics=8,
    )

    fs = info["sample_rate"]
    duration = info["duration"]

    if fs != 16000 or info["sample_width_bits"] != 16:
        raise ValueError("Expected 16 kHz, 16-bit PCM")

    print(f"\nMEETING: {meeting}")
    print(f"Duration: {duration:.3f} seconds")
    print(f"Sample rate: {fs} Hz")
    print("=" * 66)

    for mic1, mic2 in MIC_PAIRS:
        print(f"\nMicrophone pair: {mic1:02d}-{mic2:02d}")
        print("-" * 66)

        print(
            f"{'Start(s)':>10}"
            f"{'Lag(samples)':>15}"
            f"{'Delay(ms)':>15}"
            f"{'|Peak|':>12}"
        )

        for fraction in POSITION_FRACTIONS:
            start = (duration - CLIP_SECONDS) * fraction

            x1, fs1 = read_clip(meeting, mic1, start)
            x2, fs2 = read_clip(meeting, mic2, start)

            if fs1 != fs2:
                raise ValueError("Sample rates differ")

            lag, peak = estimate_delay(x1, x2, fs1)

            if lag is None:
                print(f"{start:10.1f}   LOW SIGNAL")
                continue

            delay_ms = 1000.0 * lag / fs1

            print(
                f"{start:10.1f}"
                f"{lag:15d}"
                f"{delay_ms:15.4f}"
                f"{abs(peak):12.4f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meeting",
        choices=MEETINGS + ["all"],
        default="ES2008a",
    )
    args = parser.parse_args()

    meetings = (
        MEETINGS if args.meeting == "all"
        else [args.meeting]
    )

    print("\nM2.1: MULTICHANNEL TIMING ANALYSIS")
    print("Positive lag: mic2 is delayed relative to mic1")
    print("Negative lag: mic2 is advanced relative to mic1")
    print("Lag estimates include acoustic propagation effects.")

    for meeting in meetings:
        analyze_meeting(meeting)


if __name__ == "__main__":
    main()