
"""
M2-D2: Preliminary AMI microphone-pair delay audit.

Analyze all 28 pairs of Array 1 microphones using
short audio excerpts and normalized cross-correlation.

This is a diagnostic, not a verified geometry estimator.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags


MEETING = "ES2008a"

AUDIO_DIR = (
    Path("../amicorpus") / MEETING / "audio"
)

SAMPLE_RATE = 16000
START_SECONDS = 300
DURATION_SECONDS = 5

SPEED_OF_SOUND = 343.0
MAX_ARRAY_DIAMETER_M = 0.20

MAX_PHYSICAL_DELAY_SAMPLES = (
    MAX_ARRAY_DIAMETER_M
    / SPEED_OF_SOUND
    * SAMPLE_RATE
)

# Search slightly beyond physical bound to detect
# suspicious delays or reverberation effects.
MAX_SEARCH_LAG = 16


def load_audio():
    signals = []

    for mic in range(1, 9):
        path = AUDIO_DIR / (
            f"{MEETING}.Array1-{mic:02d}.wav"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        info = sf.info(path)

        if info.samplerate != SAMPLE_RATE:
            raise ValueError(
                f"Unexpected sample rate: {path}"
            )

        start = int(START_SECONDS * SAMPLE_RATE)
        end = int(
            (START_SECONDS + DURATION_SECONDS)
            * SAMPLE_RATE
        )

        if end > info.frames:
            raise ValueError(
                f"Requested segment exceeds {path}"
            )

        signal, sr = sf.read(
            path,
            start=start,
            stop=end,
            dtype="float32",
        )

        if signal.ndim != 1:
            raise ValueError(
                f"Expected mono WAV: {path}"
            )

        signals.append(signal)

    return np.stack(signals)


def estimate_pair_delay(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)

    # Limit lag search, but normalize each lag by
    # the corresponding signal energies.
    corr = correlate(
        x,
        y,
        mode="full",
        method="fft",
    )

    lags = correlation_lags(
        len(x),
        len(y),
        mode="full",
    )

    selected = np.abs(lags) <= MAX_SEARCH_LAG

    lag_values = lags[selected]
    correlation = corr[selected]

    # Overlap energies for each candidate lag.
    x2 = np.cumsum(
        np.r_[0.0, x.astype(np.float64) ** 2]
    )
    y2 = np.cumsum(
        np.r_[0.0, y.astype(np.float64) ** 2]
    )

    normalized = np.zeros(
        len(lag_values),
        dtype=np.float64,
    )

    n = len(x)

    for k, lag in enumerate(lag_values):
        if lag >= 0:
            energy_x = x2[n] - x2[lag]
            energy_y = y2[n - lag]
        else:
            shift = -int(lag)
            energy_x = x2[n - shift]
            energy_y = y2[n] - y2[shift]

        denominator = np.sqrt(
            energy_x * energy_y
        )

        if denominator > 1e-12:
            normalized[k] = (
                correlation[k] / denominator
            )

    index = int(
        np.argmax(np.abs(normalized))
    )

    return (
        int(lag_values[index]),
        float(normalized[index]),
    )


def main():
    audio = load_audio()

    print("\nM2-D2: MICROPHONE PAIR DELAY AUDIT")
    print("=" * 72)

    print("Meeting:", MEETING)
    print("Segment:", START_SECONDS, "to",
          START_SECONDS + DURATION_SECONDS, "s")
    print("Audio shape:", audio.shape)
    print(
        "Maximum physical delay:",
        f"{MAX_PHYSICAL_DELAY_SAMPLES:.2f} samples",
    )

    print("\nMic pair     Lag(samples)  Lag(ms)   Correlation")
    print("-" * 58)

    for i, j in combinations(range(8), 2):
        lag, corr = estimate_pair_delay(
            audio[i],
            audio[j],
        )

        delay_ms = (
            lag / SAMPLE_RATE * 1000
        )

        print(
            f"{i+1:02d}-{j+1:02d}"
            f"{lag:>15d}"
            f"{delay_ms:>10.3f}"
            f"{corr:>14.4f}"
        )

    print("\nPAIR DELAY AUDIT COMPLETED")


if __name__ == "__main__":
    main()