
"""
M2-D2.5: Frequency-dependent IPD phase consistency.

Compare measured cross-spectral phase against the phase
predicted by time-domain microphone-pair delays.

Uses:
- The same 12 single-speaker windows as D2.2.
- The same STFT settings as src/features/ipd.py.
- The four microphone pairs used in M2.

The delay-based phase model is approximate in
reverberant or nonstationary speech recordings.
"""

import csv
from pathlib import Path

import numpy as np
from scipy.signal.windows import hann

from scripts.audit_m2_single_speaker_delays import (
    select_windows,
    load_window,
)
from scripts.analyze_m2_channel_phase import (
    estimate_pair_delay,
)

FS = 16000
WIN = 400
HOP = 160
NFFT = 512
LEFT_PAD = 120

PAIRS = [
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]

BANDS = [
    (0, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 8000),
]

COHERENCE_THRESHOLD = 0.5

OUTPUT = Path(
    "outputs/evaluation/m2_channel_phase/"
    "phase_consistency.csv"
)


def compute_stft(audio):
    """Match the STFT configuration in extract_ipd()."""

    n_samples = audio.shape[1]
    n_frames = round(n_samples / HOP)

    right_pad = max(
        0,
        (n_frames - 1) * HOP
        + WIN
        - LEFT_PAD
        - n_samples,
    )

    padded = np.pad(
        audio,
        ((0, 0), (LEFT_PAD, right_pad)),
        mode="constant",
    )

    indices = (
        np.arange(n_frames)[:, None] * HOP
        + np.arange(WIN)[None, :]
    )

    window = hann(
        WIN,
        sym=False,
    ).astype(np.float32)

    frames = padded[:, indices] * window

    return np.fft.rfft(
        frames,
        n=NFFT,
        axis=-1,
    )


def analyze_pair(spectrum, lag, i, j):
    """
    Compute frequency-dependent coherence and phase
    residual relative to a constant-delay hypothesis.
    """

    xi = spectrum[i]
    xj = spectrum[j]

    cross = xi * np.conj(xj)

    mean_cross = np.mean(cross, axis=0)

    power_i = np.mean(
        np.abs(xi) ** 2,
        axis=0,
    )
    power_j = np.mean(
        np.abs(xj) ** 2,
        axis=0,
    )

    coherence = (
        np.abs(mean_cross) ** 2
        / np.maximum(power_i * power_j, 1e-20)
    )

    coherence = np.clip(coherence, 0.0, 1.0)

    frequencies = np.fft.rfftfreq(
        NFFT,
        d=1.0 / FS,
    )

    measured_phase = np.angle(mean_cross)

    expected_phase = (
        -2.0 * np.pi
        * frequencies
        * lag / FS
    )

    # Circular phase residual in [-pi, pi].
    residual = np.angle(
        np.exp(
            1j * (
                measured_phase - expected_phase
            )
        )
    )

    valid_energy = (
        power_i * power_j > 1e-20
    )

    return (
        frequencies,
        coherence,
        residual,
        valid_energy,
    )


def main():
    starts = select_windows()
    records = []

    print("\nM2-D2.5: IPD PHASE CONSISTENCY")
    print("=" * 78)
    print("Windows:", len(starts))
    print("Sample rate:", FS)
    print("STFT:", WIN, "samples")
    print("Hop:", HOP, "samples")
    print("FFT:", NFFT)

    for window_number, start in enumerate(starts, 1):
        audio = load_window(start)

        spectrum = compute_stft(audio)

        for i, j in PAIRS:
            lag, correlation = estimate_pair_delay(
                audio[i], audio[j]
            )

            (
                frequencies,
                coherence,
                residual,
                valid_energy,
            ) = analyze_pair(
                spectrum, lag, i, j
            )

            for low, high in BANDS:
                if high == FS / 2:
                    band_mask = (
                        (frequencies >= low)
                        & (frequencies <= high)
                    )
                else:
                    band_mask = (
                        (frequencies >= low)
                        & (frequencies < high)
                    )

                reliable = (
                    band_mask
                    & valid_energy
                    & (coherence >= COHERENCE_THRESHOLD)
                )

                reliable_count = int(reliable.sum())
                total_count = int(band_mask.sum())

                if reliable_count:
                    abs_error_degrees = np.degrees(
                        np.abs(residual[reliable])
                    )

                    mean_error = float(
                        abs_error_degrees.mean()
                    )
                else:
                    mean_error = float("nan")

                records.append({
                    "window": window_number,
                    "start_seconds": float(start),
                    "pair": f"{i+1:02d}-{j+1:02d}",
                    "band": f"{low}-{high}",
                    "estimated_lag": int(lag),
                    "time_corr": float(correlation),
                    "mean_coherence": float(
                        coherence[band_mask].mean()
                    ),
                    "reliable_bins": reliable_count,
                    "total_bins": total_count,
                    "mean_abs_phase_error_deg": mean_error,
                })

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )

        writer.writeheader()
        writer.writerows(records)

    print("\nPHASE CONSISTENCY BY PAIR AND BAND")
    print("=" * 78)

    for i, j in PAIRS:
        pair_name = f"{i+1:02d}-{j+1:02d}"

        print(f"\nPair {pair_name}")
        print(
            f"{'Band (Hz)':<13}"
            f"{'Coherence':>12}"
            f"{'Reliable %':>14}"
            f"{'Phase error':>16}"
        )

        for low, high in BANDS:
            band = f"{low}-{high}"

            selected = [
                row for row in records
                if row["pair"] == pair_name
                and row["band"] == band
            ]

            coherence_mean = np.mean([
                row["mean_coherence"]
                for row in selected
            ])

            reliable_fraction = (
                sum(row["reliable_bins"] for row in selected)
                / sum(row["total_bins"] for row in selected)
            )

            errors = np.asarray([
                row["mean_abs_phase_error_deg"]
                for row in selected
            ])

            if np.any(np.isfinite(errors)):
                error_mean = float(
                    np.nanmean(errors)
                )
            else:
                error_mean = float("nan")

            print(
                f"{band:<13}"
                f"{coherence_mean:>12.4f}"
                f"{100 * reliable_fraction:>13.2f}%"
                f"{error_mean:>15.2f} deg"
            )

    assert len(records) == (
        len(starts) * len(PAIRS) * len(BANDS)
    )

    print("\nSaved:", OUTPUT)
    print("IPD PHASE CONSISTENCY COMPLETED")


if __name__ == "__main__":
    main()