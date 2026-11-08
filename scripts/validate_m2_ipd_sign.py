
"""
M2-D2: Validate IPD phase sign convention.

Generate a synthetic broadband signal, delay one microphone
channel by a known number of samples, and verify that:

1. Time-domain cross-correlation recovers the correct delay.
2. The IPD extractor follows the expected phase convention.
3. Phase residuals are reasonably small for a finite-window STFT.

This test does not validate real microphone geometry.
"""

import numpy as np

from scripts.analyze_m2_channel_phase import (
    estimate_pair_delay,
)
from src.features.ipd import extract_ipd


SAMPLE_RATE = 16000
DURATION_SECONDS = 1.0
KNOWN_DELAY = 4
SEED = 42

# The finite-window STFT of broadband noise does not
# produce perfect phase rotation in every frequency bin.
MAX_MEAN_PHASE_ERROR_DEG = 5.0


def validate_sinusoidal_phase():
    """Validate IPD at a known FFT-bin frequency."""

    fs = SAMPLE_RATE
    frequency = 1000.0
    delay = KNOWN_DELAY

    n_samples = int(fs * DURATION_SECONDS)
    t = np.arange(n_samples) / fs

    source = np.sin(
        2.0 * np.pi * frequency * t
    ).astype(np.float32)

    delayed = np.zeros_like(source)
    delayed[delay:] = source[:-delay]

    audio = np.zeros(
        (8, n_samples),
        dtype=np.float32,
    )

    audio[0] = delayed
    audio[4] = source

    features, _ = extract_ipd(
        audio,
        sample_rate=fs,
        mic_pairs=[(0, 4)],
        mask_threshold_db=None,
    )

    fft_bin = int(
        round(frequency * 512 / fs)
    )

    assert fft_bin == 32

    measured_phase = np.angle(
        features[5:-5, 0, fft_bin, 0]
        + 1j * features[5:-5, 0, fft_bin, 1]
    )

    expected_phase = (
        -2.0 * np.pi
        * frequency * delay / fs
    )

    residual = np.angle(
        np.exp(
            1j * (measured_phase - expected_phase)
        )
    )

    measured_mean_deg = np.degrees(
        np.angle(np.mean(np.exp(1j * measured_phase)))
    )

    error_deg = np.degrees(
        np.abs(residual)
    )

    print("\nSINGLE-FREQUENCY PHASE VALIDATION")
    print("-" * 65)

    print("Frequency:", frequency, "Hz")
    print("FFT bin:", fft_bin)
    print("Applied delay:", delay, "samples")
    print(
        f"Expected phase: "
        f"{np.degrees(expected_phase):.4f} degrees"
    )
    print(
        f"Measured phase: "
        f"{measured_mean_deg:.4f} degrees"
    )
    print(
        f"Mean absolute phase error: "
        f"{error_deg.mean():.6f} degrees"
    )

    assert error_deg.mean() < 0.1, (
        "Sinusoidal IPD phase error exceeds 0.1 degrees"
    )

    print("SINUSOIDAL IPD VALIDATION: PASS")



def main():
    print("\nM2-D2: SYNTHETIC IPD SIGN VALIDATION")
    print("=" * 65)

    # --------------------------------------------------
    # 1. Generate reproducible synthetic audio
    # --------------------------------------------------

    rng = np.random.default_rng(SEED)

    n_samples = int(
        SAMPLE_RATE * DURATION_SECONDS
    )

    source = rng.standard_normal(
        n_samples
    ).astype(np.float32)

    # Delay the source by KNOWN_DELAY samples.
    delayed = np.zeros_like(source)

    delayed[KNOWN_DELAY:] = (
        source[:-KNOWN_DELAY]
    )

    # --------------------------------------------------
    # 2. Validate cross-correlation delay
    # --------------------------------------------------

    estimated_lag, correlation = (
        estimate_pair_delay(
            delayed,
            source,
        )
    )

    print("\nTIME-DOMAIN VALIDATION")
    print("-" * 65)

    print("Expected lag:", KNOWN_DELAY)
    print("Estimated lag:", estimated_lag)
    print(f"Correlation: {correlation:.6f}")

    assert estimated_lag == KNOWN_DELAY, (
        f"Incorrect delay: expected {KNOWN_DELAY}, "
        f"obtained {estimated_lag}"
    )

    assert correlation > 0.999, (
        f"Unexpectedly low correlation: {correlation}"
    )

    print("Time-domain validation: PASS")

    # --------------------------------------------------
    # 3. Prepare eight-channel synthetic audio
    # --------------------------------------------------

    audio = np.zeros(
        (8, n_samples),
        dtype=np.float32,
    )

    audio[0] = delayed
    audio[4] = source

    # --------------------------------------------------
    # 4. Extract IPD using the actual M2 implementation
    # --------------------------------------------------

    features, times = extract_ipd(
        audio,
        sample_rate=SAMPLE_RATE,
        mic_pairs=[(0, 4)],
        mask_threshold_db=None,
    )

    print("\nIPD EXTRACTION")
    print("-" * 65)

    print("Feature shape:", features.shape)
    print("Number of timestamps:", len(times))

    expected_shape = (
        round(n_samples / 160),
        1,
        257,
        2,
    )

    assert features.shape == expected_shape

    # Reconstruct the measured phase from cosine and sine.
    measured_phase = np.angle(
        features[:, 0, :, 0]
        + 1j * features[:, 0, :, 1]
    )

    # --------------------------------------------------
    # 5. Calculate expected phase
    # --------------------------------------------------

    frequencies = np.fft.rfftfreq(
        512,
        d=1.0 / SAMPLE_RATE,
    )

    # C_ij = X_i * conj(X_j).
    # Channel i is delayed relative to channel j.
    expected_phase = (
        -2.0
        * np.pi
        * frequencies
        * KNOWN_DELAY
        / SAMPLE_RATE
    )

    # Circular phase residual in [-pi, pi].
    residual = np.angle(
        np.exp(
            1j * (
                measured_phase
                - expected_phase[None, :]
            )
        )
    )

    # --------------------------------------------------
    # 6. Exclude boundary frames and DC/Nyquist bins
    # --------------------------------------------------

    interior_residual = residual[
        2:-2,
        1:-1,
    ]

    absolute_error_deg = np.degrees(
        np.abs(interior_residual)
    )

    mean_error = float(
        np.mean(absolute_error_deg)
    )

    median_error = float(
        np.median(absolute_error_deg)
    )

    p95_error = float(
        np.percentile(
            absolute_error_deg,
            95,
        )
    )

    max_error = float(
        np.max(absolute_error_deg)
    )

    # --------------------------------------------------
    # 7. Report phase-validation results
    # --------------------------------------------------

    print("\nFREQUENCY-DOMAIN VALIDATION")
    print("-" * 65)

    print(
        f"Mean phase residual: "
        f"{mean_error:.4f} degrees"
    )

    print(
        f"Median phase residual: "
        f"{median_error:.4f} degrees"
    )

    print(
        f"95th percentile residual: "
        f"{p95_error:.4f} degrees"
    )

    print(
        f"Maximum phase residual: "
        f"{max_error:.4f} degrees"
    )

    print(
        f"Allowed mean residual: "
        f"{MAX_MEAN_PHASE_ERROR_DEG:.2f} degrees"
    )

    assert np.isfinite(
        absolute_error_deg
    ).all(), "Non-finite phase residuals"

    assert mean_error < MAX_MEAN_PHASE_ERROR_DEG, (
        f"Mean phase residual too large: "
        f"{mean_error:.4f} degrees"
    )

    print("\nFrequency-domain validation: PASS")

    validate_sinusoidal_phase()

    # --------------------------------------------------
    # 8. Final result
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("IPD SIGN CONVENTION: PASS")
    print("=" * 65)


if __name__ == "__main__":
    main()