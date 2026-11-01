
import numpy as np
from scipy.signal.windows import hann


# Candidate pairs; physical numbering must still be verified.
DEFAULT_PAIRS = [
    (0, 4),  # Array1-01 and Array1-05
    (1, 5),  # Array1-02 and Array1-06
    (2, 6),  # Array1-03 and Array1-07
    (3, 7),  # Array1-04 and Array1-08
]


def extract_ipd(
    audio,
    sample_rate=16000,
    n_frames=None,
    mic_pairs=DEFAULT_PAIRS,
):
    """
    Extract sine and cosine IPD from multichannel audio.

    Input:
        audio: (8, n_samples), float audio samples

    Output:
        features: (n_frames, n_pairs, 257, 2)
        times: (n_frames,)

    Last dimension:
        0 = cos(IPD)
        1 = sin(IPD)

    Framing matches the existing M1 log-Mel extractor.
    """
    if sample_rate != 16000:
        raise ValueError("Expected 16 kHz audio")

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 2 or audio.shape[0] != 8:
        raise ValueError(
            "Expected audio shape (8, n_samples)"
        )

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains non-finite values")

    if any(
        i < 0 or i >= 8 or j < 0 or j >= 8 or i == j
        for i, j in mic_pairs
    ):
        raise ValueError("Invalid microphone pair")

    if not mic_pairs:
        raise ValueError("At least one pair is required")

    window_length = 400
    hop_length = 160
    n_fft = 512

    if n_frames is None:
        n_frames = round(audio.shape[1] / hop_length)

    if n_frames < 1:
        raise ValueError("No frames requested")

    left_padding = 120
    right_padding = max(
        0,
        (n_frames - 1) * hop_length
        + window_length
        - left_padding
        - audio.shape[1],
    )

    padded = np.pad(
        audio,
        ((0, 0), (left_padding, right_padding)),
        mode="constant",
    )

    window = hann(
        window_length, sym=False
    ).astype(np.float32)

    n_pairs = len(mic_pairs)
    n_bins = n_fft // 2 + 1

    features = np.empty(
        (n_frames, n_pairs, n_bins, 2),
        dtype=np.float32,
    )

    batch_size = 256

    for start in range(0, n_frames, batch_size):
        end = min(start + batch_size, n_frames)

        indices = (
            np.arange(start, end)[:, None] * hop_length
            + np.arange(window_length)[None, :]
        )

        # Shape: (8, batch_frames, 400)
        frames = padded[:, indices]
        frames = frames * window

        # Shape: (8, batch_frames, 257)
        spectrum = np.fft.rfft(
            frames, n=n_fft, axis=-1
        )

        for pair_index, (i, j) in enumerate(mic_pairs):
            cross = spectrum[i] * np.conj(spectrum[j])
            magnitude = np.abs(cross)

            # Undefined phase in negligible-energy bins
            # is represented by zero.
            valid = magnitude > 1e-10

            cos_ipd = np.divide(
                cross.real,
                magnitude,
                out=np.zeros_like(magnitude),
                where=valid,
            )

            sin_ipd = np.divide(
                cross.imag,
                magnitude,
                out=np.zeros_like(magnitude),
                where=valid,
            )

            features[
                start:end, pair_index, :, 0
            ] = cos_ipd.astype(np.float32)

            features[
                start:end, pair_index, :, 1
            ] = sin_ipd.astype(np.float32)

    times = (
        np.arange(n_frames, dtype=np.float64) + 0.5
    ) * hop_length / sample_rate

    return features, times