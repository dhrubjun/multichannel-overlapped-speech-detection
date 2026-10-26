
import numpy as np
from scipy.signal.windows import hann


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def make_mel_filterbank(
    sample_rate=16000,
    n_fft=512,
    n_mels=80,
):
    """Construct an HTK-scale triangular Mel filterbank."""
    mel_points = np.linspace(
        hz_to_mel(0.0),
        hz_to_mel(sample_rate / 2),
        n_mels + 2,
    )
    hz_points = mel_to_hz(mel_points)
    frequencies = np.fft.rfftfreq(n_fft, 1 / sample_rate)

    filters = np.zeros(
        (n_mels, len(frequencies)), dtype=np.float64
    )

    for m in range(n_mels):
        left, center, right = hz_points[m:m + 3]

        rising = (frequencies - left) / (center - left)
        falling = (right - frequencies) / (right - center)

        filters[m] = np.maximum(
            0.0, np.minimum(rising, falling)
        )

    return filters


def extract_logmel(
    audio,
    sample_rate=16000,
    n_frames=None,
):
    """
    Return log-Mel features and timestamps.

    Analysis window: 400 samples (25 ms)
    Hop: 160 samples (10 ms)
    FFT: 512
    Mel bands: 80

    Frame k is nominally centered at (k + 0.5) * 0.01 s.
    """
    if sample_rate != 16000:
        raise ValueError("Expected 16 kHz audio")

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1:
        raise ValueError("Expected mono audio")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains non-finite values")

    window_length = 400
    hop_length = 160
    n_fft = 512

    if n_frames is None:
        n_frames = round(len(audio) / hop_length)

    if n_frames < 1:
        raise ValueError("No feature frames requested")

    # Center each window at the 10 ms label midpoint.
    # The even-length window has a half-sample center offset.
    left_padding = 120
    right_padding = max(
        0,
        (n_frames - 1) * hop_length
        + window_length
        - left_padding
        - len(audio),
    )

    padded = np.pad(
        audio,
        (left_padding, right_padding),
        mode="constant",
    )

    analysis_window = hann(
        window_length, sym=False
    ).astype(np.float32)

    mel_filters = make_mel_filterbank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=80,
    )

    features = np.empty((n_frames, 80), dtype=np.float32)

    # Process frames in batches to keep memory usage bounded.
    batch_size = 2048

    for start in range(0, n_frames, batch_size):
        end = min(start + batch_size, n_frames)
        indices = (
            np.arange(start, end)[:, None] * hop_length
            + np.arange(window_length)[None, :]
        )

        frames = padded[indices]
        frames = frames * analysis_window

        spectrum = np.fft.rfft(
            frames, n=n_fft, axis=1
        )
        power = np.abs(spectrum) ** 2

        mel_energy = power @ mel_filters.T
        features[start:end] = np.log(
            np.maximum(mel_energy, 1e-10)
        ).astype(np.float32)

    frame_times = (
        np.arange(n_frames, dtype=np.float64) + 0.5
    ) * hop_length / sample_rate

    return features, frame_times