
import numpy as np
from scipy.signal.windows import hann


# Candidate microphone pairs.
# Physical microphone ordering still needs verification.
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
    mic_pairs=None,
    mask_threshold_db=None,
):
    """
    Extract sine and cosine IPD from eight-channel audio.

    Parameters
    ----------
    audio : ndarray
        Audio samples with shape (8, n_samples).

    sample_rate : int
        Sampling rate. Must be 16000 Hz.

    n_frames : int or None
        Number of output frames. If None, determined
        from the audio length.

    mic_pairs : list or None
        Microphone index pairs (zero-based).
        Default: (0,4), (1,5), (2,6), (3,7).

    mask_threshold_db : float or None
        Optional relative energy threshold in dB.
        Example: -30.0
        None disables energy masking.

    Returns
    -------
    features : ndarray
        Shape (n_frames, n_pairs, 257, 2).

        Last dimension:
            0 = cos(IPD)
            1 = sin(IPD)

        Masked or undefined phase bins are set to zero.

    times : ndarray
        Frame timestamps in seconds.

    STFT configuration matches M1:
        Window: 25 ms (400 samples)
        Hop: 10 ms (160 samples)
        FFT: 512 points
        Window function: periodic Hann
    """

    # --------------------------------------------------
    # 1. Validate input
    # --------------------------------------------------

    if sample_rate != 16000:
        raise ValueError("Expected 16 kHz audio")

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 2 or audio.shape[0] != 8:
        raise ValueError(
            "Expected audio shape (8, n_samples)"
        )

    if not np.all(np.isfinite(audio)):
        raise ValueError(
            "Audio contains non-finite values"
        )

    if mic_pairs is None:
        mic_pairs = DEFAULT_PAIRS

    if len(mic_pairs) == 0:
        raise ValueError(
            "At least one microphone pair is required"
        )

    for i, j in mic_pairs:
        if not (0 <= i < 8 and 0 <= j < 8):
            raise ValueError(
                "Invalid microphone index"
            )

        if i == j:
            raise ValueError(
                "Microphone pair cannot use the same channel"
            )

    if mask_threshold_db is not None:
        if not np.isfinite(mask_threshold_db):
            raise ValueError(
                "Invalid mask threshold"
            )

        if mask_threshold_db > 0:
            raise ValueError(
                "Mask threshold must be <= 0 dB"
            )

    # --------------------------------------------------
    # 2. STFT parameters
    # --------------------------------------------------

    window_length = 400
    hop_length = 160
    n_fft = 512

    n_samples = audio.shape[1]

    if n_frames is None:
        n_frames = round(n_samples / hop_length)

    if not isinstance(n_frames, (int, np.integer)):
        raise ValueError(
            "n_frames must be an integer"
        )

    if n_frames < 1:
        raise ValueError(
            "No frames requested"
        )

    # --------------------------------------------------
    # 3. Apply the same padding as M1
    # --------------------------------------------------

    left_padding = 120

    right_padding = max(
        0,
        (n_frames - 1) * hop_length
        + window_length
        - left_padding
        - n_samples,
    )

    padded = np.pad(
        audio,
        ((0, 0), (left_padding, right_padding)),
        mode="constant",
    )

    window = hann(
        window_length,
        sym=False,
    ).astype(np.float32)

    # --------------------------------------------------
    # 4. Allocate output
    # --------------------------------------------------

    n_pairs = len(mic_pairs)
    n_bins = n_fft // 2 + 1

    features = np.empty(
        (n_frames, n_pairs, n_bins, 2),
        dtype=np.float32,
    )

    # Process in batches to limit temporary memory use.
    batch_size = 256

    # --------------------------------------------------
    # 5. Compute STFT and IPD
    # --------------------------------------------------

    for start in range(0, n_frames, batch_size):
        end = min(
            start + batch_size,
            n_frames,
        )

        indices = (
            np.arange(start, end)[:, None] * hop_length
            + np.arange(window_length)[None, :]
        )

        # Shape: (8, batch_frames, 400)
        frames = padded[:, indices]

        frames = frames * window

        # Shape: (8, batch_frames, 257)
        spectrum = np.fft.rfft(
            frames,
            n=n_fft,
            axis=-1,
        )

        # ----------------------------------------------
        # Process each microphone pair
        # ----------------------------------------------

        for pair_index, (i, j) in enumerate(mic_pairs):

            # Cross-spectrum:
            # C_ij = X_i * conjugate(X_j)
            cross = (
                spectrum[i]
                * np.conj(spectrum[j])
            )

            magnitude = np.abs(cross)

            # Phase is undefined for negligible energy.
            valid = magnitude > 1e-10

            # ------------------------------------------
            # Cosine and sine of IPD
            # ------------------------------------------

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

            # ------------------------------------------
            # Optional energy-based masking
            # ------------------------------------------

            if mask_threshold_db is not None:

                # Cross-spectrum magnitude equals
                # the product of spectral magnitudes.
                magnitude_product = magnitude

                # Maximum magnitude within each frame.
                frame_max = magnitude_product.max(
                    axis=1,
                    keepdims=True,
                )

                # Relative energy threshold:
                # 10*log10(P/Pmax) >= threshold_db
                ratio = 10.0 ** (
                    mask_threshold_db / 10.0
                )

                # Avoid masking decisions in silent
                # frames and negligible-energy bins.
                energy_mask = (
                    (frame_max > 1e-10)
                    & (
                        magnitude_product
                        >= frame_max * ratio
                    )
                    & valid
                )

                # Suppress unreliable phase bins.
                cos_ipd = np.where(
                    energy_mask,
                    cos_ipd,
                    0.0,
                )

                sin_ipd = np.where(
                    energy_mask,
                    sin_ipd,
                    0.0,
                )

            # ------------------------------------------
            # Store features
            # ------------------------------------------

            features[
                start:end,
                pair_index,
                :,
                0,
            ] = cos_ipd.astype(np.float32)

            features[
                start:end,
                pair_index,
                :,
                1,
            ] = sin_ipd.astype(np.float32)

    # --------------------------------------------------
    # 6. Frame timestamps aligned with M1
    # --------------------------------------------------

    times = (
        np.arange(n_frames, dtype=np.float64) + 0.5
    ) * hop_length / sample_rate

    return features, times