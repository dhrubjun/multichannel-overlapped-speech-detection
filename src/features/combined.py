
import numpy as np

from src.features.logmel import extract_logmel
from src.features.ipd import extract_ipd


def extract_combined_features(
    audio,
    n_frames=None,
    mask_threshold_db=-30.0,
):
    """
    Extract combined M2 spectral and spatial features.

    Parameters
    ----------
    audio : ndarray
        Eight-channel audio, shape (8, n_samples).

    n_frames : int or None
        Number of output frames.

    mask_threshold_db : float
        IPD energy-mask threshold.

    Returns
    -------
    features : ndarray
        Shape (n_frames, 2136), float32.

        Columns 0:80 contain log-Mel features.
        Columns 80:2136 contain flattened IPD features.

    frame_times : ndarray
        Frame timestamps relative to the supplied audio.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 2 or audio.shape[0] != 8:
        raise ValueError(
            "Expected audio shape (8, n_samples)"
        )

    logmel, mel_times = extract_logmel(
        audio[0],
        n_frames=n_frames,
    )

    ipd, ipd_times = extract_ipd(
        audio,
        n_frames=n_frames,
        mask_threshold_db=mask_threshold_db,
    )

    if not np.array_equal(mel_times, ipd_times):
        raise ValueError("Feature timestamps do not match")

    ipd_flat = ipd.reshape(len(ipd), -1)

    features = np.concatenate(
        (logmel, ipd_flat),
        axis=1,
    ).astype(np.float32)

    if features.shape[1] != 2136:
        raise ValueError(
            f"Unexpected feature dimension: {features.shape[1]}"
        )

    if not np.isfinite(features).all():
        raise ValueError("Non-finite combined features")

    return features, mel_times