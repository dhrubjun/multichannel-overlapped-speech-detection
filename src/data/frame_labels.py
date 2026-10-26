
import numpy as np


def generate_labels(intervals, duration, hop=0.01):
    """
    Generate frame-level labels from speaker intervals.

    Parameters
    ----------
    intervals : list
        One list of (start, end) intervals per speaker.

    duration : float
        Total recording duration in seconds.

    hop : float
        Time between consecutive frame centers.

    Returns
    -------
    labels : ndarray
        0 = no annotated speech
        1 = one speaker
        2 = two or more speakers

    activity : ndarray
        Boolean speaker activity matrix.

    frame_times : ndarray
        Center time of every frame.
    """

    if duration <= 0 or hop <= 0:
        raise ValueError("Duration and hop must be positive")

    n_frames = round(duration / hop)

    frame_times = (
        np.arange(n_frames) + 0.5
    ) * hop

    activity = np.zeros(
        (len(intervals), n_frames),
        dtype=bool
    )

    for speaker_index, speaker_intervals in enumerate(intervals):

        for start, end in speaker_intervals:

            if not 0 <= start < end <= duration:
                raise ValueError(
                    f"Invalid interval: {start}, {end}"
                )

            left = np.searchsorted(
                frame_times, start, side="left"
            )

            right = np.searchsorted(
                frame_times, end, side="left"
            )

            activity[speaker_index, left:right] = True

    speaker_count = activity.sum(axis=0)

    labels = np.minimum(
        speaker_count, 2
    ).astype(np.uint8)

    return labels, activity, frame_times



def generate_valid_mask(
    uncertain_intervals,
    frame_times,
):
    """
    Mark frames outside uncertain intervals as valid.

    True  = valid frame
    False = excluded frame
    """

    frame_times = np.asarray(frame_times)

    valid_mask = np.ones(
        len(frame_times),
        dtype=bool
    )

    for start, end in uncertain_intervals:

        if not (
            np.isfinite(start)
            and np.isfinite(end)
            and 0 <= start < end
        ):
            raise ValueError(
                f"Invalid uncertainty interval: {start}, {end}"
            )

        left = np.searchsorted(
            frame_times, start, side="left"
        )

        right = np.searchsorted(
            frame_times, end, side="left"
        )

        valid_mask[left:right] = False

    return valid_mask