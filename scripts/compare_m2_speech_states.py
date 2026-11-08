
"""
M2-D2.7: Context-corrected speech-state spatial analysis.

Compare annotated single-speaker and overlap regions
across the three AMI training meetings.

Improvements:
1. STFT uses real surrounding recording context.
2. Compares original M2 and adjacent-index pairs.
3. Matches the original IPD masking convention.
4. Selects one window per eligible speech-state run.
5. Reports meeting-specific and aggregate results.
6. Validates the local IPD calculation against
   src.features.ipd.extract_ipd.

No model training. No development/test recordings.
"""

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal.windows import hann

from src.features.ipd import extract_ipd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MEETINGS = ("ES2008a", "ES2008b", "ES2008c")

AUDIO_ROOT = Path("../amicorpus")
LABEL_ROOT = Path("outputs/labels")

OUTPUT_DIR = Path(
    "outputs/evaluation/m2_channel_phase/"
    "speech_states_context"
)

FS = 16000
HOP = 160
WIN = 400
NFFT = 512
LEFT_CONTEXT = 120
RIGHT_CONTEXT = 280

WINDOW_FRAMES = 30
MAX_PER_CLASS = 20
MIN_PHASE_FRAMES = 5

MASK_DB = -30.0
SEED = 42

M2_PAIRS = [
    (0, 4), (1, 5), (2, 6), (3, 7),
]

ADJACENT_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (4, 5), (5, 6), (6, 7), (7, 0),
]

PAIR_GROUPS = {
    "m2_original": M2_PAIRS,
    "adjacent": ADJACENT_PAIRS,
}

ALL_PAIRS = M2_PAIRS + ADJACENT_PAIRS

BANDS = [
    (0, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 8000),
]

WINDOW_SAMPLES = WINDOW_FRAMES * HOP

# Total context-backed audio length:
# (N-1)*HOP + WIN
CONTEXT_SAMPLES = (
    (WINDOW_FRAMES - 1) * HOP + WIN
)

WINDOW = hann(
    WIN, sym=False
).astype(np.float32)


# --------------------------------------------------
# Region selection
# --------------------------------------------------

def find_runs(condition):
    """Return contiguous True intervals [start, end)."""

    padded = np.concatenate((
        [False],
        condition.astype(bool),
        [False],
    ))

    changes = np.flatnonzero(
        padded[1:] != padded[:-1]
    )

    return list(zip(
        changes[::2],
        changes[1::2],
    ))


def candidate_windows(labels, valid, state):
    """One central candidate from each eligible run."""

    condition = (labels == state) & valid

    runs = find_runs(condition)
    candidates = []

    for run_id, (start, end) in enumerate(runs):
        if end - start < WINDOW_FRAMES:
            continue

        first = (
            start
            + (end - start - WINDOW_FRAMES) // 2
        )

        candidates.append({
            "run_id": run_id,
            "start_frame": int(first),
            "run_frames": int(end - start),
        })

    return candidates, len(runs)


def select_distributed(candidates, n):
    """Select distinct runs distributed in time."""

    if n <= 0:
        return []

    indices = np.rint(
        np.linspace(
            0,
            len(candidates) - 1,
            n,
        )
    ).astype(int)

    assert len(set(indices)) == n

    return [candidates[i] for i in indices]


def select_meeting_windows(meeting):
    path = LABEL_ROOT / (
        f"{meeting}_frame_labels.npz"
    )

    with np.load(path) as data:
        labels = data["labels"].copy()
        valid = data["valid_mask"].copy()
        times = data["frame_times"].copy()

    assert labels.shape == valid.shape == times.shape

    expected_times = (
        np.arange(len(labels)) + 0.5
    ) * HOP / FS

    np.testing.assert_allclose(
        times,
        expected_times,
        rtol=0,
        atol=1e-4,
    )

    c1, n1 = candidate_windows(labels, valid, 1)
    c2, n2 = candidate_windows(labels, valid, 2)

    count = min(
        MAX_PER_CLASS,
        len(c1),
        len(c2),
    )

    if count == 0:
        raise RuntimeError(
            f"{meeting}: no eligible balanced windows"
        )

    chosen = []

    for state, candidates in (
        (1, c1),
        (2, c2),
    ):
        for item in select_distributed(candidates, count):
            chosen.append({
                "meeting": meeting,
                "state": state,
                **item,
            })

    print(f"\n{meeting}")
    print(
        f"  Single-speaker runs: {n1}, "
        f"eligible: {len(c1)}"
    )
    print(
        f"  Overlap runs: {n2}, "
        f"eligible: {len(c2)}"
    )
    print(f"  Selected per class: {count}")

    return chosen


# --------------------------------------------------
# Read real audio context
# --------------------------------------------------

def read_context(meeting, start_frame):
    """
    Read exactly the samples needed to reproduce the
    original STFT frames at this global frame index.

    Output shape: (8, CONTEXT_SAMPLES).
    """

    first_sample = (
        start_frame * HOP - LEFT_CONTEXT
    )

    last_sample = first_sample + CONTEXT_SAMPLES

    channels = []

    for mic in range(1, 9):
        path = (
            AUDIO_ROOT / meeting / "audio"
            / f"{meeting}.Array1-{mic:02d}.wav"
        )

        with sf.SoundFile(path) as file:
            assert file.samplerate == FS
            assert file.channels == 1

            read_start = max(0, first_sample)
            read_end = min(file.frames, last_sample)

            if read_end <= read_start:
                raise ValueError(
                    f"Invalid audio range: {path}"
                )

            file.seek(read_start)

            signal = file.read(
                frames=read_end - read_start,
                dtype="float32",
            )

        left_pad = read_start - first_sample
        right_pad = last_sample - read_end

        signal = np.pad(
            signal,
            (left_pad, right_pad),
            mode="constant",
        )

        if len(signal) != CONTEXT_SAMPLES:
            raise RuntimeError(
                "Incorrect context length"
            )

        channels.append(signal)

    return np.stack(channels)


# --------------------------------------------------
# STFT: exact frame geometry used by M2
# --------------------------------------------------

def compute_stft(context):
    indices = (
        np.arange(WINDOW_FRAMES)[:, None] * HOP
        + np.arange(WIN)[None, :]
    )

    frames = context[:, indices] * WINDOW

    spectrum = np.fft.rfft(
        frames,
        n=NFFT,
        axis=-1,
    )

    return spectrum


def masked_ipd(cross):
    """Match the original M2 IPD mask."""

    magnitude = np.abs(cross)
    valid = magnitude > 1e-10

    frame_max = magnitude.max(
        axis=1,
        keepdims=True,
    )

    threshold_ratio = 10.0 ** (
        MASK_DB / 10.0
    )

    active = (
        (frame_max > 1e-10)
        & (magnitude >= frame_max * threshold_ratio)
        & valid
    )

    phase = np.divide(
        cross,
        magnitude,
        out=np.zeros_like(cross),
        where=valid,
    )

    return np.where(
        active,
        phase,
        0.0,
    ), active


# --------------------------------------------------
# Validation against original IPD extractor
# --------------------------------------------------

def validate_feature_equivalence():
    """
    Compare context STFT IPD against the actual
    extract_ipd implementation on synthetic audio.
    """

    rng = np.random.default_rng(SEED)

    audio = rng.standard_normal(
        (8, FS),
    ).astype(np.float32)

    start_frame = 25

    first = start_frame * HOP - LEFT_CONTEXT

    context = audio[
        :,
        first:first + CONTEXT_SAMPLES,
    ]

    assert context.shape == (
        8, CONTEXT_SAMPLES
    )

    full_ipd, _ = extract_ipd(
        audio,
        sample_rate=FS,
        mic_pairs=ALL_PAIRS,
        mask_threshold_db=MASK_DB,
    )

    reference = full_ipd[
        start_frame:start_frame + WINDOW_FRAMES
    ]

    spectrum = compute_stft(context)

    reconstructed = []

    for i, j in ALL_PAIRS:
        cross = spectrum[i] * np.conj(spectrum[j])

        phase, _ = masked_ipd(cross)

        reconstructed.append(
            np.stack(
                (phase.real, phase.imag),
                axis=-1,
            )
        )

    reconstructed = np.stack(
        reconstructed,
        axis=1,
    ).astype(np.float32)

    np.testing.assert_allclose(
        reconstructed,
        reference,
        atol=1e-6,
        rtol=1e-6,
    )

    print("Context/IPD equivalence: PASS")


# --------------------------------------------------
# Spatial measurements
# --------------------------------------------------

def calculate_metrics(context):
    spectrum = compute_stft(context)

    frequencies = np.fft.rfftfreq(
        NFFT,
        1.0 / FS,
    )

    records = []

    for group_name, pairs in PAIR_GROUPS.items():
        for i, j in pairs:
            xi = spectrum[i]
            xj = spectrum[j]

            cross = xi * np.conj(xj)

            mean_cross = cross.mean(axis=0)

            power_i = (
                np.abs(xi) ** 2
            ).mean(axis=0)

            power_j = (
                np.abs(xj) ** 2
            ).mean(axis=0)

            coherence = (
                np.abs(mean_cross) ** 2
                / np.maximum(
                    power_i * power_j,
                    1e-20,
                )
            )

            coherence = np.clip(
                coherence, 0.0, 1.0
            )

            phase, active = masked_ipd(cross)

            valid_count = active.sum(axis=0)

            resultant = np.abs(
                phase.sum(axis=0)
            )

            concentration = np.divide(
                resultant,
                valid_count,
                out=np.full(
                    len(valid_count),
                    np.nan,
                    dtype=np.float64,
                ),
                where=(
                    valid_count >= MIN_PHASE_FRAMES
                ),
            )

            for low, high in BANDS:
                if high == FS // 2:
                    band = (
                        (frequencies >= low)
                        & (frequencies <= high)
                    )
                else:
                    band = (
                        (frequencies >= low)
                        & (frequencies < high)
                    )

                values = concentration[band]
                available = np.isfinite(values)

                mean_concentration = (
                    float(values[available].mean())
                    if np.any(available)
                    else float("nan")
                )

                records.append({
                    "pair_group": group_name,
                    "pair": f"{i+1:02d}-{j+1:02d}",
                    "band": f"{low}-{high}",
                    "coherence": float(
                        coherence[band].mean()
                    ),
                    "concentration": mean_concentration,
                    "mask_retention": float(
                        active[:, band].mean()
                    ),
                    "usable_bin_fraction": float(
                        available.mean()
                    ),
                })

    return records


# --------------------------------------------------
# Reporting
# --------------------------------------------------

def save_csv(path, rows):
    if not rows:
        raise ValueError("No rows to save")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    good = np.isfinite(values)

    return (
        float(values[good].mean())
        if np.any(good)
        else float("nan")
    )


def summarize(rows):
    summary = []

    for meeting in MEETINGS:
        for group in PAIR_GROUPS:
            for state in (1, 2):
                for low, high in BANDS:
                    band = f"{low}-{high}"

                    selected = [
                        row for row in rows
                        if row["meeting"] == meeting
                        and row["pair_group"] == group
                        and row["state"] == state
                        and row["band"] == band
                    ]

                    summary.append({
                        "meeting": meeting,
                        "pair_group": group,
                        "state": state,
                        "band": band,
                        "records": len(selected),
                        "coherence": finite_mean([
                            r["coherence"]
                            for r in selected
                        ]),
                        "concentration": finite_mean([
                            r["concentration"]
                            for r in selected
                        ]),
                        "mask_retention": finite_mean([
                            r["mask_retention"]
                            for r in selected
                        ]),
                        "usable_bin_fraction": finite_mean([
                            r["usable_bin_fraction"]
                            for r in selected
                        ]),
                    })

    return summary


def print_results(summary):
    print("\nMEETING-SPECIFIC RESULTS")
    print("=" * 85)

    for meeting in MEETINGS:
        print(f"\n{meeting}")

        for group in PAIR_GROUPS:
            print(f"  {group}")

            for low, high in BANDS:
                band = f"{low}-{high}"

                entries = {
                    state: next(
                        row for row in summary
                        if row["meeting"] == meeting
                        and row["pair_group"] == group
                        and row["state"] == state
                        and row["band"] == band
                    )
                    for state in (1, 2)
                }

                single = entries[1]
                overlap = entries[2]

                delta = (
                    overlap["coherence"]
                    - single["coherence"]
                )

                print(
                    f"    {band:<10}"
                    f" single={single['coherence']:.4f}"
                    f" overlap={overlap['coherence']:.4f}"
                    f" delta={delta:+.4f}"
                )

    print("\nAGGREGATE BY PAIR GROUP")
    print("=" * 85)

    print(
        f"{'Group':<15}"
        f"{'Band':<12}"
        f"{'State':<8}"
        f"{'Coh':>8}"
        f"{'Phase':>9}"
        f"{'Mask':>9}"
        f"{'Usable':>9}"
    )

    for group in PAIR_GROUPS:
        for low, high in BANDS:
            band = f"{low}-{high}"

            for state in (1, 2):
                selected = [
                    row for row in summary
                    if row["pair_group"] == group
                    and row["state"] == state
                    and row["band"] == band
                ]

                print(
                    f"{group:<15}"
                    f"{band:<12}"
                    f"{state:<8}"
                    f"{finite_mean([r['coherence'] for r in selected]):>8.4f}"
                    f"{finite_mean([r['concentration'] for r in selected]):>9.4f}"
                    f"{finite_mean([r['mask_retention'] for r in selected]):>9.4f}"
                    f"{finite_mean([r['usable_bin_fraction'] for r in selected]):>9.4f}"
                )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("\nM2-D2.7: CONTEXT-CORRECTED SPATIAL COMPARISON")
    print("=" * 85)

    validate_feature_equivalence()

    print(
        "Context samples per window:",
        CONTEXT_SAMPLES,
    )

    all_rows = []
    selected_windows = []

    for meeting in MEETINGS:
        windows = select_meeting_windows(meeting)

        for item in windows:
            selected_windows.append(item)

            context = read_context(
                meeting,
                item["start_frame"],
            )

            metrics = calculate_metrics(context)

            for row in metrics:
                all_rows.append({
                    **item,
                    "start_seconds": (
                        item["start_frame"] * HOP / FS
                    ),
                    **row,
                })

    assert len(selected_windows) > 0

    expected_rows = (
        len(selected_windows)
        * len(ALL_PAIRS)
        * len(BANDS)
    )

    assert len(all_rows) == expected_rows

    summary = summarize(all_rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_csv(
        OUTPUT_DIR / "windows.csv",
        all_rows,
    )

    save_csv(
        OUTPUT_DIR / "summary.csv",
        summary,
    )

    save_csv(
        OUTPUT_DIR / "selected_windows.csv",
        selected_windows,
    )

    print_results(summary)

    print("\nOUTPUT")
    print("=" * 85)
    print("Selected windows:", len(selected_windows))
    print("Measurement records:", len(all_rows))
    print("Saved to:", OUTPUT_DIR)

    print("\nM2-D2.7 ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()