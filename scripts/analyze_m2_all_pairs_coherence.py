
"""
M2-D2.3: Frequency-dependent coherence analysis.

Evaluate the four M2 microphone pairs using the same
annotated single-speaker windows as M2-D2.2.

This script is diagnostic. Coherence alone does not
establish the usefulness of IPD for classification.
"""

from pathlib import Path
import csv

import numpy as np
from scipy.signal import coherence

from scripts.audit_m2_single_speaker_delays import (
    select_windows,
    load_window,
)

FS = 16000

from itertools import combinations

PAIRS = list(combinations(range(8), 2))

BANDS = [
    (0, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 8000),
]

OUTPUT = Path(
    "outputs/evaluation/m2_channel_phase/"
    "all_pairs_coherence.csv"
)


def main():
    starts = select_windows()

    records = []

    print("\nM2-D2.3: FREQUENCY-DEPENDENT COHERENCE")
    print("=" * 75)
    print("Windows:", len(starts))
    print("STFT segment: 512 samples")
    print("Segment overlap: 256 samples")

    for start in starts:
        audio = load_window(start)

        for i, j in PAIRS:
            frequencies, coherence_values = coherence(
                audio[i].astype(np.float64),
                audio[j].astype(np.float64),
                fs=FS,
                nperseg=512,
                noverlap=256,
                window="hann",
            )

            for low, high in BANDS:
                # Include Nyquist in final band.
                if high == FS / 2:
                    selected = (
                        (frequencies >= low)
                        & (frequencies <= high)
                    )
                else:
                    selected = (
                        (frequencies >= low)
                        & (frequencies < high)
                    )

                values = coherence_values[selected]

                if not len(values):
                    raise ValueError("Empty frequency band")

                records.append({
                    "start_seconds": start,
                    "pair": f"{i+1:02d}-{j+1:02d}",
                    "band": f"{low}-{high}",
                    "mean_coherence": float(values.mean()),
                    "fraction_above_05": float(
                        np.mean(values >= 0.5)
                    ),
                })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0]),
        )
        writer.writeheader()
        writer.writerows(records)

    print(
        "\nPair        Band (Hz)    "
        "Mean coherence    Fraction >= 0.5"
    )
    print("-" * 75)

    for i, j in PAIRS:
        pair = f"{i+1:02d}-{j+1:02d}"

        for low, high in BANDS:
            band = f"{low}-{high}"

            relevant = [
                row for row in records
                if row["pair"] == pair
                and row["band"] == band
            ]

            mean_coherence = np.mean([
                row["mean_coherence"]
                for row in relevant
            ])

            fraction = np.mean([
                row["fraction_above_05"]
                for row in relevant
            ])

            print(
                f"{pair:<10}"
                f"{band:<15}"
                f"{mean_coherence:>12.4f}"
                f"{fraction:>19.4f}"
            )

    assert len(records) == (
        len(starts) * len(PAIRS) * len(BANDS)
    )

    
    print("\nCOHERENCE BY CHANNEL INDEX SEPARATION")
    print("=" * 75)

    for separation in range(1, 5):
        print(f"\nIndex separation: {separation}")

        for low, high in BANDS:
            band = f"{low}-{high}"
            values = []

            for i, j in PAIRS:
                difference = j - i

                circular_separation = min(
                    difference,
                    8 - difference,
                )

                if circular_separation != separation:
                    continue

                pair_name = f"{i+1:02d}-{j+1:02d}"

                values.extend([
                    row["mean_coherence"]
                    for row in records
                    if row["pair"] == pair_name
                    and row["band"] == band
                ])

            print(
                f"  {band:<12} Hz: "
                f"mean coherence = {np.mean(values):.4f}"
            )


    print("\nCSV saved:", OUTPUT)
    print("COHERENCE ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()