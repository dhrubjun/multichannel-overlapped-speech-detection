
from pathlib import Path
import wave
import numpy as np

AUDIO_DIR = Path(
    r"C:\Master 1st Sem\Deep Learning\Project"
    r"\amicorpus\ES2008a\audio"
)

MEETING = "ES2008a"

# Different recording positions
START_TIMES = [50.0, 240.0, 420.0, 600.0, 850.0]

DURATION = 2.0
MAX_LAG_MS = 5.0

MIC_PAIRS = [
    (1, 5),
    (2, 6),
    (3, 7),
    (4, 8),
]


def read_clip(mic, start_time):
    path = AUDIO_DIR / f"{MEETING}.Array1-{mic:02d}.wav"

    with wave.open(str(path), "rb") as wav:
        fs = wav.getframerate()
        start_frame = round(start_time * fs)
        n_frames = round(DURATION * fs)

        if start_frame + n_frames > wav.getnframes():
            raise ValueError("Requested clip exceeds recording")

        wav.setpos(start_frame)
        raw = wav.readframes(n_frames)

    audio = np.frombuffer(raw, dtype=np.int16)
    return audio.astype(np.float64), fs


def estimate_delay(x1, x2, fs):
    # Remove DC offset
    x1 = x1 - np.mean(x1)
    x2 = x2 - np.mean(x2)

    # Normalize amplitudes
    x1 = x1 / (np.sqrt(np.mean(x1**2)) + 1e-12)
    x2 = x2 / (np.sqrt(np.mean(x2**2)) + 1e-12)

    max_lag = round(MAX_LAG_MS * fs / 1000)
    lags = np.arange(-max_lag, max_lag + 1)

    correlations = []

    for lag in lags:
        if lag < 0:
            a = x1[-lag:]
            b = x2[:len(x2) + lag]
        elif lag > 0:
            a = x1[:-lag]
            b = x2[lag:]
        else:
            a = x1
            b = x2

        denominator = (
            np.linalg.norm(a) * np.linalg.norm(b)
        )

        value = np.dot(a, b) / (denominator + 1e-12)
        correlations.append(value)

    correlations = np.array(correlations)
    best_index = np.argmax(np.abs(correlations))

    best_lag = int(lags[best_index])
    peak_corr = float(correlations[best_index])

    return best_lag, peak_corr



print("MULTICHANNEL TIMING ANALYSIS")
print("=" * 75)

for mic1, mic2 in MIC_PAIRS:

    print(f"\nMicrophone pair: {mic1}-{mic2}")
    print("-" * 65)

    print(
        f"{'Start (s)':>10} "
        f"{'Lag':>10} "
        f"{'Delay (ms)':>15} "
        f"{'|Correlation|':>15}"
    )

    for start_time in START_TIMES:

        x1, fs1 = read_clip(mic1, start_time)
        x2, fs2 = read_clip(mic2, start_time)

        if fs1 != fs2:
            raise ValueError("Sample rates do not match")

        lag, correlation = estimate_delay(
            x1, x2, fs1
        )

        delay_ms = 1000 * lag / fs1

        print(
            f"{start_time:10.1f} "
            f"{lag:10d} "
            f"{delay_ms:15.4f} "
            f"{abs(correlation):15.4f}"
        )