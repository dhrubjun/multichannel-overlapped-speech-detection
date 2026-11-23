
from pathlib import Path
import wave
import numpy as np
import shutil
import os

ROOT = Path(
    r"C:\Master 1st Sem\Deep Learning\Project\amicorpus"
)

AUDIO_DIR = ROOT / "ES2010d" / "audio"
BACKUP_DIR = ROOT / "ES2010d" / "original_stereo_backup"

BACKUP_DIR.mkdir(parents=True, exist_ok=True)

for mic in range(1, 9):
    filename = f"ES2010d.Array1-{mic:02d}.wav"
    source = AUDIO_DIR / filename
    backup = BACKUP_DIR / filename
    temporary = AUDIO_DIR / f"{filename}.tmp"

    if not source.is_file():
        raise FileNotFoundError(source)

    with wave.open(str(source), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()

    if channels == 1:
        print(f"SKIP already mono: {filename}")
        continue

    if channels != 2 or sample_width != 2:
        raise ValueError(f"Unexpected format: {filename}")

    # Preserve the original stereo file.
    if backup.exists():
        if backup.stat().st_size != source.stat().st_size:
            raise RuntimeError(
                f"Existing backup size mismatch: {backup}"
            )
        raise RuntimeError(
            f"Backup already exists; inspect before proceeding: {backup}"
        )

    shutil.copy2(source, backup)

    if backup.stat().st_size != source.stat().st_size:
        raise RuntimeError(f"Backup verification failed: {filename}")

    # Verify full channel equality and write mono samples.
    try:
        with wave.open(str(source), "rb") as stereo:
            with wave.open(str(temporary), "wb") as mono:
                mono.setnchannels(1)
                mono.setsampwidth(sample_width)
                mono.setframerate(sample_rate)

                while True:
                    raw = stereo.readframes(160000)
                    if not raw:
                        break

                    samples = np.frombuffer(
                        raw, dtype="<i2"
                    ).reshape(-1, 2)

                    if not np.array_equal(
                        samples[:, 0], samples[:, 1]
                    ):
                        raise RuntimeError(
                            f"Channels differ: {filename}"
                        )

                    mono.writeframes(
                        samples[:, 0].tobytes()
                    )

        with wave.open(str(temporary), "rb") as check:
            if (
                check.getnchannels() != 1
                or check.getframerate() != sample_rate
                or check.getsampwidth() != sample_width
                or check.getnframes() != frames
            ):
                raise RuntimeError(
                    f"Mono verification failed: {filename}"
                )

        os.replace(temporary, source)

    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print(f"CONVERTED: {filename} (stereo -> mono)")

print("=" * 50)
print("ES2010d conversion finished.")
print(f"Originals preserved in: {BACKUP_DIR}")