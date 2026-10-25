
from pathlib import Path
import wave

# Location of our downloaded AMI audio
AUDIO_DIR = Path(
    r"C:\Master 1st Sem\Deep Learning\Project"
    r"\amicorpus\ES2008a\audio"
)

# We are initially using Array 1
files = sorted(AUDIO_DIR.glob("ES2008a.Array1-*.wav"))

print(f"Found {len(files)} Array 1 WAV files\n")

for file in files:
    with wave.open(str(file), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        num_frames = wav.getnframes()

        duration = num_frames / sample_rate

        print(f"File: {file.name}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Channels: {channels}")
        print(f"  Sample width: {sample_width * 8} bits")
        print(f"  Frames: {num_frames}")
        print(f"  Duration: {duration:.3f} seconds")
        print()