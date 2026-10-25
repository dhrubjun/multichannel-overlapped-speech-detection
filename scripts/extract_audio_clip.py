
from pathlib import Path
import wave

AUDIO_DIR = Path(
    r"C:\Master 1st Sem\Deep Learning\Project"
    r"\amicorpus\ES2008a\audio"
)

OUTPUT_DIR = Path("outputs/audio_checks/multichannel")

START_TIME = 238.0
END_TIME = 244.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for mic in range(1, 9):

    input_file = AUDIO_DIR / (
        f"ES2008a.Array1-{mic:02d}.wav"
    )

    output_file = OUTPUT_DIR / (
        f"ES2008a_mic{mic:02d}_238-244.wav"
    )

    with wave.open(str(input_file), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()

        start_frame = round(START_TIME * sample_rate)
        end_frame = round(END_TIME * sample_rate)

        if end_frame > wav.getnframes():
            raise ValueError(f"Clip exceeds recording: {input_file}")

        wav.setpos(start_frame)
        audio_data = wav.readframes(end_frame - start_frame)

    with wave.open(str(output_file), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(sample_width)
        out.setframerate(sample_rate)
        out.writeframes(audio_data)

    print(f"Microphone {mic}: saved {output_file}")

print("\nAll eight microphone clips extracted.")