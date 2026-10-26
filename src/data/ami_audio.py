
from pathlib import Path
import wave


def validate_array_audio(meeting, audio_dir, array=1, n_mics=8):
    """
    Validate audio properties across an AMI microphone array.

    Returns a dictionary containing the shared audio properties.
    """

    audio_dir = Path(audio_dir)
    metadata = []

    for mic in range(1, n_mics + 1):
        filename = f"{meeting}.Array{array}-{mic:02d}.wav"
        path = audio_dir / filename

        if not path.is_file():
            raise FileNotFoundError(f"Missing microphone file: {path}")

        with wave.open(str(path), "rb") as wav:
            info = {
                "file": filename,
                "sample_rate": wav.getframerate(),
                "channels": wav.getnchannels(),
                "sample_width": wav.getsampwidth(),
                "frames": wav.getnframes(),
            }

        metadata.append(info)

    reference = metadata[0]

    for info in metadata:
        for field in [
            "sample_rate",
            "channels",
            "sample_width",
            "frames",
        ]:
            if info[field] != reference[field]:
                raise ValueError(
                    f"{info['file']}: {field} mismatch"
                )

        if info["channels"] != 1:
            raise ValueError(
                f"{info['file']} is not mono"
            )

    if reference["sample_rate"] <= 0:
        raise ValueError("Invalid sample rate")

    duration = (
        reference["frames"] / reference["sample_rate"]
    )

    return {
        "meeting": meeting,
        "array": array,
        "microphones": len(metadata),
        "sample_rate": reference["sample_rate"],
        "channels_per_file": reference["channels"],
        "sample_width_bits": reference["sample_width"] * 8,
        "frames": reference["frames"],
        "duration": duration,
    }