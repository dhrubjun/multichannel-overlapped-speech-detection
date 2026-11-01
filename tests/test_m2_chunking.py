
import unittest

import numpy as np

from src.features.combined import extract_combined_features


class TestM2Chunking(unittest.TestCase):

    def test_chunked_matches_full_context(self):
        rng = np.random.default_rng(42)

        # Use a sample count not divisible by 160.
        audio = (
            rng.standard_normal((8, 32037)) * 0.01
        ).astype(np.float32)

        hop = 160
        right_context = 280
        chunk_frames = 47
        n_frames = round(audio.shape[1] / hop)

        # Reference from the complete recording.
        reference, reference_times = (
            extract_combined_features(
                audio,
                n_frames=n_frames,
                mask_threshold_db=-30.0,
            )
        )

        combined_chunks = []

        for start in range(0, n_frames, chunk_frames):
            count = min(chunk_frames, n_frames - start)

            # Match the context strategy used by
            # scripts/extract_m2_combined.py.
            context_frames = 1 if start > 0 else 0

            read_start = (start - context_frames) * hop

            last_frame = start + count - 1
            read_end = min(
                audio.shape[1],
                last_frame * hop + right_context,
            )

            chunk_audio = audio[:, read_start:read_end]

            chunk, times = extract_combined_features(
                chunk_audio,
                n_frames=count + context_frames,
                mask_threshold_db=-30.0,
            )

            chunk = chunk[context_frames:]
            times = times[context_frames:]

            np.testing.assert_allclose(
                read_start / 16000.0 + times,
                reference_times[start:start + count],
                rtol=0,
                atol=1e-9,
            )

            self.assertEqual(chunk.shape, (count, 2136))

            combined_chunks.append(chunk)

        result = np.concatenate(combined_chunks, axis=0)

        np.testing.assert_allclose(
            result,
            reference,
            rtol=0,
            atol=1e-6,
        )

        self.assertEqual(result.shape, reference.shape)


if __name__ == "__main__":
    unittest.main()