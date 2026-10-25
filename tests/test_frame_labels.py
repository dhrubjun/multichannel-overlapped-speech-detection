
import unittest
import numpy as np

from src.data.frame_labels import generate_labels


class TestFrameLabels(unittest.TestCase):

    def test_silence(self):
        labels, activity, frame_times = generate_labels(
            intervals=[[], []],
            duration=2.0
        )

        self.assertTrue(np.all(labels == 0))

    def test_single_speaker(self):
        labels, activity, frame_times = generate_labels(
            intervals=[[(0.0, 2.0)]],
            duration=2.0
        )

        self.assertTrue(np.all(labels == 1))

    def test_two_speaker_overlap(self):
        labels, activity, frame_times = generate_labels(
            intervals=[
                [(1.0, 3.0)],
                [(2.0, 4.0)]
            ],
            duration=4.0
        )

        self.assertTrue(np.all(labels[:100] == 0))
        self.assertTrue(np.all(labels[100:200] == 1))
        self.assertTrue(np.all(labels[200:300] == 2))
        self.assertTrue(np.all(labels[300:400] == 1))

    def test_three_speaker_overlap(self):
        labels, activity, frame_times = generate_labels(
            intervals=[
                [(0.0, 2.0)],
                [(0.5, 1.5)],
                [(1.0, 1.2)]
            ],
            duration=2.0
        )

        self.assertTrue(np.all(labels[100:120] == 2))

    def test_touching_intervals(self):
        labels, activity, frame_times = generate_labels(
            intervals=[
                [(0.0, 1.0)],
                [(1.0, 2.0)]
            ],
            duration=2.0
        )

        self.assertTrue(np.all(labels == 1))

    
    def test_zero_duration_interval(self):
        with self.assertRaises(ValueError):
            generate_labels(
                intervals=[[(1.0, 1.0)]],
                duration=2.0
            )

    def test_interval_outside_recording(self):
        with self.assertRaises(ValueError):
            generate_labels(
                intervals=[[(1.0, 3.0)]],
                duration=2.0
            )

    def test_negative_start_time(self):
        with self.assertRaises(ValueError):
            generate_labels(
                intervals=[[(-0.5, 1.0)]],
                duration=2.0
            )



if __name__ == "__main__":
    unittest.main()