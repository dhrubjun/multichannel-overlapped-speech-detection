
import tempfile
import unittest
from pathlib import Path

from src.data.ami_annotations import load_ami_annotations


class TestAMIAnnotations(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.words_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_xml(self, speaker, elements):
        path = self.words_dir / f"TEST.{speaker}.words.xml"

        path.write_text(
            "<root>" + elements + "</root>",
            encoding="utf-8"
        )

    def load(self):
        return load_ami_annotations(
            meeting="TEST",
            words_dir=self.words_dir,
            duration=10.0,
            speakers="AB"
        )

    def test_valid_words(self):
        self.write_xml(
            "A",
            '<w starttime="1.0" endtime="2.0">Hello</w>'
        )
        self.write_xml(
            "B",
            '<w starttime="1.5" endtime="2.5">Hi</w>'
        )

        intervals, vocals, issues = self.load()

        self.assertEqual(intervals[0], [(1.0, 2.0)])
        self.assertEqual(intervals[1], [(1.5, 2.5)])
        self.assertEqual(vocals, [])
        self.assertEqual(issues, [])

    def test_punctuation_excluded(self):
        self.write_xml(
            "A",
            '<w starttime="1.0" endtime="1.0" '
            'punc="true">.</w>'
        )
        self.write_xml("B", "")

        intervals, vocals, issues = self.load()

        self.assertEqual(intervals, [[], []])
        self.assertEqual(issues, [])

    def test_missing_timestamp(self):
        self.write_xml(
            "A",
            '<w>Unknown</w>'
        )
        self.write_xml("B", "")

        intervals, vocals, issues = self.load()

        self.assertEqual(intervals, [[], []])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0][2], "missing timestamp")

    def test_vocal_sound(self):
        self.write_xml(
            "A",
            '<vocalsound starttime="2.0" '
            'endtime="3.0" type="laugh"/>'
        )
        self.write_xml("B", "")

        intervals, vocals, issues = self.load()

        self.assertEqual(vocals, [(2.0, 3.0)])
        self.assertEqual(issues, [])

    def test_zero_duration_gap(self):
        self.write_xml(
            "A",
            '<gap starttime="5.0" endtime="5.0"/>'
        )
        self.write_xml("B", "")

        intervals, vocals, issues = self.load()

        self.assertEqual(intervals, [[], []])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0][2], "zero-duration gap")

    def test_outside_duration(self):
        self.write_xml(
            "A",
            '<w starttime="9.0" endtime="11.0">Hello</w>'
        )
        self.write_xml("B", "")

        intervals, vocals, issues = self.load()

        self.assertEqual(intervals, [[], []])
        self.assertEqual(len(issues), 1)

    def test_missing_file(self):
        self.write_xml("A", "")

        with self.assertRaises(FileNotFoundError):
            self.load()

    
    def test_positive_duration_gap_is_masked(self):
        import numpy as np
        from src.data.frame_labels import generate_valid_mask

        # Speaker A has a transcription gap from 1.0 to 1.2 s
        self.write_xml(
            "A",
            '<gap starttime="1.0" endtime="1.2"/>'
        )
        self.write_xml("B", "")

        # Load annotations using our existing loader
        intervals, uncertain_intervals, issues = self.load()

        # No annotated speech is present
        self.assertEqual(intervals, [[], []])

        # The gap must be included in uncertain intervals
        self.assertEqual(uncertain_intervals, [(1.0, 1.2)])

        # The gap must still be reported in the annotation audit
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0][2], "positive-duration gap")

        # Generate a 10 ms frame grid covering 10 seconds
        frame_times = (np.arange(1000) + 0.5) * 0.01

        # Generate the uncertainty mask
        valid_mask = generate_valid_mask(
            uncertain_intervals,
            frame_times
        )

        # Frames inside the gap must be excluded
        inside_gap = (
            (frame_times >= 1.0) &
            (frame_times < 1.2)
        )

        self.assertEqual(int(inside_gap.sum()), 20)
        self.assertTrue(np.all(~valid_mask[inside_gap]))

        # Frames outside the gap must remain valid
        self.assertTrue(np.all(valid_mask[~inside_gap]))

    def test_end_before_start(self):
        self.write_xml(
            "A",
            '<vocalsound starttime="5.0" '
            'endtime="4.0" type="laugh"/>'
        )
        self.write_xml("B", "")

        intervals, uncertain, issues = self.load()

        self.assertEqual(intervals, [[], []])
        self.assertEqual(uncertain, [])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0][2], "end before start")


if __name__ == "__main__":
    unittest.main()