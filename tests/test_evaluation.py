
import unittest

import numpy as np
import torch
from torch import nn

from src.training.evaluation import (
    binary_average_precision,
    evaluate_model,
)


class FixedLogitsModel(nn.Module):
    """Return fixed logits supplied through the features."""

    def forward(self, features):
        return features


class TestEvaluation(unittest.TestCase):

    def test_perfect_average_precision(self):
        scores = np.array([0.1, 0.9, 0.2, 0.8])
        targets = np.array([0, 1, 0, 1])

        ap = binary_average_precision(scores, targets)

        self.assertAlmostEqual(ap, 1.0)

    def test_average_precision_no_positives(self):
        scores = np.array([0.1, 0.4, 0.9])
        targets = np.array([0, 0, 0])

        ap = binary_average_precision(scores, targets)

        self.assertEqual(ap, 0.0)

    def test_average_precision_with_ties(self):
        scores = np.array([0.8, 0.8, 0.2])
        targets = np.array([1, 0, 1])

        ap = binary_average_precision(scores, targets)

        # Precision at recall 1/2: 1/2
        # Precision at recall 1: 2/3
        expected = 0.5 * 0.5 + 0.5 * (2 / 3)

        self.assertAlmostEqual(ap, expected)

    def test_perfect_classification(self):
        labels = torch.tensor([[0, 1, 2, 0, 1, 2]])

        logits = torch.tensor([[
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ]])

        batch = {
            "features": logits,
            "labels": labels,
            "valid_mask": torch.ones_like(
                labels, dtype=torch.bool
            ),
        }

        results = evaluate_model(
            FixedLogitsModel(),
            [batch],
        )

        np.testing.assert_array_equal(
            results["confusion_matrix"],
            np.diag([2, 2, 2]),
        )

        self.assertEqual(results["valid_frames"], 6)
        self.assertAlmostEqual(results["overlap_f1"], 1.0)
        self.assertAlmostEqual(results["overlap_ap"], 1.0)
        self.assertAlmostEqual(results["macro_f1"], 1.0)

    def test_invalid_frames_excluded(self):
        labels = torch.tensor([[0, 2, 1, -100]])

        logits = torch.tensor([[
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
        ]])

        batch = {
            "features": logits,
            "labels": labels,
            "valid_mask": torch.tensor([
                [True, True, False, False]
            ]),
        }

        results = evaluate_model(
            FixedLogitsModel(),
            [batch],
        )

        self.assertEqual(results["valid_frames"], 2)
        self.assertEqual(
            results["confusion_matrix"].sum(), 2
        )
        self.assertAlmostEqual(results["overlap_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()