
import unittest

import torch
import torch.nn.functional as F

from src.models.tcn import (
    ResidualTCNBlock,
    SpeechTCN,
)


class TestSpeechTCN(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.model = SpeechTCN()

    def test_residual_block_shape(self):
        block = ResidualTCNBlock(
            channels=64,
            kernel_size=3,
            dilation=4,
            dropout=0.1,
        )

        x = torch.randn(2, 64, 200)
        y = block(x)

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all().item())

    def test_model_output_shape(self):
        self.model.eval()

        x = torch.randn(2, 200, 80)

        with torch.no_grad():
            logits = self.model(x)

        self.assertEqual(logits.shape, (2, 200, 3))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_receptive_field(self):
        self.assertEqual(self.model.receptive_field, 125)

    def test_parameter_count(self):
        count = sum(
            p.numel() for p in self.model.parameters()
        )
        self.assertEqual(count, 128899)

    def test_backward_propagation(self):
        self.model.train()

        x = torch.randn(2, 200, 80)
        labels = torch.randint(0, 3, (2, 200))

        logits = self.model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, 3),
            labels.reshape(-1),
        )

        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())

        for parameter in self.model.parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(
                    torch.isfinite(parameter.grad).all().item()
                )

    def test_variable_sequence_length(self):
        self.model.eval()

        for length in (100, 200, 250):
            x = torch.randn(2, length, 80)

            with torch.no_grad():
                y = self.model(x)

            self.assertEqual(y.shape, (2, length, 3))

    def test_invalid_kernel_size(self):
        with self.assertRaises(ValueError):
            ResidualTCNBlock(kernel_size=4)

    def test_invalid_dilation(self):
        with self.assertRaises(ValueError):
            ResidualTCNBlock(dilation=0)


if __name__ == "__main__":
    unittest.main()