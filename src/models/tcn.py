
import torch
from torch import nn


class ResidualTCNBlock(nn.Module):
    def __init__(
        self,
        channels=64,
        kernel_size=3,
        dilation=1,
        dropout=0.1,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        if dilation < 1:
            raise ValueError("Dilation must be positive")

        padding = dilation * (kernel_size - 1) // 2

        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.network(x)

    
class SpeechTCN(nn.Module):
    """
    Noncausal residual TCN for frame-wise
    three-class speech-state classification.

    Input:  (batch, time, 80)
    Output: (batch, time, 3)
    """

    def __init__(
        self,
        input_features=80,
        hidden_channels=64,
        num_classes=3,
        kernel_size=3,
        dilations=(1, 2, 4, 8, 16),
        dropout=0.1,
    ):
        super().__init__()

        self.input_projection = nn.Conv1d(
            in_channels=input_features,
            out_channels=hidden_channels,
            kernel_size=1,
        )

        self.tcn = nn.Sequential(
            *[
                ResidualTCNBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
                for d in dilations
            ]
        )

        self.classifier = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=num_classes,
            kernel_size=1,
        )

    def forward(self, x):
        # x: (batch, time, features)
        x = x.transpose(1, 2)

        # (batch, 80, time) -> (batch, 64, time)
        x = self.input_projection(x)

        # Temporal processing
        x = self.tcn(x)

        # (batch, 64, time) -> (batch, 3, time)
        x = self.classifier(x)

        # Return (batch, time, 3)
        return x.transpose(1, 2)

    @property
    def receptive_field(self):
        # Two convolutions per block.
        rf = 1

        for block in self.tcn:
            conv = block.network[0]
            kernel_size = conv.kernel_size[0]
            dilation = conv.dilation[0]

            rf += 2 * (kernel_size - 1) * dilation

        return rf
