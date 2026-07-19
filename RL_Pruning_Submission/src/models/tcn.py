"""
Temporal Convolutional Network (TCN) for time series forecasting.

TCNs use dilated causal convolutions to capture long-range dependencies
in sequential data while maintaining computational efficiency.

Reference:
    Bai, S., Kolter, J.Z., & Koltun, V. (2018).
    An Empirical Evaluation of Generic Convolutional and Recurrent Networks
    for Sequence Modeling. arXiv:1803.01271
"""

from typing import List

import torch
import torch.nn as nn


from .base import BaseTimeSeriesModel


class Chomp1d(nn.Module):
    """
    Remove the padding at the end to ensure causal convolution.

    This module removes the last `chomp_size` elements from the sequence,
    ensuring that the convolution is causal (doesn't look into the future).
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    A single temporal block consisting of two dilated causal convolutions
    with residual connection.

    Architecture:
        Conv1d -> Chomp1d -> ReLU -> Dropout ->
        Conv1d -> Chomp1d -> ReLU -> Dropout ->
        + Residual Connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        # First convolution
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Combine into sequential
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        # Residual connection
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using normal distribution."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network backbone.

    This is the core TCN architecture that can be used standalone
    or as part of a larger model.

    Args:
        num_inputs: Number of input channels (features).
        num_channels: List of channel sizes for each layer.
        kernel_size: Kernel size for convolutions.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, sequence_length).

        Returns:
            Output tensor of shape (batch, num_channels[-1], sequence_length).
        """
        return self.network(x)


class TCNModel(BaseTimeSeriesModel):
    """
    Complete TCN model for time series forecasting.

    This model combines the TCN backbone with a final linear layer
    for prediction.

    Args:
        input_size: Number of input features.
        output_size: Number of output features.
        sequence_length: Length of input sequences.
        num_channels: List of channel sizes for TCN layers.
        kernel_size: Kernel size for convolutions.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        sequence_length: int = 50,
        num_channels: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__(input_size, output_size, sequence_length)

        if num_channels is None:
            num_channels = [32, 64, 64, 32]

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout

        # TCN backbone
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Final prediction layer
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_size).

        Returns:
            Output tensor of shape (batch, output_size).
        """
        # TCN expects (batch, channels, sequence)
        x = x.transpose(1, 2)

        # Pass through TCN
        tcn_out = self.tcn(x)

        # Take the last timestep
        out = tcn_out[:, :, -1]

        # Final prediction
        out = self.fc(out)

        return out

    def get_receptive_field(self) -> int:
        """
        Calculate the receptive field of the TCN.

        The receptive field determines how far back in time
        the model can look to make predictions.

        Returns:
            Receptive field size in timesteps.
        """
        num_levels = len(self.num_channels)
        return 1 + 2 * (self.kernel_size - 1) * (2**num_levels - 1)
