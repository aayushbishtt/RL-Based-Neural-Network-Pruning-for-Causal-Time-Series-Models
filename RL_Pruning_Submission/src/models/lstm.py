"""
LSTM model for time series forecasting.

This module provides an LSTM-based model for sequential prediction tasks.
LSTMs are well-suited for capturing long-term dependencies in time series.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base import BaseTimeSeriesModel


class LSTMModel(BaseTimeSeriesModel):
    """
    LSTM model for time series forecasting.

    This model uses one or more LSTM layers followed by a fully connected
    layer for prediction.

    Args:
        input_size: Number of input features.
        output_size: Number of output features.
        sequence_length: Length of input sequences.
        hidden_size: Number of hidden units in LSTM.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate between LSTM layers.
        bidirectional: Whether to use bidirectional LSTM.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        sequence_length: int = 50,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__(input_size, output_size, sequence_length)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Fully connected output layer
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Linear(fc_input_size, output_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier initialization."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1)

        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_size).
            hidden: Optional initial hidden state (h_0, c_0).

        Returns:
            Output tensor of shape (batch, output_size).
        """
        batch_size = x.size(0)

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self._init_hidden(batch_size, x.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, hidden)

        # Take the last timestep output
        if self.bidirectional:
            # Concatenate forward and backward outputs
            out = lstm_out[:, -1, :]
        else:
            out = lstm_out[:, -1, :]

        # Apply dropout and final layer
        out = self.dropout(out)
        out = self.fc(out)

        return out

    def _init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state to zeros.

        Args:
            batch_size: Current batch size.
            device: Device to create tensors on.

        Returns:
            Tuple of (h_0, c_0) initial hidden states.
        """
        h_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )
        c_0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )
        return (h_0, c_0)

    def forward_with_attention(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention weights (for interpretability).

        This method computes simple temporal attention weights
        to understand which timesteps are most important.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_size).

        Returns:
            Tuple of (output, attention_weights).
        """
        batch_size = x.size(0)
        hidden = self._init_hidden(batch_size, x.device)

        # Get all LSTM outputs
        lstm_out, _ = self.lstm(x, hidden)

        # Simple attention mechanism
        # Compute attention scores
        attention_scores = torch.tanh(lstm_out)
        attention_scores = torch.mean(attention_scores, dim=-1)  # (batch, seq_len)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Weighted sum of LSTM outputs
        context = torch.bmm(
            attention_weights.unsqueeze(1), lstm_out
        ).squeeze(1)

        # Final prediction
        out = self.dropout(context)
        out = self.fc(out)

        return out, attention_weights


class GRUModel(BaseTimeSeriesModel):
    """
    GRU model for time series forecasting.

    Similar to LSTM but with fewer parameters (no cell state).

    Args:
        input_size: Number of input features.
        output_size: Number of output features.
        sequence_length: Length of input sequences.
        hidden_size: Number of hidden units in GRU.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout rate between GRU layers.
        bidirectional: Whether to use bidirectional GRU.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        sequence_length: int = 50,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__(input_size, output_size, sequence_length)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Fully connected output layer
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Linear(fc_input_size, output_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_size).
            hidden: Optional initial hidden state.

        Returns:
            Output tensor of shape (batch, output_size).
        """
        batch_size = x.size(0)

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=x.device,
            )

        # GRU forward pass
        gru_out, _ = self.gru(x, hidden)

        # Take the last timestep output
        out = gru_out[:, -1, :]

        # Apply dropout and final layer
        out = self.dropout(out)
        out = self.fc(out)

        return out
