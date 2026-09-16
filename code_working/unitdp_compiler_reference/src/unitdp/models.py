"""Small model factories used by experiment scripts."""

from __future__ import annotations

import torch
from torch import nn


class TinySequenceCNN(nn.Module):
    """A compact Opacus-friendly 1D CNN for raw window tensors."""

    def __init__(
        self,
        channels: int,
        num_classes: int,
        hidden_channels: int = 32,
        sequence_length: int = 128,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        self.channels = channels
        self.sequence_length = sequence_length
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=5, padding=2),
            nn.GroupNorm(1, hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2),
            nn.GroupNorm(1, hidden_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.reshape(x.shape[0], self.channels, self.sequence_length)
        return self.net(x)


def build_classifier(
    input_dim: int,
    num_classes: int,
    model_type: str = "linear",
    hidden_dim: int = 128,
) -> nn.Module:
    """Return a classifier compatible with the local DP-SGD trainers."""

    if model_type == "linear":
        return nn.Linear(input_dim, num_classes)
    if model_type == "mlp":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def build_sequence_classifier(
    channels: int,
    num_classes: int,
    model_type: str = "cnn1d",
    hidden_channels: int = 32,
    sequence_length: int = 128,
) -> nn.Module:
    """Return a classifier for raw sequential windows shaped (N, C, T)."""

    if model_type == "cnn1d":
        return TinySequenceCNN(
            channels=channels,
            num_classes=num_classes,
            hidden_channels=hidden_channels,
            sequence_length=sequence_length,
        )
    raise ValueError(f"Unsupported sequence model_type: {model_type}")
