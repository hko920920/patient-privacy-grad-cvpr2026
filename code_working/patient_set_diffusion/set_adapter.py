"""Small permutation-equivariant patient-set adapter for a diffusion UNet mid block."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class PatientSetAdapter(nn.Module):
    """Exchange information across records without introducing a slot position encoding.

    Input records are flattened as ``[patients * slots, channels, height, width]`` and
    ``valid_mask`` has shape ``[patients, slots]``.  Singleton patients and padded slots
    are exact identity paths.  The output projection is zero initialized so a newly
    inserted adapter is also an exact identity for multi-record patients.
    """

    def __init__(
        self,
        channels: int,
        token_dim: int = 128,
        num_heads: int = 4,
        ffn_multiplier: int = 2,
    ) -> None:
        super().__init__()
        if channels <= 0 or token_dim <= 0 or num_heads <= 0 or ffn_multiplier <= 0:
            raise ValueError("adapter dimensions must be positive")
        if token_dim % num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        self.channels = int(channels)
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        hidden_dim = token_dim * ffn_multiplier

        self.input_norm = nn.LayerNorm(channels)
        self.to_token = nn.Linear(channels, token_dim)
        self.attention = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.token_norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.to_delta = nn.Linear(token_dim, channels)
        nn.init.zeros_(self.to_delta.weight)
        nn.init.zeros_(self.to_delta.bias)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, hidden: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4:
            raise ValueError("hidden must have shape [patients*slots, channels, height, width]")
        if valid_mask.ndim != 2 or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be a boolean [patients, slots] tensor")
        patients, slots = valid_mask.shape
        if patients <= 0 or slots <= 0:
            raise ValueError("patient and slot dimensions must be nonzero")
        if hidden.shape[0] != patients * slots or hidden.shape[1] != self.channels:
            raise ValueError("hidden batch/channels do not match the declared patient-set layout")
        if valid_mask.device != hidden.device:
            raise ValueError("valid_mask and hidden must be on the same device")
        valid_counts = valid_mask.sum(dim=1)
        if bool(torch.any(valid_counts == 0)):
            raise ValueError("every patient must have at least one valid record")
        if slots == 1 or bool(torch.all(valid_counts == 1)):
            return hidden

        pooled = hidden.mean(dim=(-2, -1)).reshape(patients, slots, self.channels)
        tokens = self.to_token(self.input_norm(pooled))
        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.token_norm(tokens))
        delta = self.to_delta(tokens).reshape(patients * slots, self.channels, 1, 1)

        active = valid_mask & (valid_counts[:, None] > 1)
        delta = delta * active.reshape(patients * slots, 1, 1, 1).to(dtype=delta.dtype)
        return hidden + delta.to(dtype=hidden.dtype)


class PatientSetMidBlock(nn.Module):
    """Wrap a diffusers mid block and apply ``PatientSetAdapter`` to its tensor output."""

    def __init__(self, base_mid_block: nn.Module, set_adapter: PatientSetAdapter) -> None:
        super().__init__()
        self.base_mid_block = base_mid_block
        self.set_adapter = set_adapter
        self.has_cross_attention = bool(getattr(base_mid_block, "has_cross_attention", False))
        self._valid_mask: torch.Tensor | None = None
        self._bypass = False

    def configure(self, valid_mask: torch.Tensor, *, bypass: bool = False) -> None:
        if valid_mask.ndim != 2 or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean [patients, slots]")
        self._valid_mask = valid_mask
        self._bypass = bool(bypass)

    def clear_layout(self) -> None:
        self._valid_mask = None
        self._bypass = False

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        hidden = self.base_mid_block(*args, **kwargs)
        if self._bypass:
            return hidden
        if self._valid_mask is None:
            raise RuntimeError("patient-set layout must be configured before UNet forward")
        return self.set_adapter(hidden, self._valid_mask)
