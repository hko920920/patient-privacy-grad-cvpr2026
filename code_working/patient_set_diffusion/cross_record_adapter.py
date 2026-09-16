"""Strict leave-one-record-out adapter for patient-set diffusion experiments."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class CrossRecordOnlyAdapter(nn.Module):
    """Create each residual only from other valid records in the patient set.

    For Q=2, record ``i`` receives a learned transform of the companion record only.
    Its own token has no value/residual path into the adapter delta.  For Q>2, the
    record query may weight the other records, while the diagonal remains masked.
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
        self.head_dim = token_dim // num_heads
        hidden_dim = token_dim * ffn_multiplier

        self.input_norm = nn.LayerNorm(channels)
        self.to_token = nn.Linear(channels, token_dim)
        self.to_query = nn.Linear(token_dim, token_dim)
        self.to_key = nn.Linear(token_dim, token_dim)
        self.to_value = nn.Linear(token_dim, token_dim)
        self.context_norm = nn.LayerNorm(token_dim)
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

    def residual(self, hidden: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4:
            raise ValueError("hidden must have shape [patients*slots, channels, height, width]")
        if valid_mask.ndim != 2 or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean [patients, slots]")
        patients, slots = valid_mask.shape
        if patients <= 0 or slots <= 0:
            raise ValueError("patient and slot dimensions must be nonzero")
        if hidden.shape[0] != patients * slots or hidden.shape[1] != self.channels:
            raise ValueError("hidden does not match patient-set layout")
        if hidden.device != valid_mask.device:
            raise ValueError("hidden and valid_mask must share a device")
        valid_counts = valid_mask.sum(dim=1)
        if bool(torch.any(valid_counts == 0)):
            raise ValueError("every patient must have at least one valid record")
        if slots == 1 or bool(torch.all(valid_counts == 1)):
            return torch.zeros(
                (hidden.shape[0], self.channels, 1, 1),
                device=hidden.device,
                dtype=hidden.dtype,
            )

        pooled = hidden.mean(dim=(-2, -1)).reshape(patients, slots, self.channels)
        tokens = self.to_token(self.input_norm(pooled))

        def split_heads(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(patients, slots, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        query = split_heads(self.to_query(tokens))
        key = split_heads(self.to_key(tokens))
        value = split_heads(self.to_value(tokens))
        scores = torch.einsum("bhid,bhjd->bhij", query, key) / math.sqrt(self.head_dim)

        eye = torch.eye(slots, device=hidden.device, dtype=torch.bool)[None, None]
        valid_query = valid_mask[:, None, :, None]
        valid_key = valid_mask[:, None, None, :]
        allowed = valid_query & valid_key & ~eye
        safe_scores = scores.masked_fill(~allowed, -1e4)
        weights = torch.softmax(safe_scores, dim=-1) * allowed.to(dtype=scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        context = torch.einsum("bhij,bhjd->bhid", weights, value)
        context = context.permute(0, 2, 1, 3).reshape(patients, slots, self.token_dim)
        context = context + self.ffn(self.context_norm(context))
        delta = self.to_delta(context).reshape(patients * slots, self.channels, 1, 1)

        active = valid_mask & (valid_counts[:, None] > 1)
        delta = delta * active.reshape(patients * slots, 1, 1, 1).to(dtype=delta.dtype)
        return delta.to(dtype=hidden.dtype)

    def forward(self, hidden: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return hidden + self.residual(hidden, valid_mask)


class CrossRecordMidBlock(nn.Module):
    """Diffusers mid-block wrapper for ``CrossRecordOnlyAdapter``."""

    def __init__(self, base_mid_block: nn.Module, cross_adapter: CrossRecordOnlyAdapter) -> None:
        super().__init__()
        self.base_mid_block = base_mid_block
        self.cross_adapter = cross_adapter
        self.has_cross_attention = bool(getattr(base_mid_block, "has_cross_attention", False))
        self._valid_mask: torch.Tensor | None = None
        self._bypass = False

    def configure(self, valid_mask: torch.Tensor, *, bypass: bool = False) -> None:
        if valid_mask.ndim != 2 or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean [patients, slots]")
        self._valid_mask = valid_mask
        self._bypass = bool(bypass)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        hidden = self.base_mid_block(*args, **kwargs)
        if self._bypass:
            return hidden
        if self._valid_mask is None:
            raise RuntimeError("patient-set layout must be configured before UNet forward")
        return self.cross_adapter(hidden, self._valid_mask)
