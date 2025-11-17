"""Masking utilities for the masked autoencoder tokenizer."""

from __future__ import annotations

import torch
from torch import Tensor


def sample_random_mask(
    batch: int,
    seq_len: int,
    mask_prob_min: float,
    mask_prob_max: float,
    device: torch.device,
) -> Tensor:
    """Draw a random mask probability per sample and apply Bernoulli masking."""

    probs = torch.empty(batch, device=device).uniform_(mask_prob_min, mask_prob_max)
    masks = torch.bernoulli(probs[:, None].expand(batch, seq_len))
    return masks.bool()


def apply_mask(tokens: Tensor, mask: Tensor, mask_token: Tensor) -> Tensor:
    """Replace masked positions with a learnable mask token."""

    mask = mask.unsqueeze(-1)
    return torch.where(mask, mask_token, tokens)


__all__ = ["sample_random_mask", "apply_mask"]
