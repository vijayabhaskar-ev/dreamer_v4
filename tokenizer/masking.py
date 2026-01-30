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
    num_frames: int = 1,
) -> Tensor:
    """
    Draw a random mask probability per sample and apply Bernoulli masking.
    Implements 'Tube Masking' where the mask is consistent across time for video.
    """
    
    if num_frames > 1 and seq_len % num_frames == 0:
        patches_per_frame = seq_len // num_frames
        probs = torch.empty(batch, device=device).uniform_(mask_prob_min, mask_prob_max)
        spatial_mask = torch.bernoulli(probs[:, None].expand(batch, patches_per_frame))
        tube_mask = spatial_mask.repeat(1, num_frames)#TODO this function copies the data. Try to refactor it later
        return tube_mask.bool()
    else:
        probs = torch.empty(batch, device=device).uniform_(mask_prob_min, mask_prob_max)
        masks = torch.bernoulli(probs[:, None].expand(batch, seq_len))
        return masks.bool()


def apply_mask(tokens: Tensor, mask: Tensor, mask_token: Tensor) -> Tensor:
    """Replace masked positions with a learnable mask token."""

    mask = mask.unsqueeze(-1)
    return torch.where(mask, mask_token, tokens)


__all__ = ["sample_random_mask", "apply_mask"]
