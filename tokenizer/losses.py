"""Loss modules for the Phase 1 tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossOutputs:
    total_loss: torch.Tensor
    mse_loss: torch.Tensor
    lpips_loss: Optional[torch.Tensor]
    mask_ratio: torch.Tensor
    metrics: Dict[str, torch.Tensor]


class LossNormalizer(nn.Module):
    """Per-loss-term EMA-RMS normalization (Dreamer V4 §3.1)."""

    def __init__(self, ema_decay: float = 0.99, eps: float = 1e-6):
        super().__init__()
        self.ema_decay = ema_decay
        self.eps = eps
        self.register_buffer("ema_sq", torch.tensor(1.0))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, loss: torch.Tensor, update_ema: bool = True) -> torch.Tensor:
        if self.training and update_ema:
            with torch.no_grad():
                cur = loss.detach() ** 2
                if not bool(self.initialized):
                    self.ema_sq.copy_(cur)
                    self.initialized.fill_(True)
                else:
                    self.ema_sq.mul_(self.ema_decay).add_(cur, alpha=1 - self.ema_decay)
        rms = self.ema_sq.sqrt().clamp_min(self.eps)
        return loss / rms


class MaskedAutoencoderLoss(nn.Module):
    """Combines reconstruction losses (MSE + LPIPS) for masked autoencoding.

    Per Dreamer V4 §3.1 formula (5): total = MSE + 0.2 * LPIPS, with both
    terms RMS-EMA-normalized via `LossNormalizer` so that the 0.2 weight
    is applied to balanced quantities rather than raw magnitudes that
    differ by ~100x.
    """

    def __init__(
        self,
        lpips_module: Optional[nn.Module] = None,
        use_loss_normalization: bool = True,
    ):
        super().__init__()
        self.lpips = lpips_module
        self.use_loss_normalization = use_loss_normalization
        # Freeze the LPIPS perceptual network. Its ~138M VGG16 params should
        # produce gradients into the tokenizer reconstruction (perceptual signal)
        # but never receive their own optimizer updates. Without this, AdamW
        # creates ~1.6 GB of optimizer state for VGG.
        if self.lpips is not None:
            self.lpips.eval()
            for p in self.lpips.parameters():
                p.requires_grad_(False)

        # Per-term normalizers (paper-spec). Always constructed so checkpoints
        # are forward-compatible; behavior gated by use_loss_normalization.
        self.mse_normalizer = LossNormalizer()
        self.lpips_normalizer = LossNormalizer() if self.lpips is not None else None

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        patch_size: tuple[int, int] = (8, 8),
        normalize_by: str = "pixels",
    ) -> LossOutputs:

       
        # Masked-only MSE forces decoder to reconstruct masked patches via latents,
        # creating the gradient pressure that makes scaled tanh viable.
        # mask shape: (B, T*N) bool, True = MASKED. recon/target: (B, T, C, H, W).
        B, T, C, H, W = recon.shape
        ph, pw = patch_size
        gh, gw = H // ph, W // pw  # patch grid
        # Reshape mask (B, T*N) -> (B, T, gh, gw) and expand to pixel resolution.
        pixel_mask = mask.view(B, T, gh, gw).float()
        pixel_mask = pixel_mask.repeat_interleave(ph, dim=-2).repeat_interleave(pw, dim=-1)
        pixel_mask = pixel_mask.unsqueeze(2)  # (B, T, 1, H, W) — broadcast over C
        # MSE per masked-pixel-element. denom = (#masked pixels) * C.
        diff_sq = (recon - target).pow(2)
        masked_sq = diff_sq * pixel_mask
        denom = pixel_mask.sum().clamp_min(1.0) * C
        mse_loss = masked_sq.sum() / denom

        lpips_loss = None
        if self.lpips is not None:
            
            # Fix: hybrid recon (prediction at masked positions, target at unmasked)
            # → LPIPS gradient ONLY flows back through masked regions.
            hybrid_recon = torch.where(
                pixel_mask.bool().expand_as(recon), recon, target
            )
            b, t = target.shape[:2]
            hybrid_video = hybrid_recon.view(b * t, *hybrid_recon.shape[2:])
            target_video = target.view(b * t, *target.shape[2:])
            # LPIPS' pretrained backbone expects [-1, 1]; frames are stored as [0, 1].
            hybrid_lpips_input = hybrid_video.clamp(0.0, 1.0) * 2 - 1
            target_lpips_input = target_video * 2 - 1
            lpips_val = self.lpips(hybrid_lpips_input, target_lpips_input)
            lpips_loss = lpips_val.mean()

        # Paper-spec: each loss term is RMS-EMA-normalized before weighting.
        # The 0.2 LPIPS weight is then a clean balance between normalized
        # quantities, not a fragile magnitude offset.
        if self.use_loss_normalization:
            mse_for_total = self.mse_normalizer(mse_loss)
            if lpips_loss is not None:
                lpips_for_total = self.lpips_normalizer(lpips_loss)
            else:
                lpips_for_total = None
        else:
            mse_for_total = mse_loss
            lpips_for_total = lpips_loss

        if lpips_for_total is None:
            total = mse_for_total
        else:
            total = mse_for_total + 0.2 * lpips_for_total

        metrics = {
            "loss/mse": mse_loss.detach(),
            "loss/total": total.detach(),
        }
        if lpips_loss is not None:
            metrics["loss/lpips"] = lpips_loss.detach()
        if self.use_loss_normalization:
            metrics["loss/mse_rms"] = self.mse_normalizer.ema_sq.sqrt().detach()
            if self.lpips_normalizer is not None:
                metrics["loss/lpips_rms"] = self.lpips_normalizer.ema_sq.sqrt().detach()

        return LossOutputs(
            total_loss=total,
            mse_loss=mse_loss,
            lpips_loss=lpips_loss,
            mask_ratio=mask.float().mean(),
            metrics=metrics,
        )


__all__ = ["MaskedAutoencoderLoss", "LossOutputs", "LossNormalizer"]
