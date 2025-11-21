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


class MaskedAutoencoderLoss(nn.Module): #TODO Need to refactor including handling of total with LPIPS
    """Combines reconstruction losses (MSE + LPIPS) for masked autoencoding."""

    def __init__(self, lpips_module: Optional[nn.Module] = None):
        super().__init__()
        self.lpips = lpips_module

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        normalize_by: str = "pixels",
    ) -> LossOutputs:
        mse = F.mse_loss(recon, target, reduction="none")
        if normalize_by == "pixels":
            mse = mse.mean()
        else:
            mse = (mse.view(mse.size(0), -1).mean(dim=-1) * mask.float().mean(dim=-1)).mean()

        lpips_loss = None
        if self.lpips is not None:
            # LPIPS expects NCHW, flatten temporal dimension.
            b, t = target.shape[:2]
            recon_video = recon.view(b * t, *recon.shape[2:])
            target_video = target.view(b * t, *target.shape[2:])
            lpips_val = self.lpips(recon_video, target_video)
            lpips_loss = lpips_val.mean()
        total = mse if lpips_loss is None else mse + lpips_loss

        metrics = {
            "loss/mse": mse.detach(),
            "loss/total": total.detach(),
        }
        if lpips_loss is not None:
            metrics["loss/lpips"] = lpips_loss.detach()

        return LossOutputs(
            total_loss=total,
            mse_loss=mse,
            lpips_loss=lpips_loss,
            mask_ratio=mask.float().mean(),
            metrics=metrics,
        )


__all__ = ["MaskedAutoencoderLoss", "LossOutputs"]
