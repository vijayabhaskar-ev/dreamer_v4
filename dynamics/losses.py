"""Loss module for the Dreamer V4 dynamics model (flow matching).

Flow matching objective: the model receives noised latents
z_noised = (1 - tau) * noise + tau * z_clean and predicts z_clean.
Training target is MSE(predicted, z_clean).

Reference: See flow_matching.py for the noise schedule (add_noise, sample_tau_and_d).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DynamicsLossOutputs:
    total_loss: torch.Tensor
    mse_loss: torch.Tensor
    metrics: Dict[str, torch.Tensor]


class FlowMatchingLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        z_predicted: torch.Tensor,
        z_clean: torch.Tensor,
    ) -> DynamicsLossOutputs:
        
        mse_loss = F.mse_loss(z_predicted, z_clean)
        metrics = {
            "loss/mse": mse_loss.detach(),
        }
        return DynamicsLossOutputs(total_loss=mse_loss, mse_loss=mse_loss, metrics=metrics)



__all__ = ["FlowMatchingLoss", "DynamicsLossOutputs"]
