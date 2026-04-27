"""Phase 3: Imagination Training (Dreamer V4, Section 3.3).

Trains policy_head + value_head via PMPO + TD(λ) on rollouts generated
by the frozen Phase 2 world model. No environment interaction required.
"""

from .config import ImaginationConfig
from .algorithms import (
    compute_lambda_returns,
    compute_advantages,
    value_loss,
    pmpo_policy_loss,
)
from .rollout import imagine_rollout
from .trainer import ImaginationTrainer

__all__ = [
    "ImaginationConfig",
    "ImaginationTrainer",
    "compute_lambda_returns",
    "compute_advantages",
    "value_loss",
    "pmpo_policy_loss",
    "imagine_rollout",
]
