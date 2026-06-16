"""Configuration for Phase 3 imagination training (Dreamer V4, Section 3.3).

Hyperparameters follow the paper defaults:
  - gamma = 0.997   (discount factor)
  - lambda = 0.95   (TD(λ) parameter, standard Dreamer value)
  - alpha = 0.5     (PMPO positive/negative advantage balance, Eq. 11)
  - beta = 0.3      (PMPO reverse KL coefficient, Eq. 11)
  - K = 4           (denoising steps per imagined frame)
  - tau_ctx = 0.1   (context corruption level, Section 3.2)

The value head reuses the RewardHead architecture (symexp twohot, 255 bins).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ImaginationConfig:
    # ── RL hyperparameters (paper Section 3.3, Eq. 10 & 11) ────────────
    gamma: float = 0.997
    lambda_: float = 0.95
    imagination_horizon: int = 15       # H — number of imagined steps per context
    pmpo_alpha: float = 0.5
    pmpo_beta: float = 0.3

    # ── Flow denoising during imagination (paper Section 3.2) ──────────
    K_imagination: int = 4              # Shortcut forcing: K=4 steps per frame
    tau_ctx: float = 0.1                # Context corruption level
    K_max: int = 64                     # Used to compute d_ctx = 1/K_max

    # ── Context seeding ────────────────────────────────────────────────
    num_context_frames: int = 4         # Real frames used to seed imagination
    context_window: int = 16            # Sliding window for dynamics attention

    # ── Value head architecture (same as RewardHead) ───────────────────
    value_num_bins: int = 255           # Symexp twohot bins
    value_hidden_dim: int = 512
    value_num_layers: int = 4

    # ── Optimizer ──────────────────────────────────────────────────────
    lr: float = 3e-5                    # Learning rate for policy + value heads
    min_lr: float = 1e-6
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 10.0

    # ── Training loop ──────────────────────────────────────────────────
    epochs: int = 100
    batch_size: int = 16
    steps_per_epoch: int = 200
    device: str = "cuda"

    # ── Logging & checkpointing ────────────────────────────────────────
    log_interval: int = 10
    checkpoint_interval: int = 5
    log_smooth_window: int = 50
