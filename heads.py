"""Prediction heads for the Dreamer V4 world model.

Reward head:   latent -> symexp twohot distribution (255-bin categorical)
Continue head: latent -> sigmoid probability that the episode continues
Policy head:   latent -> Gaussian action distribution (μ, σ) for behavior cloning

Following DreamerV3 (Hafner et al., 2024), the reward head uses the symexp
twohot parameterization: a categorical distribution over 255 exponentially-
spaced bins, trained with categorical cross-entropy on twohot-encoded targets.
This decouples gradient scale from target magnitude, enabling robust learning
across diverse reward scales.

Following DreamerV4, the heads support multi-token prediction (MTP) with
L+1 independent output layers sharing a common MLP backbone.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Symlog / Symexp transforms ───────────────────────────────────────────

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithmic compression: sign(x) * ln(|x| + 1).

    Compresses large magnitudes while preserving sign and zero.
    """
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1).

    Recovers real-valued quantities from symlog-space representations.
    """
    return torch.sign(x) * (torch.exp(x.abs()) - 1.0)


# ── Twohot encoding ─────────────────────────────────────────────────────

def twohot_encode(x: torch.Tensor, bins_symlog: torch.Tensor) -> torch.Tensor:
    """Encode real-valued scalars as twohot vectors over symlog bins.

    Targets are mapped to symlog space and interpolated between the two
    nearest bins.  The bins are assumed uniformly spaced in symlog space
    (e.g. linspace(-20, 20, 255)).

    Args:
        x:            (*batch) real-valued scalar targets
        bins_symlog:  (N,) uniformly spaced bin positions in symlog space
    Returns:
        (*batch, N) twohot encoded soft-target vectors
    """
    N = bins_symlog.shape[0]
    x_symlog = symlog(x)

    lo = bins_symlog[0]
    hi = bins_symlog[-1]
    # Continuous position in the bin grid: [0, N-1]
    pos = (x_symlog - lo) / (hi - lo) * (N - 1)
    pos = pos.clamp(0, N - 1)

    k = pos.long().clamp(max=N - 2)   # lower bin index
    w = pos - k.float()               # interpolation weight ∈ [0, 1]

    # XLA triggers graph recompilation if shapes vary. `scatter_` is a single fixed-graph operation that XLA can compile once and reuse — it's the TPU-friendly way to do indexed writes.   
    result = x.new_zeros(*x.shape, N)
    result.scatter_(-1, k.unsqueeze(-1), (1.0 - w).unsqueeze(-1))
    result.scatter_add_(-1, (k + 1).unsqueeze(-1), w.unsqueeze(-1))

    return result


# ── MLP backbone ─────────────────────────────────────────────────────────

class MLPBackbone(nn.Module):
    """Shared MLP backbone: (num_layers - 1) hidden layers.

    Architecture per layer:  Linear -> LayerNorm -> SiLU
    The final output projection is handled separately by the head.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512,
                 num_layers: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Reward Head ──────────────────────────────────────────────────────────

class RewardHead(nn.Module):
    """Predicts scalar reward via symexp twohot distribution.

    DreamerV3/V4: outputs logits for a categorical distribution over
    ``num_bins`` exponentially-spaced bins.  Trained with cross-entropy
    on twohot-encoded targets.  Predictions are the expected value
    (softmax-weighted sum of real-space bin positions).

    Supports multi-token prediction (MTP) with independent output
    layers per temporal offset, sharing a common MLP backbone.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        num_bins: int = 255,
        num_layers: int = 4,
        mtp_length: int = 0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.mtp_length = mtp_length
        num_outputs = mtp_length + 1 if mtp_length > 0 else 1

        bins_symlog = torch.linspace(-20.0, 20.0, num_bins)
        self.register_buffer("bins_symlog", bins_symlog)
        self.register_buffer("bins_real", symexp(bins_symlog))

        self.backbone = MLPBackbone(latent_dim, hidden_dim, num_layers)

        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_bins) for _ in range(num_outputs)
        ])

        # Zero-init output weights — prevents large early predictions
        # (DreamerV3 Section "Critic learning")
        for head in self.output_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, h: torch.Tensor, mtp_offset: int = 0) -> torch.Tensor:
        """Get logits for a specific MTP offset.

        Args:
            h:          (B, T, D) or (B, D) latent states
            mtp_offset: which MTP output layer to use (0 = current timestep)
        Returns:
            (B, T, num_bins) or (B, num_bins) logits
        """
        features = self.backbone(h)
        return self.output_heads[mtp_offset](features)

    def loss(self, h: torch.Tensor, reward_true: torch.Tensor) -> torch.Tensor:
        """Categorical cross-entropy with twohot targets (single-step).

        Uses only the first output head (mtp_offset=0).  For MTP training,
        use loss_mtp() instead.

        Args:
            h:           (B, T, D) latent states
            reward_true: (B, T) real-valued rewards
        Returns:
            scalar loss
        """
        #TODO Need to modify this for muti head training after initial pipeline
        logits = self.forward(h, mtp_offset=0)                     # (B, T, num_bins)
        target = twohot_encode(reward_true, self.bins_symlog)      # (B, T, num_bins)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target * log_probs).sum(dim=-1).mean()

    def loss_mtp(self, h: torch.Tensor, rewards_future: torch.Tensor) -> torch.Tensor:
        """MTP loss: sum of cross-entropy losses across temporal offsets.

        Args:
            h:              (B, T, D) latent states at each position
            rewards_future: (B, T, L+1) rewards at offsets 0..L from each position
        Returns:
            scalar loss (averaged over batch and time, summed over offsets)
        """
        features = self.backbone(h)                                # (B, T, hidden)
        num_offsets = min(rewards_future.shape[-1], len(self.output_heads))
        losses = []
        for n in range(num_offsets):
            logits = self.output_heads[n](features)                # (B, T, num_bins)
            target = twohot_encode(rewards_future[..., n], self.bins_symlog)
            log_probs = F.log_softmax(logits, dim=-1)
            losses.append(-(target * log_probs).sum(dim=-1).mean())
        return torch.stack(losses).sum()

    def predict(self, h: torch.Tensor, mtp_offset: int = 0) -> torch.Tensor:
        """Predict real-valued reward (expected value under distribution).

        Args:
            h: (B, T, D) or (B, D)
            mtp_offset: which MTP output to use
        Returns:
            same shape as input minus last dim
        """
        logits = self.forward(h, mtp_offset)
        probs = F.softmax(logits, dim=-1)
        return (probs * self.bins_real).sum(dim=-1)


# ── Continue Head ────────────────────────────────────────────────────────

class ContinueHead(nn.Module):
    """Predicts probability that the episode continues (1 - done).

    Training:  loss = BCE(head(h), 1.0 - done)
    Inference: continue_prob = sigmoid(head(h))

    Supports MTP with independent output layers.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 4,
        mtp_length: int = 0,
    ):
        super().__init__()
        self.mtp_length = mtp_length
        num_outputs = mtp_length + 1 if mtp_length > 0 else 1

        self.backbone = MLPBackbone(latent_dim, hidden_dim, num_layers)

        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_outputs)
        ])

        for head in self.output_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, h: torch.Tensor, mtp_offset: int = 0) -> torch.Tensor:
        """Raw logits (pre-sigmoid).

        Args:
            h: (B, T, D) or (B, D)
        Returns:
            (B, T, 1) or (B, 1) logits
        """
        features = self.backbone(h)
        return self.output_heads[mtp_offset](features)

    def loss(self, h: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy against continue target = 1 - done.

        Args:
            h:    (B, T, D) latent states
            done: (B, T) boolean or float (1.0 = terminal)
        Returns:
            scalar loss
        """
        logits = self.forward(h, mtp_offset=0).squeeze(-1)         # (B, T)
        target = 1.0 - done.float()                                # 1 = continue
        return F.binary_cross_entropy_with_logits(logits, target)

    def loss_mtp(self, h: torch.Tensor, dones_future: torch.Tensor) -> torch.Tensor:
        """MTP loss across temporal offsets.

        Args:
            h:            (B, T, D) latent states
            dones_future: (B, T, L+1) done flags at offsets 0..L
        Returns:
            scalar loss
        """
        features = self.backbone(h)
        num_offsets = min(dones_future.shape[-1], len(self.output_heads))
        losses = []
        for n in range(num_offsets):
            logits = self.output_heads[n](features).squeeze(-1)
            target = 1.0 - dones_future[..., n].float()
            losses.append(F.binary_cross_entropy_with_logits(logits, target))
        return torch.stack(losses).sum()

    def predict(self, h: torch.Tensor, mtp_offset: int = 0) -> torch.Tensor:
        """Predict continue probability in [0, 1].

        Args:
            h: (B, T, D) or (B, D)
        Returns:
            same shape as input minus last dim
        """
        return torch.sigmoid(self.forward(h, mtp_offset).squeeze(-1))


# ── Policy Head ─────────────────────────────────────────────────────────

class PolicyHead(nn.Module):
    """Predicts continuous actions via diagonal Gaussian distribution.

    Training (Phase 2):  L_bc = -log π(a_t | z_{≤t})  (behavior cloning)
    Inference (Phase 3): a ~ N(μ, σ²)  (reparameterized sampling)

    Supports MTP with independent output layers per temporal offset.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        latent_dim: int = 512,
        action_dim: int = 6,
        hidden_dim: int = 512,
        num_layers: int = 4,
        mtp_length: int = 0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.mtp_length = mtp_length
        num_outputs = mtp_length + 1 if mtp_length > 0 else 1

        self.backbone = MLPBackbone(latent_dim, hidden_dim, num_layers)

        # Each output head produces (μ, log_σ) per action dim → 2 * action_dim
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim * 2) for _ in range(num_outputs)
        ])

        for head in self.output_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, h: torch.Tensor, mtp_offset: int = 0):
        """Get (μ, log_σ) for a specific MTP offset.

        Args:
            h:          (B, T, D) or (B, D) latent states
            mtp_offset: which MTP output layer to use
        Returns:
            mu:      (*batch, action_dim)
            log_std: (*batch, action_dim) clamped to [LOG_STD_MIN, LOG_STD_MAX]
        """
        features = self.backbone(h)
        raw = self.output_heads[mtp_offset](features)
        mu, log_std = raw.split(self.action_dim, dim=-1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def _log_prob(self, mu, log_std, actions):
        """Diagonal Gaussian log-probability.

        Args:
            mu:      (*batch, action_dim)
            log_std: (*batch, action_dim)
            actions: (*batch, action_dim)
        Returns:
            (*batch) log-probability summed over action dims
        """
        var = (2 * log_std).exp()
        # -0.5 * (((a - μ)² / σ²) + 2*log_σ + log(2π))
        log_p = -0.5 * (((actions - mu) ** 2) / var + 2 * log_std + math.log(2 * math.pi))
        return log_p.sum(dim=-1)

    def loss(self, h: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Behavior cloning loss: negative log-probability (single-step).

        Args:
            h:       (B, T, D) latent states
            actions: (B, T, action_dim) ground-truth actions
        Returns:
            scalar loss
        """
        mu, log_std = self.forward(h, mtp_offset=0)
        return -self._log_prob(mu, log_std, actions).mean()

    def loss_mtp(self, h: torch.Tensor, actions_future: torch.Tensor) -> torch.Tensor:
        """MTP loss: sum of negative log-prob across temporal offsets.

        Args:
            h:              (B, T, D) latent states
            actions_future: (B, T, L+1, action_dim) actions at offsets 0..L
        Returns:
            scalar loss (averaged over batch/time, summed over offsets)
        """
        features = self.backbone(h)
        num_offsets = min(actions_future.shape[-2], len(self.output_heads))
        losses = []
        for n in range(num_offsets):
            raw = self.output_heads[n](features)
            mu, log_std = raw.split(self.action_dim, dim=-1)
            log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
            losses.append(-self._log_prob(mu, log_std, actions_future[..., n, :]).mean())
        return torch.stack(losses).sum()

    def sample(self, h: torch.Tensor, mtp_offset: int = 0) -> torch.Tensor:
        """Reparameterized sample for Phase 3 imagination.

        Args:
            h: (B, T, D) or (B, D)
        Returns:
            (*batch, action_dim) sampled actions
        """
        mu, log_std = self.forward(h, mtp_offset)
        std = log_std.exp()
        return mu + std * torch.randn_like(std)

    def predict(self, h: torch.Tensor, mtp_offset: int = 0) -> torch.Tensor:
        """Deterministic action prediction (mean of distribution).

        Args:
            h: (B, T, D) or (B, D)
        Returns:
            (*batch, action_dim) mean actions
        """
        mu, _ = self.forward(h, mtp_offset)
        return mu


# ── Utility: mean-pool latent tokens ─────────────────────────────────────

# def pool_latents(z: torch.Tensor) -> torch.Tensor:
#     """Mean-pool across latent tokens: (B, T, S, D) -> (B, T, D).

#     S=32 latent tokens each capture different spatial aspects of the frame.
#     Mean-pooling creates a single summary vector per timestep.
#     """
#     return z.mean(dim=-2)
