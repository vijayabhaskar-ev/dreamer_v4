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


# ── MTP padding mask ─────────────────────────────────────────────────────

def _mtp_valid_mean(per_pos_loss: torch.Tensor, offset: int) -> torch.Tensor:
    """Mean of a (B, T) per-position loss over MTP-valid positions only.

    Multi-token-prediction targets at offset `n` come from timestep t+n, so the
    last `n` positions of a length-T sequence have no real target — they were
    zero/terminal-padded by build_mtp_targets / build_mtp_action_targets
    (reward→0, done→1, action→0). Averaging over them trains the heads toward the
    padding value and blows up the high-offset losses. Exclude them so each
    per-offset loss is averaged over valid positions only.
    """
    seq_len = per_pos_loss.shape[1]
    pos = torch.arange(seq_len, device=per_pos_loss.device)
    valid = (pos < (seq_len - offset)).to(per_pos_loss.dtype)   # (T,)  last `offset` are padding
    valid = valid.unsqueeze(0).expand_as(per_pos_loss)          # (B, T)
    denom = valid.sum().clamp(min=1.0)
    return (per_pos_loss * valid).sum() / denom


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

    # `scatter_` is a single fixed-shape operation, so it's an efficient way to do indexed writes without dynamic-shape branching.
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

    def loss(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Categorical cross-entropy with twohot targets (single-step).

        Uses only the first output head (mtp_offset=0).  For MTP training,
        use loss_mtp() instead.

        Args:
            h:      (B, T, D) latent states
            target: (B, T) real-valued scalar targets
                    (rewards for reward_head, λ-returns for value_head)
        Returns:
            scalar loss
        """
        #TODO Need to modify this for muti head training after initial pipeline
        logits = self.forward(h, mtp_offset=0)                   # (B, T, num_bins)
        twohot_target = twohot_encode(target, self.bins_symlog)  # (B, T, num_bins)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(twohot_target * log_probs).sum(dim=-1).mean()
    
    def loss_mtp(self, h: torch.Tensor, targets_future: torch.Tensor) -> torch.Tensor:
        """MTP loss: sum of cross-entropy losses across temporal offsets.

        Args:
            h:              (B, T, D) latent states at each position
            targets_future: (B, T, L+1) scalar targets at offsets 0..L
                            (rewards for reward_head, λ-returns for value_head)
        Returns:
            scalar loss (averaged over batch and time, summed over offsets)
        """
        features = self.backbone(h)                                # (B, T, hidden)
        num_offsets = min(targets_future.shape[-1], len(self.output_heads))
        losses = []
        for n in range(num_offsets):
            logits = self.output_heads[n](features)                # (B, T, num_bins)
            twohot_target = twohot_encode(targets_future[..., n], self.bins_symlog)
            log_probs = F.log_softmax(logits, dim=-1)
            ce = -(twohot_target * log_probs).sum(dim=-1)          # (B, T) per-position
            losses.append(_mtp_valid_mean(ce, n))                  # exclude padded tail (t+n >= T)
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
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")  # (B, T)
            losses.append(_mtp_valid_mean(bce, n))                 # exclude padded tail (t+n >= T)
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
            nll = -self._log_prob(mu, log_std, actions_future[..., n, :])   # (B, T) per-position
            losses.append(_mtp_valid_mean(nll, n))                          # exclude padded tail
        return torch.stack(losses).sum()

    def sample(self, h: torch.Tensor, mtp_offset: int = 0) -> torch.Tensor:
        """Draw a stochastic action ~ N(μ, σ²) for Phase 3 imagination rollouts.

        Plain sampling — NOT a pathwise/reparameterized gradient path. Every
        call site runs under @torch.no_grad() (the rollout) or in eval, and
        PMPO consumes the action as a fixed label for its score-function
        gradient (the policy gradient flows only through _log_prob's μ/log_std
        on a fresh forward). There is no action to backprop through: Dreamer V4
        uses PMPO precisely to avoid the pathwise actor gradients that an
        explicit μ + σ·ε reparameterization would enable.

        Args:
            h: (B, T, D) or (B, D)
        Returns:
            (*batch, action_dim) sampled actions
        """
        mu, log_std = self.forward(h, mtp_offset)
        return torch.normal(mu, log_std.exp())

    def predict(self, h: torch.Tensor, mtp_offset: int = 0) -> torch.Tensor:
        """Deterministic action prediction (mean of distribution).

        Args:
            h: (B, T, D) or (B, D)
        Returns:
            (*batch, action_dim) mean actions
        """
        mu, _ = self.forward(h, mtp_offset)
        return mu

    # ── head-agnostic interface (mirrored by CategoricalPolicyHead, for PMPO) ──
    def log_prob(self, h: torch.Tensor, actions: torch.Tensor, mtp_offset: int = 0) -> torch.Tensor:
        """Σ_dims log π(a | h) — diagonal-Gaussian log-prob. (*batch).

        Lets PMPO treat the Gaussian and categorical heads uniformly.
        """
        mu, log_std = self.forward(h, mtp_offset)
        return self._log_prob(mu, log_std, actions)

    def kl_to(self, prior: "PolicyHead", h: torch.Tensor, mtp_offset: int = 0) -> torch.Tensor:
        """KL[N_self(·|h) ‖ N_prior(·|h)] summed over action dims. (*batch).

        Closed-form diagonal-Gaussian KL — the term previously inlined in
        pmpo_policy_loss.
        """
        mu, log_std = self.forward(h, mtp_offset)
        with torch.no_grad():
            mu_prior, log_std_prior = prior.forward(h, mtp_offset)
        sigma = log_std.exp()
        sigma_prior = log_std_prior.exp()
        kl = (log_std_prior - log_std) + (sigma**2 + (mu - mu_prior)**2) / (2 * sigma_prior**2) - 0.5  #TODO Maybe avoid the square operation and save one exp per forward
        return kl.sum(dim=-1)

    def action_std(self, h: torch.Tensor, mtp_offset: int = 0) -> torch.Tensor:
        """Mean per-dim action std σ — scalar collapse/exploration diagnostic.

        Head-agnostic: the categorical head reports the same quantity (the std
        of its bin distribution in action units), so `stats/policy_std` stays
        comparable across head types. → 0 signals a collapsed/deterministic policy.
        """
        _, log_std = self.forward(h, mtp_offset)
        return log_std.exp().mean()


# ── Categorical Policy Head (Dreamer V4 §3.3 — paper-faithful discrete head) ──

class CategoricalPolicyHead(nn.Module):
    """Per-dimension categorical action head — Dreamer V4's policy parameterization.

    The paper (§3.3) parameterizes the policy as a "categorical or vectorized
    binary distribution"; for continuous control we port that by discretizing
    each action dim into ``num_bins`` bins uniformly spaced in
    [-action_bound, action_bound]. The joint policy is the product of these
    independent per-dim categoricals (a "vectorized"/factored categorical).

    Why this over a diagonal Gaussian: a categorical is inherently MULTIMODAL —
    it can put mass on swing-left AND swing-right with ~0 between, the exact
    bimodal target an MLE Gaussian collapses to its mean (μ→0, the documented
    BC mode-averaging failure). It also models the action spread honestly (no
    single σ) and needs no tanh/Jacobian machinery.

    Drop-in for ``PolicyHead`` on the BC + eval paths (same constructor and
    ``forward``/``loss``/``loss_mtp``/``sample``/``predict``). For Phase-3 it
    exposes the HEAD-AGNOSTIC ``log_prob(h, a)`` and ``kl_to(prior, h)``;
    ``pmpo_policy_loss`` must be refactored to call these instead of assuming
    Gaussian (μ, log_σ) params + the inlined Gaussian KL.

    NOTE: discretization snaps actions to the nearest bin (≤ bound/(num_bins-1)
    error). num_bins=41 over [-1, 1] → ±0.025; raise num_bins for finer control.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        action_dim: int = 6,
        hidden_dim: int = 512,
        num_layers: int = 4,
        mtp_length: int = 0,
        num_bins: int = 41,
        action_bound: float = 1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.mtp_length = mtp_length
        self.action_bound = float(action_bound)
        num_outputs = mtp_length + 1 if mtp_length > 0 else 1

        self.backbone = MLPBackbone(latent_dim, hidden_dim, num_layers)
        # Each output head produces num_bins logits per action dim.
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim * num_bins) for _ in range(num_outputs)
        ])
        for head in self.output_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        # Uniform bin centers in [-bound, bound]; a buffer so .to(device) moves it.
        self.register_buffer(
            "bin_centers", torch.linspace(-self.action_bound, self.action_bound, num_bins)
        )

    # ── distribution params ──────────────────────────────────────────────
    def _logits(self, h: torch.Tensor, mtp_offset: int = 0) -> torch.Tensor:
        """(*batch, action_dim, num_bins) per-dim logits."""
        raw = self.output_heads[mtp_offset](self.backbone(h))         # (*b, A*K)
        return raw.unflatten(-1, (self.action_dim, self.num_bins))    # (*b, A, K)

    def forward(self, h: torch.Tensor, mtp_offset: int = 0) -> torch.Tensor:
        """Per-dim logits (*batch, action_dim, num_bins).

        Unlike the Gaussian PolicyHead (which returns (μ, log_σ)), this returns
        logits — so PMPO must use ``log_prob``/``kl_to``, not (μ, log_σ).
        """
        return self._logits(h, mtp_offset)

    def _nearest_bin(self, actions: torch.Tensor) -> torch.Tensor:
        """Real actions → nearest bin index per dim. (*batch, action_dim)."""
        a = actions.clamp(-self.action_bound, self.action_bound)
        pos = (a + self.action_bound) / (2 * self.action_bound) * (self.num_bins - 1)
        return pos.round().long().clamp(0, self.num_bins - 1)

    # ── log-prob (head-agnostic interface) ───────────────────────────────
    def log_prob(self, h: torch.Tensor, actions: torch.Tensor, mtp_offset: int = 0) -> torch.Tensor:
        """Σ_dims log π(a_d | h) — factored categorical log-prob. (*batch)."""
        logp = self._logits(h, mtp_offset).log_softmax(dim=-1)        # (*b, A, K)
        bins = self._nearest_bin(actions)                            # (*b, A)
        lp = logp.gather(-1, bins.unsqueeze(-1)).squeeze(-1)         # (*b, A)
        return lp.sum(-1)                                            # (*b)

    # ── BC losses ────────────────────────────────────────────────────────
    def loss(self, h: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Behavior-cloning NLL (single-step, offset 0)."""
        return -self.log_prob(h, actions, mtp_offset=0).mean()

    def loss_mtp(self, h: torch.Tensor, actions_future: torch.Tensor) -> torch.Tensor:
        """MTP NLL: sum over offsets of masked per-position negative log-prob.

        actions_future: (B, T, L+1, action_dim).
        """
        feats = self.backbone(h)
        num_offsets = min(actions_future.shape[-2], len(self.output_heads))
        losses = []
        for n in range(num_offsets):
            raw = self.output_heads[n](feats).unflatten(-1, (self.action_dim, self.num_bins))
            logp = raw.log_softmax(dim=-1)
            bins = self._nearest_bin(actions_future[..., n, :])
            lp = logp.gather(-1, bins.unsqueeze(-1)).squeeze(-1).sum(-1)   # (B, T)
            losses.append(_mtp_valid_mean(-lp, n))                         # exclude padded tail
        return torch.stack(losses).sum()

    # ── sampling / deterministic readout ─────────────────────────────────
    def sample(self, h: torch.Tensor, mtp_offset: int = 0) -> torch.Tensor:
        """Sample one bin per dim ~ Categorical(softmax(logits)) → bin center.

        Multimodal: split mass across bins yields genuinely different draws.
        """
        probs = self._logits(h, mtp_offset).softmax(dim=-1)          # (*b, A, K)
        idx = torch.multinomial(probs.reshape(-1, self.num_bins), 1)
        idx = idx.reshape(probs.shape[:-1])                          # (*b, A)
        return self.bin_centers[idx]

    def predict(self, h: torch.Tensor, mtp_offset: int = 0) -> torch.Tensor:
        """Deterministic action: argmax bin per dim → bin center. (*batch, A)."""
        idx = self._logits(h, mtp_offset).argmax(dim=-1)             # (*b, A)
        return self.bin_centers[idx]

    # ── reverse KL to a prior (head-agnostic interface for PMPO) ──────────
    def kl_to(self, prior: "CategoricalPolicyHead", h: torch.Tensor, mtp_offset: int = 0) -> torch.Tensor:
        """KL[π_self(·|h) ‖ π_prior(·|h)] summed over action dims. (*batch).

        Exact closed form for factored categoricals — replaces the Gaussian KL
        inlined in pmpo_policy_loss.
        """
        logp = self._logits(h, mtp_offset).log_softmax(dim=-1)       # (*b, A, K)
        with torch.no_grad():
            logq = prior._logits(h, mtp_offset).log_softmax(dim=-1)
        kl = (logp.exp() * (logp - logq)).sum(-1)                    # (*b, A)
        return kl.sum(-1)                                            # (*b)

    def action_std(self, h: torch.Tensor, mtp_offset: int = 0) -> torch.Tensor:
        """Mean per-dim action std σ — analytic std of each per-dim categorical
        over its bin centers, in action units (head-agnostic counterpart of the
        Gaussian head's σ). → 0 as the policy collapses onto a single bin.
        """
        probs = self._logits(h, mtp_offset).softmax(dim=-1)          # (*b, A, K)
        centers = self.bin_centers                                   # (K,)
        mean = (probs * centers).sum(-1, keepdim=True)               # (*b, A, 1)
        var = (probs * (centers - mean) ** 2).sum(-1)                # (*b, A)
        return var.clamp_min(0).sqrt().mean()


# ── Utility: mean-pool latent tokens ─────────────────────────────────────

# def pool_latents(z: torch.Tensor) -> torch.Tensor:
#     """Mean-pool across latent tokens: (B, T, S, D) -> (B, T, D).

#     S=32 latent tokens each capture different spatial aspects of the frame.
#     Mean-pooling creates a single summary vector per timestep.
#     """
#     return z.mean(dim=-2)
