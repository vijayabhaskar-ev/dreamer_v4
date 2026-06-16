from __future__ import annotations

import torch
import torch.nn as nn


# ════════════════════════════════════════════════════════════════════════
# 1. λ-returns (Paper Eq. 10)
# ════════════════════════════════════════════════════════════════════════

# λ-returns are pure targets: @torch.no_grad() keeps the whole computation
# outside the autograd graph, so gradients can never flow back through the
# rollout's rewards/values/continues (detach-at-source).
@torch.no_grad()
def compute_lambda_returns(
    rewards: torch.Tensor,      # (B, H)
    values: torch.Tensor,       # (B, H+1)  — values[:, -1] is bootstrap v_T
    continues: torch.Tensor,    # (B, H)    — c_t ∈ [0, 1]
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    """Compute TD(λ) returns for imagined trajectories.

    Paper Eq. 10:
        R^λ_t = r_t + γ · c_t · ((1 - λ) · v_{t+1} + λ · R^λ_{t+1})
        R^λ_T = v_T       (bootstrap — terminal value)

    Args:
        rewards:   (B, H)   imagined rewards r_t for t ∈ [0, H-1]
        values:    (B, H+1) imagined values v_t for t ∈ [0, H]; the last
                            column is the bootstrap value for the terminal
                            state (after the final imagination step).
        continues: (B, H)   continue probabilities c_t ∈ [0, 1]. If c_t = 0,
                            the return truncates at t (no future credit).
                            c_t must share rewards' indexing convention: both
                            are same-frame predictions (Phase-2 heads trained
                            on same-row rewards[t]/dones[t]), so the gate
                            zeroes exactly the future beyond r_t. Only a
                            MIXED convention (rewards and dones indexed
                            differently in the training data) would shift
                            the gate by one step.
        gamma:     discount factor (paper: 0.997)
        lambda_:   TD(λ) mixing parameter (0 = 1-step TD, 1 = full MC)

    Returns:
        (B, H) tensor of λ-returns R^λ_t for t ∈ [0, H-1]
    """
    B, H = rewards.shape
    # Full-tuple checks: dim-1-only checks would still let a (1, H+1) tensor
    # broadcast silently across the batch. Too-short inputs IndexError loudly,
    # but too-LONG values are the silent killer — the bootstrap (values[:, -1])
    # and the recursion (values[:, t+1]) would read from inconsistent columns.
    assert values.shape == (B, H + 1), (
        f"values must be (B, H+1)=({B}, {H + 1}) with bootstrap v_T in the "
        f"last column; got {tuple(values.shape)}"
    )
    assert continues.shape == rewards.shape, (
        f"continues must match rewards shape ({B}, {H}); got {tuple(continues.shape)}"
    )

    lambda_returns = torch.zeros_like(rewards)
    acc = values[:, -1]

    for t in reversed(range(H)):
        acc = rewards[:, t] + gamma * continues[:, t] * ((1 - lambda_) * values[:, t+1] + lambda_ * acc)
        lambda_returns[:, t] = acc

    return lambda_returns


# ════════════════════════════════════════════════════════════════════════
# 2. Advantages (Paper Eq. 11 precursor)
# ════════════════════════════════════════════════════════════════════════

# Advantages are pure targets too — PMPO consumes only their sign, never
# gradients. @torch.no_grad() enforces detach-at-source instead of relying
# on caller discipline.
@torch.no_grad()
def compute_advantages(
    lambda_returns: torch.Tensor,  # (B, H)
    values: torch.Tensor,          # (B, H+1)
) -> torch.Tensor:
    """Compute advantages A_t = R^λ_t - V(s_t).

    Args:
        lambda_returns: (B, H)   TD(λ) returns from compute_lambda_returns()
        values:         (B, H+1) value predictions (only first H used)

    Returns:
        (B, H) advantages
    """
    B, H = lambda_returns.shape
    assert values.shape == (B, H + 1), (
        f"values must be (B, H+1)=({B}, {H + 1}); got {tuple(values.shape)}"
    )
    return lambda_returns - values[:, :H]


# ════════════════════════════════════════════════════════════════════════
# 3. Value loss (Paper Eq. 10)
# ════════════════════════════════════════════════════════════════════════

def value_loss(
    value_head: nn.Module,        # RewardHead instance (symexp twohot, 255 bins)
    states: torch.Tensor,         # (B, H, D_embed) detached agent_out states
    lambda_returns: torch.Tensor, # (B, H) detached target returns
) -> torch.Tensor:
    """Cross-entropy value loss against twohot-encoded λ-return targets.

    Args:
        states:         (B, H, D) detached latent states from imagination
        lambda_returns: (B, H) detached target λ-returns

    Returns:
        scalar loss (averaged over batch and time)
    """
    return value_head.loss(states, lambda_returns)



# ════════════════════════════════════════════════════════════════════════
# 4. PMPO policy loss (Paper Eq. 11)
# ════════════════════════════════════════════════════════════════════════

def pmpo_policy_loss(
    policy_head: nn.Module,        # Current (trainable) PolicyHead
    policy_prior: nn.Module,       # Frozen copy of Phase 2 PolicyHead
    states: torch.Tensor,          # (B, H, D_embed) detached latent states
    actions: torch.Tensor,         # (B, H, action_dim) actions sampled during rollout
    advantages: torch.Tensor,      # (B, H) detached advantages
    alpha: float,                  # 0.5 (paper)
    beta: float,                   # 0.3 (paper)
) -> torch.Tensor:
    """Preference-based MPO policy loss (Paper Eq. 11).

        L_policy = (1-α)/|D-| · Σ_{D-} ln π_θ(a|s)       (minimize log π on bad actions)
                 -  α / |D+|  · Σ_{D+} ln π_θ(a|s)       (maximize log π on good actions)
                 +  β / N     · Σ_i KL[π_θ(·|s_i) ‖ π_prior(·|s_i)]

    where D+ = {(s,a) : A_t ≥ 0} and D- = {(s,a) : A_t < 0}.

    PMPO uses only the SIGN of the advantage, not its magnitude — this
    eliminates the need for advantage normalization.

    KL is REVERSE KL (π_θ ‖ π_prior), constraining the policy to stay inside
    the prior's support and preventing exploitation of world-model errors
    in low-probability regions.

    For diagonal Gaussians N(μ₁, σ₁²) ‖ N(μ₂, σ₂²) per action dim:
        KL = log(σ₂/σ₁) + (σ₁² + (μ₁ - μ₂)²) / (2 σ₂²) - 0.5

    Args:
        policy_head:  trainable policy (produces π_θ)
        policy_prior: frozen policy from Phase 2 (produces π_prior)
        states:       (B, H, D) detached latent states
        actions:      (B, H, action_dim) actions sampled during rollout
        advantages:   (B, H) detached advantages
        alpha:        weight for positive-advantage term
        beta:         coefficient for reverse KL regularization

    Returns:
        scalar loss
    """
    mu, log_std = policy_head(states)

    with torch.no_grad():
        mu_prior, log_std_prior = policy_prior(states)

    log_prob = policy_head._log_prob(mu, log_std, actions)

    # Float masks (not boolean indexing) keep the tensor shapes static and avoid data-dependent control flow.
    pos_mask = (advantages >= 0).float()
    neg_mask = 1 - pos_mask
    pos_term = -(alpha * (pos_mask * log_prob).sum()) / pos_mask.sum().clamp(min=1.0)
    neg_term = (1 - alpha) * (neg_mask * log_prob).sum() / neg_mask.sum().clamp(min=1.0)

    sigma = log_std.exp()
    sigma_prior = log_std_prior.exp()
    kl = (log_std_prior - log_std) + (sigma**2 + (mu - mu_prior)**2) / (2 * sigma_prior**2) - 0.5  #TODO May be avoid the square operation and save one exp per forward
    kl = kl.sum(dim=-1)
    kl_term = beta * kl.mean()

    return pos_term + neg_term + kl_term





