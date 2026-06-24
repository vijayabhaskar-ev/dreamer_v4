"""Unit tests for imagination/algorithms.py — plain runnable script.

Run from project root:
    python -m tests.test_imagination_algorithms

Prints PASS/FAIL per test. Exits 1 if any test fails.
"""

import copy
import sys

import torch

from heads import RewardHead, PolicyHead
from imagination.algorithms import (
    compute_lambda_returns,
    compute_advantages,
    value_loss,
    pmpo_policy_loss,
)


torch.manual_seed(0)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"PASS: {name}")
    else:
        print(f"FAIL: {name} — {detail}")
        failures.append(name)


# ═══════════════════════════════════════════════════════════════════════
# Stage 1 — compute_lambda_returns
# ═══════════════════════════════════════════════════════════════════════

# 1a — MC at γ=λ=1: with zero bootstrap and unit rewards, λ-return at t is
# just the remaining-steps count.
rewards = torch.ones(1, 5)
values = torch.zeros(1, 6)
continues = torch.ones(1, 5)
got = compute_lambda_returns(rewards, values, continues, gamma=1.0, lambda_=1.0)
expected = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
check("1a MC γ=λ=1", torch.allclose(got, expected), f"got={got.tolist()}")

# 1b — discounted MC: γ=0.5, λ=1. Recurrence: R_t = 1 + 0.5 · R_{t+1}, R_4 = 1.
# Backward: R_4=1, R_3=1.5, R_2=1.75, R_1=1.875, R_0=1.9375.
rewards = torch.ones(1, 5)
values = torch.zeros(1, 6)
continues = torch.ones(1, 5)
got = compute_lambda_returns(rewards, values, continues, gamma=0.5, lambda_=1.0)
expected = torch.tensor([[1.9375, 1.875, 1.75, 1.5, 1.0]])
check("1b discounted MC γ=0.5", torch.allclose(got, expected), f"got={got.tolist()}")

# 1c — terminal truncation: continues=[1, 0, 1] at γ=λ=1.
# Backward with bootstrap v_3=0:
#   t=2: R = 1 + 1·1·(0·0 + 1·0) = 1  (acc updates to 1)
#   t=1: R = 1 + 1·0·(...)        = 1  (continues kills future credit, acc=1)
#   t=0: R = 1 + 1·1·(0·1 + 1·1)  = 2
rewards = torch.ones(1, 3)
continues = torch.tensor([[1.0, 0.0, 1.0]])
values = torch.zeros(1, 4)
got = compute_lambda_returns(rewards, values, continues, gamma=1.0, lambda_=1.0)
expected = torch.tensor([[2.0, 1.0, 1.0]])
check("1c terminal truncation", torch.allclose(got, expected), f"got={got.tolist()}")

# 1d — detach invariant: even if values has requires_grad, output must not.
values_grad = torch.zeros(1, 6, requires_grad=True)
got = compute_lambda_returns(
    torch.ones(1, 5), values_grad, torch.ones(1, 5), gamma=1.0, lambda_=1.0,
)
check(
    "1d detach invariant",
    not got.requires_grad,
    f"output.requires_grad = {got.requires_grad}",
)


# ═══════════════════════════════════════════════════════════════════════
# Stage 2 — compute_advantages
# ═══════════════════════════════════════════════════════════════════════

# 2a — basic: lambda_returns - values[:, :H]
lambda_returns = torch.ones(2, 3)
values = torch.zeros(2, 4)
got = compute_advantages(lambda_returns, values)
check("2a basic advantages", torch.allclose(got, torch.ones(2, 3)), f"got={got.tolist()}")

# 2b — off-by-one check: the slicing MUST be values[:, :H], not values[:, 1:].
# With these inputs, the two choices give numerically different results:
#   values[:, :H] → [0 - 10, 0 - 20, 0 - 30] = [-10, -20, -30]   ← correct
#   values[:, 1:] → [0 - 20, 0 - 30, 0 - 40] = [-20, -30, -40]   ← WRONG (TD error)
lambda_returns = torch.zeros(1, 3)
values = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
got = compute_advantages(lambda_returns, values)
expected_correct = torch.tensor([[-10.0, -20.0, -30.0]])
expected_wrong = torch.tensor([[-20.0, -30.0, -40.0]])
check(
    "2b off-by-one check (values[:, :H])",
    torch.allclose(got, expected_correct) and not torch.allclose(got, expected_wrong),
    f"got={got.tolist()}",
)


# ═══════════════════════════════════════════════════════════════════════
# Stage 3 — value_loss
# ═══════════════════════════════════════════════════════════════════════

B, H, D = 2, 3, 512
value_head = RewardHead(latent_dim=D)
states = torch.randn(B, H, D)
lambda_returns = torch.randn(B, H)

# 3a — returns scalar tensor
loss = value_loss(value_head, states, lambda_returns)
check(
    "3a value_loss scalar",
    loss.dim() == 0,
    f"loss.shape = {tuple(loss.shape)}",
)

# 3b — loss decreases under SGD. Because RewardHead is zero-initialized, the
# softmax is uniform → loss ≈ log(num_bins) regardless of target. But after
# a few SGD steps with fixed target, the model should learn that target and
# the loss should drop below the uniform baseline log(255) ≈ 5.54.
fresh_value_head = RewardHead(latent_dim=D)
opt = torch.optim.SGD(fresh_value_head.parameters(), lr=0.1)
fixed_target = torch.full((B, H), 2.0)
initial_loss = value_loss(fresh_value_head, states, fixed_target).item()
for _ in range(30):
    opt.zero_grad()
    l = value_loss(fresh_value_head, states, fixed_target)
    l.backward()
    opt.step()
final_loss = value_loss(fresh_value_head, states, fixed_target).item()
check(
    "3b value_loss decreases under SGD",
    final_loss < initial_loss - 0.1,
    f"initial={initial_loss:.4f}, final={final_loss:.4f}",
)


# ═══════════════════════════════════════════════════════════════════════
# Stage 4 — pmpo_policy_loss
# ═══════════════════════════════════════════════════════════════════════

B, H, D, A = 2, 3, 512, 6
policy_head = PolicyHead(latent_dim=D, action_dim=A)
states = torch.randn(B, H, D)
actions = torch.randn(B, H, A)

# Compute the log_prob the same way pmpo_policy_loss does, to use as the
# ground-truth expected value in 4a/4b.
with torch.no_grad():
    mu, log_std = policy_head(states)
    log_prob = policy_head._log_prob(mu, log_std, actions)

# 4a — α=0, β=0, all adv < 0 → loss == log_prob.mean()
advantages = -torch.ones(B, H)
policy_prior_4a = copy.deepcopy(policy_head)
loss = pmpo_policy_loss(
    policy_head, policy_prior_4a, states, actions, advantages,
    alpha=0.0, beta=0.0,
)
expected = log_prob.mean()
check(
    "4a α=0, β=0, all-neg adv → mean(log_prob)",
    torch.allclose(loss, expected, atol=1e-5),
    f"loss={loss.item():.6f}, expected={expected.item():.6f}",
)

# 4b — α=1, β=0, all adv ≥ 0 → loss == -log_prob.mean()
advantages = torch.ones(B, H)
policy_prior_4b = copy.deepcopy(policy_head)
loss = pmpo_policy_loss(
    policy_head, policy_prior_4b, states, actions, advantages,
    alpha=1.0, beta=0.0,
)
expected = -log_prob.mean()
check(
    "4b α=1, β=0, all-pos adv → -mean(log_prob)",
    torch.allclose(loss, expected, atol=1e-5),
    f"loss={loss.item():.6f}, expected={expected.item():.6f}",
)

# 4c — identical policies, β=1, σ ≠ 1 → KL=0 exactly, loss == -log_prob.mean().
# This is the test that catches the operator-precedence bug in the KL formula.
# Force σ ≠ 1 by setting the log_std half of the output bias to 0.5, then
# deepcopy so prior has matching parameters.
policy_head_sigma = PolicyHead(latent_dim=D, action_dim=A)
with torch.no_grad():
    policy_head_sigma.output_heads[0].bias[A:].fill_(0.5)  # log_std bias → 0.5, σ ≈ 1.65
policy_prior_sigma = copy.deepcopy(policy_head_sigma)

with torch.no_grad():
    mu_s, log_std_s = policy_head_sigma(states)
    log_prob_s = policy_head_sigma._log_prob(mu_s, log_std_s, actions)

advantages = torch.ones(B, H)
loss = pmpo_policy_loss(
    policy_head_sigma, policy_prior_sigma, states, actions, advantages,
    alpha=1.0, beta=1.0,
)
expected = -log_prob_s.mean()
check(
    "4c identical-policies KL=0 (σ≠1)",
    torch.allclose(loss, expected, atol=1e-5),
    f"loss={loss.item():.6f}, expected={expected.item():.6f}, "
    f"diff={abs(loss.item() - expected.item()):.6e} (should be ≈0)",
)

# 4d — prior gradient isolation: after backward, policy_prior must have no grads.
policy_head_4d = PolicyHead(latent_dim=D, action_dim=A)
policy_prior_4d = copy.deepcopy(policy_head_4d)
for p in policy_prior_4d.parameters():
    p.grad = None
advantages = torch.randn(B, H)
loss = pmpo_policy_loss(
    policy_head_4d, policy_prior_4d, states, actions, advantages,
    alpha=0.5, beta=0.3,
)
loss.backward()
prior_clean = all(
    p.grad is None or p.grad.abs().sum().item() == 0.0
    for p in policy_prior_4d.parameters()
)
check(
    "4d prior gradient isolation (no_grad works)",
    prior_clean,
    "some policy_prior parameter received a non-zero gradient",
)


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════

if failures:
    print(f"\n{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("\nALL PASS")
