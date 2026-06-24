"""Smoke test for imagination/rollout.py — plain runnable script.

Run from project root:
    python test_imagination_rollout.py

Builds a tiny but real DynamicsModel + tokenizer + heads, runs
`imagine_rollout`, and checks the output contract end-to-end:

  S1   output dict has the five expected keys
  S2-S6 each output tensor has the right shape  (S6 = bootstrap slot)
  S7   no NaN / Inf anywhere
  S8   continue probabilities lie in [0, 1]
  S9   outputs are detached (rollout is @torch.no_grad)
  S10  dynamics-model parameters have no grad after the rollout
  S11  values[:, H] equals value_head.predict(states[:, H-1]) (bootstrap)
  S12  sliding-window slice still works when horizon > context_window
  S13  pmpo_policy_loss backward routes grads to policy_head only
  S14  value_loss backward routes grads to value_head only

Prints PASS/FAIL per check. Exits 1 if any check fails.
"""

import copy
import sys

import torch

from dynamics.config import DynamicsConfig
from dynamics.dynamic_model import DynamicsModel
from heads import ContinueHead, PolicyHead, RewardHead
from imagination.algorithms import (
    compute_advantages,
    compute_lambda_returns,
    pmpo_policy_loss,
    value_loss,
)
from imagination.rollout import imagine_rollout
from tokenizer.config import TokenizerConfig
from tokenizer.tokenizer import MaskedAutoencoderTokenizer


torch.manual_seed(0)

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"PASS: {name}")
    else:
        print(f"FAIL: {name} — {detail}")
        failures.append(name)


def all_grads_zero_or_none(module: torch.nn.Module) -> tuple[bool, str]:
    """Return (ok, first_offender_detail) — True iff every param has no grad
    or a zero-norm grad."""
    for n, p in module.named_parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0.0:
            return False, f"{n} has |grad|.sum()={p.grad.abs().sum().item():.4e}"
    return True, ""


def any_grad_nonzero(module: torch.nn.Module) -> tuple[bool, str]:
    """Return (ok, summary) — True iff at least one param has a non-zero grad."""
    for n, p in module.named_parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0.0:
            return True, f"{n} got |grad|.sum()={p.grad.abs().sum().item():.4e}"
    return False, "no parameter received a non-zero gradient"


# ═══════════════════════════════════════════════════════════════════════
# Build tiny stack — same minimal-dim recipe as test_gradient_isolation.py
# ═══════════════════════════════════════════════════════════════════════

EMBED_DIM = 512
LATENT_DIM = 128
S_Z = 32
ACTION_DIM = 6

tok_cfg = TokenizerConfig(
    image_size=(64, 64), patch_size=(8, 8),
    latent_dim=LATENT_DIM, num_latent_tokens=S_Z, embed_dim=EMBED_DIM,
)
tokenizer = MaskedAutoencoderTokenizer(tok_cfg)

dyn_cfg = DynamicsConfig.from_tokenizer(
    tok_cfg, embed_dim=EMBED_DIM, depth=2, num_heads=8, mtp_length=0,
)
dynamics = DynamicsModel(dyn_cfg, tokenizer)
dynamics.enable_agent_tokens(num_tasks=1)

policy_head = PolicyHead(latent_dim=EMBED_DIM, action_dim=ACTION_DIM)
reward_head = RewardHead(latent_dim=EMBED_DIM)
continue_head = ContinueHead(latent_dim=EMBED_DIM)
value_head = RewardHead(latent_dim=EMBED_DIM)


# ═══════════════════════════════════════════════════════════════════════
# Single rollout — feeds checks S1–S11
# ═══════════════════════════════════════════════════════════════════════

B, C, H = 2, 4, 3
K = 2  # smaller than paper K=4 for test speed
TAU_CTX = 0.1
K_MAX = 64
CONTEXT_WINDOW = 4

z_context = torch.randn(B, C, S_Z, LATENT_DIM)
actions_context = torch.randn(B, C - 1, ACTION_DIM)

# Zero gradients on every module so S10 is meaningful
for m in (dynamics, policy_head, reward_head, continue_head, value_head):
    m.zero_grad(set_to_none=True)

out = imagine_rollout(
    dynamics_model=dynamics,
    policy_head=policy_head,
    reward_head=reward_head,
    continue_head=continue_head,
    value_head=value_head,
    z_context=z_context,
    actions_context=actions_context,
    horizon=H,
    K=K,
    tau_ctx=TAU_CTX,
    K_max=K_MAX,
    context_window=CONTEXT_WINDOW,
)


# ── S1: output structure ────────────────────────────────────────────────

expected_keys = {"states", "actions", "rewards", "continues", "values"}
check(
    "S1 output dict has expected keys",
    isinstance(out, dict) and set(out.keys()) == expected_keys,
    f"got keys={set(out.keys()) if isinstance(out, dict) else type(out)}",
)


# ── S2–S6: output shapes ────────────────────────────────────────────────

check(
    "S2 states.shape == (B, H, D_embed)",
    tuple(out["states"].shape) == (B, H, EMBED_DIM),
    f"got {tuple(out['states'].shape)}",
)
check(
    "S3 actions.shape == (B, H, action_dim)",
    tuple(out["actions"].shape) == (B, H, ACTION_DIM),
    f"got {tuple(out['actions'].shape)}",
)
check(
    "S4 rewards.shape == (B, H)",
    tuple(out["rewards"].shape) == (B, H),
    f"got {tuple(out['rewards'].shape)}",
)
check(
    "S5 continues.shape == (B, H)",
    tuple(out["continues"].shape) == (B, H),
    f"got {tuple(out['continues'].shape)}",
)
check(
    "S6 values.shape == (B, H+1)  [bootstrap slot present]",
    tuple(out["values"].shape) == (B, H + 1),
    f"got {tuple(out['values'].shape)}",
)


# ── S7: numerical sanity ────────────────────────────────────────────────

all_finite = all(
    torch.isfinite(t).all().item() for t in out.values()
)
check("S7 no NaN / Inf in any output", all_finite,
      "at least one output tensor contains NaN or Inf")


# ── S8: continue probabilities ∈ [0, 1] ─────────────────────────────────

c = out["continues"]
check(
    "S8 continues ∈ [0, 1]",
    (c >= 0).all().item() and (c <= 1).all().item(),
    f"min={c.min().item():.4f}, max={c.max().item():.4f}",
)


# ── S9: outputs are detached (no grad) ──────────────────────────────────

no_grad_required = all(not t.requires_grad for t in out.values())
check(
    "S9 all outputs have requires_grad=False",
    no_grad_required,
    "at least one output retained a grad — @torch.no_grad contract broken",
)


# ── S10: dynamics-model gradients are clean after rollout ──────────────

ok, detail = all_grads_zero_or_none(dynamics)
check("S10 dynamics-model has no .grad after rollout", ok, detail)


# ── S11: bootstrap value matches value_head.predict(last state) ─────────

with torch.no_grad():
    expected_v_T = value_head.predict(out["states"][:, H - 1])  # (B,)
got_v_T = out["values"][:, H]
check(
    "S11 values[:, H] == value_head.predict(states[:, H-1])",
    torch.allclose(got_v_T, expected_v_T, atol=1e-5),
    f"max diff = {(got_v_T - expected_v_T).abs().max().item():.4e}",
)


# ═══════════════════════════════════════════════════════════════════════
# S12: sliding-window edge case (horizon > context_window)
# ═══════════════════════════════════════════════════════════════════════

H_big = 6
CW_small = 4

for m in (dynamics, policy_head, reward_head, continue_head, value_head):
    m.zero_grad(set_to_none=True)

out_big = imagine_rollout(
    dynamics_model=dynamics,
    policy_head=policy_head,
    reward_head=reward_head,
    continue_head=continue_head,
    value_head=value_head,
    z_context=z_context,
    actions_context=actions_context,
    horizon=H_big,
    K=K,
    tau_ctx=TAU_CTX,
    K_max=K_MAX,
    context_window=CW_small,
)

shapes_ok = (
    tuple(out_big["states"].shape) == (B, H_big, EMBED_DIM)
    and tuple(out_big["actions"].shape) == (B, H_big, ACTION_DIM)
    and tuple(out_big["values"].shape) == (B, H_big + 1)
)
finite_ok = all(torch.isfinite(t).all().item() for t in out_big.values())
check(
    "S12 sliding window (horizon=6 > context_window=4) produces sane output",
    shapes_ok and finite_ok,
    f"shapes_ok={shapes_ok}, finite_ok={finite_ok}",
)


# ═══════════════════════════════════════════════════════════════════════
# S13: end-to-end pmpo_policy_loss → grads route to policy_head only
# ═══════════════════════════════════════════════════════════════════════

for m in (dynamics, policy_head, reward_head, continue_head, value_head):
    m.zero_grad(set_to_none=True)

policy_prior = copy.deepcopy(policy_head)
for p in policy_prior.parameters():
    p.requires_grad_(False)

lambda_returns = compute_lambda_returns(
    out["rewards"], out["values"], out["continues"],
    gamma=0.997, lambda_=0.95,
)
advantages = compute_advantages(lambda_returns, out["values"])

policy_loss = pmpo_policy_loss(
    policy_head, policy_prior,
    out["states"], out["actions"], advantages,
    alpha=0.5, beta=0.3,
)
policy_loss.backward()

policy_got_grad, policy_detail = any_grad_nonzero(policy_head)
dyn_clean,    dyn_detail    = all_grads_zero_or_none(dynamics)
reward_clean, reward_detail = all_grads_zero_or_none(reward_head)
cont_clean,   cont_detail   = all_grads_zero_or_none(continue_head)
prior_clean,  prior_detail  = all_grads_zero_or_none(policy_prior)

check(
    "S13a policy_head receives gradients",
    policy_got_grad,
    policy_detail,
)
check(
    "S13b dynamics model stays frozen (no grad)",
    dyn_clean,
    dyn_detail,
)
check(
    "S13c reward_head stays frozen (no grad)",
    reward_clean,
    reward_detail,
)
check(
    "S13d continue_head stays frozen (no grad)",
    cont_clean,
    cont_detail,
)
check(
    "S13e policy_prior stays frozen (no grad)",
    prior_clean,
    prior_detail,
)


# ═══════════════════════════════════════════════════════════════════════
# S14: end-to-end value_loss → grads route to value_head only
# ═══════════════════════════════════════════════════════════════════════

for m in (dynamics, policy_head, reward_head, continue_head, value_head):
    m.zero_grad(set_to_none=True)

# Recompute lambda_returns since we cleared grads (the tensor itself is fine,
# but the recompute documents intent for readers).
lambda_returns = compute_lambda_returns(
    out["rewards"], out["values"], out["continues"],
    gamma=0.997, lambda_=0.95,
)

v_loss = value_loss(value_head, out["states"], lambda_returns)
v_loss.backward()

value_got_grad, value_detail = any_grad_nonzero(value_head)
dyn_clean,    dyn_detail    = all_grads_zero_or_none(dynamics)
reward_clean, reward_detail = all_grads_zero_or_none(reward_head)
policy_clean, policy_detail = all_grads_zero_or_none(policy_head)

check(
    "S14a value_head receives gradients",
    value_got_grad,
    value_detail,
)
check(
    "S14b dynamics model stays frozen (no grad)",
    dyn_clean,
    dyn_detail,
)
check(
    "S14c reward_head stays frozen (no grad)",
    reward_clean,
    reward_detail,
)
check(
    "S14d policy_head stays frozen (no grad)",
    policy_clean,
    policy_detail,
)


# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════

if failures:
    print(f"\n{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("\nALL PASS")
