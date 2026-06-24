"""Forward-mask firewall test — DreamerV4 Phase-2 (Option-A).

Verifies the *asymmetric agent-token attention mask*: world-model tokens cannot
attend to agent tokens, so the world-model prediction ``z_hat`` is
value-independent of the agent token. Concretely:

  * ∂ z_hat / ∂ agent_embedding == 0   (the firewall — predictions can't leak)
  * ∂ agent_out / ∂ agent_embedding > 0 (the agent stream *does* read the world)
  * the flow loss still trains the world-model backbone

This is the real causal firewall under Option-A (NO ``detach()`` — the head
losses finetune the backbone jointly; the mask, not a gradient stop, is what
prevents the agent stream from contaminating the world model's predictions).

Run from project root:  python -m tests.test_gradient_isolation
"""
import torch

from dynamics.config import DynamicsConfig
from dynamics.dynamic_model import DynamicsModel
from tokenizer.config import TokenizerConfig
from tokenizer.tokenizer import MaskedAutoencoderTokenizer

# ── minimal model with agent tokens enabled ───────────────────────────────
tok_cfg = TokenizerConfig(image_size=(64, 64), patch_size=(8, 8),
                          latent_dim=128, num_latent_tokens=32, embed_dim=512)
tok = MaskedAutoencoderTokenizer(tok_cfg)
dyn_cfg = DynamicsConfig.from_tokenizer(tok_cfg, embed_dim=512, depth=2,
                                        num_heads=8, mtp_length=0)
model = DynamicsModel(dyn_cfg, tok)
model.enable_agent_tokens(num_tasks=1)

agent_params = [(n, p) for n, p in model.named_parameters() if "agent_embedding" in n]
assert agent_params, "agent_embedding parameter not found — is enable_agent_tokens() wired up?"

B, T, S_z, D = 2, 4, 32, 128
z_noised = torch.randn(B, T, S_z, D)
actions = torch.randn(B, T - 1, dyn_cfg.action_dim)
tau = torch.rand(B, T)
d = torch.full((B, T), 1.0 / 64)


def _agent_grad_magnitude() -> float:
    return sum(p.grad.abs().sum().item() for _, p in agent_params if p.grad is not None)


# ── Test 1: z_hat must NOT depend on the agent token (the firewall) ────────
model.zero_grad(set_to_none=True)
out = model(z_noised, actions, tau, d, use_agent_tokens=True)
out.z_hat.sum().backward()
leak = _agent_grad_magnitude()
print(f"{'PASS' if leak == 0 else 'FAIL'}: z_hat is value-independent of the agent token "
      f"(|∂z_hat/∂agent_embedding| = {leak:.6f}, expect 0)")

# ── Test 2: agent_out MUST depend on the agent token (it has to, to be useful) ──
model.zero_grad(set_to_none=True)
out = model(z_noised, actions, tau, d, use_agent_tokens=True)
out.agent_out.sum().backward()
signal = _agent_grad_magnitude()
print(f"{'PASS' if signal > 0 else 'FAIL'}: agent_out depends on the agent token "
      f"(|∂agent_out/∂agent_embedding| = {signal:.4f}, expect > 0)")

# ── Test 3: the flow loss still trains the world-model backbone (Option-A) ──
model.zero_grad(set_to_none=True)
out = model(z_noised, actions, tau, d, use_agent_tokens=True)
z_clean = torch.randn_like(z_noised)
((out.z_hat - z_clean) ** 2).mean().backward()
backbone_trains = any(
    p.grad is not None and p.grad.abs().sum() > 0
    for n, p in model.named_parameters()
    if "agent_embedding" not in n and "proj" in n
)
print(f"{'PASS' if backbone_trains else 'FAIL'}: world-model backbone (proj_in/proj_out) "
      f"trains under the flow loss")
