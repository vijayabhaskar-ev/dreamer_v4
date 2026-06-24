import torch
from dynamics.config import DynamicsConfig
from dynamics.dynamic_model import DynamicsModel
from tokenizer.config import TokenizerConfig
from tokenizer.tokenizer import MaskedAutoencoderTokenizer
from heads import RewardHead

# Build minimal model
tok_cfg = TokenizerConfig(image_size=(64,64), patch_size=(8,8),
                          latent_dim=128, num_latent_tokens=32, embed_dim=512)
tok = MaskedAutoencoderTokenizer(tok_cfg)
dyn_cfg = DynamicsConfig.from_tokenizer(tok_cfg, embed_dim=512, depth=2,
                                         num_heads=8, mtp_length=0)
model = DynamicsModel(dyn_cfg, tok)
model.enable_agent_tokens(num_tasks=1)

reward_head = RewardHead(latent_dim=512, hidden_dim=512)

# Dummy forward
B, T, S_z, D = 2, 4, 32, 128
z_noised = torch.randn(B, T, S_z, D)
actions = torch.randn(B, T-1, 6)
tau = torch.rand(B, T)
d = torch.full((B, T), 1/64)

output = model(z_noised, actions, tau, d, use_agent_tokens=True)

# ── Test: reward loss through detached agent_out ──
h = output.agent_out.detach()  # <-- the critical line
rewards = torch.randn(B, T)
loss = reward_head.loss(h, rewards)
loss.backward()

# Check: dynamics backbone should have ZERO gradients
for name, p in model.named_parameters():
    if p.grad is not None and p.grad.abs().sum() > 0:
        if 'agent_embedding' not in name:
            print(f"FAIL: {name} has non-zero grad = {p.grad.abs().sum():.6f}")
            break
else:
    print("PASS: No dynamics backbone gradients from reward head loss")

# ── Test: flow loss DOES flow through backbone ──
model.zero_grad()
reward_head.zero_grad()
z_clean = torch.randn_like(z_noised)
output2 = model(z_noised, actions, tau, d, use_agent_tokens=True)
flow_loss = ((output2.z_hat - z_clean) ** 2).mean()
flow_loss.backward()

has_grad = any(
    p.grad is not None and p.grad.abs().sum() > 0
    for n, p in model.named_parameters()
    if 'agent_embedding' not in n and 'proj' in n
)
print(f"{'PASS' if has_grad else 'FAIL'}: proj_in/proj_out get flow gradients")
