# Dreamer V4

A from-scratch PyTorch implementation of **DreamerV4** (Hafner et al., DeepMind, 2024) — a model-based reinforcement learning agent that learns by *imagining* trajectories inside a learned world model. This repo covers all three training phases of the paper:

1. **Tokenizer** — a masked autoencoder that compresses 64×64 RGB frames into a small set of latent tokens.
2. **Dynamics model** — a block-causal transformer trained with a flow-matching objective (with bootstrap loss and curriculum) to predict future latents.
3. **Imagination + RL** — agent tokens, value/policy/reward/continue heads, λ-returns, and **PMPO** (preference-based MPO) policy optimization rolled out inside the frozen world model.

The implementation is written for **TPU v4 / torch_xla**, with a CUDA fallback. A large fraction of the code is dedicated to keeping XLA graph counts small and host RAM bounded — see [TPU/XLA notes](#tpu--xla-notes) below.

---

## Status

| Phase | Component | Status |
|---|---|---|
| 1 | Tokenizer model + losses + masking | Complete |
| 1 | Tokenizer trainer + entrypoint + eval | Complete |
| 2 | Dynamics transformer + flow matching + embeddings | Complete |
| 2 | Bootstrap loss + warmup→ramp→full curriculum | Complete |
| 2 | Agent tokens + MTP heads (reward / continue / policy) | Complete |
| 2 | Trainer + entrypoint + autoregressive eval | Complete |
| 3 | `imagine_rollout` (Euler denoise + sliding context) | Complete |
| 3 | λ-returns, advantages, value loss, **PMPO** policy loss | Complete |
| 3 | End-to-end imagination trainer + entrypoint | **In progress** |
| — | TPU v4-8 stability, compile-cache and host-RAM management | Hardened |
| — | CUDA distributed-training fallback | Not implemented |

---

## Architecture at a glance

```
   raw frames (B,T,3,64,64)
            │
            ▼
   ┌──────────────────────┐
   │  Tokenizer  (MAE)    │   spatial 8×8 patches → 32 latent tokens / frame
   │  encoder ─► z_clean  │   bottleneck: latent_dim = 128, tanh
   │  decoder ─► recon    │   loss = MSE + 0.2 · LPIPS  (tube masking)
   └──────────┬───────────┘
              │ frozen after Phase 1
              ▼
   ┌──────────────────────┐
   │  Dynamics            │   12-layer transformer, embed_dim 512, GQA
   │  (flow matching)     │   spatial attn every layer, temporal every 4
   │  z_noised, τ, d, a   │   block-causal across time, sliding window C=16
   │  ─► ẑ                │   loss = flow MSE + bootstrap (curriculum-mixed)
   │  + agent tokens ─► h │   asymmetric mask: agent sees all, world ignores agent
   └──────────┬───────────┘
              │ frozen after Phase 2
              ▼
   ┌──────────────────────┐
   │  Heads (on h)        │   reward: 255 symexp twohot bins
   │                      │   continue: Bernoulli
   │                      │   value: 255 symexp twohot bins  (Phase 3)
   │                      │   policy: diagonal Gaussian       (Phase 3)
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │  Imagination loop    │   H = 15 steps, K = 4 Euler denoise sub-steps
   │  (Phase 3)           │   λ-returns (γ=0.997, λ=0.95)
   │                      │   PMPO policy loss (α=0.5, β=0.3, reverse KL)
   └──────────────────────┘
```

---

## Repository layout

```
dreamer_v4/
├── tokenizer/                 Phase 1 — masked autoencoder
│   ├── tokenizer.py           encoder + decoder + latent bottleneck
│   ├── layers.py              RoPE, GQA, QK-norm, soft-capped flex_attention
│   ├── masking.py             tube masking (spatial mask shared across T)
│   ├── losses.py              MSE + optional LPIPS
│   ├── trainer.py             training loop, metrics, checkpointing
│   ├── train_tokenizer.py     CLI entrypoint
│   └── config.py
├── dynamics/                  Phase 2 — flow-matching world model
│   ├── dynamic_model.py       transformer + register tokens + agent tokens
│   ├── dynamic_block.py       spatial + (periodic) temporal + FF block
│   ├── flow_matching.py       add_noise, sample_tau_and_d (on-device RNG)
│   ├── embedding.py           action / agent / (τ,d) embeddings
│   ├── trainer.py             curriculum, MTP, XLA-safe grad clip
│   ├── train_dynamics.py      CLI entrypoint
│   ├── evaluate_dynamics.py   K-step denoise + autoregressive rollout + GIFs
│   └── config.py
├── imagination/               Phase 3 — RL inside the world model
│   ├── rollout.py             imagine_rollout: H-step Euler + sliding buffer
│   ├── algorithms.py          λ-returns, advantages, value loss, PMPO
│   ├── trainer.py             (WIP) imagination training loop
│   ├── train_imagination.py   (WIP) CLI entrypoint
│   └── config.py
├── heads.py                   reward / continue / policy heads + symlog twohot
├── device_utils.py            unified TPU / CUDA / CPU device abstraction
├── _env_setup.py              must-be-first import: env vars for XLA / wandb / tmp
├── generate_dataset.py        dm_control → .npz episodes
├── mock_data.py               synthetic moving-square videos (for smoke tests)
├── run_training_loop.sh       4h45m auto-restart wrapper (torch_xla 2.5 leak)
├── test_imagination_algorithms.py   λ-returns / advantage / loss unit tests
├── test_gradient_isolation.py       proves agent_out detach blocks dynamics grads
└── requirements.txt
```

---

## Phase 1 — Tokenizer

A spatio-temporal **masked autoencoder** that compresses video frames into a small set of latent tokens used by the dynamics model.

- **Patch embedding** (`tokenizer/layers.py`): 8×8 spatial patches projected to `embed_dim = 512`.
- **Latent tokens** (`tokenizer/tokenizer.py`): 32 learnable tokens per frame cross-attend to patches under a **block-causal mask** (latent at time *t* may not see future patches).
- **Encoder / decoder**: 8 transformer layers each, GQA with 2 KV heads, RMSNorm pre-norm, RoPE, QK-norm, soft-capped attention (tanh at 30, Gemma-2 style), drop-path.
- **Periodic temporal attention**: every 4 layers, to keep compute bounded.
- **Bottleneck**: `latent_dim = 128`, tanh activation.
- **Masking** (`tokenizer/masking.py`): per-sample mask probability ∼ Uniform[0, 0.9] with **tube consistency** — the same spatial pattern is masked across all frames in a clip, which prevents temporal flickering in reconstructions.
- **Loss** (`tokenizer/losses.py`): pixel MSE + optional 0.2 × LPIPS (auto-disabled if weights unavailable).
- **Encode-only path** (`tokenizer/tokenizer.py`): skips the decoder during dynamics training (~50 % less compute).

Training:

```bash
python -m tokenizer.train_tokenizer \
  --dataset offline --data-path data/episodes.npz \
  --epochs 100 --batch-size 32
```

---

## Phase 2 — Dynamics (flow matching)

A **block-causal latent transformer** trained to denoise corrupted latents, given previous latents, actions, and the (τ, d) flow-matching parameters.

- **Model** (`dynamics/dynamic_model.py`): 12 transformer blocks, embed_dim 512, 8 heads, GQA (2 KV heads), 4 register tokens, sliding-window causal mask of length C = 16.
- **Flow matching** (`dynamics/flow_matching.py`):
  - `z_noised = (1 - τ) · noise + τ · z_clean`
  - τ and `d = 1 / 2^k` sampled **on-device** to avoid CPU↔TPU syncs.
  - Inference uses a small fixed step count `K_inference = 4` for fast rollouts.
- **Bootstrap loss + curriculum** (`dynamics/trainer.py`): warmup (flow-only) → ramp (gradually mix bootstrap, 0 → 1) → full. The curriculum mix is a persistent on-device tensor updated via `.fill_()` so it does **not** trigger XLA recompilation.
- **Sequence-length alternation**: 85 % short batches (T₁) and 15 % long (T₂) — implemented with a single fixed-shape graph plus a per-frame loss mask, so XLA only compiles one variant instead of two.
- **Agent tokens** (Phase 2 finetuning): per-frame learnable tokens with an **asymmetric** spatial mask — agent tokens attend to everything; world-model tokens cannot attend to agent tokens. This is what lets Phase 3 train the policy without contaminating the frozen world model.
- **Multi-token prediction (MTP) heads**: reward / continue / policy share a backbone but emit one output layer per temporal offset.

Training:

```bash
python -m dynamics.train_dynamics \
  --tokenizer-ckpt checkpoints/tokenizer/final.pt \
  --dataset offline --data-path data/episodes.npz \
  --curriculum-warmup-steps 5000 --curriculum-ramp-steps 15000
```

Evaluation produces autoregressive rollout GIFs and per-τ-bin reconstruction metrics:

```bash
python -m dynamics.evaluate_dynamics \
  --dynamics-ckpt checkpoints/dynamics/final.pt \
  --tokenizer-ckpt checkpoints/tokenizer/final.pt
```

---

## Phase 3 — Imagination + PMPO RL

Algorithms and the rollout primitive are **complete and unit-tested**; the end-to-end trainer is in progress.

- **`imagine_rollout`** (`imagination/rollout.py`):
  1. Denoise `z_{t+1}` with K = 4 Euler steps using a context buffer corrupted to `τ_ctx = 0.1`.
  2. Run dynamics with `use_agent_tokens=True` to obtain the agent hidden state `h_{t+1}`.
  3. Sample `a_{t+1} ∼ policy_head(h_{t+1})` (reparameterized).
  4. Predict reward, continue, value from `h_{t+1}`.
  5. Slide the context buffer; **detach** before storing so the XLA graph stays compact.

- **`compute_lambda_returns`** (`imagination/algorithms.py`): TD(λ) with γ = 0.997, λ = 0.95, returns are **detached at the source** so a misuse downstream cannot leak gradients into the value targets.

- **`compute_advantages`** = λ_returns − values[:, :H].

- **`value_loss`**: reuses the reward head's symexp twohot encoding (255 bins) and categorical cross-entropy.

- **`pmpo_policy_loss`** — the paper's preference-based MPO objective:
  - Partition the batch by `sign(advantage)` into D⁺ (good) and D⁻ (bad).
  - Loss = `(1 - α)/|D⁻| · Σ ln π(a|s)_bad − α/|D⁺| · Σ ln π(a|s)_good + β · KL(π_θ ‖ π_prior)` (reverse KL).
  - α = 0.5, β = 0.3.

Tests live at the repo root:

```bash
python test_imagination_algorithms.py   # λ-returns / advantage / loss algebra
python test_gradient_isolation.py       # confirms detach(agent_out) blocks grads
```

---

## Installation

```bash
git clone https://github.com/vijayabhaskar-ev/dreamer_v4.git
cd dreamer_v4
pip install -r requirements.txt
# optional, for TPU:
pip install torch_xla
# optional, for dm_control envs and LPIPS:
pip install dm_control lpips
```

Generate a dataset (or skip and use `mock_data.py` for smoke tests):

```bash
python generate_dataset.py --domain cheetah --task run --episodes 2000 --seq-len 50
```

---

## TPU / XLA notes

A non-trivial fraction of this repo is plumbing to keep TPU runs healthy. The constraints that drove the design:

1. **Different tensor shapes ⇒ different XLA graphs.** Each compiled graph for the 47 M-param dynamics model with AdamW pins ≈ 0.5–1 GB of host RAM, so on a v4-8 the budget is ~15 graphs total. Code paths use fixed-shape tensors with float masks rather than boolean indexing or dynamic shapes.
2. **LR is a compile-time constant.** Continuous LR schedulers compile a new graph every step, so the scheduler is quantized into 2–3 buckets.
3. **Per-step `.item()` triggers a device→host sync.** Metrics are batched and only synced at log intervals; gradient clipping uses an on-device `_xla_safe_clip_grad_norm`.
4. **All RNG runs on-device.** `torch.randint`, `torch.arange`, etc. are always called with `device=self.device` to avoid CPU↔TPU traffic.
5. **Constant masks are cached** keyed on (shape, num_frames) — temporal causal masks, latent cross masks, and the bool→float conversions inside `AttentionMask`.
6. **`torch_xla 2.5` `MpDeviceLoader` thread leak**: `run_training_loop.sh` wraps the trainer in 4h45m cycles with auto-resume, so threads never accumulate past ~6500. To be removed when ≥ 2.8 lands.

`device_utils.py` provides a `get_device("auto")` abstraction that prefers TPU > CUDA > CPU and a `mark_step()` shim that's a no-op off XLA — most training code is device-agnostic.

---

## What's noteworthy in this implementation

- **Asymmetric agent-token mask** in `dynamics/dynamic_model.py` — the cleanest way to add an agent stream to a pretrained world model without leaking agent state into world-model predictions.
- **On-device flow-matching RNG** in `dynamics/flow_matching.py` — eliminates a CPU↔TPU sync per training step.
- **Fixed-shape, mask-everything XLA path** in `dynamics/trainer.py` — the 70/30 short/long batch split that was doubling all compilations is collapsed into a single graph plus a loss mask.
- **PMPO** in `imagination/algorithms.py` — preference-based MPO with reverse KL toward a frozen prior, partitioned by advantage sign rather than via softmax weights.
- **Operational hardening**: `_env_setup.py` redirects `/tmp`, caps inductor compile workers, and pins the XLA cache to `~/xla_cache` so restarts are warm.

---

## References

- Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T. — *Mastering Diverse Control Tasks Through World Models* / DreamerV4 (2024).
- Lipman, Y. et al. — *Flow Matching for Generative Modeling* (2023).
- Abdolmaleki, A. et al. — *Maximum a Posteriori Policy Optimization* (2018).
- Hafner, D. et al. — DreamerV3 (2023).

---

## License

MIT — see `LICENSE`.
