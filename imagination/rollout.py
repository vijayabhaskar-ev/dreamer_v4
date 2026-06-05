"""Imagined rollout generation for Phase 3 (Dreamer V4, Section 3.3).

Unrolls the frozen dynamics model autoregressively to produce imagined
trajectories: at each step, the next latent is generated via K-step Euler
flow-matching denoising, the dynamics model is run with `use_agent_tokens=True`
to expose the agent state `h`, and the policy/reward/continue/value heads
are queried on `h`. Returned trajectories feed PMPO + TD(λ) training.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from dynamics.flow_matching import add_noise


@torch.no_grad()
def imagine_rollout(
    dynamics_model: nn.Module,      # Frozen DynamicsModel (Phase 2)
    policy_head: nn.Module,         # Current trainable PolicyHead
    reward_head: nn.Module,         # Frozen Phase 2 RewardHead
    continue_head: nn.Module,       # Frozen Phase 2 ContinueHead
    value_head: nn.Module,          # Trainable RewardHead acting as ValueHead
    z_context: torch.Tensor,        # (B, C, S_z, D_lat) real encoded context (clean)
    actions_context: torch.Tensor,  # (B, C-1, A) real context actions
    horizon: int,                   # H imagination steps
    K: int,                         # Denoising steps per frame (paper: 4)
    tau_ctx: float,                 # Context corruption level (paper: 0.1)
    K_max: int,                     # Used for d_ctx = 1/K_max
    context_window: int,            # Sliding-window size
) -> Dict[str, torch.Tensor]:
    """Unroll the dynamics model to generate an imagined trajectory.

    Per imagination step t ∈ [0, H-1]:
      1. Denoise next latent z_{t+1} via K-step Euler flow matching,
         conditioned on the sliding-window context and the previous action.
      2. Forward dynamics with `use_agent_tokens=True` and extract
         h_{t+1} = agent_out[:, -1].
      3. Sample a_{t+1} ~ π_θ(·|h_{t+1}); predict r_{t+1}, c_{t+1}, v_{t+1}.
      4. Append (z_{t+1}, a_{t+1}) (detached) to the sliding buffer.

    Heads run with mtp_offset=0 only — no MTP in imagination.

    Args:
        dynamics_model:  frozen DynamicsModel
        policy_head:     trainable PolicyHead
        reward_head:     frozen reward predictor
        continue_head:   frozen continue predictor
        value_head:      trainable value head (RewardHead architecture)
        z_context:       (B, C, S_z, D_lat) clean encoded context latents
        actions_context: (B, C-1, A) real context actions
        horizon:         number of imagination steps H
        K:               denoising Euler steps per frame
        tau_ctx:         context corruption level (signal level = 1 - tau_ctx)
        K_max:           used to set d_ctx = 1/K_max for context frames
        context_window:  maximum frames kept in the sliding window

    Returns:
        dict with keys:
            'states':    (B, H, D_embed)    agent_out at each imagined step
            'actions':   (B, H, action_dim) sampled actions
            'rewards':   (B, H)             reward_head.predict(states)
            'continues': (B, H)             continue_head.predict(states)
            'values':    (B, H+1)           value_head.predict(states) + bootstrap

    Gradient flow: the whole rollout is under @torch.no_grad(), so no graph
    is retained. The trainer re-runs policy_head.forward(states) with
    gradients enabled inside pmpo_policy_loss; the rollout's only role is
    to produce the (states, actions, advantages) triple.
    """
    B, C, S_z, D_lat = z_context.shape
    action_dim = policy_head.action_dim
    D_embed = dynamics_model.config.embed_dim

    device = z_context.device

    tau_ctx_val = 1 - tau_ctx
    tau_vec = torch.full((B, C),tau_ctx_val, device=device)
    z_ctx_noised, _ = add_noise(z_context, tau_vec)

    states_buf   = torch.zeros(B, horizon, D_embed, device=device)
    actions_buf = torch.zeros(B, horizon, action_dim, device=device)
    rewards_buf  = torch.zeros(B, horizon, device=device)
    continues_buf= torch.zeros(B, horizon, device=device)
    values_buf   = torch.zeros(B, horizon + 1, device=device)

    max_buf = C + horizon
    actions_buffer = torch.zeros(B, max_buf, action_dim, device=device)
    actions_buffer[:, :C-1] = actions_context  # slot C-1 stays zero → null first imagined action
    z_buffer   = torch.zeros(B, max_buf, S_z, D_lat, device=device)
    tau_buffer = torch.full((B, max_buf), tau_ctx_val, device=device)
    d_buffer   = torch.full((B, max_buf), 1.0 / K_max, device=device)
    z_buffer[:, :C] = z_ctx_noised
    buf_len = C

    # Constant (B, 1) signal level for re-noising each new frame. A view of
    # tau_buffer (permanently == tau_ctx_val), reused instead of allocating a
    # fresh torch.full every step — the per-step alloc was XLA-era boilerplate.
    tau_new_const = tau_buffer[:, :1]
    v_bootstrap = None  # set each step; final value is the TD(λ) bootstrap

    for h in range(horizon):
      win_start = max(0, buf_len - context_window)
      z_win = z_buffer[:, win_start:buf_len]
      tau_win = tau_buffer[:, win_start:buf_len]
      d_win   = d_buffer[:, win_start:buf_len]
      a_win   = actions_buffer[:, win_start:buf_len]

      z_new, h_out = denoise_one_frame(dynamics_model, z_win, tau_win, d_win, a_win, K)

      a_new  = policy_head.sample(h_out)      # (B, action_dim)
      r_new  = reward_head.predict(h_out)     # (B,)
      c_new  = continue_head.predict(h_out)   # (B,)
      v_new  = value_head.predict(h_out)      # (B,)

      states_buf[:, h] = h_out
      actions_buf[:, h]  = a_new
      rewards_buf[:, h]  = r_new
      continues_buf[:, h]= c_new
      values_buf[:, h]   = v_new
      v_bootstrap = v_new  # last imagined value → TD(λ) bootstrap (values_buf has horizon+1 slots)

      z_new_noised, _ = add_noise(z_new.detach(), tau_new_const)
      z_buffer[:, buf_len:buf_len + 1] = z_new_noised
      actions_buffer[:, buf_len] = a_new

      buf_len = buf_len + 1

    values_buf[:, horizon] = v_bootstrap


    return {
                 'states':    states_buf,
                 'actions':   actions_buf,
                 'rewards':   rewards_buf,
                 'continues': continues_buf,
                 'values':    values_buf,
             }


@torch.no_grad()
def denoise_one_frame(
    model,
    z_win: torch.Tensor,
    tau_win: torch.Tensor,
    d_win: torch.Tensor,
    a_win: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """Generate one new frame via K-step Euler denoising."""
    B, C, S_z, D_lat  = z_win.shape
    device = z_win.device
    d_step = 1.0 / K

    z_current = torch.randn(B, 1, S_z, D_lat, device=device)
    h_new = None

    # k-invariant across the K Euler steps → build once (these were rebuilt every
    # step as XLA static-shape boilerplate; inert on eager CUDA).
    d_new = torch.full((B, 1), d_step, device=device)
    d_seq = torch.cat([d_win, d_new], dim=1)        # (B, C+1), constant across k
    tau_seq = torch.empty(B, C + 1, device=device)
    tau_seq[:, :C] = tau_win                         # window tau is constant across k

    for k in range(K):
        tau_k = k / K
        tau_seq[:, C] = tau_k                        # only the new-frame slot varies per k

        z_seq = torch.cat([z_win, z_current], dim=1)  # z_current changes per k → rebuild
        is_last = (k == K - 1)
        output = model(z_seq, a_win, tau_seq, d_seq, use_agent_tokens=is_last)
        z_hat_new = output.z_hat[:, -1:]
        if is_last:
          h_new = output.agent_out[:, -1]         # (B, D_embed)

        v = (z_hat_new - z_current) / max(1.0 - tau_k, 1e-4)
        z_current = z_current + v * d_step

    return z_current, h_new