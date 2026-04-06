from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from .embedding import ActionEmbedding, TauDEmbedding, AgentTokenEmbedding
from .dynamic_block import DynamicsTransformerBlock
from .config import DynamicsConfig
from tokenizer.layers import RotaryPositionEmbedding, AttentionMask


class DynamicsOutput(NamedTuple):
    """Return type for DynamicsModel.forward().

    z_hat:     (B, T, S_z, D_latent)  — denoised latent prediction
    agent_out: (B, T, D_embed) or None — agent token output (Phase 2 only)
    """
    z_hat: torch.Tensor
    agent_out: Optional[torch.Tensor]


class DynamicsModel(nn.Module):
    def __init__(self, config: DynamicsConfig, tokenizer):
        super().__init__()
        self.config = config
        self.config.validate_against_tokenizer(tokenizer.config)

        self.tokenizer = tokenizer
        self.tokenizer.requires_grad_(False)
        self.tokenizer.eval()

        self.action_embedding = ActionEmbedding(config.action_dim, config.embed_dim)
        self.tau_d_embedding = TauDEmbedding(config.embed_dim)

        self.proj_in = nn.Linear(config.latent_input_dim, config.embed_dim)
        self.proj_out = nn.Linear(config.embed_dim, config.latent_input_dim)

        self.register_tokens = nn.Parameter(torch.randn(1, config.num_register_tokens, config.embed_dim) * 0.02 )

        # Agent token embedding — None until enable_agent_tokens() is called
        self.agent_embedding: Optional[AgentTokenEmbedding] = None

        self._base_tokens_per_frame = 1 + 1 + config.num_latent_tokens + config.num_register_tokens  # a + τd + z̃ + reg
        self.rope_spatial = RotaryPositionEmbedding(config.embed_dim // config.num_heads, max_positions=self._base_tokens_per_frame)
        self.rope_temporal = RotaryPositionEmbedding(
            config.embed_dim // config.num_heads,
            max_positions=max(config.seq_len_long, config.seq_len_short, 256),
        )

        self.blocks = nn.ModuleList([
            DynamicsTransformerBlock(
                config.embed_dim, config.num_heads, config.mlp_ratio, config.dropout, config.drop_path,
                use_temporal=(i % config.temporal_interval == 0),
                num_kv_heads=config.num_kv_heads,
                rope_spatial=self.rope_spatial,
                rope_temporal=self.rope_temporal
            ) for i in range(config.depth)
        ])

        self.norm = nn.RMSNorm(config.embed_dim)

        # Sliding window temporal mask cache — keyed by T, built lazily
        self._context_length = config.context_length
        self._temporal_mask_cache: dict[int, AttentionMask] = {}

        # Asymmetric spatial mask cache — keyed by tokens_per_frame
        self._spatial_mask_cache: dict[int, torch.Tensor] = {}

    # ── Agent token lifecycle ───────────────────────────────────────────

    def enable_agent_tokens(self, num_tasks: int = 1) -> None:
        """Activate agent tokens for Phase 2 finetuning.

        Creates the agent token embedding and rebuilds spatial RoPE
        to accommodate the extra token position.  Call this once when
        transitioning from Phase 1 to Phase 2.
        """
        self.agent_embedding = AgentTokenEmbedding(self.config.embed_dim, num_tasks=num_tasks)

        # Rebuild spatial RoPE with +num_agent_tokens positions
        tokens_with_agent = self._base_tokens_per_frame + self.config.num_agent_tokens
        new_rope = RotaryPositionEmbedding(
            self.config.embed_dim // self.config.num_heads,
            max_positions=tokens_with_agent,
        )
        self.rope_spatial = new_rope

        # Update all blocks to use the new RoPE
        for block in self.blocks:
            block.spatial_attn.rope = new_rope

        # Clear caches — new token count invalidates them
        self._spatial_mask_cache.clear()

    # ── Mask builders ───────────────────────────────────────────────────

    def _get_sliding_window_causal_mask(self, T: int, device: torch.device) -> AttentionMask:
        """Get or build a sliding window causal mask for T frames.

        The mask allows frame q to attend to frame k iff:
            k <= q          (causal — no future)
            q - k <= C      (window — not too far back)

        Cached per T value. For two fixed T values (T₁, T₂) this means
        exactly two cached masks — no XLA recompilation from mask changes.
        """
        if T not in self._temporal_mask_cache:
            C = self._context_length
            q_idx = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
            k_idx = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
            blocked = (k_idx > q_idx) | (q_idx - k_idx > C)
            mask = AttentionMask(mask=blocked)
            mask.apply_to_sdpa((1, 1, T, T))
            self._temporal_mask_cache[T] = mask
        return self._temporal_mask_cache[T]

    def _get_spatial_agent_mask(self, tokens_per_frame: int, device: torch.device) -> torch.Tensor:
        """Build and cache asymmetric spatial mask for agent tokens.

        Layout per frame: [action, tau_d, z_latent..., register..., agent...]

        Rule (paper Section 3.3):
          - Agent tokens can attend to everything (including themselves)
          - All other tokens CANNOT attend to agent tokens

        This prevents causal confusion: the world model's future predictions
        depend only on actions, not on the current task embedding.

        Returns:
            (N, N) float mask: 0.0 where attention is allowed, -inf where blocked.
        """
        if tokens_per_frame in self._spatial_mask_cache:
            return self._spatial_mask_cache[tokens_per_frame]

        N = tokens_per_frame
        n_agent = self.config.num_agent_tokens
        n_base = N - n_agent  # non-agent token count

        mask = torch.zeros(N, N, device=device)
        mask[:n_base, n_base:] = float("-inf")

        self._spatial_mask_cache[tokens_per_frame] = mask
        return mask

    # ── Core methods ────────────────────────────────────────────────────

    def encode_frames(self, frames):
        """Use frozen tokenizer to get bottleneck latents.

        Uses encode_only() which skips the decoder — ~50% less tokenizer compute.
        """
        with torch.no_grad():
            B, T = frames.shape[:2]
            z = self.tokenizer.encode_only(frames)  # (B, T*S_z, latent_dim)
            z = z.view(B, T, -1, z.shape[-1])
            return z

    def forward(self, z_noised, actions, tau, d,
                use_agent_tokens: bool = False) -> DynamicsOutput:
        """
        Args:
            z_noised: (B, T, S_z, D_latent) — corrupted latent tokens
            actions:  (B, T-1, action_dim) or None
            tau:      (B, T) — per-frame signal levels
            d:        (B, T) — per-frame step sizes
            use_agent_tokens: whether to insert agent tokens (Phase 2)

        Returns:
            DynamicsOutput with:
                z_hat:     (B, T, S_z, D_latent) — denoised prediction
                agent_out: (B, T, D_embed) or None
        """
        B, T, S_z, D_latent = z_noised.shape

        z_up = self.proj_in(z_noised)  # (B, T, S_z, D_embed)
        tau_d = self.tau_d_embedding(tau, d)  # (B, T, D_embed)

        has_agent = use_agent_tokens and self.agent_embedding is not None
        n_agent = self.config.num_agent_tokens if has_agent else 0
        total_tokens_per_frame = self._base_tokens_per_frame + n_agent

        if has_agent:
            agent_tok = self.agent_embedding(batch_size=B)  # (B, 1, D_embed)

        frame_blocks = []
        #TODO Current loop looks inefficient. Need to vectorize into a single cat with some reshaping
        for t in range(T):
            tokens_this_frame = []

            if t == 0 or actions is None:
                a_token = self.action_embedding(None, batch_size=B)
            else:
                a_token = self.action_embedding(actions[:, t-1 : t], batch_size=B)
            tokens_this_frame.append(a_token)

            tokens_this_frame.append(tau_d[:, t:t+1])  # (B, 1, D)

            tokens_this_frame.append(z_up[:, t])  # (B, S_z, D)

            registers = self.register_tokens.expand(B, -1, -1)  # (B, S_r, D)
            tokens_this_frame.append(registers)

            if has_agent:
                tokens_this_frame.append(agent_tok)  # (B, 1, D)

            frame_block = torch.cat(tokens_this_frame, dim=1)  # (B, N, D)
            frame_blocks.append(frame_block)

        x = torch.cat(frame_blocks, dim=1)  # (B, T*N, D)

        temporal_mask = self._get_sliding_window_causal_mask(T, x.device)
        spatial_mask = self._get_spatial_agent_mask(total_tokens_per_frame, x.device) if has_agent else None

        for block in self.blocks:
            x = block(x, num_frames=T, temporal_mask=temporal_mask, spatial_mask=spatial_mask)
        x = self.norm(x)

        x = x.view(B, T, total_tokens_per_frame, -1)

        z_prediction = x[:, :, 2:(2 + self.config.num_latent_tokens), :]
        z_down = self.proj_out(z_prediction)

        agent_out = None
        if has_agent:
            agent_out = x[:, :, -n_agent:, :].reshape(B, T, -1)

        return DynamicsOutput(z_hat=z_down, agent_out=agent_out)
