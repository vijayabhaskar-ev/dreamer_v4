"""Masked autoencoder tokenizer skeleton for Dreamer V4 Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import TokenizerConfig
from .layers import AttentionMask, RotaryPositionEmbedding, TransformerBlock, set_soft_cap_enabled
from .masking import apply_mask, sample_random_mask


@dataclass
class TokenizerOutputs:
    reconstructed: torch.Tensor
    mask: torch.Tensor
    mask_token: torch.Tensor
    latent_tokens: torch.Tensor


class PatchEmbed(nn.Module):
    """Patchify images into spatio-temporal tokens.

    Uses ``nn.Conv2d`` with ``stride == kernel_size`` — the standard ViT
    patchify idiom. Mathematically equivalent to unfold + linear projection
    (same parameter count, same operation).
    """

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        ph, pw = config.patch_size
        self.patch_size = config.patch_size
        self.grid_size = (
            config.image_size[0] // ph,
            config.image_size[1] // pw,
        )
        self.proj = nn.Conv2d(
            config.in_channels,
            config.embed_dim,
            kernel_size=(ph, pw),
            stride=(ph, pw),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = x.reshape(b, t, -1, x.size(-1))
        x = x.flatten(1, 2)
        return x

    def num_patches(self) -> int:
        gh, gw = self.grid_size
        return gh * gw


class LatentTokenEmbedding(nn.Module):
    """Learnable latent tokens used as compression targets."""

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        # Init bumped from 0.02 to 0.1 alongside the RoPE indexing fix in layers.py.
        # Gives Q vectors measurable per-slot distinction in the first forward pass
        # before QKV projections are trained. Pairs with the structural fix; not a
        # standalone cure (RMSNorm would absorb magnitude differences across blocks).
        self.latent_tokens = nn.Parameter(
            torch.randn(1, config.num_latent_tokens, config.embed_dim) * 0.1
        )

    def forward(self, batch: int, num_frames:int) -> torch.Tensor:
        latents = self.latent_tokens.expand(batch, -1, -1)
        latents = latents.unsqueeze(1).expand(-1, num_frames, -1, -1)  # (B, T, num_latent_tokens, D)
        latents = latents.flatten(1, 2) 
        return latents


class MaskedAutoencoderTokenizer(nn.Module):
    def __init__(self, config: TokenizerConfig):
        super().__init__()
        # Iter 46 (2026-05-27): toggle module-level soft-cap flag BEFORE constructing
        # blocks so attention modules pick up the setting consistently. Default False
        # matches Hansen's reference (which omits paper §3.4 attention soft-cap).
        set_soft_cap_enabled(getattr(config, "use_attention_soft_cap", False))
        self.config = config
        self.patch_embed = PatchEmbed(config)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.latent_tokens = LatentTokenEmbedding(config)

        head_dim = config.embed_dim // config.num_heads
        # Sized to num_latent_tokens + num_patches so Real Fix 2 (spatial attention over the
        # full [latents, patches] per-frame sequence) has enough RoPE positions. Each latent
        # slot occupies positions 0..L-1; patches occupy L..L+N-1.
        self.rope_spatial = RotaryPositionEmbedding(
            dim=head_dim,
            max_positions=config.num_latent_tokens + self.patch_embed.num_patches(),
            base=10000.0,
        )
        # Sized to L_q + num_patches for the latent cross-attention modules after the
        # RoPE indexing fix (layers.py: LatentCrossAttention and PatchToLatentCrossAttention
        # now use global token indices up to L_q + num_patches - 1). 512 leaves headroom
        # for larger seq_len / image_size in future configs.
        # TODO: rename rope_temporal — post-fix it encodes mixed sequence position, not
        # time-only. Kept for now to avoid churn during the debugging cycle.
        self.rope_temporal = RotaryPositionEmbedding(
            dim=head_dim,
            max_positions=512,
            base=10000.0,
        )
        
      
        self.temporal_interval = 4  # TODO Make this a config param
        encoder_blocks = []
        for i in range(config.depth):
            use_temporal = ((i + 1) % self.temporal_interval == 0)  
            encoder_blocks.append(
                TransformerBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    drop_path=config.drop_path,
                    use_temporal=use_temporal,
                    num_kv_heads=config.num_kv_heads,
                    rope_spatial=self.rope_spatial,
                    rope_temporal=self.rope_temporal                )
            )
        self.blocks = nn.ModuleList(encoder_blocks)
        self.norm = nn.RMSNorm(config.embed_dim)

        # Bottleneck
        self.latent_proj = nn.Linear(config.embed_dim, config.latent_dim)
        self.latent_expand = nn.Linear(config.latent_dim, config.embed_dim)


        # Decoder Setup — Iter 45 (2026-05-21): replaced per-position decoder_queries
        # with a single shared mask_token. The previous per-position parameter had
        # ~33K free dims (1 × num_patches × embed_dim) that could learn to memorize
        # dataset-mean appearance per position — a bypass route allowing the decoder
        # to satisfy masked-MSE + masked-LPIPS losses WITHOUT using latents (just
        # output the dataset mean at each masked position). Single mask_token forces
        # ALL positional differentiation through RoPE in the decoder attention blocks
        # → no per-position content can be cached. Matches Hansen's reference pattern.
        self.decoder_mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        torch.nn.init.normal_(self.decoder_mask_token, std=0.02)

        decoder_blocks = []
        for i in range(config.depth):
            use_temporal = ((i + 1) % self.temporal_interval == 0)  
            decoder_blocks.append(
                TransformerBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    drop_path=config.drop_path,
                    use_temporal=use_temporal,
                    is_decoder=True,
                    num_kv_heads=config.num_kv_heads,
                    rope_spatial=self.rope_spatial,
                    rope_temporal=self.rope_temporal,
                )
            )
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.decoder_norm = nn.RMSNorm(config.embed_dim) 
        
        self.to_pixels = nn.Linear(
            config.embed_dim,
            self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1] * config.in_channels
        )

        # Mask caches — avoids creating new AttentionMask objects every forward call.
        # Keyed on t (num_frames); since t is constant during training, these are
        # built once and reused, including pre-warmed float mask caches.
        self._cached_temporal_attn_mask: Optional[AttentionMask] = None
        self._cached_latent_cross_attn_mask: Optional[AttentionMask] = None
        self._cached_encoder_modality_mask: Optional[torch.Tensor] = None
        self._cached_t: int = -1

    def _get_cached_masks(self, t: int, num_patches: int, device: torch.device):
        """Return cached attention masks, rebuilding only when t changes.

        Returns (temporal_attn_mask, latent_cross_attn_mask, encoder_modality_mask).
        The third mask is a float (L_pf+N_pf, L_pf+N_pf) tensor for paper §3.1
        modality-restricted encoder spatial attention.
        """
        if self._cached_t != t:
            temporal_mask = self._build_temporal_causal_mask(t, device)
            self._cached_temporal_attn_mask = AttentionMask(is_causal=False, mask=temporal_mask)

            latents_per_frame = self.config.num_latent_tokens
            total_latents = t * latents_per_frame
            cross_mask = self._build_latent_cross_mask(latents_per_frame, num_patches, t, device)
            self._cached_latent_cross_attn_mask = AttentionMask(is_causal=False, mask=cross_mask)

            patches_per_frame = num_patches // t
            self._cached_encoder_modality_mask = self._build_encoder_modality_mask(
                latents_per_frame, patches_per_frame, device
            )

            # Pre-warm float cache so apply_to_sdpa() never re-allocates
            self._cached_temporal_attn_mask.apply_to_sdpa((1, 1, t, t))
            self._cached_latent_cross_attn_mask.apply_to_sdpa(
                (1, 1, total_latents, total_latents + num_patches))
            self._cached_t = t
        return (
            self._cached_temporal_attn_mask,
            self._cached_latent_cross_attn_mask,
            self._cached_encoder_modality_mask,
        )

    def _build_encoder_modality_mask(
        self,
        latents_per_frame: int,
        patches_per_frame: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Build the paper §3.1 modality mask for encoder spatial attention.

        Within a single frame, tokens are laid out as [latents (L_pf), patches (N_pf)].
        Latent queries attend to all keys; patch queries attend ONLY to patch keys.

        Returns a float (L_pf+N_pf, L_pf+N_pf) tensor added to attention scores:
          - 0.0 where attention is allowed
          - large negative (-inf) where attention is blocked

        Rule: mask[q, k] = -inf iff (q is patch) AND (k is latent),
              i.e. q >= L_pf AND k < L_pf.
        """
        total = latents_per_frame + patches_per_frame
        mask = torch.zeros(total, total, dtype=dtype, device=device)
        block_val = torch.finfo(dtype).min
        mask[latents_per_frame:, :latents_per_frame] = block_val
        return mask

    def forward(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> TokenizerOutputs:
        batch, t, c, h, w = frames.shape
        
        patches = self.patch_embed(frames)  # (B, T*N, D)
               
        latent_tokens = self.latent_tokens(batch, t)  # (B, L, D)
        latents_per_frame = self.config.num_latent_tokens  
        total_latents = latent_tokens.size(1) 
        
        if mask is None:
             mask = sample_random_mask(
                batch=batch,
                seq_len=patches.size(1),
                mask_prob_min=self.config.mask_prob_min,
                mask_prob_max=self.config.mask_prob_max,
                device=patches.device,
                num_frames=t,
            )
        
        masked_patches = apply_mask(
            tokens=patches,
            mask=mask,
            mask_token=self.mask_token.expand(batch, patches.size(1), -1),
        )
        
        encoder_sequence = torch.cat([latent_tokens, masked_patches], dim=1) 
        

        num_patches = masked_patches.size(1)
        temporal_attn_mask, latent_cross_attn_mask, encoder_modality_mask = (
            self._get_cached_masks(t, num_patches, frames.device)
        )

        for block in self.blocks:
            encoder_sequence = block(
                encoder_sequence,
                num_frames=t,
                temporal_mask=temporal_attn_mask,
                latent_cross_mask=latent_cross_attn_mask,
                num_latents=total_latents,
                encoder_modality_mask=encoder_modality_mask,
            )
        encoder_sequence = self.norm(encoder_sequence)

        z_latents = encoder_sequence[:, :total_latents, :]
        z_latents = torch.tanh(self.latent_proj(z_latents))
        z_expanded = self.latent_expand(z_latents)

        N_patches = self.patch_embed.num_patches()
        decoder_queries = self.decoder_mask_token.expand(batch, t * N_patches, -1)  # (B, T*N, D)
        
        decoder_sequence = torch.cat([z_expanded, decoder_queries], dim=1)
        
        x = decoder_sequence
        for block in self.decoder_blocks:
            x = block(
                x,
                num_frames=t,
                temporal_mask=temporal_attn_mask,
                latent_cross_mask=latent_cross_attn_mask,
                num_latents=total_latents,
            )
        x = self.decoder_norm(x)
        
        recon_tokens = x[:, total_latents:, :]
        recon_patches = self.to_pixels(recon_tokens)
        recon_frames = self._unpatchify(recon_patches, frames.shape)

        return TokenizerOutputs(
            reconstructed=recon_frames,
            mask=mask,
            mask_token=self.mask_token,
            latent_tokens=z_latents,
        )




    def _build_temporal_causal_mask(self, t: int, device: torch.device) -> torch.Tensor:
        frame_idx = torch.arange(t, device=device)
        q_frames = frame_idx.unsqueeze(1)  # (T, 1)
        k_frames = frame_idx.unsqueeze(0)  # (1, T)
        mask = k_frames > q_frames  # (T, T)
        
        return mask

    def _build_latent_cross_mask(
        self, 
        latents_per_frame: int, 
        num_patches: int, 
        t: int, 
        device: torch.device
    ) -> torch.Tensor:
        """
        Build block causal mask for latent cross-attention.
        
        Latents (queries) attend to all tokens (keys) with block causal constraints:
        - Latents can attend to all other latents (no mask logic applied here for latents keys)
        - Latents can attend to patches with block causal (past/current frames only)
        
        Shape: (L, L+T*N) where L=num_latents, T*N=num_patches
        """
        total_latents = t * latents_per_frame  
        total_kv = total_latents + num_patches
        patches_per_frame = num_patches // t

        
        mask = torch.zeros(total_latents, total_kv, dtype=torch.bool, device=device)
        
        
        # Latent frame indices: shape (L, 1)
        latent_frames_q = torch.arange(total_latents, device=device) // latents_per_frame
        latent_frames_q = latent_frames_q.unsqueeze(1)
        
        # Latent key frame indices: shape (1, L)
        latent_frames_k = torch.arange(total_latents, device=device) // latents_per_frame
        latent_frames_k = latent_frames_k.unsqueeze(0)
        
        # Patch frame indices: shape (1, P)
        patch_frames = torch.arange(num_patches, device=device) // patches_per_frame
        patch_frames = patch_frames.unsqueeze(0)
        
        patch_mask = patch_frames > latent_frames_q
        
        latent_mask = latent_frames_k > latent_frames_q
        
        mask[:, :total_latents] = latent_mask
        mask[:, total_latents:] = patch_mask
        
        return mask

    def _unpatchify(self, patches: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        b, t, c, h, w = shape
        ph, pw = self.patch_embed.patch_size
        gh, gw = h // ph, w // pw

        # Unpatchify via reshape+permute. For non-overlapping patches
        # (stride == kernel_size), fold is exactly the inverse of patchify
        # and reduces to dimension shuffling.
        patches = patches.reshape(b * t, gh, gw, c, ph, pw)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        frames = patches.reshape(b, t, c, h, w)
        return frames

    def encode_only(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames to latent tokens WITHOUT running the decoder.

        Skips masking, decoder blocks, and un-patchify — saves ~50% compute.
        Used by the dynamics model which only needs latent representations.

        Returns: (B, T*num_latent_tokens, latent_dim) latent tokens.
        """
        batch, t, c, h, w = frames.shape

        patches = self.patch_embed(frames)  # (B, T*N, D)
        latent_tokens = self.latent_tokens(batch, t)  # (B, L, D)
        total_latents = latent_tokens.size(1)
        num_patches = patches.size(1)

        encoder_sequence = torch.cat([latent_tokens, patches], dim=1)

        temporal_attn_mask, latent_cross_attn_mask, encoder_modality_mask = (
            self._get_cached_masks(t, num_patches, frames.device)
        )

        for block in self.blocks:
            encoder_sequence = block(
                encoder_sequence,
                num_frames=t,
                temporal_mask=temporal_attn_mask,
                latent_cross_mask=latent_cross_attn_mask,
                num_latents=total_latents,
                encoder_modality_mask=encoder_modality_mask,
            )
        encoder_sequence = self.norm(encoder_sequence)

        z_latents = encoder_sequence[:, :total_latents, :]
        z_latents = torch.tanh(self.latent_proj(z_latents))
        return z_latents

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(frames)
        return outputs.latent_tokens