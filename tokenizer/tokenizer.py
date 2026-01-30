"""Masked autoencoder tokenizer skeleton for Dreamer V4 Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import TokenizerConfig
from .layers import AttentionMask, RotaryPositionEmbedding, TransformerBlock
from .masking import apply_mask, sample_random_mask


@dataclass
class TokenizerOutputs:
    reconstructed: torch.Tensor
    mask: torch.Tensor
    mask_token: torch.Tensor
    latent_tokens: torch.Tensor


class PatchEmbed(nn.Module):
    """Patchify images into spatio-temporal tokens."""

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        ph, pw = config.patch_size
        self.patch_size = config.patch_size
        self.grid_size = (
            config.image_size[0] // ph,
            config.image_size[1] // pw,
        )
        patch_dim = config.in_channels * ph * pw
        self.proj = nn.Linear(patch_dim, config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b, t, c, h, w = x.shape
        ph, pw = self.patch_size
        x = x.reshape(b * t, c, h, w)
        patches = torch.nn.functional.unfold(x, kernel_size=(ph, pw), stride=(ph, pw))
        patches = patches.transpose(1, 2)  # (B*T, num_patches, patch_dim)
        patches = self.proj(patches)
        patches = patches.reshape(b, t, -1, patches.size(-1))
        patches = patches.flatten(1, 2)  # (B, T*num_patches, dim)
        return patches

    def num_patches(self) -> int:
        gh, gw = self.grid_size
        return gh * gw


class LatentTokenEmbedding(nn.Module):
    """Learnable latent tokens used as compression targets."""

    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.latent_tokens = nn.Parameter( 
            torch.randn(1, config.num_latent_tokens, config.embed_dim) * 0.02
        )

    def forward(self, batch: int, num_frames:int) -> torch.Tensor:
        latents = self.latent_tokens.expand(batch, -1, -1)
        latents = latents.unsqueeze(1).expand(-1, num_frames, -1, -1)  # (B, T, 32, D)
        latents = latents.flatten(1, 2) 
        return latents


class MaskedAutoencoderTokenizer(nn.Module):
    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.latent_tokens = LatentTokenEmbedding(config)

        # RoPE for relative position encoding in attention
        head_dim = config.embed_dim // config.num_heads
        self.rope_spatial = RotaryPositionEmbedding(
            dim=head_dim,
            max_positions=self.patch_embed.num_patches(),
            base=10000.0,
        )
        self.rope_temporal = RotaryPositionEmbedding(
            dim=head_dim,
            max_positions=64,  
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


        # Decoder Setup
        self.decoder_queries = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches(), config.embed_dim))
        torch.nn.init.normal_(self.decoder_queries, std=0.02)

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
        

        temporal_causal_mask = self._build_temporal_causal_mask(t, frames.device)
        temporal_attn_mask = AttentionMask(is_causal=False, mask=temporal_causal_mask)
        
        num_latents = latent_tokens.size(1)
        num_patches = masked_patches.size(1)
        latent_cross_mask = self._build_latent_cross_mask(latents_per_frame, num_patches, t, frames.device)
        latent_cross_attn_mask = AttentionMask(is_causal=False, mask=latent_cross_mask)
        
        for block in self.blocks:
            encoder_sequence = block(
                encoder_sequence, 
                num_frames=t, 
                temporal_mask=temporal_attn_mask,
                latent_cross_mask=latent_cross_attn_mask,
                num_latents=total_latents            )
        encoder_sequence = self.norm(encoder_sequence)

        z_latents = encoder_sequence[:, :total_latents, :]
        z_latents = torch.tanh(self.latent_proj(z_latents))
        z_expanded = self.latent_expand(z_latents) 

        decoder_queries = self.decoder_queries.expand(batch, -1, -1)
        decoder_queries = decoder_queries.unsqueeze(1).expand(-1, t, -1, -1)  # (B, T, N, D)
        decoder_queries = decoder_queries.flatten(1, 2)  # (B, T*N, D)
        
        decoder_sequence = torch.cat([z_expanded, decoder_queries], dim=1)
        
        x = decoder_sequence
        for block in self.decoder_blocks:
            x = block(
                x, 
                num_frames=t, 
                temporal_mask=temporal_attn_mask,
                latent_cross_mask=latent_cross_attn_mask,
                num_latents=total_latents
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
        num_patches = self.patch_embed.num_patches()
        
        patches = patches.view(b * t, num_patches, -1)
        patches = patches.transpose(1, 2)
        frames = torch.nn.functional.fold(
            patches,
            output_size=(h, w),
            kernel_size=(ph, pw),
            stride=(ph, pw),
        )
        frames = frames.view(b, t, -1, h, w)
        return frames

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(frames)
        return outputs.latent_tokens