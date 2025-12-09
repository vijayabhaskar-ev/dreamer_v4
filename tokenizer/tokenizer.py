"""Masked autoencoder tokenizer skeleton for Dreamer V4 Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import TokenizerConfig
from .layers import AttentionMask, TransformerBlock
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
        # x: (B, T, C, H, W)
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

    def forward(self, batch: int) -> torch.Tensor:
        return self.latent_tokens.expand(batch, -1, -1)


class MaskedAutoencoderTokenizer(nn.Module):
    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)
        self.pos_embed = nn.Parameter( #TODO add ROPE
            torch.randn(1, self.patch_embed.num_patches(), config.embed_dim) * 0.02
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.latent_tokens = LatentTokenEmbedding(config)

        # Encoder Blocks
        encoder_blocks = []
        for _ in range(config.depth):
            encoder_blocks.append(
                TransformerBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    drop_path=config.drop_path,
                )
            )
        self.blocks = nn.ModuleList(encoder_blocks)
        self.norm = nn.LayerNorm(config.embed_dim)

        # Temporal Embedding
        self.max_frames = 32 #TODO add as config param
        self.temporal_embed = nn.Parameter(torch.zeros(1, self.max_frames, 1, config.embed_dim))
        torch.nn.init.trunc_normal_(self.temporal_embed, std=0.02)

        # Bottleneck
        self.latent_proj = nn.Linear(config.embed_dim, config.embed_dim)

        # Decoder Setup
        self.decoder_queries = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        torch.nn.init.normal_(self.decoder_queries, std=0.02)

        decoder_blocks = []
        for _ in range(config.depth):
            decoder_blocks.append(
                TransformerBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    drop_path=config.drop_path,
                )
            )
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.decoder_norm = nn.LayerNorm(config.embed_dim) #why two serprate layernorm for encoder and decoder?
        
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
        
        # --- 1. Embeddings ---
        patches = self.patch_embed(frames) # (B, T*N, D)
        
        # Add Spatial Positional Embeddings
        n_patches_per_frame = self.patch_embed.num_patches() #TODO Refactor position embedding for both encoder and decoder
        pos = self.pos_embed[:, :n_patches_per_frame, :].repeat(1, t, 1) #TODO add ROPE
        patches = patches + pos

        # Add Temporal Embeddings
        temp_embed = self._get_temporal_embed(t) # (1, T, 1, D)
        # Broadcast temporal embed to all patches in that frame
        temp_broadcast = temp_embed.expand(-1, -1, n_patches_per_frame, -1).reshape(1, t * n_patches_per_frame, -1)
        patches = patches + temp_broadcast

        # --- 2. Encoder Preparation ---
        latent_tokens = self.latent_tokens(batch) # (B, L, D)
        
        if mask is None:
             mask = sample_random_mask(
                batch=batch,
                seq_len=patches.size(1),
                mask_prob_min=self.config.mask_prob_min,
                mask_prob_max=self.config.mask_prob_max,
                device=patches.device,
            )
        
        masked_patches = apply_mask(
            tokens=patches,
            mask=mask,
            mask_token=self.mask_token.expand(batch, patches.size(1), -1),
        )
        
        # Sequence: [Latents, Masked Patches]
        encoder_sequence = torch.cat([latent_tokens, masked_patches], dim=1) 
        
        # --- 3. Encoder Masking (Block Causal) ---
        # Generate boolean mask for SDPA
        enc_mask_bool = self._build_block_causal_mask(
            num_latents=latent_tokens.size(1),
            num_patches=masked_patches.size(1),
            t=t,
            device=frames.device
        )
        enc_attn_mask = AttentionMask(is_causal=False, mask=enc_mask_bool)
        
        # --- 4. Encoder Forward ---
        for block in self.blocks:
            encoder_sequence = block(encoder_sequence, enc_attn_mask)
        encoder_sequence = self.norm(encoder_sequence)

        # Extract Latents and bottleneck
        num_latents = latent_tokens.size(1)
        z_latents = encoder_sequence[:, :num_latents, :]
        z_latents = torch.tanh(self.latent_proj(z_latents))

        # --- 5. Decoder Preparation ---
        # Decoder Input: [Latents, Decoder Queries]
        # Queries need pos + temp info
        decoder_queries = self.decoder_queries.expand(batch, t * n_patches_per_frame, -1) 
        decoder_queries = decoder_queries + pos + temp_broadcast
        
        decoder_sequence = torch.cat([z_latents, decoder_queries], dim=1)
        
        # --- 6. Decoder Masking (Block Causal) ---
        dec_mask_bool = self._build_block_causal_mask(
            num_latents=z_latents.size(1),
            num_patches=decoder_queries.size(1),
            t=t,
            device=frames.device
        )
        dec_attn_mask = AttentionMask(is_causal=False, mask=dec_mask_bool)
        
        # --- 7. Decoder Forward ---
        x = decoder_sequence
        for block in self.decoder_blocks:
            x = block(x, dec_attn_mask)
        x = self.decoder_norm(x)
        
        # Extract pixels from query positions
        recon_tokens = x[:, num_latents:, :]
        recon_patches = self.to_pixels(recon_tokens)
        recon_frames = self._unpatchify(recon_patches, frames.shape)

        return TokenizerOutputs(
            reconstructed=recon_frames,
            mask=mask,
            mask_token=self.mask_token,
            latent_tokens=z_latents,
        )




    def _get_temporal_embed(self, t: int) -> torch.Tensor:
        """Safe retrieval of temporal embeddings with truncation/interpolation."""
        if t <= self.max_frames:
            return self.temporal_embed[:, :t, :, :]
        else:
            # Simple interpolation for T > max_frames
            # (1, MaxT, 1, D) -> (1, D, MaxT, 1) -> Interpolate -> (1, T, 1, D)
            permuted = self.temporal_embed.permute(0, 3, 1, 2)
            interp = torch.nn.functional.interpolate(permuted, size=(t, 1), mode='bilinear')
            return interp.permute(0, 2, 3, 1)

    def _build_block_causal_mask(self, num_latents: int, num_patches: int, t: int, device: torch.device) -> torch.Tensor:
        """       
        Dreamer V4 Logic:
        1. Latents can attend to EVERYTHING (Mask = False)
        2. Patches can attend to ALL Latents (Mask = False)
        3. Patches can attend to Patches only if Frame i >= Frame j (Block Causal)
        """
        total_tokens = num_latents + num_patches
        
        mask = torch.zeros(total_tokens, total_tokens, dtype=torch.bool, device=device)
        
        patches_per_frame = num_patches // t
        patch_idx = torch.arange(num_patches, device=device)
        patch_frames = patch_idx // patches_per_frame
        
        patch_q = patch_frames.unsqueeze(1) # (N, 1)
        patch_k = patch_frames.unsqueeze(0) # (1, N)
        
        patch_mask = patch_k > patch_q
        
        mask[num_latents:, num_latents:] = patch_mask
        
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