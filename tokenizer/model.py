"""Masked autoencoder tokenizer skeleton for Dreamer V4 Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import TokenizerConfig
from .layers import AttentionMask, TransformerBlock
from .masking import apply_mask, sample_random_mask


@dataclass
class TokenizerOutputs:
    token_embeddings: torch.Tensor
    reconstructed: torch.Tensor
    mask: torch.Tensor
    mask_token: torch.Tensor
    latent_tokens: torch.Tensor


class PatchEmbed(nn.Module):
    """Patchify images into spatio-temporal tokens using 3D or 2D convs."""

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
        self.cls_tokens = nn.Parameter( #TODO No noeed for cls tokens in dreamer v4
            torch.randn(
                1,
                config.learned_cls_tokens,
                config.embed_dim,
            )
            * 0.02
        )

    def forward(self, batch: int) -> torch.Tensor:
        tokens = torch.cat([self.cls_tokens, self.latent_tokens], dim=1) #TODO No need for cls tokens in dreamer v4
        return tokens.expand(batch, -1, -1)


class MaskedAutoencoderTokenizer(nn.Module):
    def __init__(self, config: TokenizerConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.num_patches() * 1, config.embed_dim) * 0.02    #TODO Neeed to implement ROPE  and temporal embedding after initial implementation
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.latent_tokens = LatentTokenEmbedding(config)

        blocks = []
        for _ in range(config.depth):
            blocks.append(
                TransformerBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    drop_path=config.drop_path,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.to_patch = nn.Linear(config.embed_dim, config.embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1] * config.in_channels),
        )

    def forward(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> TokenizerOutputs:
        batch = frames.size(0)
        patches = self.patch_embed(frames)
        base_pos = self.pos_embed
        pos = base_pos.repeat(1, frames.size(1), 1)[:, : patches.size(1), :] #TODO Need to implement ROPE and temporal embedding after initial implementation
        patches = patches + pos #TODO Implemented just to make sure MAE pipeline is working
        #patches = patches + self.pos_embed[:, : patches.size(1), :]

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
        latent_tokens = self.latent_tokens(batch)
        sequence = torch.cat([masked_patches, latent_tokens], dim=1) #(B, N + L, D) where N = patches per frame (256) and L = latent tokens (32).

        attn_mask = AttentionMask(causal=True)
        for block in self.blocks:
            sequence = block(sequence, attn_mask)
        sequence = self.norm(sequence)

        token_embeddings = sequence[:, : patches.size(1), :]
        decoder_inputs = self.to_patch(token_embeddings)
        recon_patches = self.decoder(decoder_inputs)
        recon_frames = self._unpatchify(recon_patches, frames.shape)

        return TokenizerOutputs(
            token_embeddings=token_embeddings,
            reconstructed=recon_frames,
            mask=mask,
            mask_token=self.mask_token,
            latent_tokens=sequence[:, patches.size(1) :, :],
        )

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


__all__ = ["MaskedAutoencoderTokenizer", "TokenizerOutputs"]
