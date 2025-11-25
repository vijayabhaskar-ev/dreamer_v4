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
        """
        Batches are combined to increase parallelism while extracting the patches.
        """
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
        for _ in range(config.depth): #TODO Need to impleement casual and relative positional embeddings based on dreamer v4 paper
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

        # Temporal Embedding
        # We assume a max number of frames to learn positions for.
        # If input has more frames, we might need to interpolate or error out.
        self.max_frames = 32  # Reasonable default for video clips
        self.temporal_embed = nn.Parameter(torch.zeros(1, self.max_frames, 1, config.embed_dim))
        torch.nn.init.trunc_normal_(self.temporal_embed, std=0.02)

        # Bottleneck
        # Project latents and apply tanh
        self.latent_proj = nn.Linear(config.embed_dim, config.embed_dim)

        # Decoder
        # Learned tokens for the decoder (separate from mask_token)
        self.decoder_queries = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        torch.nn.init.normal_(self.decoder_queries, std=0.02)

        # In DreamerV4 (and standard MAE), the decoder takes [Latents, Mask Tokens]
        # and processes them with Self-Attention.
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
        self.decoder_norm = nn.LayerNorm(config.embed_dim)
        
        self.to_pixels = nn.Linear(config.embed_dim, self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1] * config.in_channels)

    def forward(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> TokenizerOutputs:
        # frames: (B, T, C, H, W)
        batch, t, c, h, w = frames.shape
        
        # 1. Patchify & Embed
        patches = self.patch_embed(frames) # (B, T*N, D)
        
        # 2. Add Spatial Positional Embeddings
        # pos_embed is (1, N, D). We repeat it for T frames.
        n_patches = self.patch_embed.num_patches()
        pos = self.pos_embed[:, :n_patches, :].repeat(1, t, 1) # (1, T*N, D)
        patches = patches + pos

        # 3. Add Temporal Embeddings
        # temporal_embed is (1, MaxT, 1, D). We slice to T and broadcast to N patches.
        if t <= self.max_frames:
            temp = self.temporal_embed[:, :t, :, :] # (1, T, 1, D)
            temp = temp.expand(-1, -1, n_patches, -1).reshape(1, t * n_patches, -1) # (1, T*N, D)
            patches = patches + temp
        else:
            # Fallback or error if T > max_frames. For now, just slice and repeat if needed or clamp.
            # Simple truncation for safety:
            temp = self.temporal_embed[:, : self.max_frames, :, :]
            # If t is larger, we just cycle or clamp? Let's just use what we have and warn implicitly by logic.
            # Proper way: interpolate. For now: assume t <= max_frames.
            pass

        # 4. Encoder (Bottleneck)
        # Input: [Latent Tokens, Patches]
        # We want the Latent Tokens to attend to the Patches (and themselves).
        # If using standard Self-Attention, we just concat them.
        
        latent_tokens = self.latent_tokens(batch) # (B, L, D)
        
        # We don't mask the input patches in a Bottleneck MAE usually, 
        # OR we do mask them and ask latents to reconstruct from partial view.
        # Standard MAE masks patches. Perceiver usually sees all.
        # DreamerV4 Tokenizer usually sees all pixels to compress them.
        # Let's stick to the user's "MaskedAutoencoder" name: we mask the patches.
        
        if mask is None:
             mask = sample_random_mask(
                batch=batch,
                seq_len=patches.size(1),
                mask_prob_min=self.config.mask_prob_min,
                mask_prob_max=self.config.mask_prob_max,
                device=patches.device,
            )
        
        # Apply mask to patches: replace masked patches with mask_token or drop them?
        # Standard MAE drops them. 
        # But here we want to compress the *visible* patches into latents.
        # So we should drop masked patches from the sequence to save compute, 
        # OR keep them as mask tokens if we want latents to know "this is missing".
        # Let's use the `apply_mask` from masking.py which replaces with mask_token.
        
        masked_patches = apply_mask(
            tokens=patches,
            mask=mask,
            mask_token=self.mask_token.expand(batch, patches.size(1), -1),
        )
        
        # Concat: [Latents, Masked Patches]
        # Note: We put Latents FIRST so we can easily extract them.
        sequence = torch.cat([latent_tokens, masked_patches], dim=1) 
        
        attn_mask = AttentionMask(causal=True) # Causal attention
        for block in self.blocks:
            sequence = block(sequence, attn_mask)
        sequence = self.norm(sequence)

        # Extract Latents (the compressed representation)
        num_latents = latent_tokens.size(1)
        z_latents = sequence[:, :num_latents, :] # (B, L, D)
        
        # Bottleneck: Tanh activation
        z_latents = torch.tanh(self.latent_proj(z_latents))

        # 5. Decoder
        # We want to reconstruct the FULL image (or just masked parts).
        # We concatenate [Latents, Mask Tokens] and run Self-Attention.
        
        # Create Decoder Queries (Learned Tokens)
        # They need positional info to know "where" they are.
        # We use self.decoder_queries instead of self.mask_token
        decoder_queries = self.decoder_queries.expand(batch, t * n_patches, -1) + pos 
        if t <= self.max_frames:
             decoder_queries = decoder_queries + temp
        
        # Concatenate: [Latents, Decoder Queries]
        decoder_sequence = torch.cat([z_latents, decoder_queries], dim=1)
        
        # Apply Decoder Blocks (Self-Attention)
        # Note: Decoder is also causal in time
        decoder_attn_mask = AttentionMask(causal=True)
        
        x = decoder_sequence
        for block in self.decoder_blocks:
            x = block(x, decoder_attn_mask)
        x = self.decoder_norm(x)
        
        # Extract only the reconstructed patches (ignore latents)
        # The latents were at the start, so we take from num_latents onwards.
        recon_tokens = x[:, num_latents:, :]
        
        # Project to pixels
        recon_patches = self.to_pixels(recon_tokens)
        recon_frames = self._unpatchify(recon_patches, frames.shape)

        return TokenizerOutputs(
            reconstructed=recon_frames,
            mask=mask,
            mask_token=self.mask_token,
            latent_tokens=z_latents,
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
        frames = frames.view(b, t, -1, h, w) #torch.Size([32, 4, 3, 64, 64])
        return frames

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(frames)
        return outputs.latent_tokens


__all__ = ["MaskedAutoencoderTokenizer", "TokenizerOutputs"]
