"""Core transformer layers for the Dreamer V4 tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionMask:
    """
    Wrapper for handling masks in Scaled Dot Product Attention.
    For Flash Attention (SDPA), a boolean mask (True = -inf, False = 0) is preferred.
    """
    is_causal: bool = False
    mask: Optional[torch.Tensor] = None  # Expected shape: (L, L) or (B, 1, L, L)

    def apply_to_sdpa(self, size: tuple) -> Optional[torch.Tensor]:
        """Returns the mask argument formatted for F.scaled_dot_product_attention."""
        # If we have a custom mask (like block causal), we return it.
        # SDPA expects a float mask (added to scores) or a bool mask (True positions are masked out).
        if self.mask is not None:
            return self.mask
        return None 


class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim 

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True) #TODO Is it really efficient? Need to do more research on this later.
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: AttentionMask) -> torch.Tensor:
        B, L, C = x.shape
        qkv = self.qkv(x)
        
        # Reshape to (B, L, 3, H, D) -> Permute to (3, B, H, L, D)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Get mask for SDPA
        sdpa_mask = attn_mask.apply_to_sdpa((B, self.num_heads, L, L))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=attn_mask.is_causal if sdpa_mask is None else False 
        )

        out = out.transpose(1, 2).reshape(B, L, C)
        out = self.out_proj(out)
        return out


class LatentCrossAttention(nn.Module):
    """
    Cross-attention for latent tokens in DreamerV4 tokenizer.
    
    Latent tokens attend to ALL tokens (latents + patches) with block causal masking.
    This allows latents to compress information from patches across frames.
    
    Q: from latents only
    K, V: from all tokens (latents + patches)
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        latents: torch.Tensor, 
        context: torch.Tensor,
        attn_mask: Optional[AttentionMask] = None
    ) -> torch.Tensor:
        """
        Args:
            latents: Latent tokens (B, L, D) - these are the queries
            context: All tokens [latents + patches] (B, L+T*N, D) - these are K, V
            attn_mask: Block causal mask for latent-to-all attention
        
        Returns:
            Updated latent tokens (B, L, D)
        """
        B, L_q, C = latents.shape
        _, L_kv, _ = context.shape
        
        # Project Q from latents, K/V from context
        q = self.q_proj(latents)
        k = self.k_proj(context)
        v = self.v_proj(context)
        
        # Reshape for multi-head attention
        q = q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L_q, D)
        k = k.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L_kv, D)
        v = v.view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L_kv, D)
        
        # Get mask for SDPA
        sdpa_mask = None
        if attn_mask is not None:
            sdpa_mask = attn_mask.apply_to_sdpa((B, self.num_heads, L_q, L_kv))
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        
        out = out.transpose(1, 2).reshape(B, L_q, C)
        out = self.out_proj(out)
        return out


class SpatialAttention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        B, L, C = x.shape
        patches_per_frame = L // num_frames
        
        x = x.view(B * num_frames, patches_per_frame, C)
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(B * num_frames, patches_per_frame, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*T, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # No mask needed - all patches in same frame attend freely to each other
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        
        out = out.transpose(1, 2).reshape(B * num_frames, patches_per_frame, C)
        out = out.view(B, L, C)
        
        out = self.out_proj(out)
        return out


class TemporalAttention(nn.Module):
    """
    Temporal Attention: Attends to the same spatial position across all frames.
    
    For video input with T frames and N patches per frame, each patch token
    at spatial position i attends to all patches at position i across all T frames.
    This creates N independent attention computations, each over T tokens.
    
    Input shape: (B, T*N, D) where T=frames, N=patches_per_frame
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        num_frames: int,
        attn_mask: Optional[AttentionMask] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T*N, D)
            num_frames: Number of frames T
            attn_mask: Optional attention mask (e.g., causal masking across time)
        
        Returns:
            Output tensor of shape (B, T*N, D)
        """
        B, L, C = x.shape
        T = num_frames
        patches_per_frame = L // T
        N = patches_per_frame
        
        # Reshape to (B, T, N, D) then transpose to (B, N, T, D)
        # This groups same spatial positions across frames
        x = x.view(B, T, N, C).transpose(1, 2).contiguous()  # (B, N, T, D)
        x = x.view(B * N, T, C)  # (B*N, T, D) - each spatial position is a sequence
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(B * N, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*N, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply scaled dot product attention across frames for each spatial position
        sdpa_mask = None
        if attn_mask is not None:
            sdpa_mask = attn_mask.apply_to_sdpa((B * N, self.num_heads, T, T))
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=attn_mask.is_causal if (attn_mask is not None and sdpa_mask is None) else False
        )
        
        # Reshape back: (B*N, H, T, D) -> (B*N, T, C) -> (B, N, T, D) -> (B, T, N, D) -> (B, T*N, D)
        out = out.transpose(1, 2).reshape(B * N, T, C)
        out = out.view(B, N, T, C).transpose(1, 2).contiguous()  # (B, T, N, D)
        out = out.view(B, L, C)
        
        out = self.out_proj(out)
        return out




class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gate, x_val = self.fc1(x).chunk(2, dim=-1)
        x = x_val * F.silu(x_gate) 
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    DreamerV4-style Transformer Block with factorized spatial-temporal attention.
    
    Architecture per DreamerV4 paper:
    - Latent tokens: cross-attention to ALL tokens (latents + patches) with block causal mask
    - Patch tokens: factorized spatial + temporal attention
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
        use_temporal: bool = False, 
    ):
        super().__init__()
        self.use_temporal = use_temporal
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # For latent tokens: cross-attention to all tokens
        self.latent_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.latent_cross_attn = LatentCrossAttention(embed_dim, num_heads, dropout)
        
        # For patch tokens: spatial attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.spatial_attn = SpatialAttention(embed_dim, num_heads, dropout)
        
        # For patch tokens: temporal attention (every Nth layer)
        if use_temporal:
            self.norm_temporal = nn.LayerNorm(embed_dim)
            self.temporal_attn = TemporalAttention(embed_dim, num_heads, dropout)
        
        # Shared feed-forward
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, mlp_ratio, dropout)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self, 
        x: torch.Tensor, 
        num_frames: int,
        temporal_mask: Optional[AttentionMask] = None,
        latent_cross_mask: Optional[AttentionMask] = None,
        num_latents: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, L+T*N, D) where L=num_latents
            num_frames: Number of frames T
            temporal_mask: Causal mask for temporal attention (T, T)
            latent_cross_mask: Block causal mask for latent-to-all attention (L, L+T*N)
            num_latents: Number of latent tokens at the start of sequence
        """
        if num_latents > 0:
            latents = x[:, :num_latents, :]
            patches = x[:, num_latents:, :]
            
            # 1. Latent cross-attention: latents attend to ALL tokens
            context = torch.cat([latents, patches], dim=1)  # Full sequence
            latents = latents + self.drop_path(
                self.latent_cross_attn(
                    self.latent_norm(latents), 
                    self.context_norm(context),
                    latent_cross_mask
                )
            )
            
            # 2. Spatial attention: patches attend to same-frame patches
            patches = patches + self.drop_path(
                self.spatial_attn(self.norm1(patches), num_frames)
            )
            
            # 3. Temporal attention: same position across frames (if enabled)
            if self.use_temporal:
                patches = patches + self.drop_path(
                    self.temporal_attn(self.norm_temporal(patches), num_frames, temporal_mask)
                )
            
            x = torch.cat([latents, patches], dim=1)
        else: 
            # No latent tokens, just patches
            x = x + self.drop_path(self.spatial_attn(self.norm1(x), num_frames))
            
            if self.use_temporal:
                x = x + self.drop_path(
                    self.temporal_attn(self.norm_temporal(x), num_frames, temporal_mask)
                )
        
        # 4. Feed-forward (applied to all tokens)
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x