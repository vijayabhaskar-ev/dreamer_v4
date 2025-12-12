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

        # Packed QKV is standard and efficient
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

        # Flash Attention / SDPA
        # Note: If is_causal=True in SDPA, it applies strict triangular masking. 
        # Since we use Block Causal (custom), we pass is_causal=False and provide the explicit mask.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=attn_mask.is_causal if sdpa_mask is None else False 
        )

        out = out.transpose(1, 2).reshape(B, L, C)
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
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, mlp_ratio, dropout)
        # Identity used if drop_path is 0 to avoid graph overhead of unused Dropout layer
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: AttentionMask) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask))
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x