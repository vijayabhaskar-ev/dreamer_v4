"""Core transformer layers for the Dreamer V4 tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionMask:
    causal: bool
    mask: Optional[torch.Tensor] = None

    def apply(self, attn_scores: torch.Tensor) -> torch.Tensor:
        if self.mask is not None:
            attn_scores = attn_scores + self.mask
        if self.causal:
            L = attn_scores.size(-1)
            causal_mask = torch.triu(torch.ones(L, L, device=attn_scores.device), diagonal=1)
            attn_scores = attn_scores.masked_fill(causal_mask.bool(), float("-inf"))
        return attn_scores


class MultiheadSelfAttention(nn.Module):
    """Block-causal self-attention with optional additional masking."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: AttentionMask) -> torch.Tensor:
        B, L, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_scores = attn_mask.apply(attn_scores)
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, C)
        out = self.out_proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Standard Pre-LN transformer block with block-causal self-attention."""

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
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: AttentionMask) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask))
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


__all__ = [
    "AttentionMask",
    "MultiheadSelfAttention",
    "TransformerBlock",
]
