import torch
import torch.nn as nn
from typing import Optional
from tokenizer.layers import SpatialAttention, TemporalAttention, FeedForward, RotaryPositionEmbedding, AttentionMask

class DynamicsTransformerBlock(nn.Module):
    """
    Transformer block for the Dynamics Model (Latent Transformer).
    """
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        mlp_ratio: float, 
        dropout: float, 
        drop_path: float, 
        use_temporal: bool = False,
        num_kv_heads: Optional[int] = None,
        rope_spatial: Optional[RotaryPositionEmbedding] = None,
        rope_temporal: Optional[RotaryPositionEmbedding] = None
    ):
        super().__init__()
        self.use_temporal = use_temporal
        self.embed_dim = embed_dim
        
        self.norm_spatial = nn.RMSNorm(embed_dim)
        self.norm_ff = nn.RMSNorm(embed_dim)
        
        self.spatial_attn = SpatialAttention(
            embed_dim, num_heads, num_kv_heads=num_kv_heads, 
            dropout=dropout, rope=rope_spatial
        )
        
        if use_temporal:
            self.norm_temporal = nn.RMSNorm(embed_dim)
            self.temporal_attn = TemporalAttention(
                embed_dim, num_heads, num_kv_heads=num_kv_heads, 
                dropout=dropout, rope=rope_temporal
            )
            
        self.ff = FeedForward(embed_dim, mlp_ratio, dropout)
        
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()
        
    def forward(self, x, num_frames, temporal_mask: Optional[AttentionMask] = None):
        """
        Args:
            x: (B, T * tokens_per_frame, D)
            num_frames: T
            temporal_mask: Causal mask for temporal attention
        """
        x = x + self.drop_path(self.spatial_attn(self.norm_spatial(x), num_frames)) # shape  (B, T*tokens_per_frame, D)
        
        if self.use_temporal:
            x = x + self.drop_path(
                self.temporal_attn(self.norm_temporal(x), num_frames, attn_mask=temporal_mask)
            )
            
        x = x + self.drop_path(self.ff(self.norm_ff(x)))
        
        return x