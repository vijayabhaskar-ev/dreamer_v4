# block_causal.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# -------------------------
# Utilities
# -------------------------
def masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1, eps: float = 1e-9):
    if mask is None:
        return F.softmax(logits, dim=dim)
    # mask: broadcastable boolean (True for allowed positions)
    neg_inf = -1e9
    logits = logits.masked_fill(~mask, neg_inf)
    return F.softmax(logits, dim=dim)

# -------------------------
# RMSNorm
# -------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.scale

# -------------------------
# Rotary positional embeddings (RoPE)
# -------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device=None):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq, dim)
        return emb  # to be used with sin/cos

def apply_rope(x: torch.Tensor, rope_emb: torch.Tensor):
    # x: (..., seq, dim)
    # rope_emb: (seq, dim)
    sin = rope_emb.sin()[None, :, None, :]  # broadcast: (1, seq, 1, dim)
    cos = rope_emb.cos()[None, :, None, :]
    # split dim into pairs
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot

# -------------------------
# SwiGLU MLP
# -------------------------
class SwiGLU(nn.Module):
    def __init__(self, dim, mult=4.0):
        super().__init__()
        hidden = int(dim * mult)
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

# -------------------------
# Grouped Query Attention (GQA) / Multihead Attention
# -------------------------
class GQAAttention(nn.Module):
    def __init__(self, dim, n_heads=8, head_dim=None, n_query_groups=1, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_groups = n_query_groups
        if head_dim is None:
            assert dim % n_heads == 0
            head_dim = dim // n_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        # Query grouping: group queries into n_groups (reduce Q dims)
        self.q = nn.Linear(dim, n_heads * head_dim)
        self.kv = nn.Linear(dim, 2 * n_heads * head_dim)
        self.proj = nn.Linear(n_heads * head_dim, dim)
        self.dropout = nn.Dropout(dropout)

        # if grouped queries < heads, we will repeat q across groups later

    def forward(self, q_in, k_in, v_in, mask: Optional[torch.Tensor] = None, rope: Optional[torch.Tensor] = None):
        """
        q_in: (B, Lq, D)
        k_in: (B, Lk, D)
        v_in: (B, Lk, D)
        mask: (B, Lq, Lk) boolean where True means allowed
        rope: (Lk, D) or (Lq, D) rotary embeddings (optional)
        """
        B, Lq, _ = q_in.shape
        _, Lk, _ = k_in.shape

        q = self.q(q_in)  # (B, Lq, H*Hd)
        kv = self.kv(k_in)  # (B, Lk, 2*H*Hd)
        k, v = torch.chunk(kv, 2, dim=-1)

        # reshape to heads
        q = q.view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, Lq, Hd)
        k = k.view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, Lk, Hd)
        v = v.view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, Lk, Hd)

        # Optionally apply RoPE: assume rope is (L, D) and D == head_dim*2 (even)
        # Apply separate for q and k if provided
        if rope is not None:
            # rope for q and k may differ in length
            if rope.shape[0] == Lq:
                q = _apply_rope_to_heads(q, rope)
            if rope.shape[0] == Lk:
                k = _apply_rope_to_heads(k, rope)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, Lq, Lk)

        if mask is not None:
            # mask expected shape: (B, Lq, Lk) or (Lq, Lk)
            if mask.dim() == 2:
                mask2 = mask[None, None, :, :].expand(B, self.n_heads, -1, -1)
            else:
                mask2 = mask[:, None, :, :].expand(-1, self.n_heads, -1, -1)
            attn = masked_softmax(attn_logits, mask2, dim=-1)
        else:
            attn = F.softmax(attn_logits, dim=-1)

        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, Lq, Hd)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.n_heads * self.head_dim)
        out = self.proj(out)
        return out

def _apply_rope_to_heads(x_heads: torch.Tensor, rope_emb: torch.Tensor):
    # x_heads: (B, H, L, Hd) where Hd must be even
    B, H, L, Hd = x_heads.shape
    x = x_heads.permute(2, 0, 1, 3)  # (L, B, H, Hd)
    rope = rope_emb[:, None, None, :]  # (L, 1, 1, Hd)
    # perform pair-wise rotary on last dim: we need to interleave dims pairs
    x = x.reshape(L, B, H, Hd)
    # rotate pairs
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    sin = rope[..., ::2]
    cos = rope[..., 1::2]
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos
    x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1).reshape(L, B, H, Hd)
    return x_rot.permute(1, 2, 0, 3)  # back to (B, H, L, Hd)

# -------------------------
# Attention block wrapper
# -------------------------
class AttentionBlock(nn.Module):
    def __init__(self, dim, n_heads=8, head_dim=None, gqa_groups=1, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.attn = GQAAttention(dim, n_heads=n_heads, head_dim=head_dim, n_query_groups=gqa_groups, dropout=dropout)
        self.proj_ln = RMSNorm(dim)

    def forward(self, x_q, x_kv, mask: Optional[torch.Tensor] = None, rope: Optional[torch.Tensor] = None):
        # x_q: (B, Lq, D), x_kv: (B, Lk, D)
        x = self.norm(x_q)
        attn_out = self.attn(x, x_kv, x_kv, mask=mask, rope=rope)
        return x_q + attn_out

# -------------------------
# Space-only Block (per time-step, per-modality)
# -------------------------
class SpaceBlock(nn.Module):
    """
    Applies attention **within each modality and within the same time step**.
    To run over a full batch of T time steps, we iterate over time dimension
    and apply attention on the tokens of each modality separately.
    """
    def __init__(self, dim, n_heads=8, head_dim=None, gqa_groups=1, dropout=0.0):
        super().__init__()
        self.attn_block = AttentionBlock(dim, n_heads=n_heads, head_dim=head_dim, gqa_groups=gqa_groups, dropout=dropout)
        self.mlp = SwiGLU(dim)
        self.mlp_norm = RMSNorm(dim)

    def forward(self, tokens_by_modality):
        """
        tokens_by_modality: dict[modal_name] = Tensor shape (B, T, L_mod, D)
         - We apply self-attention **within** the (L_mod) tokens for each time step independently.
        Returns: updated tokens_by_modality (same shapes)
        """
        out = {}
        B, T = next(iter(tokens_by_modality.values())).shape[:2]
        for mod, x in tokens_by_modality.items():
            # x: (B, T, L, D)
            B, T, L, D = x.shape
            x_flat = x.view(B * T, L, D)  # collapse time for per-frame attention
            x_flat = self.attn_block(x_flat, x_flat, mask=None, rope=None)
            x_flat = x_flat + self.mlp(self.mlp_norm(x_flat))
            out[mod] = x_flat.view(B, T, L, D)
        return out

# -------------------------
# Time-only (causal) Block — operates on latents across time
# -------------------------
class TimeBlock(nn.Module):
    """
    Applies causal attention across time for latent tokens.
    We assume latents shaped (B, T, K, D). We'll allow each latent token
    at time t to attend to latents at times <= t (causal / triangular mask).
    """
    def __init__(self, dim, n_heads=8, head_dim=None, gqa_groups=1, dropout=0.0):
        super().__init__()
        self.attn_block = AttentionBlock(dim, n_heads=n_heads, head_dim=head_dim, gqa_groups=gqa_groups, dropout=dropout)
        self.mlp = SwiGLU(dim)
        self.mlp_norm = RMSNorm(dim)

    def _causal_mask(self, B, T, K, device):
        # Build mask of shape (B, T*K, T*K) where only positions with time_j <= time_i are allowed
        idx = torch.arange(T, device=device)
        ti = (idx[:, None] * K + torch.arange(K, device=device)[None, :]).reshape(-1)  # mapping time->chunk indices
        # easier: build block mask by times
        mask = torch.zeros(T * K, T * K, dtype=torch.bool, device=device)
        for t in range(T):
            i_start = t * K
            i_end = i_start + K
            for s in range(0, t + 1):
                j_start = s * K
                j_end = j_start + K
                mask[i_start:i_end, j_start:j_end] = True
        mask = mask.unsqueeze(0).expand(B, -1, -1)  # (B, Lq, Lk)
        return mask

    def forward(self, latents):
        # latents: (B, T, K, D)
        B, T, K, D = latents.shape
        x = latents.view(B, T * K, D)  # treat each time-chunk as group of tokens
        mask = self._causal_mask(B, T, K, latents.device)  # (B, L, L)
        x = self.attn_block(x, x, mask=mask, rope=None)
        x = x + self.mlp(self.mlp_norm(x))
        return x.view(B, T, K, D)

# -------------------------
# BlockCausalTransformer wrapper (alternating blocks)
# -------------------------
class BlockCausalTransformer(nn.Module):
    def __init__(self, dim, n_layers=12, space_every=2, n_heads=8, head_dim=None, gqa_groups=1, dropout=0.0, latent_k=16):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.space_every = space_every
        self.latent_k = latent_k

        # build layers
        self.space_layers = nn.ModuleList()
        self.time_layers = nn.ModuleList()
        for i in range(n_layers):
            if (i + 1) % (space_every + 1) != 0:
                self.space_layers.append(SpaceBlock(dim, n_heads=n_heads, head_dim=head_dim, gqa_groups=gqa_groups, dropout=dropout))
                self.time_layers.append(nn.Identity())
            else:
                self.space_layers.append(nn.Identity())
                self.time_layers.append(TimeBlock(dim, n_heads=n_heads, head_dim=head_dim, gqa_groups=gqa_groups, dropout=dropout))

    def forward(self, tokens_by_modality, latents):
        """
        tokens_by_modality: dict(mod)->Tensor (B, T, L_mod, D)
        latents: Tensor (B, T, K, D)
        """
        x_mod = tokens_by_modality
        x_lat = latents
        for space_blk, time_blk in zip(self.space_layers, self.time_layers):
            # apply space block if not identity
            if not isinstance(space_blk, nn.Identity):
                x_mod = space_blk(x_mod)  # updates each modality per time
            # apply time block if not identity
            if not isinstance(time_blk, nn.Identity):
                # time block expects latents only
                x_lat = time_blk(x_lat)
            # After a time block we may want to fuse latents into modalities (encoder latents attend to modalities).
            # For simplicity, we skip cross-attention here. In practice you'd implement a cross-attn step where
            # latents read modality tokens (encoder fusion) -> latents = LatentRead(latents, modalities)
        return x_mod, x_lat

# -------------------------
# Simple example usage
# -------------------------
if __name__ == "__main__":
    B = 2
    T = 4
    L_img = 16   # patches per frame
    L_proprio = 4
    K = 8        # latents per time
    D = 128

    # Dummy inputs
    img_tokens = torch.randn(B, T, L_img, D)        # image patch tokens
    proprio_tokens = torch.randn(B, T, L_proprio, D)
    latents = torch.randn(B, T, K, D)

    tokens_by_mod = {
        "image": img_tokens,
        "proprio": proprio_tokens,
    }

    model = BlockCausalTransformer(dim=D, n_layers=6, space_every=2, n_heads=8, head_dim=16, gqa_groups=1, dropout=0.1, latent_k=K)
    out_mod, out_lat = model(tokens_by_mod, latents)
    print("out_mod['image']:", out_mod["image"].shape)   # (B, T, L_img, D)
    print("out_lat:", out_lat.shape)                    # (B, T, K, D)
