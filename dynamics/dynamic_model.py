import torch
import torch.nn as nn
from .embedding import ActionEmbedding, TauDEmbedding
from .dynamic_block import DynamicsTransformerBlock
from .config import DynamicsConfig
from tokenizer.layers import RotaryPositionEmbedding, AttentionMask

class DynamicsModel(nn.Module):
    def __init__(self, config: DynamicsConfig, tokenizer):
        super().__init__()
        self.config = config
        self.config.validate_against_tokenizer(tokenizer.config)

        self.tokenizer = tokenizer
        self.tokenizer.requires_grad_(False)
        self.tokenizer.eval()

        self.action_embedding = ActionEmbedding(config.action_dim, config.embed_dim)
        self.tau_d_embedding = TauDEmbedding(config.embed_dim)
        
        self.proj_in = nn.Linear(config.latent_input_dim, config.embed_dim)
        self.proj_out = nn.Linear(config.embed_dim, config.latent_input_dim)
        
        self.register_tokens = nn.Parameter(torch.randn(1, config.num_register_tokens, config.embed_dim) * 0.02 )
        
        tokens_per_frame = 1 + 1 + config.num_latent_tokens + config.num_register_tokens  # a + τd + z̃ + reg
        self.rope_spatial = RotaryPositionEmbedding(config.embed_dim // config.num_heads, max_positions=tokens_per_frame)
        self.rope_temporal = RotaryPositionEmbedding(config.embed_dim // config.num_heads, max_positions=config.seq_len)
        
        self.blocks = nn.ModuleList([
            DynamicsTransformerBlock(
                config.embed_dim, config.num_heads, config.mlp_ratio, config.dropout, config.drop_path,
                use_temporal=(i % config.temporal_interval == 0),
                num_kv_heads=config.num_kv_heads,
                rope_spatial=self.rope_spatial,
                rope_temporal=self.rope_temporal
            ) for i in range(config.depth)
        ])
        
        self.norm = nn.RMSNorm(config.embed_dim)
        
    def encode_frames(self, frames):
        """Use frozen tokenizer to get bottleneck latents."""
        with torch.no_grad():
            B, T = frames.shape[:2]
            output = self.tokenizer(frames)
            z = output.latent_tokens   #Shape: (B, num_latent_tokens, latent_dim)
            z = z.view(B, T, -1,z.shape[-1])
            return z
        
    def forward(self, z_noised, actions, tau, d):

        B, T, S_z, D_latent = z_noised.shape

        z_up = self.proj_in(z_noised) #Shape: (B, T, S_z, D_latent) -> (B, T, S_z, D_embed)
        tau_d = self.tau_d_embedding(tau, d) #Shape: (B, T, D_embed) 

        total_tokens_per_frame = 1 + 1 + self.config.num_latent_tokens + self.config.num_register_tokens  # a + τd + z̃ + reg

        frame_blocks = []
        for t in range(T):
            tokens_this_frame = []
            
            if t == 0 or actions is None:
                a_token = self.action_embedding(None, batch_size=B)
            else:
                a_token = self.action_embedding(actions[:, t-1 : t], batch_size=B)
            tokens_this_frame.append(a_token)
            
            tokens_this_frame.append(tau_d[:,t:t+1])  # (B, 1, D)  
            
            
            tokens_this_frame.append(z_up[:, t])  # (B, S_z, D)
            
            registers = self.register_tokens.expand(B, -1, -1)  # (1, S_r, D) → (B, S_r, D)
            tokens_this_frame.append(registers)
            
            frame_block = torch.cat(tokens_this_frame, dim=1)  # (B, 38, D)
            frame_blocks.append(frame_block)

        x = torch.cat(frame_blocks, dim=1)  # (B, T*38, D)
        # Frame 0: [a₀, τd₀, z̃₀¹, z̃₀², ..., z̃₀³², reg₀¹, ..., reg₀⁴]  ← 38 tokens
        # Frame 1: [a₁, τd₁, z̃₁¹, z̃₁², ..., z̃₁³², reg₁¹, ..., reg₁⁴]  ← 38 tokens
        # Frame 2: [a₂, τd₂, z̃₂¹, z̃₂², ..., z̃₂³², reg₂¹, ..., reg₂⁴]  ← 38 tokens
        # Frame 3: [a₃, τd₃, z̃₃¹, z̃₃², ..., z̃₃³², reg₃¹, ..., reg₃⁴]  ← 38 tokens

        # Flatten → ONE sequence of 4 × 38 = 152 tokens: (B, 152, D)
        temporal_attn_mask = AttentionMask(is_causal=True)

        for block in self.blocks:
            x = block(x, num_frames=T, temporal_mask=temporal_attn_mask)
        x = self.norm(x) # (B, T * (2 + S_z + S_r), D_embed)

        x = x.view(B , T, total_tokens_per_frame, -1)

        z_prediction = x[:,:, 2:(1 + 1 + self.config.num_latent_tokens ), :]

        z_down = self.proj_out(z_prediction)
        return  z_down
