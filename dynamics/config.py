@dataclass
class DynamicsConfig:
    embed_dim: int = 512          # Internal model dimension
    latent_input_dim: int = 64    # From tokenizer bottleneck
    depth: int = 12               # More layers than tokenizer
    num_heads: int = 8
    num_kv_heads: int = 2         # GQA
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4  # Paper: register tokens
    temporal_interval: int = 4    # Temporal attn every 4 layers
    
    # Flow matching
    K_max: int = 64               # Max steps for training
    K_inference: int = 4          # Steps at inference
    tau_ctx: float = 0.1          # Context corruption
    
    # Action space (DM Control)
    action_dim: int = 6           # Task-dependent
    num_action_tokens: int = 1    # S_a in the paper
    
    # Training
    dropout: float = 0.1
    drop_path: float = 0.1
    
    # From tokenizer (must match)
    num_latent_tokens: int = 32   # S_z spatial tokens per frame