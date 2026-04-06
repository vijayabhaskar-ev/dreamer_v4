from dataclasses import dataclass
from typing import Any

from tokenizer.config import TokenizerConfig


@dataclass
class DynamicsConfig:
    embed_dim: int = 512          # Internal model dimension
    latent_input_dim: int = 128   # Must match tokenizer latent_dim
    depth: int = 12               
    num_heads: int = 8
    num_kv_heads: int = 2         # GQA
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4  # register tokens
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
    num_latent_tokens: int = 32

    # Sequence lengths and context window
    seq_len_short: int = 8       
    seq_len_long: int = 32      
    context_length: int = 16      

    # Keep seq_len as an alias for backward compatibility (defaults to seq_len_short)
    seq_len: int = 8

    # Agent tokens (Phase 2: agent finetuning)
    num_agent_tokens: int = 1         # S_agent per frame (paper Section 3.3)

    # Prediction heads
    head_hidden_dim: int = 512        # MLP hidden dim for reward/continue heads
    head_num_layers: int = 4          # Number of MLP layers in heads
    num_reward_bins: int = 255        # Symexp twohot bins for reward head
    mtp_length: int = 0              # Multi-token prediction horizon (0 = disabled, 8 for Phase 2)
    reward_loss_scale: float = 1.0    # Weight for reward head loss
    continue_loss_scale: float = 1.0  # Weight for continue head loss
    bc_loss_scale: float = 1.0        # Weight for behavior cloning loss

    @classmethod
    def from_tokenizer(cls, tokenizer_cfg: TokenizerConfig, **overrides: Any) -> "DynamicsConfig":
        """Build dynamics config with shared tokenizer fields copied automatically."""
        shared = {"latent_input_dim", "num_latent_tokens"}
        conflicts = shared & overrides.keys()
        if conflicts:
            raise ValueError(
                f"Cannot override shared fields {conflicts}; "
                "they are derived from TokenizerConfig"
            )
        return cls(
            latent_input_dim=tokenizer_cfg.latent_dim,
            num_latent_tokens=tokenizer_cfg.num_latent_tokens,
            **overrides,
        )

    def validate_against_tokenizer(self, tokenizer_cfg: TokenizerConfig) -> None:
        """Raise if shared dimensions between tokenizer and dynamics are inconsistent."""
        mismatches = []

        if self.latent_input_dim != tokenizer_cfg.latent_dim:
            mismatches.append(
                f"latent_input_dim={self.latent_input_dim} but tokenizer.latent_dim={tokenizer_cfg.latent_dim}"
            )

        if self.num_latent_tokens != tokenizer_cfg.num_latent_tokens:
            mismatches.append(
                "num_latent_tokens="
                f"{self.num_latent_tokens} but tokenizer.num_latent_tokens={tokenizer_cfg.num_latent_tokens}"
            )

        if mismatches:
            raise ValueError(
                "DynamicsConfig is inconsistent with TokenizerConfig:\n- " + "\n- ".join(mismatches)
            )
