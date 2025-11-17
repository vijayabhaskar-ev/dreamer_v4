from dataclasses import dataclass
from typing import Literal, Tuple


@dataclass
class TokenizerConfig:
    """Configuration hyperparameters for the Dreamer V4 tokenizer."""

    image_size: Tuple[int, int] = (84, 84)
    patch_size: Tuple[int, int] = (6, 6)
    in_channels: int = 3
    embed_dim: int = 512
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    drop_path: float = 0.1
    mask_prob_min: float = 0.0
    mask_prob_max: float = 0.9
    latent_dim: int = 1024
    num_latent_tokens: int = 32
    learned_cls_tokens: int = 1
    reconstruction_loss: Literal["mse_lpips"] = "mse_lpips"
    norm_loss_by: Literal["pixels", "tokens"] = "pixels"
    tokenizer_lr: float = 1e-4
    weight_decay: float = 0.05
    lpips_net: Literal["vgg", "alex", "squeeze"] = "vgg"
    use_grad_checkpoint: bool = False


__all__ = ["TokenizerConfig"]
