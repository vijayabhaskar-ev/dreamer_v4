from dataclasses import dataclass
from typing import Literal, Tuple


@dataclass
class TokenizerConfig:
    """Configuration hyperparameters for the Dreamer V4 tokenizer."""

    image_size: Tuple[int, int] = (64, 64)
    patch_size: Tuple[int, int] = (8, 8)
    in_channels: int = 3
    embed_dim: int = 512
    depth: int = 8
    num_heads: int = 8
    # Dataset
    dataset_name: str = "dm_control"
    task_name: str = "cheetah_run"
    action_repeat: int = 2
    seq_len: int = 4 # Number of frames per sequence

    # Optimization
    mlp_ratio: float = 4.0 #TODO Need to check the implementation of mlp_ratio after completion of the tokenizer
    dropout: float = 0.1
    drop_path: float = 0.1 #TODO Dreawer v4 uses droppath instead of dropout. Need to check the implementation of drop_path after completion of the tokenizer
    mask_prob_min: float = 0.0
    mask_prob_max: float = 0.9
    latent_dim: int = 1024
    num_latent_tokens: int = 32 #TODO Should mopdify the num_latent_tokens based on the image size and patch size. Maybe compute it dyamically?
    learned_cls_tokens: int = 0 #TODO Not used in dreamer v4/ MAE
    reconstruction_loss: Literal["mse_lpips"] = "mse" #TODO FOr inital debugging chnageing this pnly to MSE and not mse_lpips
    norm_loss_by: Literal["pixels", "tokens"] = "pixels"
    tokenizer_lr: float = 1e-4 #TODO Adjust base don the normaal size. Need to check the implementation of tokenizer_lr after completion of the tokenizer
    weight_decay: float = 0.05
    lpips_net: Literal["vgg", "alex", "squeeze"] = "vgg" #TODO Need to check the implementation of lpips_net after  completion of the tokenizer
    use_grad_checkpoint: bool = False #TODO Do more research on this


__all__ = ["TokenizerConfig"]
