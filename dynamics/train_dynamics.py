"""
Usage:
    python -m dynamics.train_dynamics \
        --tokenizer-ckpt checkpoints/tokenizer/final.pt \
        --epochs 20 \
        --batch-size 16

Requires a pretrained tokenizer checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .config import DynamicsConfig
from .trainer import DynamicsTrainer, DynamicsTrainingConfig
from tokenizer.config import TokenizerConfig
from tokenizer.dataset import DatasetFactory
from device_utils import get_device, should_use_xla, is_master, wrap_loader
import wandb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Dreamer V4 dynamics model")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--weight-decay-heavy", type=float, default=0.1,
                        help="Weight decay for attention+FF layers (higher to prevent drift)")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup-steps", type=int, default=500)   
    parser.add_argument("--min-lr", type=float, default=1e-6) 
    parser.add_argument("--curriculum-warmup-steps", type=int, default=2000,
                        help="Flow-only phase steps before bootstrap ramp")
    parser.add_argument("--curriculum-ramp-steps", type=int, default=4000,
                        help="End of gradual bootstrap introduction")

    # Paths
    parser.add_argument("--tokenizer-ckpt", type=str, required=True,
                        help="Path to pretrained tokenizer checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/dynamics")
    parser.add_argument("--save-final", type=str, default="checkpoints/dynamics/final.pt")
    parser.add_argument("--resume-from", type=str, default=None)

    # Tokenizer config overrides (only needed if NOT loading from checkpoint)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-latent-tokens", type=int, default=32)
    parser.add_argument("--tokenizer-embed-dim", type=int, default=512)

    # Dynamics config
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--K-max", type=int, default=64)
    parser.add_argument("--K-inference", type=int, default=4)
    parser.add_argument("--seq-len-short", type=int, default=8,
                        help="T₁: short batch sequence length (used most of the time)")
    parser.add_argument("--seq-len-long", type=int, default=32,
                        help="T₂: long batch sequence length (occasional, must be > context-length)")
    parser.add_argument("--context-length", type=int, default=16,
                        help="C: sliding window size for temporal attention")
    parser.add_argument("--long-batch-ratio", type=float, default=0.15,
                        help="Fraction of training steps using long sequences")
    parser.add_argument("--num-register-tokens", type=int, default=4)
    parser.add_argument("--temporal-interval", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=6)

    # Logging
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--log-model-stats-interval", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=None)

    # Dataset
    parser.add_argument("--dataset", type=str, default="dm_control",
                        help="Dataset type: dm_control, offline, moving_square")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Path to .npz file (required when --dataset=offline)")
    parser.add_argument("--task", type=str, default="cheetah_run")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--val-steps-per-epoch", type=int, default=None)

    # WandB
    parser.add_argument("--wandb-project", type=str, default="dreamer-v4-dynamics")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--wandb-disabled", action="store_true")

    return parser




def load_tokenizer_config_from_ckpt(ckpt_path: str, device: str = "cpu") -> Optional[TokenizerConfig]:
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "tokenizer_cfg" in state:
            return state["tokenizer_cfg"]
    except Exception:
        pass
    return None



def _train_fn(index=0, args=None):
    """Per-device training function (called by xmp.spawn or directly)."""
    opts = args if args is not None else build_parser().parse_args()

    ckpt_dir = Path(opts.checkpoint_dir)
    if is_master():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Try loading from checkpoint first (safest — guaranteed to match)
    tokenizer_cfg = load_tokenizer_config_from_ckpt(opts.tokenizer_ckpt, device="cpu")

    if tokenizer_cfg is None:
        # Fall back to CLI args — user must ensure these match the checkpoint
        if is_master():
            print("[WARN] Could not load TokenizerConfig from checkpoint. "
                  "Using CLI args — make sure they match the pretrained tokenizer!")
        tokenizer_cfg = TokenizerConfig(
            image_size=(opts.image_size, opts.image_size),
            patch_size=(opts.patch_size, opts.patch_size),
            latent_dim=opts.latent_dim,
            num_latent_tokens=opts.num_latent_tokens,
            embed_dim=opts.tokenizer_embed_dim,
            seq_len=opts.seq_len_short,
            dataset_name=opts.dataset,
            task_name=opts.task,
        )
    else:
        if is_master():
            print(f"[INFO] Loaded TokenizerConfig from checkpoint: "
                  f"latent_dim={tokenizer_cfg.latent_dim}, "
                  f"num_latent_tokens={tokenizer_cfg.num_latent_tokens}")

    dynamics_cfg = DynamicsConfig.from_tokenizer(
        tokenizer_cfg,
        embed_dim=opts.embed_dim,
        depth=opts.depth,
        num_heads=opts.num_heads,
        num_kv_heads=opts.num_kv_heads,
        K_max=opts.K_max,
        K_inference=opts.K_inference,
        seq_len_short=opts.seq_len_short,
        seq_len_long=opts.seq_len_long,
        seq_len=opts.seq_len_short,
        context_length=opts.context_length,
        num_register_tokens=opts.num_register_tokens,
        temporal_interval=opts.temporal_interval,
        action_dim=opts.action_dim,
    )

    checkpoint_interval = opts.checkpoint_interval
    if checkpoint_interval is None:
        checkpoint_interval = max(1, opts.epochs // 5)

    training_cfg = DynamicsTrainingConfig(
        epochs=opts.epochs,
        batch_size=opts.batch_size,
        lr=opts.lr,
        weight_decay=opts.weight_decay,
        weight_decay_heavy=opts.weight_decay_heavy,
        grad_clip=opts.grad_clip,
        amp=opts.amp,
        device=opts.device,
        checkpoint_interval=checkpoint_interval,
        log_interval=opts.log_interval,
        log_model_stats_interval=opts.log_model_stats_interval,
        warmup_steps=opts.warmup_steps,
        min_lr=opts.min_lr,
        curriculum_warmup_steps=opts.curriculum_warmup_steps,
        curriculum_ramp_steps=opts.curriculum_ramp_steps,
        steps_per_epoch=opts.steps_per_epoch,
        long_batch_ratio=opts.long_batch_ratio,
    )

    # Only master initializes wandb; workers get disabled mode
    if is_master():
        if opts.wandb_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{opts.dataset}_{opts.task}_{timestamp}"
        else:
            run_name = opts.wandb_name
        wandb_mode = "disabled" if opts.wandb_disabled else ("offline" if opts.wandb_offline else "online")
        wandb.init(
            project=opts.wandb_project,
            entity=opts.wandb_entity,
            name=run_name,
            config={
                "dynamics": vars(dynamics_cfg) if hasattr(dynamics_cfg, '__dict__') else str(dynamics_cfg),
                "training": vars(training_cfg),
                "tokenizer": vars(tokenizer_cfg) if hasattr(tokenizer_cfg, '__dict__') else str(tokenizer_cfg),
                "task": opts.task,
                "dataset": opts.dataset,
            },
            mode=wandb_mode,
        )
    else:
        wandb.init(mode="disabled")


    # Short-sequence dataset (T₁) — used for most training steps
    dataset_cfg_short = replace(
        tokenizer_cfg,
        dataset_name=opts.dataset,
        task_name=opts.task,
        seq_len=dynamics_cfg.seq_len_short,
    )
    # Long-sequence dataset (T₂) — used occasionally for length generalization
    dataset_cfg_long = replace(
        tokenizer_cfg,
        dataset_name=opts.dataset,
        task_name=opts.task,
        seq_len=dynamics_cfg.seq_len_long,
    )

    train_steps_per_worker = opts.steps_per_epoch
    if opts.num_workers > 0:
        train_steps_per_worker = max(1, opts.steps_per_epoch // opts.num_workers)

    # Short and long training datasets
    train_dataset_short = DatasetFactory.get_dataset(
        dataset_cfg_short,
        batch_size=training_cfg.batch_size,
        steps_per_epoch=train_steps_per_worker,
        dataset_path=opts.dataset_path,
    )
    train_dataset_long = DatasetFactory.get_dataset(
        dataset_cfg_long,
        batch_size=training_cfg.batch_size,
        steps_per_epoch=train_steps_per_worker,
        dataset_path=opts.dataset_path,
    )

    # Validation uses short sequences only
    val_steps = opts.val_steps_per_epoch
    if val_steps is None:
        val_steps = max(1, train_steps_per_worker // 10)

    val_dataset = None
    if val_steps > 0:
        val_dataset = DatasetFactory.get_dataset(
            dataset_cfg_short,
            batch_size=training_cfg.batch_size,
            steps_per_epoch=val_steps,
            dataset_path=opts.dataset_path,
        )

    device = get_device(opts.device)
    use_pin_memory = device.type == "cuda"

    train_loader_short_raw = DataLoader(
        train_dataset_short,
        batch_size=None,
        num_workers=opts.num_workers,
        pin_memory=use_pin_memory,
        multiprocessing_context='spawn' if opts.num_workers > 0 else None,
    )
    train_loader_long_raw = DataLoader(
        train_dataset_long,
        batch_size=None,
        num_workers=opts.num_workers,
        pin_memory=use_pin_memory,
        multiprocessing_context='spawn' if opts.num_workers > 0 else None,
    )

    val_loader_raw = DataLoader(
        val_dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=use_pin_memory,
    ) if val_dataset is not None else None

    # Wrap with MpDeviceLoader for async host→device transfer on TPU
    train_loader_short = wrap_loader(train_loader_short_raw, device)
    train_loader_long = wrap_loader(train_loader_long_raw, device)
    val_loader = wrap_loader(val_loader_raw, device) if val_loader_raw is not None else None

    trainer = DynamicsTrainer(
        dynamics_cfg=dynamics_cfg,
        tokenizer_cfg=tokenizer_cfg,
        training_cfg=training_cfg,
        tokenizer_ckpt=opts.tokenizer_ckpt,
    )

    start_epoch = 1
    if opts.resume_from is not None:
        if is_master():
            print(f"[INFO] Resuming from {opts.resume_from}")
        start_epoch = trainer.load_checkpoint(opts.resume_from)
        if is_master():
            print(f"[INFO] Resuming from epoch {start_epoch}")

    if is_master():
        total_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"Dreamer V4 Dynamics Training")
        print(f"{'='*60}")
        print(f"  Task:           {opts.task}")
        print(f"  Trainable params: {total_params:,}")
        print(f"  Latent shape:   ({dynamics_cfg.num_latent_tokens}, {dynamics_cfg.latent_input_dim})")
        print(f"  Embed dim:      {dynamics_cfg.embed_dim}")
        print(f"  Depth:          {dynamics_cfg.depth}")
        print(f"  T_short (T₁):   {dynamics_cfg.seq_len_short}")
        print(f"  T_long  (T₂):   {dynamics_cfg.seq_len_long}")
        print(f"  Context (C):    {dynamics_cfg.context_length}")
        print(f"  Long ratio:     {training_cfg.long_batch_ratio}")
        print(f"  K_max:          {dynamics_cfg.K_max}")
        print(f"  Batch size:     {training_cfg.batch_size}")
        print(f"  LR:             {training_cfg.lr}")
        print(f"  Epochs:         {training_cfg.epochs}")
        print(f"  Device:         {trainer.device}")
        print(f"{'='*60}\n")

    trainer.fit(
        train_loader_short=train_loader_short,
        train_loader_long=train_loader_long,
        val_loader=val_loader,
        checkpoint_dir=str(ckpt_dir),
        start_epoch=start_epoch,
    )

    final_path = opts.save_final or str(ckpt_dir / "final.pt")
    trainer.save_checkpoint(final_path, epoch=training_cfg.epochs)
    if is_master():
        print(f"\n[DONE] Final checkpoint saved to {final_path}")

    wandb.finish()


def main(args: Optional[list[str]] = None) -> None:
    opts = build_parser().parse_args(args)

    if should_use_xla(opts.device):
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(_train_fn, args=(opts,), nprocs=None)
    else:
        #TODO Implement multi-GPU training
        _train_fn(index=0, args=opts)


if __name__ == "__main__":
    main()
