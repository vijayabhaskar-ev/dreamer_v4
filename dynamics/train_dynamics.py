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
import wandb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Dreamer V4 dynamics model")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

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
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--num-register-tokens", type=int, default=4)
    parser.add_argument("--temporal-interval", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=6)

    # Logging
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--log-model-stats-interval", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=None)

    # Dataset
    parser.add_argument("--dataset", type=str, default="dm_control")
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



def main(args: Optional[list[str]] = None) -> None:
    parser = build_parser()
    opts = parser.parse_args(args)

    ckpt_dir = Path(opts.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Try loading from checkpoint first (safest — guaranteed to match)
    tokenizer_cfg = load_tokenizer_config_from_ckpt(opts.tokenizer_ckpt, device=opts.device)

    if tokenizer_cfg is None:
        # Fall back to CLI args — user must ensure these match the checkpoint
        print("[WARN] Could not load TokenizerConfig from checkpoint. "
              "Using CLI args — make sure they match the pretrained tokenizer!")
        tokenizer_cfg = TokenizerConfig(
            image_size=(opts.image_size, opts.image_size),
            patch_size=(opts.patch_size, opts.patch_size),
            latent_dim=opts.latent_dim,
            num_latent_tokens=opts.num_latent_tokens,
            embed_dim=opts.tokenizer_embed_dim,
            seq_len=opts.seq_len,
            dataset_name=opts.dataset,
            task_name=opts.task,
        )
    else:
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
        seq_len=opts.seq_len,
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
        grad_clip=opts.grad_clip,
        amp=opts.amp,
        device=opts.device,
        checkpoint_interval=checkpoint_interval,
        log_interval=opts.log_interval,
        log_model_stats_interval=opts.log_model_stats_interval,
    )

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


    dataset_cfg = replace(
        tokenizer_cfg,
        dataset_name=opts.dataset,
        task_name=opts.task,
        seq_len=dynamics_cfg.seq_len,
    )
    train_steps_per_worker = opts.steps_per_epoch
    if opts.num_workers > 0:
        train_steps_per_worker = max(1, opts.steps_per_epoch // opts.num_workers)

    train_dataset = DatasetFactory.get_dataset(
        dataset_cfg,
        batch_size=training_cfg.batch_size,
        steps_per_epoch=train_steps_per_worker,
    )

    val_steps = opts.val_steps_per_epoch
    if val_steps is None:
        val_steps = max(1, train_steps_per_worker // 10)

    val_dataset = None
    if val_steps > 0:
        val_dataset = DatasetFactory.get_dataset(
            dataset_cfg,
            batch_size=training_cfg.batch_size,
            steps_per_epoch=val_steps,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=opts.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=True,
    ) if val_dataset is not None else None

    trainer = DynamicsTrainer(
        dynamics_cfg=dynamics_cfg,
        tokenizer_cfg=tokenizer_cfg,
        training_cfg=training_cfg,
        tokenizer_ckpt=opts.tokenizer_ckpt,
    )

    start_epoch = 1
    if opts.resume_from is not None:
        print(f"[INFO] Resuming from {opts.resume_from}")
        start_epoch = trainer.load_checkpoint(opts.resume_from)
        print(f"[INFO] Resuming from epoch {start_epoch}")

    total_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"Dreamer V4 Dynamics Training")
    print(f"{'='*60}")
    print(f"  Task:           {opts.task}")
    print(f"  Trainable params: {total_params:,}")
    print(f"  Latent shape:   ({dynamics_cfg.num_latent_tokens}, {dynamics_cfg.latent_input_dim})")
    print(f"  Embed dim:      {dynamics_cfg.embed_dim}")
    print(f"  Depth:          {dynamics_cfg.depth}")
    print(f"  Seq len (T):    {dynamics_cfg.seq_len}")
    print(f"  K_max:          {dynamics_cfg.K_max}")
    print(f"  Batch size:     {training_cfg.batch_size}")
    print(f"  LR:             {training_cfg.lr}")
    print(f"  Epochs:         {training_cfg.epochs}")
    print(f"  Device:         {trainer.device}")
    print(f"{'='*60}\n")

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=str(ckpt_dir),
        start_epoch=start_epoch,
    )

    final_path = opts.save_final or str(ckpt_dir / "final.pt")
    trainer.save_checkpoint(final_path, epoch=training_cfg.epochs)
    print(f"\n[DONE] Final checkpoint saved to {final_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
