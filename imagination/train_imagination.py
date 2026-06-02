"""Phase 3 entry point: imagination-based RL training.

Loads a Phase 2 checkpoint (dynamics + reward_head + continue_head + policy_head)
and runs imagination training: policy_head and a new value_head are optimized
via PMPO + TD(λ) on imagined rollouts from the frozen world model.

Usage:
    python -m imagination.train_imagination \
        --tokenizer-ckpt checkpoints/tokenizer/final.pt \
        --phase2-ckpt checkpoints/dynamics/final.pt \
        --horizon 15 \
        --epochs 100 \
        --batch-size 16 \
        --num-context-frames 4

"""

from __future__ import annotations

# MUST be first: sets env vars that PyTorch reads at import time.
import _env_setup  # noqa: F401

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

import wandb

from dynamics.config import DynamicsConfig
from tokenizer.config import TokenizerConfig
from tokenizer.dataset import DatasetFactory
from device_utils import get_device, is_master
from wandb_utils import init_wandb, add_wandb_args

from .config import ImaginationConfig
from .trainer import ImaginationTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dreamer V4 Phase 3: Imagination Training")

    # Phase 3 specific
    parser.add_argument("--phase2-ckpt", type=str, required=True,
                        help="Path to Phase 2 checkpoint (dynamics + heads)")
    parser.add_argument("--tokenizer-ckpt", type=str, required=True)
    parser.add_argument("--num-tasks", type=int, default=1)

    # RL hyperparameters
    parser.add_argument("--horizon", type=int, default=15,
                        help="Imagination horizon H")
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--lambda", type=float, default=0.95, dest="lambda_")
    parser.add_argument("--pmpo-alpha", type=float, default=0.5)
    parser.add_argument("--pmpo-beta", type=float, default=0.3)
    parser.add_argument("--K-imagination", type=int, default=4,
                        help="Denoising steps per imagined frame")
    parser.add_argument("--tau-ctx", type=float, default=0.1)
    parser.add_argument("--K-max", type=int, default=64)
    parser.add_argument("--num-context-frames", type=int, default=4)
    parser.add_argument("--context-window", type=int, default=16)

    # Optimizer / training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    # Dynamics config (needed to reconstruct the model; should match Phase 2)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--seq-len-short", type=int, default=8)
    parser.add_argument("--seq-len-long", type=int, default=32)
    parser.add_argument("--num-register-tokens", type=int, default=4)
    parser.add_argument("--temporal-interval", type=int, default=4)
    parser.add_argument("--action-dim", type=int, default=6)

    # Dataset
    parser.add_argument("--dataset", type=str, default="dm_control")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--task", type=str, default="cheetah_run")
    parser.add_argument("--num-workers", type=int, default=0)

    # Checkpointing / logging
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/imagination")
    parser.add_argument("--save-final", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=5)

    # WandB
    add_wandb_args(parser, default_project="dreamer-v4-imagination")

    return parser


def _load_tokenizer_config(ckpt_path: str) -> Optional[TokenizerConfig]:
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "tokenizer_cfg" in state:
            return state["tokenizer_cfg"]
    except Exception:
        pass
    return None


def _train_fn(index=0, args=None):
    opts = args if args is not None else build_parser().parse_args()

    ckpt_dir = Path(opts.checkpoint_dir)
    if is_master():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Configs ─────────────────────────────────────────────────────
    tokenizer_cfg = _load_tokenizer_config(opts.tokenizer_ckpt)
    if tokenizer_cfg is None:
        tokenizer_cfg = TokenizerConfig(
            dataset_name=opts.dataset,
            task_name=opts.task,
            seq_len=opts.num_context_frames,
        )

    dynamics_cfg = DynamicsConfig.from_tokenizer(
        tokenizer_cfg,
        embed_dim=opts.embed_dim,
        depth=opts.depth,
        num_heads=opts.num_heads,
        num_kv_heads=opts.num_kv_heads,
        seq_len_short=opts.seq_len_short,
        seq_len_long=opts.seq_len_long,
        seq_len=opts.num_context_frames,
        context_length=opts.context_window,
        num_register_tokens=opts.num_register_tokens,
        temporal_interval=opts.temporal_interval,
        action_dim=opts.action_dim,
        mtp_length=0,  # No MTP in Phase 3
    )

    imagination_cfg = ImaginationConfig(
        gamma=opts.gamma,
        lambda_=opts.lambda_,
        imagination_horizon=opts.horizon,
        pmpo_alpha=opts.pmpo_alpha,
        pmpo_beta=opts.pmpo_beta,
        K_imagination=opts.K_imagination,
        tau_ctx=opts.tau_ctx,
        K_max=opts.K_max,
        num_context_frames=opts.num_context_frames,
        context_window=opts.context_window,
        lr=opts.lr,
        min_lr=opts.min_lr,
        warmup_steps=opts.warmup_steps,
        weight_decay=opts.weight_decay,
        grad_clip=opts.grad_clip,
        epochs=opts.epochs,
        batch_size=opts.batch_size,
        steps_per_epoch=opts.steps_per_epoch,
        amp=opts.amp,
        device=opts.device,
        log_interval=opts.log_interval,
        checkpoint_interval=opts.checkpoint_interval,
    )

    # ── WandB ───────────────────────────────────────────────────────
    init_wandb(
        opts,
        run_name_prefix=f"phase3_{opts.task}",
        config={
            "phase": 3,
            "imagination": vars(imagination_cfg),
            "dynamics": vars(dynamics_cfg),
            "task": opts.task,
            "phase2_ckpt": opts.phase2_ckpt,
        },
    )

    # ── Dataset ─────────────────────────────────────────────────────
    dataset_cfg = replace(
        tokenizer_cfg,
        dataset_name=opts.dataset,
        task_name=opts.task,
        seq_len=opts.num_context_frames,
    )

    steps_per_worker = opts.steps_per_epoch
    if opts.num_workers > 0:
        steps_per_worker = max(1, opts.steps_per_epoch // opts.num_workers)

    train_dataset = DatasetFactory.get_dataset(
        dataset_cfg,
        batch_size=imagination_cfg.batch_size,
        steps_per_epoch=steps_per_worker,
        dataset_path=opts.dataset_path,
        expected_action_dim=opts.action_dim,
    )

    device = get_device(opts.device)
    use_pin_memory = device.type == "cuda"
    train_loader_raw = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=opts.num_workers,
        pin_memory=use_pin_memory,
        multiprocessing_context="spawn" if opts.num_workers > 0 else None,
    )
    train_loader = train_loader_raw

    # ── Trainer ─────────────────────────────────────────────────────
    trainer = ImaginationTrainer(
        dynamics_cfg=dynamics_cfg,
        tokenizer_cfg=tokenizer_cfg,
        imagination_cfg=imagination_cfg,
        tokenizer_ckpt=opts.tokenizer_ckpt,
        phase2_ckpt=opts.phase2_ckpt,
        num_tasks=opts.num_tasks,
    )

    start_epoch = 1
    if opts.resume_from is not None:
        start_epoch = trainer.load_checkpoint(opts.resume_from)

    if is_master():
        policy_params = sum(p.numel() for p in trainer.policy_head.parameters() if p.requires_grad)
        value_params = sum(p.numel() for p in trainer.value_head.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"Dreamer V4 Phase 3: Imagination Training")
        print(f"{'='*60}")
        print(f"  Task:              {opts.task}")
        print(f"  Policy params:     {policy_params:,}")
        print(f"  Value params:      {value_params:,}")
        print(f"  Horizon:           {imagination_cfg.imagination_horizon}")
        print(f"  gamma / lambda:    {imagination_cfg.gamma} / {imagination_cfg.lambda_}")
        print(f"  PMPO α / β:        {imagination_cfg.pmpo_alpha} / {imagination_cfg.pmpo_beta}")
        print(f"  K (denoise steps): {imagination_cfg.K_imagination}")
        print(f"  Batch size:        {imagination_cfg.batch_size}")
        print(f"  LR:                {imagination_cfg.lr}")
        print(f"  Epochs:            {imagination_cfg.epochs}")
        print(f"  Device:            {trainer.device}")
        print(f"  Phase 2 ckpt:      {opts.phase2_ckpt}")
        print(f"{'='*60}\n")

    # ── Train ───────────────────────────────────────────────────────
    trainer.fit(
        train_loader=train_loader,
        checkpoint_dir=str(ckpt_dir),
        start_epoch=start_epoch,
    )

    final_path = opts.save_final or str(ckpt_dir / "final.pt")
    trainer.save_checkpoint(final_path, epoch=imagination_cfg.epochs)
    if is_master():
        print(f"\n[DONE] Phase 3 final checkpoint saved to {final_path}")

    wandb.finish()


def main(args: Optional[list[str]] = None) -> None:
    opts = build_parser().parse_args(args)

    _train_fn(index=0, args=opts)


if __name__ == "__main__":
    main()
