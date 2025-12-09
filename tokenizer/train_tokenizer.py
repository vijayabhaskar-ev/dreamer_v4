"""CLI entry point for training the Dreamer V4 tokenizer on synthetic data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from .config import TokenizerConfig
from .trainer import TokenizerTrainer, TokenizerTrainingConfig, MaskedAutoencoderLoss
from .dataset import DatasetFactory
import wandb

try:
    from mock_data import MovingSquareDataset
except ImportError as exc:  # pragma: no cover - fallback when module missing
    raise ImportError(
        "mock_data.MovingSquareDataset not found. Ensure mock_data.py is on PYTHONPATH "
        "or provide your own dataset implementation."
    ) from exc


class MovingSquareIterableDataset(IterableDataset):
    """Wraps MovingSquareDataset to yield full batches for each iterator step."""
    # Deprecated: Use DatasetFactory instead
    def __init__(self, dataset, batch_size: int, steps_per_epoch: int):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            yield self.dataset.sample(self.batch_size)


def build_parser() -> argparse.ArgumentParser: #TODO Need to check the defeault values of the arguments
    parser = argparse.ArgumentParser(description="Train the Dreamer V4 tokenizer (MAE)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/tokenizer")
    parser.add_argument("--save-final", type=str, default="checkpoints/tokenizer/final.pt")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--lpips", type=str, choices=["none", "vgg", "alex", "squeeze"], default="vgg")
    parser.add_argument("--dataset", type=str, default="dm_control", help="Dataset name (dm_control, moving_square)")
    parser.add_argument("--task", type=str, default="cheetah_run", help="DMControl task name")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation only")
    # WandB arguments
    parser.add_argument("--wandb-project", type=str, default="dreamer-v4-tokenizer", help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity (user/team)")
    parser.add_argument("--wandb-name", type=str, default=None, help="WandB run name")
    parser.add_argument("--wandb-offline", action="store_true", help="Run WandB in offline mode")
    return parser


def main(args: Optional[list[str]] = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    wandb.init(
        project=parsed.wandb_project,
        entity=parsed.wandb_entity,
        name=parsed.wandb_name,
        mode="offline" if parsed.wandb_offline else "online",
        config=vars(parsed),
    )

    Path(parsed.checkpoint_dir).mkdir(parents=True, exist_ok=True) #TODO makie sure reading checkpoint_dir is wroking properly
    if parsed.save_final:
        Path(parsed.save_final).parent.mkdir(parents=True, exist_ok=True) #TODO Need to check the implementation of save_final

    tokenizer_cfg = TokenizerConfig(
        image_size=(parsed.image_size, parsed.image_size),
        patch_size=(parsed.patch_size, parsed.patch_size),
        embed_dim=parsed.embed_dim,
        dataset_name=parsed.dataset,
        task_name=parsed.task,
    )

    training_cfg = TokenizerTrainingConfig(
        epochs=parsed.epochs,
        batch_size=parsed.batch_size,
        lr=parsed.lr,
        weight_decay=parsed.weight_decay,
        grad_clip=parsed.grad_clip,
        amp=parsed.amp,
        checkpoint_interval=max(1, parsed.epochs // 5),
        device=parsed.device,
    )

    lpips_module = None
    if parsed.lpips != "none":
        try:
            import lpips

            lpips_module = lpips.LPIPS(net=parsed.lpips).to(parsed.device) 
        except ImportError:
            raise ImportError(
                "LPIPS requested but library not installed. Install `lpips` package or rerun with --lpips none."
            )

    # When using multiple workers with an IterableDataset, each worker produces 'steps_per_epoch' batches.
    # We divide by num_workers so the TOTAL batches per epoch roughly equals the requested steps_per_epoch.
    steps_per_worker = parsed.steps_per_epoch
    if parsed.num_workers > 0:
        steps_per_worker = max(1, parsed.steps_per_epoch // parsed.num_workers)

    train_dataset = DatasetFactory.get_dataset(tokenizer_cfg, parsed.batch_size, steps_per_worker)
    
    # Create DataLoader
    # Note: For IterableDataset, batch_size must be None in DataLoader if the dataset yields batches
    # But our factory returns a dataset that yields batches (B, T, C, H, W)
    # So we set batch_size=None
    #TODO Maybe need to modify the dataset factory to yield batches of size batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=parsed.num_workers, #TODO Need to check the implementation of num_workers. Right now num_workers  > 0 is getting stuck
        pin_memory=True,
    )

    trainer = TokenizerTrainer(
        tokenizer_cfg=tokenizer_cfg,
        training_cfg=training_cfg,
        loss_module=MaskedAutoencoderLoss(lpips_module=lpips_module),
    )

    if parsed.evaluate:
        if not parsed.resume_from:
            raise ValueError("Must provide --resume-from when --evaluate is set.")
        trainer.load_checkpoint(parsed.resume_from, strict=True) #TODO Refactor to properly evalute with validationdataset. Read more on this in the docs
        trainer.evaluate(train_loader, output_dir="evaluation_results")
        return
    
    if parsed.resume_from:
        trainer.load_checkpoint(parsed.resume_from, strict=False)

    trainer.fit(
        train_loader=train_loader,
        val_loader=None,
        checkpoint_dir=parsed.checkpoint_dir,
    )

    if parsed.save_final:
        trainer.save_checkpoint(parsed.save_final)


if __name__ == "__main__":
    main()
