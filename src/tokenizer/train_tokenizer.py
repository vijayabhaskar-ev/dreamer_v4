"""CLI entry point for training the Dreamer V4 tokenizer on synthetic data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import IterableDataset

from .config import TokenizerConfig
from .trainer import (
    TokenizerTrainer,
    TokenizerTrainingConfig,
    create_dataloader,
    MaskedAutoencoderLoss,
)

try:
    from mock_data import MovingSquareDataset
except ImportError as exc:  # pragma: no cover - fallback when module missing
    raise ImportError(
        "mock_data.MovingSquareDataset not found. Ensure mock_data.py is on PYTHONPATH "
        "or provide your own dataset implementation."
    ) from exc


class MovingSquareIterableDataset(IterableDataset):
    """Wraps MovingSquareDataset to yield full batches for each iterator step."""

    def __init__(self, dataset: MovingSquareDataset, batch_size: int, steps_per_epoch: int):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            yield self.dataset.sample(self.batch_size)


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--image-size", type=int, nargs=2, default=(84, 84))
    parser.add_argument("--patch-size", type=int, nargs=2, default=(6, 6))
    parser.add_argument("--seq-length", type=int, default=4)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--square-size", type=int, default=8)
    parser.add_argument("--mask-prob-min", type=float, default=0.0)
    parser.add_argument("--mask-prob-max", type=float, default=0.9)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--latent-tokens", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--lpips", type=str, choices=["none", "vgg", "alex", "squeeze"], default="none")
    return parser


def main(args: Optional[list[str]] = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    Path(parsed.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if parsed.save_final:
        Path(parsed.save_final).parent.mkdir(parents=True, exist_ok=True)

    tokenizer_cfg = TokenizerConfig(
        image_size=tuple(parsed.image_size),
        patch_size=tuple(parsed.patch_size),
        in_channels=parsed.channels,
        embed_dim=parsed.embed_dim,
        depth=parsed.depth,
        num_heads=parsed.num_heads,
        mlp_ratio=parsed.mlp_ratio,
        dropout=parsed.dropout,
        drop_path=parsed.drop_path,
        mask_prob_min=parsed.mask_prob_min,
        mask_prob_max=parsed.mask_prob_max,
        num_latent_tokens=parsed.latent_tokens,
        lpips_net=(parsed.lpips if parsed.lpips != "none" else "vgg"),
        use_grad_checkpoint=parsed.grad_checkpoint,
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

    moving_square = MovingSquareDataset(
        H=parsed.image_size[0],
        W=parsed.image_size[1],
        T=parsed.seq_length,
        C=parsed.channels,
        square_size=parsed.square_size,
    )
    iterable_dataset = MovingSquareIterableDataset(
        moving_square,
        batch_size=parsed.batch_size,
        steps_per_epoch=parsed.steps_per_epoch,
    )
    train_loader = create_dataloader(iterable_dataset, batch_size=parsed.batch_size, shuffle=True)

    trainer = TokenizerTrainer(
        tokenizer_cfg=tokenizer_cfg,
        training_cfg=training_cfg,
        loss_module=MaskedAutoencoderLoss(lpips_module=lpips_module),
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=None,
        checkpoint_dir=parsed.checkpoint_dir,
    )

    if parsed.save_final:
        trainer.save_checkpoint(parsed.save_final)


if __name__ == "__main__":
    main()
