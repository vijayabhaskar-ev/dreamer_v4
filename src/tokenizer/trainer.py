"""Standalone training utilities for the Dreamer V4 tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from .config import TokenizerConfig
from .losses import MaskedAutoencoderLoss
from .model import MaskedAutoencoderTokenizer


@dataclass
class TokenizerTrainingConfig:
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    amp: bool = False
    checkpoint_interval: int = 1
    device: str = "cuda"


class TokenizerTrainer:
    """Utility class for training the masked autoencoder tokenizer."""

    def __init__(
        self,
        tokenizer_cfg: TokenizerConfig,
        training_cfg: TokenizerTrainingConfig,
        loss_module: Optional[MaskedAutoencoderLoss] = None,
    ) -> None:
        self.tokenizer_cfg = tokenizer_cfg
        self.training_cfg = training_cfg

        self.device = torch.device(training_cfg.device if torch.cuda.is_available() else "cpu")
        self.model = MaskedAutoencoderTokenizer(tokenizer_cfg).to(self.device)
        self.loss_module = loss_module or MaskedAutoencoderLoss(lpips_module=None)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_cfg.lr,
            weight_decay=training_cfg.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=training_cfg.amp)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        for epoch in range(1, self.training_cfg.epochs + 1):
            train_metrics = self._run_epoch(train_loader, epoch, training=True)
            val_metrics = None
            if val_loader is not None:
                val_metrics = self._run_epoch(val_loader, epoch, training=False)

            print(
                f"Epoch {epoch}: train_loss={train_metrics['loss/tokenizer_total']:.4f}"
                + (f" val_loss={val_metrics['loss/tokenizer_total']:.4f}" if val_metrics else "")
            )

            if (
                checkpoint_dir is not None
                and self.training_cfg.checkpoint_interval > 0
                and epoch % self.training_cfg.checkpoint_interval == 0
            ):
                ckpt_path = f"{checkpoint_dir}/tokenizer_epoch_{epoch:03d}.pt"
                self.save_checkpoint(ckpt_path)

    def save_checkpoint(self, path: str) -> None:
        state = {
            "tokenizer_cfg": self.tokenizer_cfg,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, path)
        print(f"Saved tokenizer checkpoint to {path}")

    def load_checkpoint(self, path: str, strict: bool = True) -> None:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"], strict=strict)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])

    def _run_epoch(
        self, loader: DataLoader, epoch: int, training: bool
    ) -> Dict[str, float]:
        self.model.train(training)
        total_loss = 0.0
        total_mse = 0.0
        total_lpips = 0.0
        total_steps = 0

        for batch in loader:
            frames = batch.to(self.device)
            if frames.dim() == 4:
                frames = frames.unsqueeze(1)

            with torch.cuda.amp.autocast(enabled=self.training_cfg.amp):
                outputs = self.model(frames)
                loss_outputs = self.loss_module(
                    recon=outputs.reconstructed,
                    target=frames,
                    mask=outputs.mask,
                    normalize_by=self.tokenizer_cfg.norm_loss_by,
                )
                loss = loss_outputs.total_loss

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_cfg.grad_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss += loss_outputs.total_loss.item()
            total_mse += loss_outputs.mse_loss.item()
            if loss_outputs.lpips_loss is not None:
                total_lpips += loss_outputs.lpips_loss.item()
            total_steps += 1

        metrics = {
            "loss/tokenizer_total": total_loss / max(total_steps, 1),
            "loss/tokenizer_mse": total_mse / max(total_steps, 1),
        }
        if total_lpips > 0:
            metrics["loss/tokenizer_lpips"] = total_lpips / max(total_steps, 1)
        return metrics


def create_dataloader(dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    if isinstance(dataset, IterableDataset):
        return DataLoader(dataset, batch_size=None)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
