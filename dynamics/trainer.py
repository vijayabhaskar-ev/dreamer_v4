
from __future__ import annotations

from dataclasses import dataclass
import pickle
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from .config import DynamicsConfig
from .losses import FlowMatchingLoss
from .dynamic_model import DynamicsModel
from .flow_matching import add_noise, sample_tau_and_d
from tokenizer.config import TokenizerConfig
from tokenizer.tokenizer import MaskedAutoencoderTokenizer
from tokenizer.metrics import MetricsBuffer, ModelStatistics, GPUMemoryTracker, ThroughputTracker
import wandb


@dataclass
class DynamicsTrainingConfig:
    """
    TODO Review these defaults against the Dreamer V4 paper.
         The tokenizer uses lr=1e-4, weight_decay=0.05.
         Dynamics may need different values.
    """
    epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    amp: bool = False
    checkpoint_interval: int = 1
    device: str = "cuda"
    # Logging
    log_interval: int = 10
    log_smooth_window: int = 10
    log_model_stats: bool = True
    log_model_stats_interval: int = 50
    log_memory: bool = True


class DynamicsTrainer:
    """
    Training loop for the dynamics model using flow matching.
    """

    def __init__(
        self,
        dynamics_cfg: DynamicsConfig,
        tokenizer_cfg: TokenizerConfig,
        training_cfg: DynamicsTrainingConfig,
        tokenizer_ckpt: str,
    ) -> None:
        self.dynamics_cfg = dynamics_cfg
        self.tokenizer_cfg = tokenizer_cfg
        self.training_cfg = training_cfg

        self.device = torch.device(training_cfg.device if torch.cuda.is_available() else "cpu")

        self.tokenizer = MaskedAutoencoderTokenizer(tokenizer_cfg).to(self.device)
        self._load_tokenizer_checkpoint(tokenizer_ckpt)
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

        self.model = DynamicsModel(dynamics_cfg, self.tokenizer).to(self.device)
        self.loss = FlowMatchingLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_cfg.lr,
            weight_decay=training_cfg.weight_decay,
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=training_cfg.amp)

        self.global_step = 0
        self.metrics_buffer = MetricsBuffer(window=training_cfg.log_smooth_window)
        self.throughput_tracker = ThroughputTracker()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: Optional[str] = None,
        start_epoch: int = 1,
    ) -> None:

        for epoch in range(start_epoch, self.training_cfg.epochs + 1):
            train_metrics = self._run_epoch(train_loader, epoch, training=True)
            val_metrics = None
            if val_loader is not None:
                val_metrics = self._run_epoch(val_loader, epoch, training=False)

            print(
                f"Epoch {epoch}: train_loss={train_metrics['loss/dynamics_total']:.4f}"
                + (f" val_loss={val_metrics['loss/dynamics_total']:.4f}" if val_metrics else "")
            )

            if (
                checkpoint_dir is not None
                and self.training_cfg.checkpoint_interval > 0
                and epoch % self.training_cfg.checkpoint_interval == 0
            ):
                ckpt_path = f"{checkpoint_dir}/dynamics_epoch_{epoch:03d}.pt"
                self.save_checkpoint(ckpt_path, epoch)


    def save_checkpoint(self, path: str, epoch: int) -> None:
        state = {
                "epoch": epoch,
                "dynamic_cfg": self.dynamics_cfg,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
        torch.save(state, path)
        print(f"Saved dynamics checkpoint to {path}")

    def load_checkpoint(self, path: str, strict: bool = True) -> int:
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
        except pickle.UnpicklingError:
            state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"], strict=strict)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        
        return state.get("epoch", 0) + 1

    def _run_epoch(
        self, loader: DataLoader, epoch: int, training: bool
    ) -> Dict[str, float]:

        self.model.train(training)
        total_loss = 0.0
        total_mse = 0.0
        total_steps = 0

        log_interval = self.training_cfg.log_interval
        model_stats_interval = self.training_cfg.log_model_stats_interval

        for batch in loader:
            frames, actions = batch
            frames = frames.to(self.device)
            actions = actions.to(self.device)

            if frames.dim() == 4:
                frames = frames.unsqueeze(1)

            B, T = frames.shape[0:2]

            z_clean = self.model.encode_frames(frames)  # (B, T, S_z, latent_dim)

            tau, d = sample_tau_and_d(B, T, K_max=self.dynamics_cfg.K_max, device=self.device)

            z_noised, _ = add_noise(z_clean, tau)

            with torch.cuda.amp.autocast(enabled=self.training_cfg.amp):
                z_predicted = self.model(z_noised, actions, tau, d)
                loss_outputs = self.loss(z_predicted, z_clean)
                loss = loss_outputs.total_loss

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_cfg.grad_clip,
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.global_step += 1

                # Update smoothing buffer every step
                self.metrics_buffer.update({
                    "train/loss": loss.item(),
                    "train/mse": loss_outputs.mse_loss.item(),
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "train/tau_mean": tau.mean().item(),
                    "train/d_mean": d.mean().item(),
                })

                # Log smoothed metrics periodically
                if self.global_step % log_interval == 0:
                    metrics = self.metrics_buffer.get_averages()
                    metrics.update({
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/scale": self.scaler.get_scale(),
                        "epoch": epoch,
                        "global_step": self.global_step,
                    })

                    metrics.update(self.throughput_tracker.step(B))

                    if self.training_cfg.log_memory:
                        metrics.update(GPUMemoryTracker.get_memory_stats(self.device))

                    if self.training_cfg.log_model_stats and self.global_step % model_stats_interval == 0:
                        metrics.update(ModelStatistics.compute_weight_stats(self.model))
                        metrics.update(ModelStatistics.compute_gradient_stats(self.model))

                    wandb.log(metrics, step=self.global_step)

            total_loss += loss_outputs.total_loss.item()
            total_mse += loss_outputs.mse_loss.item()
            total_steps += 1

        return {
            "loss/dynamics_total": total_loss / max(total_steps, 1),
            "loss/dynamics_mse": total_mse / max(total_steps, 1),
        }




    def _load_tokenizer_checkpoint(self, ckpt_path) -> None:
        if ckpt_path is None:
            raise ValueError(
                "DreamerAgent expects a pretrained tokenizer checkpoint. "
                "Set `tokenizer_ckpt` in the config to load weights before training the agent."
            )

        state = torch.load(ckpt_path, map_location=self.device)
        if "model" in state:
            state = state["model"]
        missing, unexpected = self.tokenizer.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Tokenizer checkpoint missing parameters: {missing}")
        if unexpected:
            raise RuntimeError(f"Tokenizer checkpoint has unexpected parameters: {unexpected}")

