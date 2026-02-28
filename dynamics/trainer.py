
from __future__ import annotations

from dataclasses import dataclass
import pickle
from typing import Dict, Optional
import math

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .config import DynamicsConfig
from .dynamic_model import DynamicsModel
from .flow_matching import add_noise, sample_tau_and_d
from tokenizer.config import TokenizerConfig
from tokenizer.tokenizer import MaskedAutoencoderTokenizer
from tokenizer.metrics import MetricsBuffer, ModelStatistics, GPUMemoryTracker, ThroughputTracker
from device_utils import get_device, get_device_type, make_grad_scaler, save_checkpoint as save_ckpt, is_master
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
    lr: float = 1e-4
    warmup_steps: int = 500       
    min_lr: float = 1e-6  
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    amp: bool = False
    checkpoint_interval: int = 1
    device: str = "cuda"
    # Curriculum: bootstrap ramp-up
    curriculum_warmup_steps: int = 2000   
    curriculum_ramp_steps: int = 4000    
    # Logging
    steps_per_epoch: int = 100
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

        self.device = get_device(training_cfg.device)

        self.rms_flow = RMSNormalizer(decay=0.99)
        self.rms_bootstrap = RMSNormalizer(decay=0.99)

        self.tokenizer = MaskedAutoencoderTokenizer(tokenizer_cfg).to(self.device)
        self._load_tokenizer_checkpoint(tokenizer_ckpt)
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

        self.model = DynamicsModel(dynamics_cfg, self.tokenizer).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_cfg.lr,
            weight_decay=training_cfg.weight_decay,
        )

        self.scaler = make_grad_scaler(self.device, enabled=training_cfg.amp)
        self.global_step = 0
        self._last_lr_bucket = -1
        self.metrics_buffer = MetricsBuffer(window=training_cfg.log_smooth_window)
        self.throughput_tracker = ThroughputTracker()



    def _build_scheduler(self, total_steps: int) -> None:
        """Store schedule params. LR is set via _set_lr_from_schedule() each step.

        Using LambdaLR with XLA causes recompilation every step because the LR
        value is embedded as a compile-time constant in the XLA graph. Instead,
        we compute LR in Python and set it before optimizer.step() so the value
        is baked into each step's graph. Since we change LR infrequently
        (only at coarse intervals), this avoids most recompilations.
        """
        self._schedule_total_steps = total_steps
        self._schedule_warmup = self.training_cfg.warmup_steps
        self._schedule_min_lr = self.training_cfg.min_lr
        self._schedule_base_lr = self.training_cfg.lr
        self._last_lr_bucket = -1  # track when LR actually changes

    def _set_lr_from_schedule(self) -> None:
        """Set optimizer LR from schedule, quantized to reduce XLA recompilations.

        Each unique LR value creates a new XLA graph compilation (~50-100MB
        cache entry). With 4 code path variants (full/single × boot/no-boot),
        each LR bucket costs ~4 compilations. We use only 10 buckets → max
        ~40 compiled graphs total, keeping XLA cache under ~2-4GB.
        """
        step = self.global_step
        warmup = self._schedule_warmup
        total = self._schedule_total_steps
        base_lr = self._schedule_base_lr
        min_lr = self._schedule_min_lr

        if step < warmup:
            frac = step / max(warmup, 1)
        else:
            progress = (step - warmup) / max(total - warmup, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            frac = max(min_lr / base_lr, cosine)

        # Quantize to 10 buckets → max ~40 XLA compilations (10 × 4 code paths)
        bucket = int(frac * 10)
        if bucket == self._last_lr_bucket:
            return  # Same bucket, no LR change, no recompilation
        self._last_lr_bucket = bucket

        lr = base_lr * (bucket / 10.0)
        for group in self.optimizer.param_groups:
            group['lr'] = lr



    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: Optional[str] = None,
        start_epoch: int = 1,
    ) -> None:

        steps_per_epoch = self.training_cfg.steps_per_epoch
        total_steps = steps_per_epoch * self.training_cfg.epochs
        self._build_scheduler(total_steps)


        for epoch in range(start_epoch, self.training_cfg.epochs + 1):
            train_metrics = self._run_epoch(train_loader, epoch, training=True)
            val_metrics = None
            if val_loader is not None:
                val_metrics = self._run_epoch(val_loader, epoch, training=False)

            if is_master():
                epoch_log = {
                    "epoch": epoch,
                    "epoch/train_loss": train_metrics["loss/dynamics_total"],
                    "epoch/train_flow": train_metrics["loss/dynamics_flow"],
                    "epoch/train_bootstrap": train_metrics["loss/dynamics_bootstrap"],
                    "global_step": self.global_step,
                }
                if val_metrics is not None:
                    epoch_log["epoch/val_loss"] = val_metrics["loss/dynamics_total"]
                    epoch_log["epoch/val_flow"] = val_metrics["loss/dynamics_flow"]
                    epoch_log["epoch/val_bootstrap"] = val_metrics["loss/dynamics_bootstrap"]
                wandb.log(epoch_log, step=self.global_step)

                print(
                    f"Epoch {epoch}: train_loss={train_metrics['loss/dynamics_total']:.4f} "
                    f"flow={train_metrics['loss/dynamics_flow']:.4f} "
                    f"bootstrap={train_metrics['loss/dynamics_bootstrap']:.4f}"
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
                "global_step": self.global_step,                          
                "dynamic_cfg": self.dynamics_cfg,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler_bucket": self._last_lr_bucket,
                "rms_normalizers": {
                    "flow": self.rms_flow.state_dict(),
                    "bootstrap": self.rms_bootstrap.state_dict(),
                },
            }
        save_ckpt(state, path, self.device)
        if is_master():
            print(f"Saved dynamics checkpoint to {path}")

    def load_checkpoint(self, path: str, strict: bool = True) -> int:
        try:
            state = torch.load(path, map_location='cpu', weights_only=True)
        except pickle.UnpicklingError:
            state = torch.load(path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(state["model"], strict=strict)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "rms_normalizers" in state:
            rms_state = state["rms_normalizers"]
            if "flow" in rms_state:
                self.rms_flow.load_state_dict(rms_state["flow"])
            if "bootstrap" in rms_state:
                self.rms_bootstrap.load_state_dict(rms_state["bootstrap"])
        if "global_step" in state:
            self.global_step = state["global_step"]
        if "scheduler_bucket" in state:
            self._last_lr_bucket = state["scheduler_bucket"]

        return state.get("epoch", 0) + 1

    def _run_epoch(
        self, loader: DataLoader, epoch: int, training: bool
    ) -> Dict[str, float]:

        self.model.train(training)
        total_loss = torch.tensor(0.0, device=self.device)
        total_flow = torch.tensor(0.0, device=self.device)
        total_bootstrap = torch.tensor(0.0, device=self.device)
        total_steps = 0

        log_interval = self.training_cfg.log_interval
        model_stats_interval = self.training_cfg.log_model_stats_interval

        # On-device accumulators for smoothed logging (reset every log_interval)
        _log_loss = torch.tensor(0.0, device=self.device)
        _log_flow = torch.tensor(0.0, device=self.device)
        _log_bootstrap = torch.tensor(0.0, device=self.device)
        _log_grad_norm = torch.tensor(0.0, device=self.device)
        _log_tau = torch.tensor(0.0, device=self.device)
        _log_d = torch.tensor(0.0, device=self.device)
        _log_count = 0

        # Curriculum boundaries (Python ints — no XLA sync)
        warmup_end = self.training_cfg.curriculum_warmup_steps
        ramp_end = self.training_cfg.curriculum_ramp_steps

        for batch in loader:
            frames, actions = batch
            frames = frames.to(self.device)
            actions = actions.to(self.device)

            if frames.dim() == 4:
                frames = frames.unsqueeze(1)

            B, T = frames.shape[0:2]
            assert B >= 4, "Batch size too small for mixed training"

            # Python-level gate: skip expensive bootstrap forward passes
            # during flow-only warmup (no XLA sync, no dynamic shapes)
            need_bootstrap = self.global_step >= warmup_end

            if training and T > 1:
                n_single = max(1, int(B * 0.3))
                n_full = B - n_single

                frames_full = frames[n_single :]
                actions_full = actions[n_single:]

                z_clean_f = self.model.encode_frames(frames_full)  # (n_full, T, S_z, latent_dim)
                tau_f, d_f = sample_tau_and_d(n_full, T, K_max=self.dynamics_cfg.K_max, device=self.device)

                # Warm-up: flow-only → gradual bootstrap ramp → full mix
                if self.global_step < warmup_end:
                    d_f = torch.full_like(d_f, 1.0 / self.dynamics_cfg.K_max)
                elif self.global_step < ramp_end:
                    # Quantize ramp to 10 levels to avoid XLA recompilation
                    # (ramp_progress as a Python float is a compile-time constant)
                    ramp_frac = (self.global_step - warmup_end) / (ramp_end - warmup_end)
                    ramp_bucket = int(ramp_frac * 10) / 10.0  # 0.0, 0.1, ..., 1.0
                    mask = torch.rand_like(d_f) < ramp_bucket
                    d_min_tensor = torch.full_like(d_f, 1.0 / self.dynamics_cfg.K_max)
                    d_f = torch.where(mask, d_f, d_min_tensor)

                # (step ramp_end+): Normal sampling, no override
                z_noised_f, _ = add_noise(z_clean_f, tau_f)

                with torch.amp.autocast(device_type=get_device_type(self.device), enabled=self.training_cfg.amp):
                    loss_full, metrics_full = self._compute_loss(
                        z_clean_f, z_noised_f, actions_full, tau_f, d_f,
                        training=training, compute_bootstrap=need_bootstrap,
                    )

                 #=========30% Single-Frame Training==========

                random_t = torch.randint(0, T, (n_single,), device=self.device)
                frames_single = frames[:n_single][torch.arange(n_single, device=self.device), random_t].unsqueeze(1)

                z_clean_s = self.model.encode_frames(frames_single)  # *(n_single, T, S_z, latent_dim)
                tau_s, d_s = sample_tau_and_d(n_single, 1, K_max=self.dynamics_cfg.K_max, device=self.device)

                if self.global_step < warmup_end:
                    d_s = torch.full_like(d_s, 1.0 / self.dynamics_cfg.K_max)
                elif self.global_step < ramp_end:
                    ramp_frac = (self.global_step - warmup_end) / (ramp_end - warmup_end)
                    ramp_bucket = int(ramp_frac * 10) / 10.0
                    mask = torch.rand_like(d_s) < ramp_bucket
                    d_min_tensor = torch.full_like(d_s, 1.0 / self.dynamics_cfg.K_max)
                    d_s = torch.where(mask, d_s, d_min_tensor)

                z_noised_s, _ = add_noise(z_clean_s, tau_s)

                # Log tau & d from BOTH paths (proper weighted average)
                tau_for_log = torch.cat([tau_f.flatten(), tau_s.flatten()])
                d_for_log = torch.cat([d_f.flatten(), d_s.flatten()])

                with torch.amp.autocast(device_type=get_device_type(self.device), enabled=self.training_cfg.amp):
                    loss_single, metrics_single = self._compute_loss(
                        z_clean_s, z_noised_s, None, tau_s, d_s,
                        training=training, compute_bootstrap=need_bootstrap,
                    )

                loss = (n_full * loss_full + n_single * loss_single) / B
                metrics = {
                    "loss_flow": (n_full * metrics_full["loss_flow"] + n_single * metrics_single["loss_flow"]) / B,
                    "loss_bootstrap": (n_full * metrics_full["loss_bootstrap"] + n_single * metrics_single["loss_bootstrap"]) / B,
                }

            else:
                # Validation or T=1 already: standard path
                z_clean = self.model.encode_frames(frames)
                tau, d = sample_tau_and_d(B, T, K_max=self.dynamics_cfg.K_max, device=self.device)
                z_noised, _ = add_noise(z_clean, tau)
                tau_for_log, d_for_log = tau, d
                with torch.amp.autocast(device_type=get_device_type(self.device), enabled=self.training_cfg.amp):
                    loss, metrics = self._compute_loss(
                        z_clean, z_noised, actions, tau, d,
                        training=training, compute_bootstrap=need_bootstrap,
                    )




                
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_cfg.grad_clip,
                )

                # Compute LR manually as a tensor so XLA doesn't recompile
                # on every LR change (scheduler embeds LR as compile-time constant)
                self._set_lr_from_schedule()

                self.scaler.step(self.optimizer)  # xm.optimizer_step on TPU (includes mark_step)
                self.scaler.update()

                self.global_step += 1

                # Accumulate on-device — no .item() sync per step
                _log_loss += loss.detach()
                _log_flow += metrics["loss_flow"]
                _log_bootstrap += metrics["loss_bootstrap"]
                _log_grad_norm += grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else grad_norm
                _log_tau += tau_for_log.mean().detach()
                _log_d += d_for_log.mean().detach()
                _log_count += 1

                if self.global_step % log_interval == 0:
                    # Master logs; ALL processes reset accumulators
                    if is_master():
                        n = max(_log_count, 1)
                        self.metrics_buffer.update({
                            "train/loss": (_log_loss / n).item(),
                            "train/flow": (_log_flow / n).item(),
                            "train/bootstrap": (_log_bootstrap / n).item(),
                            "train/grad_norm": (_log_grad_norm / n).item() if isinstance(_log_grad_norm, torch.Tensor) else _log_grad_norm / n,
                            "train/tau_mean": (_log_tau / n).item(),
                            "train/d_mean": (_log_d / n).item(),
                        })

                        smoothed_metrics = self.metrics_buffer.get_averages()
                        smoothed_metrics.update({
                            "train/lr": self.optimizer.param_groups[0]["lr"],
                            "train/scale": self.scaler.get_scale(),
                            "epoch": epoch,
                            "global_step": self.global_step,
                        })

                        smoothed_metrics.update(self.throughput_tracker.step(B))

                        if self.training_cfg.log_memory:
                            smoothed_metrics.update(GPUMemoryTracker.get_memory_stats(self.device))

                        if self.training_cfg.log_model_stats and self.global_step % model_stats_interval == 0:
                            smoothed_metrics.update(ModelStatistics.compute_weight_stats(self.model))
                            smoothed_metrics.update(ModelStatistics.compute_gradient_stats(self.model))

                        wandb.log(smoothed_metrics, step=self.global_step)

                    # Reset on ALL processes — prevents unbounded accumulation
                    _log_loss.zero_()
                    _log_flow.zero_()
                    _log_bootstrap.zero_()
                    _log_grad_norm.zero_()
                    _log_tau.zero_()
                    _log_d.zero_()
                    _log_count = 0

            total_loss += loss.detach()
            total_flow += metrics["loss_flow"]
            total_bootstrap += metrics["loss_bootstrap"]
            total_steps += 1

        # Single .item() at epoch end — one sync instead of N
        return {
            "loss/dynamics_total": (total_loss / max(total_steps, 1)).item(),
            "loss/dynamics_flow": (total_flow / max(total_steps, 1)).item(),
            "loss/dynamics_bootstrap": (total_bootstrap / max(total_steps, 1)).item(),
        }


    def _compute_loss(self, z_clean, z_noised, actions, tau, d, training=True,
                      compute_bootstrap=True):
        """Compute flow matching + bootstrap loss.

        Uses masked arithmetic instead of boolean indexing so all tensors
        keep fixed (B, T, ...) shapes — critical for XLA/TPU which
        recompiles the entire graph on shape changes.

        The ``compute_bootstrap`` flag is a **Python-level gate** controlled
        by the caller based on curriculum stage (pure int comparison, no XLA
        sync).  When False the two extra model forward passes for the
        bootstrap target are skipped entirely, avoiding ~2x wasted compute
        during the flow-only warmup phase.

        Args:
            z_clean:  (B, T, S_z, D)  — target clean latents
            z_noised: (B, T, S_z, D)  — corrupted input
            actions:  (B, T-1, action_dim)
            tau:      (B, T)  — per-frame signal levels
            d:        (B, T)  — per-frame step sizes
            compute_bootstrap: whether to run the bootstrap forward passes

        Returns:
            loss, metrics_dict
        """
        d_min = 1 / self.dynamics_cfg.K_max

        z_hat = self.model(z_noised, actions, tau, d)  # (B, T, S_z, D)

        # Float masks — no boolean indexing, no dynamic shapes
        flow_mask = (d == d_min).float()   # (B, T)  1.0 for flow, 0.0 for bootstrap
        boot_mask = 1.0 - flow_mask        # (B, T)
        w = 0.9 * tau + 0.1               # (B, T)

        # ── Flow loss (masked) ──────────────────────────────────────
        per_sample_flow = ((z_hat - z_clean) ** 2).mean(dim=(-2, -1))  # (B, T)
        n_flow = flow_mask.sum().clamp(min=1.0)
        loss_flow = (flow_mask * w * per_sample_flow).sum() / n_flow
        loss_flow_normed = self.rms_flow.normalize(loss_flow, update=training)

        # ── Bootstrap loss (masked) ─────────────────────────────────
        if compute_bootstrap:
            tau_4d = tau[:, :, None, None]   # (B, T, 1, 1)
            d_4d = d[:, :, None, None]
            v_pred = (z_hat - z_noised) / (1 - tau_4d).clamp(min=1e-4)

            with torch.no_grad():
                z_hat_half1 = self.model(z_noised, actions, tau, d / 2)
                v1 = (z_hat_half1 - z_noised) / (1 - tau_4d).clamp(min=1e-4)

                z_mid = z_noised + v1 * (d_4d / 2)

                z_hat_half2 = self.model(z_mid, actions, tau + d / 2, d / 2)
                v2 = (z_hat_half2 - z_mid) / (1 - (tau_4d + d_4d / 2)).clamp(min=1e-4)

                v_target = (v1 + v2) / 2

            weight_tau_sq = (1 - tau[:, :, None, None]) ** 2  # (B, T, 1, 1)
            per_sample_boot = (
                weight_tau_sq * (v_pred - v_target) ** 2
            ).mean(dim=(-2, -1))  # (B, T)

            n_boot = boot_mask.sum().clamp(min=1.0)
            loss_bootstrap = (boot_mask * w * per_sample_boot).sum() / n_boot
            loss_bootstrap_normed = self.rms_bootstrap.normalize(loss_bootstrap, update=training)
        else:
            loss_bootstrap = torch.tensor(0.0, device=z_hat.device)
            loss_bootstrap_normed = loss_bootstrap
            n_boot = torch.tensor(0.0, device=z_hat.device)

        loss_total = loss_flow_normed + loss_bootstrap_normed

        # Return detached tensors — .item() called only at log intervals
        metrics = {
            "loss_flow": loss_flow.detach(),
            "loss_bootstrap": loss_bootstrap.detach(),
            "n_flow": n_flow.detach(),
            "n_bootstrap": n_boot.detach(),
        }

        return loss_total, metrics




    def _load_tokenizer_checkpoint(self, ckpt_path) -> None:
        if ckpt_path is None:
            raise ValueError(
                "DreamerAgent expects a pretrained tokenizer checkpoint. "
                "Set `tokenizer_ckpt` in the config to load weights before training the agent."
            )

        try:
            state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        except pickle.UnpicklingError:
            state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if "model" in state:
            state = state["model"]
        missing, unexpected = self.tokenizer.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Tokenizer checkpoint missing parameters: {missing}")
        if unexpected:
            raise RuntimeError(f"Tokenizer checkpoint has unexpected parameters: {unexpected}")






class RMSNormalizer:

    def __init__(self, decay=0.99, epsilon=1e-8):
        self.decay = decay
        self.epsilon = epsilon
        self.rms_sq = torch.tensor(1.0)
        self._on_device = False

    def normalize(self, loss: torch.Tensor, update: bool = True) -> torch.Tensor:
        loss_sq = loss.detach() ** 2
        if not self._on_device:
            self.rms_sq = self.rms_sq.to(loss_sq.device)
            self._on_device = True

        if update:
            self.rms_sq = self.decay * self.rms_sq + (1 - self.decay) * loss_sq

        rms = torch.sqrt(self.rms_sq + self.epsilon)
        return loss / rms
    

    def state_dict(self):
        return {"rms_sq": self.rms_sq}

    def load_state_dict(self, state):
        self.rms_sq = state["rms_sq"]
        self._on_device = False
