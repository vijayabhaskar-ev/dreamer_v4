
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
    weight_decay_heavy: float = 0.1   # attention + FF layers (drifted in charts)
    grad_clip: float = 5.0
    amp: bool = False
    checkpoint_interval: int = 1
    device: str = "cuda"
    # Curriculum: bootstrap ramp-up
    curriculum_warmup_steps: int = 2000   
    curriculum_ramp_steps: int = 4000    
    # Alternating batch lengths
    long_batch_ratio: float = 0.15  # fraction of training steps using T₂ (long sequences)
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

        # Per-group weight decay: attention+FF get heavy decay (these grew
        # unboundedly in prior runs), norms/biases get zero, rest gets default.
        heavy_decay_params = []
        no_decay_params = []
        default_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            name_lower = name.lower()
            if param.dim() <= 1 or 'norm' in name_lower:
                no_decay_params.append(param)
            elif any(p in name_lower for p in ['spatial_attn', 'temporal_attn', '.ff.']):
                heavy_decay_params.append(param)
            else:
                default_decay_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {"params": heavy_decay_params, "weight_decay": training_cfg.weight_decay_heavy},
            {"params": no_decay_params, "weight_decay": 0.0},
            {"params": default_decay_params, "weight_decay": training_cfg.weight_decay},
        ], lr=training_cfg.lr)

        self.scaler = make_grad_scaler(self.device, enabled=training_cfg.amp)
        self.global_step = 0
        self._last_lr_bucket = -1
        self.metrics_buffer = MetricsBuffer(window=training_cfg.log_smooth_window)
        self.throughput_tracker = ThroughputTracker()

        # Cached device tensors — avoids creating new XLA graph nodes per step
        self._zero = torch.tensor(0.0, device=self.device)
        self._ramp_tensor = torch.tensor(0.0, device=self.device)



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
        train_loader_short: DataLoader,
        train_loader_long: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: Optional[str] = None,
        start_epoch: int = 1,
    ) -> None:

        steps_per_epoch = self.training_cfg.steps_per_epoch
        total_steps = steps_per_epoch * self.training_cfg.epochs
        self._build_scheduler(total_steps)


        for epoch in range(start_epoch, self.training_cfg.epochs + 1):
            train_metrics = self._run_epoch(
                train_loader_short, train_loader_long, epoch, training=True,
            )
            val_metrics = None
            if val_loader is not None:
                val_metrics = self._run_epoch(
                    val_loader, None, epoch, training=False,
                )

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
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except (ValueError, KeyError) as e:
                if is_master():
                    print(f"[WARN] Could not load optimizer state ({e}). "
                          "Using fresh optimizer (expected when changing param groups).")
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
        self,
        loader_short: DataLoader,
        loader_long: Optional[DataLoader],
        epoch: int,
        training: bool,
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

        # Alternating batch lengths: iterate short loader, draw from long loader
        # with probability long_batch_ratio. During validation (loader_long=None),
        # only the short loader is used.
        long_ratio = self.training_cfg.long_batch_ratio if (training and loader_long is not None) else 0.0
        long_iter = iter(loader_long) if loader_long is not None else None

        for batch_short in loader_short:
            # Decide whether to use short or long batch this step
            # Python-level branch — both T values compile once, no per-step recompilation
            use_long = training and long_iter is not None and (torch.rand(1).item() < long_ratio)
            if use_long:
                try:
                    batch = next(long_iter)
                except StopIteration:
                    long_iter = iter(loader_long)
                    batch = next(long_iter)
            else:
                batch = batch_short

            frames, actions = batch
            frames = frames.to(self.device)
            actions = actions.to(self.device)

            if frames.dim() == 4:
                frames = frames.unsqueeze(1)

            B, T = frames.shape[0:2]

            # Python-level gate: skip expensive bootstrap forward passes
            # during flow-only warmup (no XLA sync, no dynamic shapes)
            need_bootstrap = self.global_step >= warmup_end

            z_clean = self.model.encode_frames(frames)
            tau, d = sample_tau_and_d(B, T, K_max=self.dynamics_cfg.K_max, device=self.device)
            
            # Context frame: First frame always gets near-clean noise level
            tau[:, 0] = 1.0 - self.dynamics_cfg.tau_ctx

            # Curriculum d-override
            if self.global_step < warmup_end:
                d = torch.full_like(d, 1.0 / self.dynamics_cfg.K_max)
            elif self.global_step < ramp_end:
                ramp_frac = (self.global_step - warmup_end) / (ramp_end - warmup_end)
                self._ramp_tensor.fill_(int(ramp_frac * 10) / 10.0)
                mask = torch.rand_like(d) < self._ramp_tensor
                d_min = torch.full_like(d, 1.0 / self.dynamics_cfg.K_max)
                d = torch.where(mask, d, d_min)

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
                            "train/seq_len": float(T),
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
                            smoothed_metrics.update(self._compute_wd_metrics())

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


    def _compute_wd_metrics(self) -> Dict[str, float]:
        """Per-group effective weight decay: wd * sum(||p||^2).

        Shows how much regularization each group contributes.
        Only called at model_stats_interval so .item() syncs are acceptable.
        """
        names = ["attn_ff", "no_decay", "default"]
        metrics = {}
        for name, group in zip(names, self.optimizer.param_groups):
            wd = group["weight_decay"]
            if wd == 0.0:
                continue
            sq_sum = sum(p.data.norm(2) ** 2 for p in group["params"] if p.requires_grad)
            metrics[f"model/wd_{name}"] = (wd * sq_sum).item()
        return metrics

    def _compute_loss(self, z_clean, z_noised, actions, tau, d, training=True,
                      compute_bootstrap=True):
        """Compute flow matching + bootstrap loss.

        Uses masked arithmetic instead of boolean indexing so all tensors
        keep fixed (B, T, ...) shapes — critical for XLA/TPU which
        recompiles the entire graph on shape changes. T can be T₁ or T₂
        (two fixed values from alternating batch lengths).

        Args:
            z_clean:  (B, T, S_z, D)  — target clean latents
            z_noised: (B, T, S_z, D)  — corrupted input
            actions:  (B, T-1, action_dim)
            tau:      (B, T)  — per-frame signal levels
            d:        (B, T)  — per-frame step sizes
            compute_bootstrap: Python-level gate to skip bootstrap forward passes

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
            loss_bootstrap = self._zero
            loss_bootstrap_normed = self._zero
            n_boot = self._zero

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
            # In-place update — avoids creating new XLA graph nodes per step
            self.rms_sq.mul_(self.decay).add_(loss_sq, alpha=(1 - self.decay))

        rms = torch.sqrt(self.rms_sq + self.epsilon)
        return loss / rms
    

    def state_dict(self):
        return {"rms_sq": self.rms_sq}

    def load_state_dict(self, state):
        self.rms_sq = state["rms_sq"]
        self._on_device = False
