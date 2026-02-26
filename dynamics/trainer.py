
from __future__ import annotations

from dataclasses import dataclass
import pickle
from typing import Dict, Optional
import math

from torch.optim.lr_scheduler import LambdaLR
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .config import DynamicsConfig
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
    lr: float = 1e-4
    warmup_steps: int = 500       
    min_lr: float = 1e-6  
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

        self.scaler = torch.cuda.amp.GradScaler(enabled=training_cfg.amp)
        self.scheduler = None
        self.global_step = 0
        self.metrics_buffer = MetricsBuffer(window=training_cfg.log_smooth_window)
        self.throughput_tracker = ThroughputTracker()



    def _build_scheduler(self, total_steps: int) -> LambdaLR:
            """Linear warmup for warmup_steps, then cosine decay to min_lr.
            
            lr_lambda returns a *multiplier* on base_lr:
            - step 0 → 0.0 (lr = 0)
            - step warmup_steps → 1.0 (lr = base_lr)  
            - step total_steps → min_lr/base_lr (lr = min_lr)
            """
            warmup_steps = self.training_cfg.warmup_steps
            min_lr = self.training_cfg.min_lr
            base_lr = self.training_cfg.lr

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return step / max(warmup_steps, 1)
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return max(min_lr / base_lr, cosine)

            return LambdaLR(self.optimizer, lr_lambda)



    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: Optional[str] = None,
        start_epoch: int = 1,
    ) -> None:

        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.training_cfg.epochs
        self.scheduler = self._build_scheduler(total_steps)

        # If resuming, restore scheduler state from checkpoint   
        #TODO Need to refactor this. May be move this to common place where we are loading checkpoint
        if hasattr(self, '_pending_scheduler_state') and self._pending_scheduler_state is not None:
            self.scheduler.load_state_dict(self._pending_scheduler_state)
            self._pending_scheduler_state = None


        for epoch in range(start_epoch, self.training_cfg.epochs + 1):
            train_metrics = self._run_epoch(train_loader, epoch, training=True)
            val_metrics = None
            if val_loader is not None:
                val_metrics = self._run_epoch(val_loader, epoch, training=False)

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
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,  
                "rms_normalizers": {
                    "flow": self.rms_flow.state_dict(),
                    "bootstrap": self.rms_bootstrap.state_dict(),
                },
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
        if "rms_normalizers" in state:
            rms_state = state["rms_normalizers"]
            if "flow" in rms_state:
                self.rms_flow.load_state_dict(rms_state["flow"])
            if "bootstrap" in rms_state:
                self.rms_bootstrap.load_state_dict(rms_state["bootstrap"])
        if "global_step" in state:                                    
            self.global_step = state["global_step"]                  
        self._pending_scheduler_state = state.get("scheduler", None)  
        
        return state.get("epoch", 0) + 1

    def _run_epoch(
        self, loader: DataLoader, epoch: int, training: bool
    ) -> Dict[str, float]:

        self.model.train(training)
        total_loss = 0.0
        total_flow = 0.0
        total_bootstrap = 0.0
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
            assert B >= 4, "Batch size too small for mixed training"

            if training and T > 1:
                n_single = max(1, int(B * 0.3))
                n_full = B - n_single

                frames_full = frames[n_single :]
                actions_full = actions[n_single:]

                z_clean_f = self.model.encode_frames(frames_full)  # (n_full, T, S_z, latent_dim)
                tau_f, d_f = sample_tau_and_d(n_full, T, K_max=self.dynamics_cfg.K_max, device=self.device)

                # Warm-up: flow-only → gradual bootstrap ramp → full mix
                warmup_end = 2000
                ramp_end = 4000
                if self.global_step < warmup_end:
                    d_f = torch.full_like(d_f, 1.0 / self.dynamics_cfg.K_max) #TODO 
                elif self.global_step < ramp_end: #TODO Added this to support training from scratch fro smaller steps. This is not needed for training at scale.
                    #  Linearly ramp bootstrap probability from 0% to 100%
                    ramp_progress = (self.global_step - warmup_end) / (ramp_end - warmup_end)
                    mask = torch.rand_like(d_f) < ramp_progress
                    d_min_tensor = torch.full_like(d_f, 1.0 / self.dynamics_cfg.K_max)
                    d_f = torch.where(mask, d_f, d_min_tensor)

                # (step 4000+): Normal sampling, no override
                z_noised_f, _ = add_noise(z_clean_f, tau_f)        
                tau_for_log, d_for_log = tau_f, d_f

                with torch.cuda.amp.autocast(enabled=self.training_cfg.amp):
                    loss_full, metrics_full = self._compute_loss(z_clean_f,z_noised_f, actions_full, tau_f, d_f, training = training )

                 #=========30% Single-Frame Training==========

                random_t = torch.randint(0, T, (n_single,))
                frames_single = frames[:n_single][torch.arange(n_single),random_t].unsqueeze(1)

                z_clean_s = self.model.encode_frames(frames_single)  # *(n_single, T, S_z, latent_dim)
                tau_s, d_s = sample_tau_and_d(n_single, 1, K_max=self.dynamics_cfg.K_max, device=self.device)

                if self.global_step < warmup_end:
                    d_s = torch.full_like(d_s, 1.0 / self.dynamics_cfg.K_max)
                elif self.global_step < ramp_end:#TODO Added this to support training from scratch fro smaller steps. This is not needed for training at scale.
                    ramp_progress = (self.global_step - warmup_end) / (ramp_end - warmup_end)
                    mask = torch.rand_like(d_s) < ramp_progress
                    d_min_tensor = torch.full_like(d_s, 1.0 / self.dynamics_cfg.K_max)
                    d_s = torch.where(mask, d_s, d_min_tensor)

                z_noised_s, _ = add_noise(z_clean_s, tau_s)        
                tau_for_log, d_for_log = tau_f, d_s

                with torch.cuda.amp.autocast(enabled=self.training_cfg.amp):
                    loss_single, metrics_single = self._compute_loss(z_clean_s,z_noised_s, None, tau_s, d_s, training = training )

                loss = (n_full * loss_full + n_single * loss_single) / B
                metrics = { #TODO verify this metrc
                    "loss_flow": (n_full * metrics_full["loss_flow"] + n_single * metrics_single["loss_flow"]) / B,
                    "loss_bootstrap": (n_full * metrics_full["loss_bootstrap"] + n_single * metrics_single["loss_bootstrap"]) / B,
                }

            else:
                # Validation or T=1 already: standard path
                z_clean = self.model.encode_frames(frames)
                tau, d = sample_tau_and_d(B, T, K_max=self.dynamics_cfg.K_max, device=self.device)
                z_noised, _ = add_noise(z_clean, tau)
                tau_for_log, d_for_log = tau, d
                with torch.cuda.amp.autocast(enabled=self.training_cfg.amp):

                    loss, metrics = self._compute_loss(z_clean, z_noised, actions, tau, d, training=training)




                
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
                self.scheduler.step() 

                self.global_step += 1

                self.metrics_buffer.update({
                    "train/loss": loss.item(),
                    "train/flow": metrics["loss_flow"],
                    "train/bootstrap": metrics["loss_bootstrap"],
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "train/tau_mean": tau_for_log.mean().item(),
                    "train/d_mean": d_for_log.mean().item(),
                })

                if self.global_step % log_interval == 0:
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

            total_loss += loss.item()
            total_flow += metrics["loss_flow"]
            total_bootstrap += metrics["loss_bootstrap"]
            total_steps += 1

        return {
            "loss/dynamics_total": total_loss / max(total_steps, 1),
            "loss/dynamics_flow": total_flow / max(total_steps, 1),
            "loss/dynamics_bootstrap": total_bootstrap / max(total_steps, 1),
        }


    def _compute_loss(self, z_clean, z_noised, actions, tau, d, training = True):
        """Compute flow matching + bootstrap loss.
        
        Args:
            z_clean:  (B, T, S_z, D)  — target clean latents
            z_noised: (B, T, S_z, D)  — corrupted input
            actions:  (B, T-1, action_dim)
            tau:      (B, T)  — per-frame signal levels
            d:        (B, T)  — per-frame step sizes
        
        Returns:
            loss, metrics_dict #TODO modify this based on the final change
        """
        
        d_min = 1/self.dynamics_cfg.K_max

        z_hat = self.model(z_noised, actions, tau, d)  # (B, T, S_z, D)

        is_flow = (d == d_min)  # (B, T)
        is_bootstrap = ~ is_flow

        # assert is_flow.any(), f"No flow samples in batch! is_flow.sum()={is_flow.sum()}"
        # assert is_bootstrap.any(), f"No bootstrap samples! is_bootstrap.sum()={is_bootstrap.sum()}"
        w = 0.9 * tau + 0.1  # (B, T)

        if is_flow.any():
            z_hat_flow  = z_hat[is_flow]
            z_clean_flow = z_clean[is_flow]
            w_flow = w[is_flow]  # (N_flow,)
            per_sample = ((z_hat_flow - z_clean_flow) ** 2).mean(dim=(-2, -1))  # (N_flow,)
            loss_flow = (w_flow * per_sample).mean()
            loss_flow_normed = self.rms_flow.normalize(loss_flow, update = training)

        else:
            loss_flow = torch.tensor(0.0, device=z_hat.device)  
            loss_flow_normed = torch.tensor(0.0, device=z_hat.device)


        if is_bootstrap.any():
            tau_4d = tau[:, :, None, None]      # (B, T, 1, 1)
            d_4d = d[:, :, None, None]   
            v_pred = (z_hat - z_noised) / (1 - tau_4d)

            with torch.no_grad():
                #TODO Need to test this code block as is userful to reduce the gpu memory usage spikes
                
                # z_noised_bs = z_noised[is_bootstrap]
                # actions_bs = actions[is_bootstrap] if actions is not None else None
                # tau_bs = tau[is_bootstrap]
                # d_bs = d[is_bootstrap]
                # z_hat_half1 = self.model(z_noised_bs, actions_bs, tau_bs, d_bs/2)
                z_hat_half1 = self.model(z_noised, actions, tau, d/2)
                v1 = (z_hat_half1 - z_noised) / (1-tau_4d)

                z_mid = z_noised + v1 * (d_4d/2)

                z_hat_half2 = self.model(z_mid, actions, tau + d/2, d/2)
                v2 = (z_hat_half2 - z_mid) / (1 - (tau_4d + d_4d/2))

                v_target = (v1 + v2) / 2


            v_pred_boot = v_pred[is_bootstrap]      # (N_boot, S_z, D)
            v_target_boot = v_target[is_bootstrap]  # (N_boot, S_z, D)
            tau_boot = tau[is_bootstrap]   
            w_boot = (0.9 * tau_boot + 0.1)  # (N_boot,)

            weight_tau_sq = (1 - tau_boot[:, None, None]) ** 2   # (N_boot, 1, 1)

            per_sample = (
                weight_tau_sq 
                * (v_pred_boot - v_target_boot) ** 2
            ).mean(dim=(-2, -1))  # (N_boot,)

            loss_bootstrap = (w_boot * per_sample).mean()
            loss_bootstrap_normed = self.rms_bootstrap.normalize(loss_bootstrap, update = training)

        else:
            loss_bootstrap = torch.tensor(0.0, device=z_hat.device)
            loss_bootstrap_normed = torch.tensor(0.0, device=z_hat.device)




        loss_total = loss_flow_normed + loss_bootstrap_normed
    
        metrics = {
            "loss_flow": loss_flow.item(),
            "loss_bootstrap": loss_bootstrap.item(),
            "n_flow": is_flow.sum().item(),
            "n_bootstrap": is_bootstrap.sum().item(),
        }
    
        return loss_total, metrics




    def _load_tokenizer_checkpoint(self, ckpt_path) -> None:
        if ckpt_path is None:
            raise ValueError(
                "DreamerAgent expects a pretrained tokenizer checkpoint. "
                "Set `tokenizer_ckpt` in the config to load weights before training the agent."
            )

        try:
            state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        except pickle.UnpicklingError:
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
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
       # self.rms_sq = None  
        self.rms_sq = torch.tensor(1.0)  
    
    def normalize(self, loss: torch.Tensor, update: bool = True) -> torch.Tensor:
        loss_sq = loss.detach() ** 2
        self.rms_sq = self.rms_sq.to(loss_sq.device)  

        # if self.rms_sq is None:
        #     self.rms_sq = loss_sq  
        if update:
            self.rms_sq = self.decay * self.rms_sq + (1 - self.decay) * loss_sq
        
        rms = torch.sqrt(self.rms_sq + self.epsilon)
        return loss / rms
    

    def state_dict(self):
        return {"rms_sq": self.rms_sq}

    def load_state_dict(self, state):
        self.rms_sq = state["rms_sq"]
