"""Standalone training utilities for the Dreamer V4 tokenizer."""

from __future__ import annotations

import gc
import pickle
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from .config import TokenizerConfig
from .losses import MaskedAutoencoderLoss
from .tokenizer import MaskedAutoencoderTokenizer
from .metrics import MetricsBuffer, ModelStatistics, GPUMemoryTracker, ThroughputTracker
from device_utils import get_device, make_grad_scaler, save_checkpoint, is_master
from PIL import Image
import wandb


@dataclass
class TokenizerTrainingConfig: #TODO Need to move this to config.py
    epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    amp: bool = False
    checkpoint_interval: int = 1
    device: str = "cuda"
    # Logging configuration
    log_interval: int = 10          # Log every N steps
    log_smooth_window: int = 10     # Rolling average window size
    log_model_stats: bool = True    # Log weight/gradient statistics
    log_model_stats_interval: int = 50   # Model stats every N steps
    log_memory: bool = True         # Track GPU memory


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



        self.device = get_device(training_cfg.device)
        self.model = MaskedAutoencoderTokenizer(tokenizer_cfg).to(self.device)
        self.loss_module = loss_module or MaskedAutoencoderLoss(lpips_module=None)
        self.loss_module = self.loss_module.to(self.device)

        # Parameter-group weight decay split (standard MAE / paper recipe).
        # Apply weight_decay only to matrix weights (ndim >= 2). Exclude
        # norms, biases, embeddings, and learnable token banks. Uniform decay
        # on those creates a tug-of-war with the recon-loss-driven scale and
        # progressively de-normalizes the network — a contributing factor to
        # the saturation regime documented in Iter 23/36.
        decay_params, no_decay_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # No-decay rules (specific, not substring):
            #   - 1D params (biases, norm gains)
            #   - RMSNorm / LayerNorm weights
            #   - Learnable token banks: mask_token, decoder_queries, latent_tokens
            #   - Explicit ".bias" parameters
            # NOTE: do NOT use a loose "embed" substring match — it would also
            # catch patch_embed.proj.weight (a Conv2d kernel, ndim=4) which
            # should receive decay. There are no nn.Embedding lookup tables in
            # the tokenizer, so explicit token-bank names suffice.
            if (
                p.ndim < 2
                or "norm" in n.lower()
                or "latent_tokens" in n
                or "decoder_queries" in n
                or "mask_token" in n
                or n.endswith(".bias")
            ):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        param_groups = [
            {"params": decay_params, "weight_decay": training_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=training_cfg.lr,
        )
        self.scaler = make_grad_scaler(self.device, enabled=training_cfg.amp)

        self.global_step = 0
        self.metrics_buffer = MetricsBuffer(window=training_cfg.log_smooth_window)
        self.throughput_tracker = ThroughputTracker()
        self._rss_warning_emitted = False

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: Optional[str] = None,
        start_epoch: int = 1,
    ) -> None:
        for epoch in range(start_epoch, self.training_cfg.epochs + 1):
            train_metrics = self._run_epoch(train_loader, epoch, training=True) #TODO Need to refacto r this to make sure it runs only duyirng traingin
            val_metrics = None
            if val_loader is not None: #TODO Need to refactor this including _run_epoch
                # Wrap eval in torch.no_grad() — prevents autograd graph from
                # building during forward pass. Without this, LPIPS' VGG16
                # activations get pinned in GPU memory expecting a backward
                # pass that never comes. OOMs on the 4090 at production batch
                # size (run on 2026-05-11: 22 GB pinned during eval LPIPS).
                # No effect on training path; val_loss values unchanged.
                with torch.no_grad():
                    val_metrics = self._run_epoch(val_loader, epoch, training=False)

                # In-training latent diagnostics — logs effective rank,
                # n95 components, norm std within sequence, and abs_max to
                # wandb each epoch. Spot saturation lock-in / mode collapse
                # live instead of waiting for training to finish and running
                # latent_temporal_diagnostic.py separately. ~1s overhead.
                self._log_latent_metrics(val_loader)

            if is_master():
                print(
                    f"Epoch {epoch}: train_loss={train_metrics['loss/tokenizer_total']:.4f}"
                    + (f" val_loss={val_metrics['loss/tokenizer_total']:.4f}" if val_metrics else "")
                )

            if (
                checkpoint_dir is not None
                and self.training_cfg.checkpoint_interval > 0
                and epoch % self.training_cfg.checkpoint_interval == 0
            ):
                ckpt_path = f"{checkpoint_dir}/tokenizer_epoch_{epoch:03d}.pt" # TODO: refactor to use a more robust checkpointing strategy
                self.save_checkpoint(ckpt_path, epoch)
                
                if val_loader is not None:
                    self.visualize_reconstruction(val_loader, epoch, checkpoint_dir)

    def save_checkpoint(self, path: str, epoch: int) -> None:
        state = {
            "epoch": epoch,
            "tokenizer_cfg": self.tokenizer_cfg,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        save_checkpoint(state, path, self.device)
        del state
        gc.collect()
        if is_master():
            print(f"Saved tokenizer checkpoint to {path}")



    #TODO refactor this so that it doesnt affect the training speed and remove the data loading inside this function.
    @torch.no_grad()
    def _log_latent_metrics(self, val_loader: DataLoader) -> None:
        """Lightweight in-training latent diagnostic. Logs 4 metrics to wandb:

        - latent/effective_rank: Shannon-entropy effective rank of latent SVD.
          Catches collapse — should be > 5 on a healthy tokenizer; values near
          1.0 mean total mode collapse.
        - latent/n95_components: number of components for 95% variance.
        - latent/norm_std_within_seq: std of latent norm across timesteps within
          a single sequence. Catches tanh saturation lock-in — when tanh pins
          the bottleneck at max norm sqrt(D), this value goes to exactly 0.
        - latent/abs_max: peak latent magnitude. With tanh active should stay
          near 1.0; without tanh tracks the encoder's natural scale.

        Runs in ~1 second on RTX 4090 over 3 val batches. Logged each epoch.
        """
        if not is_master():
            return

        was_training = self.model.training
        self.model.eval()

        all_latents = []
        norm_stds = []
        try:
            for i, batch in enumerate(val_loader):
                if i >= 3:  # 3 batches is enough for a stable rank estimate
                    break
                frames = batch[0] if isinstance(batch, (list, tuple)) else batch
                frames = frames.to(self.device)
                if frames.dim() == 4:
                    frames = frames.unsqueeze(1)
                z = self.model.encode_only(frames)  # (B, T*L, D)
                B = z.shape[0]
                T = frames.shape[1]
                L = self.tokenizer_cfg.num_latent_tokens
                D = self.tokenizer_cfg.latent_dim
                z_4d = z.view(B, T, L, D)
                # Norm trajectory: first latent slot per frame, norm per t, std over T
                first_slot = z_4d[:, :, 0, :]                       # (B, T, D)
                norms_per_t = first_slot.norm(dim=-1)               # (B, T)
                norm_stds.append(norms_per_t.std(dim=1).mean().item())
                all_latents.append(z_4d.reshape(-1, D).cpu())
        finally:
            if was_training:
                self.model.train()

        if not all_latents:
            return

        latents_flat = torch.cat(all_latents, dim=0)               # (N, D)
        centered = latents_flat - latents_flat.mean(dim=0, keepdim=True)
        S = torch.linalg.svdvals(centered)
        var = S ** 2
        total = var.sum().clamp_min(1e-12)
        var_norm = var / total
        eff_rank = float(torch.exp(-(var_norm * (var_norm + 1e-12).log()).sum()))
        cum = torch.cumsum(var_norm, dim=0)
        n_95 = int((cum < 0.95).sum().item()) + 1

        metrics = {
            "latent/effective_rank": eff_rank,
            "latent/n95_components": float(n_95),
            "latent/norm_std_within_seq": float(sum(norm_stds) / max(1, len(norm_stds))),
            "latent/abs_max": float(latents_flat.abs().max().item()),
        }
        wandb.log(metrics, step=self.global_step)

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
        """        
        Logs include:
        - Smoothed losses (MSE, LPIPS, total)
        - Gradient norms (useful for detecting exploding/vanishing gradients)
        - Optimizer state (learning rate, AMP scale)
        - Model statistics (weight norms by layer type)
        - GPU memory utilization
        - Training throughput (samples/sec)
        """
        self.model.train(training)
        total_loss = torch.tensor(0.0, device=self.device)
        total_mse = torch.tensor(0.0, device=self.device)
        total_lpips = torch.tensor(0.0, device=self.device)
        total_steps = 0
        
        log_interval = self.training_cfg.log_interval
        model_stats_interval = self.training_cfg.log_model_stats_interval

        # On-device accumulators for smoothed logging (reset every log_interval)
        _log_loss = torch.tensor(0.0, device=self.device)
        _log_mse = torch.tensor(0.0, device=self.device)
        _log_lpips = torch.tensor(0.0, device=self.device)
        _log_grad_norm = torch.tensor(0.0, device=self.device)
        _log_mask_ratio = torch.tensor(0.0, device=self.device)
        _log_count = 0

        for batch in loader:
            #save_debug_gif(batch, "test_pipeline.gif")
            frames, *_ = batch
            frames = frames.to(self.device)
            if frames.dim() == 4: #TODO change it after intial implementation
                frames = frames.unsqueeze(1)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.training_cfg.amp):
                outputs = self.model(frames)
                loss_outputs = self.loss_module(
                    recon=outputs.reconstructed,
                    target=frames,
                    mask=outputs.mask,
                    patch_size=self.tokenizer_cfg.patch_size,
                    normalize_by=self.tokenizer_cfg.norm_loss_by,
                )
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

                # Accumulate on-device — no .item() sync per step
                _log_loss.add_(loss.detach())
                _log_mse.add_(loss_outputs.mse_loss.detach())
                if loss_outputs.lpips_loss is not None:
                    _log_lpips.add_(loss_outputs.lpips_loss.detach())
                _log_grad_norm.add_(grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                _log_mask_ratio.add_(loss_outputs.mask_ratio.detach())
                _log_count += 1

                if self.global_step % log_interval == 0:
                    # Host-side resource counters from /proc/self/status (passive — no GPU sync).
                    _local_rss = -1.0
                    _local_threads = -1
                    _local_fds = -1
                    _local_shm = -1
                    try:
                        with open('/proc/self/status') as _f:
                            for _line in _f:
                                if _line.startswith('RssAnon:'):
                                    _local_rss = int(_line.split()[1]) / (1024 ** 2)
                                elif _line.startswith('Threads:'):
                                    _local_threads = int(_line.split()[1])
                    except OSError:
                        pass
                    try:
                        import os as _os_probe
                        _local_fds = len(_os_probe.listdir('/proc/self/fd'))
                        _local_shm = len(_os_probe.listdir('/dev/shm'))
                    except OSError:
                        pass

                    # RSS warning — early OOM signal. Threshold 55 GB matches dynamics.
                    if _local_rss > 55 and not self._rss_warning_emitted:
                        print(
                            f"[WARN] RSS {_local_rss:.1f} GB — near OOM-kill threshold.",
                            flush=True,
                        )
                        self._rss_warning_emitted = True

                    if is_master():
                        n = max(_log_count, 1)
                        # Single device→host transfer instead of 5 separate .item() calls.
                        log_vec = torch.stack([
                            _log_loss / n,
                            _log_mse / n,
                            _log_lpips / n,
                            _log_grad_norm / n,
                            _log_mask_ratio / n,
                        ]).cpu().tolist()
                        loss_v, mse_v, lpips_v, grad_v, mask_v = log_vec
                        self.metrics_buffer.update({
                            "train/loss": loss_v,
                            "train/mse": mse_v,
                            "train/lpips": lpips_v,
                            "train/grad_norm": grad_v,
                            "train/mask_ratio": mask_v,
                        })

                        metrics = self.metrics_buffer.get_averages()
                        metrics.update({
                            "train/lr": self.optimizer.param_groups[0]['lr'],
                            "train/scale": self.scaler.get_scale(),
                            "epoch": epoch,
                            "global_step": self.global_step,
                            "samples_seen": self.global_step * frames.size(0),
                        })
                        metrics.update(self.throughput_tracker.step(frames.size(0)))

                        if self.training_cfg.log_memory:
                            metrics.update(GPUMemoryTracker.get_memory_stats(self.device))
                        if self.training_cfg.log_model_stats and self.global_step % model_stats_interval == 0:
                            metrics.update(ModelStatistics.compute_weight_stats(self.model))
                            metrics.update(ModelStatistics.compute_gradient_stats(self.model))

                        # Host-side resource metrics.
                        if _local_rss >= 0:
                            metrics['host/rss_anon_gb'] = _local_rss
                        if _local_threads >= 0:
                            metrics['host/threads'] = _local_threads
                        if _local_fds >= 0:
                            metrics['host/fd_count'] = _local_fds
                        if _local_shm >= 0:
                            metrics['host/shm_entries'] = _local_shm

                        wandb.log(metrics, step=self.global_step)

                    # Reset accumulators — prevents unbounded accumulation
                    _log_loss.zero_()
                    _log_mse.zero_()
                    _log_lpips.zero_()
                    _log_grad_norm.zero_()
                    _log_mask_ratio.zero_()
                    _log_count = 0

            # Accumulate on-device — no .item() sync per step
            total_loss.add_(loss_outputs.total_loss.detach())
            total_mse.add_(loss_outputs.mse_loss.detach())
            if loss_outputs.lpips_loss is not None:
                total_lpips.add_(loss_outputs.lpips_loss.detach())
            total_steps += 1

        # Single batched device→host transfer — one sync instead of 3-4
        denom = max(total_steps, 1)
        loss_v, mse_v, lpips_v = torch.stack([
            total_loss / denom,
            total_mse / denom,
            total_lpips / denom,
        ]).cpu().tolist()
        epoch_metrics = {
            "loss/tokenizer_total": loss_v,
            "loss/tokenizer_mse": mse_v,
        }
        if lpips_v > 0:
            epoch_metrics["loss/tokenizer_lpips"] = lpips_v
        return epoch_metrics

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, output_dir: str = "evaluation_results") -> Dict[str, float]:
        """
        Evaluates the model on the validation set and saves visualizations.
        """
        self.model.eval()
        # On-device accumulators — avoid per-batch .item() syncs in the eval loop.
        total_loss = torch.zeros((), device=self.device)
        total_mse = torch.zeros((), device=self.device)
        total_lpips = torch.zeros((), device=self.device)
        total_steps = 0  # host int — no sync needed

        import os
        os.makedirs(output_dir, exist_ok=True)

        for i, batch in enumerate(val_loader):
            frames, *_ = batch
            frames = frames.to(self.device)
            if frames.dim() == 4:
                frames = frames.unsqueeze(1)

            # Calculate sequence length dynamically
            b, t, c, h, w = frames.shape
            ph, pw = self.tokenizer_cfg.patch_size
            num_patches = (h // ph) * (w // pw)
            seq_len = t * num_patches

            # Use a fixed mask ratio for evaluation consistency
            from .masking import sample_random_mask
            mask = sample_random_mask(frames.size(0), seq_len, 0.75, 0.75, self.device, num_frames=t) #TODO need to check the mask ratio for evaluation
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=self.training_cfg.amp):
                outputs = self.model(frames, mask=mask)
                loss_outputs = self.loss_module(
                    recon=outputs.reconstructed,
                    target=frames,
                    mask=outputs.mask,
                    patch_size=self.tokenizer_cfg.patch_size,
                    normalize_by=self.tokenizer_cfg.norm_loss_by,
                )

            total_loss.add_(loss_outputs.total_loss.detach())
            total_mse.add_(loss_outputs.mse_loss.detach())
            if loss_outputs.lpips_loss is not None:
                total_lpips.add_(loss_outputs.lpips_loss.detach())
            total_steps += 1

            # Visualize first batch
        
            self.visualize_reconstruction(val_loader, i, output_dir)

        # Single batched device→host transfer at eval end (was 3*N per-batch syncs).
        denom = max(total_steps, 1)
        loss_v, mse_v, lpips_v = torch.stack([
            total_loss / denom,
            total_mse / denom,
            total_lpips / denom,
        ]).cpu().tolist()
        metrics = {
            "eval/loss": loss_v,
            "eval/mse": mse_v,
        }
        if lpips_v > 0:
            metrics["eval/lpips"] = lpips_v

        print("Evaluation Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        wandb.log(metrics)

        return metrics

    @torch.no_grad()
    def visualize_reconstruction(self, val_loader: DataLoader, epoch: int, save_dir: str):
        self.model.eval()
        batch = next(iter(val_loader))
        frames, *_ = batch
        frames = frames.to(self.device)

        if frames.dim() == 4:
            frames = frames.unsqueeze(1)

        b, t, c, h, w = frames.shape
        ph, pw = self.tokenizer_cfg.patch_size
        num_patches = (h // ph) * (w // pw)
        seq_len = t * num_patches

        from .masking import sample_random_mask
        import wandb

        original = frames[0]

        # === 1. TRUE RECONSTRUCTION (no masking) ===
        # This shows the actual autoencoder quality through the bottleneck
        mask_none = torch.zeros(b, seq_len, dtype=torch.bool, device=self.device)
        outputs_clean = self.model(frames, mask=mask_none)

        recon_clean = outputs_clean.reconstructed[0]
        combined_clean = torch.cat([original, recon_clean], dim=3)

        clean_path = f"{save_dir}/epoch_{epoch}_recon_clean.gif"
        save_video_grid(combined_clean, clean_path)
        wandb.log({
            "eval/reconstruction_clean": wandb.Video(clean_path, caption=f"Epoch {epoch} - No Mask", fps=10, format="gif")
        })

        # === 2. MASKED RECONSTRUCTION (75% masking) ===
        # This shows the model's ability to fill in missing information
        mask_high = sample_random_mask(b, seq_len, 0.75, 0.75, self.device, num_frames=t)
        outputs_masked = self.model(frames, mask=mask_high)

        recon_masked = outputs_masked.reconstructed[0]
        combined_masked = torch.cat([original, recon_masked], dim=3)

        masked_path = f"{save_dir}/epoch_{epoch}_recon_masked.gif"
        save_video_grid(combined_masked, masked_path)
        wandb.log({
            "eval/reconstruction_masked": wandb.Video(masked_path, caption=f"Epoch {epoch} - 75% Masked", fps=10, format="gif")
        })


def create_dataloader(dataset, batch_size: int, shuffle: bool = True) -> DataLoader: #TODO Need to refactor this after completion of the tokenizer
    if isinstance(dataset, IterableDataset):
        return DataLoader(dataset, batch_size=None)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



def save_video_grid(tensor, path):
    """
    Input: (T, C, H, W) tensor, normalized 0-1
    Saves a GIF.
    """
    # Ensure tensor is on CPU
    tensor = tensor.cpu()
    
    frames = []
    for t in range(tensor.shape[0]):
        # Convert to numpy 0-255
        # (C, H, W) -> (H, W, C)
        img = tensor[t].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy()
        frames.append(Image.fromarray(img))
    
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    # print(f"Saved video to {path}")
