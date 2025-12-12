"""Standalone training utilities for the Dreamer V4 tokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from .config import TokenizerConfig
from .losses import MaskedAutoencoderLoss
from .tokenizer import MaskedAutoencoderTokenizer
from PIL import Image
import wandb


@dataclass
class TokenizerTrainingConfig: #TODO Need to move this to config.py
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
        start_epoch: int = 1,
    ) -> None:
        for epoch in range(start_epoch, self.training_cfg.epochs + 1):
            train_metrics = self._run_epoch(train_loader, epoch, training=True) #TODO Need to refacto r this to make sure it runs only duyirng traingin
            val_metrics = None
            if val_loader is not None: #TODO Need to refactor this including _run_epoch
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
        torch.save(state, path)
        print(f"Saved tokenizer checkpoint to {path}")

    def load_checkpoint(self, path: str, strict: bool = True) -> int:
        state = torch.load(path, map_location=self.device, weights_only=False) #TODO Need to check the weights_only
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
        total_lpips = 0.0
        total_steps = 0
        
        for batch in loader:
            #save_debug_gif(batch, "test_pipeline.gif")
            frames = batch.to(self.device)
            if frames.dim() == 4: #TODO change it after intial implementation
                frames = frames.unsqueeze(1)

            with torch.cuda.amp.autocast(enabled=self.training_cfg.amp):
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
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_cfg.grad_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                wandb.log({
                    "train/loss": loss.item(),
                    "train/mse": loss_outputs.mse_loss.item(),
                    "train/lpips": loss_outputs.lpips_loss.item() if loss_outputs.lpips_loss is not None else 0.0,
                    "epoch": epoch,
                })

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

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, output_dir: str = "evaluation_results") -> Dict[str, float]:
        """
        Evaluates the model on the validation set and saves visualizations.
        """
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_lpips = 0.0
        total_steps = 0
        
        import os
        os.makedirs(output_dir, exist_ok=True)

        for i, batch in enumerate(val_loader):
            frames = batch.to(self.device)
            if frames.dim() == 4:
                frames = frames.unsqueeze(1)

            # Calculate sequence length dynamically
            b, t, c, h, w = frames.shape
            ph, pw = self.tokenizer_cfg.patch_size
            num_patches = (h // ph) * (w // pw)
            seq_len = t * num_patches

            # Use a fixed mask ratio for evaluation consistency
            from .masking import sample_random_mask
            mask = sample_random_mask(frames.size(0), seq_len, 0.75, 0.75, self.device) #TODO need to check the mask ratio for evaluation
            
            with torch.cuda.amp.autocast(enabled=self.training_cfg.amp):
                outputs = self.model(frames, mask=mask)
                loss_outputs = self.loss_module(
                    recon=outputs.reconstructed,
                    target=frames,
                    mask=outputs.mask,
                    patch_size=self.tokenizer_cfg.patch_size,
                    normalize_by=self.tokenizer_cfg.norm_loss_by,
                )

            total_loss += loss_outputs.total_loss.item()
            total_mse += loss_outputs.mse_loss.item()
            if loss_outputs.lpips_loss is not None:
                total_lpips += loss_outputs.lpips_loss.item()
            total_steps += 1

            # Visualize first batch
        
            self.visualize_reconstruction(val_loader, i, output_dir)

        metrics = {
            "eval/loss": total_loss / max(total_steps, 1),
            "eval/mse": total_mse / max(total_steps, 1),
        }
        if total_lpips > 0:
            metrics["eval/lpips"] = total_lpips / max(total_steps, 1)
            
        print("Evaluation Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        
        wandb.log(metrics)
            
        return metrics

    @torch.no_grad()
    def visualize_reconstruction(self, val_loader: DataLoader, epoch: int, save_dir: str):
        self.model.eval()
        batch = next(iter(val_loader))
        frames = batch.to(self.device)
        if frames.dim() == 4: frames = frames.unsqueeze(1)

        # Calculate sequence length dynamically
        b, t, c, h, w = frames.shape
        ph, pw = self.tokenizer_cfg.patch_size
        num_patches = (h // ph) * (w // pw)
        seq_len = t * num_patches

        # Use a high mask ratio to see if the model actually "learned" structure
        # or is just copying unmasked patches.
        from .masking import sample_random_mask
        mask = sample_random_mask(frames.size(0), seq_len, 0.75, 0.75, self.device) #TODO Need to do more research  on the masking ratio during evaluatiom'
        
        outputs = self.model(frames, mask=mask)
        
        # Unnormalize if your data was normalized (e.g., mean/std)
        # Assuming data is 0-1 or -1 to 1.
        
        recon = outputs.reconstructed
        
        import wandb
        from PIL import Image
        
        # Take first video of first batch item
        # frames: (B, T, C, H, W) -> (T, C, H, W)
        original = frames[0]
        reconstruction = recon[0]
        
        # Stitch side-by-side: (T, C, H, W*2)
        combined = torch.cat([original, reconstruction], dim=3) 
        
        save_path = f"{save_dir}/epoch_{epoch}_recon.gif"
        save_video_grid(combined, save_path)
        
        # Log to WandB as a video
        # WandB expects (T, C, H, W) or path
        wandb.log({"eval/reconstruction": wandb.Video(save_path, caption=f"Epoch {epoch} Reconstruction", fps=10, format="gif")})


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