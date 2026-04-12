"""
Evaluate a trained dynamics model with latent metrics and visual diagnostics.

Example:
    python -m dynamics.evaluate_dynamics \
        --dynamics-ckpt checkpoints/dynamics/final.pt \
        --tokenizer-ckpt checkpoints/tokenizer/final.pt \
        --dataset dm_control \
        --task cheetah_run \
        --batch-size 8 \
        --steps 20 \
        --device tpu \
        --output-dir eval/dynamics
"""

from __future__ import annotations

# MUST be first: sets env vars (inductor thread count, XLA cache dir) that
# PyTorch reads at import time. Placing this after `import torch` is too late.
import _env_setup  # noqa: F401  (side-effect import)

import argparse
import csv
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path
import pickle
from typing import Optional

import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import wandb

from .flow_matching import add_noise, sample_tau_and_d
from .trainer import DynamicsTrainer, DynamicsTrainingConfig
from tokenizer.config import TokenizerConfig
from tokenizer.dataset import DatasetFactory
from tokenizer.layers import AttentionMask
from device_utils import get_device, mark_step, is_xla_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Dreamer V4 dynamics model")
    parser.add_argument("--dynamics-ckpt", type=str, required=True, help="Path to trained dynamics checkpoint")
    parser.add_argument("--tokenizer-ckpt", type=str, required=True, help="Path to tokenizer checkpoint")
    parser.add_argument("--dataset", type=str, default="dm_control")
    parser.add_argument("--task", type=str, default="cheetah_run")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Path to .npz file (required when --dataset=offline)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20, help="Evaluation batches to process")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-dir", type=str, default="evaluation/dynamics")
    parser.add_argument("--tau-bins", type=int, default=5)
    parser.add_argument("--max-gifs", type=int, default=4)
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--wandb-project", type=str, default="dreamer-v4-dynamics-eval")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--wandb-disabled", action="store_true")
    # Autoregressive rollout evaluation
    parser.add_argument("--num-context-frames", type=int, default=1,
                        help="Number of GT frames provided as context before rollout")
    parser.add_argument("--rollout-horizon", type=int, default=0,
                        help="Frames to generate autoregressively (0 = seq_len - context)")
    parser.add_argument("--rollout-batches", type=int, default=10,
                        help="Batches for rollout eval (expensive: K*H forward passes each)")
    return parser


def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, TypeError):
        return torch.load(path, map_location='cpu', weights_only=False)


def resolve_device(requested: str) -> torch.device:
    return get_device(requested)


def load_tokenizer_config_from_ckpt(ckpt_path: str, device: torch.device) -> TokenizerConfig:
    state = safe_torch_load(ckpt_path, device)
    if "tokenizer_cfg" not in state:
        raise ValueError(
            f"`tokenizer_cfg` not found in {ckpt_path}. "
            "Tokenizer checkpoint must include tokenizer config."
        )
    return state["tokenizer_cfg"]


def load_dynamics_config_from_ckpt(ckpt_path: str, device: torch.device):
    state = safe_torch_load(ckpt_path, device)
    cfg = state.get("dynamic_cfg") or state.get("dynamics_cfg")
    if cfg is None:
        raise ValueError(
            f"No dynamics config found in {ckpt_path}. Expected `dynamic_cfg` or `dynamics_cfg`."
        )
    return cfg


@torch.no_grad()
def decode_latents_to_frames(tokenizer, z_latents_4d: torch.Tensor) -> torch.Tensor:
    """Decode latent tokens back to frames through tokenizer decoder."""
    B, T, S_z, D_latent = z_latents_4d.shape
    z_latents = z_latents_4d.reshape(B, T * S_z, D_latent)

    z_expanded = tokenizer.latent_expand(z_latents)
    decoder_queries = tokenizer.decoder_queries.expand(B, -1, -1)
    decoder_queries = decoder_queries.unsqueeze(1).expand(-1, T, -1, -1).flatten(1, 2)
    decoder_sequence = torch.cat([z_expanded, decoder_queries], dim=1)

    temporal_causal_mask = tokenizer._build_temporal_causal_mask(T, z_latents.device)
    temporal_attn_mask = AttentionMask(is_causal=False, mask=temporal_causal_mask)

    num_patches = T * tokenizer.patch_embed.num_patches()
    latent_cross_mask = tokenizer._build_latent_cross_mask(
        latents_per_frame=tokenizer.config.num_latent_tokens,
        num_patches=num_patches,
        t=T,
        device=z_latents.device,
    )
    latent_cross_attn_mask = AttentionMask(is_causal=False, mask=latent_cross_mask)

    x = decoder_sequence
    for block in tokenizer.decoder_blocks:
        x = block(
            x,
            num_frames=T,
            temporal_mask=temporal_attn_mask,
            latent_cross_mask=latent_cross_attn_mask,
            num_latents=z_latents.size(1),
        )
    x = tokenizer.decoder_norm(x)

    recon_tokens = x[:, z_latents.size(1):, :]
    recon_patches = tokenizer.to_pixels(recon_tokens)
    H, W = tokenizer.config.image_size
    shape = torch.Size((B, T, tokenizer.config.in_channels, H, W))
    return tokenizer._unpatchify(recon_patches, shape)


def save_video_gif(video: torch.Tensor, path: Path, fps: int = 8) -> None:
    """Save (T, C, H, W) tensor in [0,1] as GIF."""
    video = video.detach().cpu().clamp(0.0, 1.0)
    frames = []
    for t in range(video.shape[0]):
        img = (
            video[t]
            .mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to(torch.uint8)
            .numpy()
        )
        frames.append(Image.fromarray(img))
    duration_ms = max(1, int(1000 / max(1, fps)))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def save_line_plot(
    path: Path,
    title: str,
    x_values: list[int],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    y_label: str,
) -> None:
    """Draw a simple line chart using Pillow (no matplotlib dependency)."""
    width, height = 900, 520
    margin_l, margin_r, margin_t, margin_b = 80, 30, 60, 80
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    all_y = [y for _, values, _ in series for y in values]
    y_min = min(all_y) if all_y else 0.0
    y_max = max(all_y) if all_y else 1.0
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1e-6

    draw.rectangle([margin_l, margin_t, margin_l + plot_w, margin_t + plot_h], outline=(0, 0, 0), width=2)
    draw.text((margin_l, 20), title, fill=(0, 0, 0))
    draw.text((15, margin_t + plot_h // 2), y_label, fill=(0, 0, 0))
    draw.text((margin_l + plot_w // 2 - 35, height - 30), "Timestep", fill=(0, 0, 0))

    n = max(1, len(x_values))
    x_den = max(1, n - 1)

    for idx, x in enumerate(x_values):
        px = margin_l + int(plot_w * (idx / x_den))
        draw.line([px, margin_t + plot_h, px, margin_t + plot_h + 5], fill=(0, 0, 0), width=1)
        draw.text((px - 5, margin_t + plot_h + 10), str(x), fill=(0, 0, 0))

    for tick in range(5):
        frac = tick / 4
        y_val = y_min + (1 - frac) * (y_max - y_min)
        py = margin_t + int(plot_h * frac)
        draw.line([margin_l - 5, py, margin_l, py], fill=(0, 0, 0), width=1)
        draw.text((8, py - 6), f"{y_val:.3f}", fill=(0, 0, 0))

    legend_x = margin_l + 10
    legend_y = margin_t + 10

    for name, values, color in series:
        points = []
        for idx, y in enumerate(values):
            px = margin_l + int(plot_w * (idx / x_den))
            y_norm = (y - y_min) / (y_max - y_min)
            py = margin_t + int(plot_h * (1 - y_norm))
            points.append((px, py))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for px, py in points:
            r = 3
            draw.ellipse([px - r, py - r, px + r, py + r], fill=color)

        draw.rectangle([legend_x, legend_y, legend_x + 18, legend_y + 12], fill=color)
        draw.text((legend_x + 24, legend_y - 2), name, fill=(0, 0, 0))
        legend_y += 18

    img.save(path)


@torch.no_grad()
def denoise_one_frame(
    model,
    z_context: torch.Tensor,
    tau_context: torch.Tensor,
    d_context: torch.Tensor,
    actions_context: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """Generate one new frame via K-step Euler denoising."""
    B = z_context.shape[0]
    device = z_context.device
    d_step = 1.0 / K

    z_current = torch.randn(B, 1, z_context.shape[2], z_context.shape[3], device=device)

    for k in range(K):
        tau_k = k / K

        z_seq = torch.cat([z_context, z_current], dim=1)
        tau_new = torch.full((B, 1), tau_k, device=device)
        d_new = torch.full((B, 1), d_step, device=device)
        tau_seq = torch.cat([tau_context, tau_new], dim=1)
        d_seq = torch.cat([d_context, d_new], dim=1)

        output = model(z_seq, actions_context, tau_seq, d_seq)
        z_hat_new = output.z_hat[:, -1:]

        v = (z_hat_new - z_current) / max(1.0 - tau_k, 1e-4)
        z_current = z_current + v * d_step
        mark_step()

    return z_current


@torch.no_grad()
def autoregressive_rollout(
    model,
    tokenizer,
    z_clean_full: torch.Tensor,
    actions_full: torch.Tensor,
    num_context: int,
    rollout_horizon: int,
    K: int,
    tau_ctx: float,
    context_length: int,
    K_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autoregressively generate future frames and measure MSE vs ground truth."""
    device = z_clean_full.device
    B, T_total, S_z, D_lat = z_clean_full.shape
    tau_ctx_val = 1.0 - tau_ctx
    d_ctx_val = 1.0 / K_max

    z_ctx_clean = z_clean_full[:, :num_context]
    tau_ctx_vec = torch.full((B, num_context), tau_ctx_val, device=device)
    z_ctx_noised, _ = add_noise(z_ctx_clean, tau_ctx_vec)

    # Pre-allocate fixed-size buffer to avoid dynamic torch.cat
    max_buf_len = num_context + rollout_horizon
    z_buffer = torch.zeros(B, max_buf_len, S_z, D_lat, device=device)
    tau_buffer = torch.full((B, max_buf_len), tau_ctx_val, device=device)
    d_buffer = torch.full((B, max_buf_len), d_ctx_val, device=device)

    z_buffer[:, :num_context] = z_ctx_noised
    buf_len = num_context

    generated = []
    mse_per_step = torch.zeros(rollout_horizon, device=device)

    for h in range(rollout_horizon):
        t_target = num_context + h
        if t_target >= T_total:
            break

        win_start = max(0, buf_len - context_length)
        z_win = z_buffer[:, win_start:buf_len]
        tau_win = tau_buffer[:, win_start:buf_len]
        d_win = d_buffer[:, win_start:buf_len]

        abs_start = win_start
        abs_end = t_target
        actions_win = actions_full[:, abs_start:abs_end]

        z_gen = denoise_one_frame(
            model, z_win, tau_win, d_win, actions_win, K,
        )

        z_gt = z_clean_full[:, t_target:t_target + 1]
        mse_per_step[h] = ((z_gen - z_gt) ** 2).mean()
        # Detach before storing — prevents holding the denoise graph across
        # the whole rollout horizon.
        generated.append(z_gen.detach())

        tau_gen = torch.full((B, 1), tau_ctx_val, device=device)
        z_gen_noised, _ = add_noise(z_gen, tau_gen)
        z_buffer[:, buf_len:buf_len + 1] = z_gen_noised
        buf_len += 1

        mark_step()

        if (h + 1) % 5 == 0:
            print(f"  [rollout] generated frame {h + 1}/{rollout_horizon}")

    z_rollout = torch.cat(generated, dim=1) if generated else z_clean_full[:, :0]
    return z_rollout, mse_per_step


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main(args: Optional[list[str]] = None) -> None:
    parser = build_parser()
    opts = parser.parse_args(args)

    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if opts.wandb_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"eval_{opts.dataset}_{opts.task}_{timestamp}"
    else:
        run_name = opts.wandb_name
    wandb_mode = "disabled" if opts.wandb_disabled else ("offline" if opts.wandb_offline else "online")
    wandb.init(
        project=opts.wandb_project,
        entity=opts.wandb_entity,
        name=run_name,
        mode=wandb_mode,
        config=vars(opts),
    )

    device = resolve_device(opts.device)
    print(f"[INFO] Using device: {device}")
    _is_xla = is_xla_device(device)

    tokenizer_cfg = load_tokenizer_config_from_ckpt(opts.tokenizer_ckpt, device)
    dynamics_cfg = load_dynamics_config_from_ckpt(opts.dynamics_ckpt, device)
    dynamics_cfg.validate_against_tokenizer(tokenizer_cfg)

    eval_train_cfg = DynamicsTrainingConfig(
        epochs=1,
        batch_size=opts.batch_size,
        amp=opts.amp,
        device=str(device),
        log_model_stats=False,
        log_memory=False,
        log_interval=10_000,
        log_model_stats_interval=10_000,
    )

    trainer = DynamicsTrainer(
        dynamics_cfg=dynamics_cfg,
        tokenizer_cfg=tokenizer_cfg,
        training_cfg=eval_train_cfg,
        tokenizer_ckpt=opts.tokenizer_ckpt,
    )
    trainer.load_checkpoint(opts.dynamics_ckpt, strict=True)
    trainer.model.to(device)
    trainer.model.eval()
    trainer.tokenizer.eval()

    dataset_cfg = replace(
        tokenizer_cfg,
        dataset_name=opts.dataset,
        task_name=opts.task,
        seq_len=dynamics_cfg.seq_len,
    )

    steps_per_worker = opts.steps
    if opts.num_workers > 0:
        steps_per_worker = max(1, opts.steps // opts.num_workers)

    dataset = DatasetFactory.get_dataset(
        dataset_cfg,
        batch_size=opts.batch_size,
        steps_per_epoch=steps_per_worker,
        dataset_path=opts.dataset_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=opts.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    tau_edges = torch.linspace(0.0, 1.0, steps=opts.tau_bins + 1, device=device)
    tau_sum = torch.zeros(opts.tau_bins, dtype=torch.float64, device=device)
    tau_count = torch.zeros(opts.tau_bins, dtype=torch.int32, device=device)

    log2_kmax = int(math.log2(dynamics_cfg.K_max))
    d_values = torch.tensor([1.0 / (2 ** k) for k in range(log2_kmax + 1)], device=device)
    num_d = d_values.numel()
    d_sum = torch.zeros(num_d, dtype=torch.float64, device=device)
    d_count = torch.zeros(num_d, dtype=torch.int32, device=device)

    joint_sum = torch.zeros((opts.tau_bins, num_d), dtype=torch.float64, device=device)
    joint_count = torch.zeros((opts.tau_bins, num_d), dtype=torch.int32, device=device)

    # Accumulate on-device — no .item() calls in the loop
    overall_sum_t = torch.tensor(0.0, dtype=torch.float64, device=device)
    overall_count_t = torch.tensor(0, dtype=torch.int32, device=device)

    action_delta_sum = None
    action_true_mse_sum = None
    action_shuf_mse_sum = None
    action_batches = 0

    # Store data for GIF generation AFTER the main loop
    gif_data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    processed_steps = 0

    with torch.no_grad():
        for step_idx, batch in enumerate(loader):
            if processed_steps >= opts.steps:
                break

            frames, actions = batch[0], batch[1]
            frames = frames.to(device)
            actions = actions.to(device)

            if frames.dim() == 4:
                frames = frames.unsqueeze(1)

            B, T = frames.shape[0:2]

            z_clean = trainer.model.encode_frames(frames)
            tau, d = sample_tau_and_d(B, T, K_max=dynamics_cfg.K_max, device=device)
            tau[:, 0] = 1.0 - dynamics_cfg.tau_ctx
            z_noised, _ = add_noise(z_clean, tau)
            z_hat = trainer.model(z_noised, actions, tau, d).z_hat

            # --- Accumulate metrics on-device (no .item()) ---
            mse_bt = ((z_hat - z_clean) ** 2).mean(dim=(-2, -1))  # (B, T)
            overall_sum_t += mse_bt.sum().double()
            overall_count_t += mse_bt.numel()

            # Tau-bucket metrics — vectorized, no Python loop with .any()
            # Digitize tau into bins: bin index for each (B, T) element
            tau_flat = tau.reshape(-1)
            mse_flat = mse_bt.reshape(-1)
            tau_bin_idx = torch.bucketize(tau_flat, tau_edges[1:-1])  # [0, num_bins-1]
            tau_bin_idx = tau_bin_idx.clamp(0, opts.tau_bins - 1)

            for i in range(opts.tau_bins):
                mask_i = (tau_bin_idx == i)
                tau_sum[i] += (mse_flat * mask_i.float()).sum().double()
                tau_count[i] += mask_i.float().sum().int()

            # D-bucket metrics — vectorized
            d_flat = d.reshape(-1)
            for j in range(num_d):
                mask_j = torch.isclose(d_flat, d_values[j])
                d_sum[j] += (mse_flat * mask_j.float()).sum().double()
                d_count[j] += mask_j.float().sum().int()

            # Joint tau x d metrics — vectorized
            for i in range(opts.tau_bins):
                mask_i = (tau_bin_idx == i)
                for j in range(num_d):
                    mask_j = torch.isclose(d_flat, d_values[j])
                    mask_ij = mask_i & mask_j
                    joint_sum[i, j] += (mse_flat * mask_ij.float()).sum().double()
                    joint_count[i, j] += mask_ij.float().sum().int()

            # Action shuffle sensitivity
            if B > 1:
                perm = torch.randperm(B).to(device)
                shuffled_actions = actions[perm]
                z_hat_shuf = trainer.model(z_noised, shuffled_actions, tau, d).z_hat

                delta_t = (z_hat - z_hat_shuf).abs().mean(dim=(0, 2, 3))
                mse_true_t = ((z_hat - z_clean) ** 2).mean(dim=(0, 2, 3))
                mse_shuf_t = ((z_hat_shuf - z_clean) ** 2).mean(dim=(0, 2, 3))

                if action_delta_sum is None:
                    action_delta_sum = torch.zeros_like(delta_t)
                    action_true_mse_sum = torch.zeros_like(mse_true_t)
                    action_shuf_mse_sum = torch.zeros_like(mse_shuf_t)

                action_delta_sum += delta_t
                action_true_mse_sum += mse_true_t
                action_shuf_mse_sum += mse_shuf_t
                action_batches += 1

            # Store GIF data on CPU for later (don't decode in the hot loop)
            if len(gif_data) < opts.max_gifs:
                gif_data.append((
                    frames[:1].cpu(),
                    z_clean[:1].cpu(),
                    z_hat[:1].cpu(),
                ))

            # Trigger XLA execution
            mark_step()

            processed_steps += 1
            if processed_steps % 5 == 0 or processed_steps == opts.steps:
                print(f"[INFO] Processed {processed_steps}/{opts.steps} batches")

    # --- Transfer accumulated metrics to CPU once ---
    overall_sum = overall_sum_t.item()
    overall_count = overall_count_t.item()

    if overall_count == 0:
        raise RuntimeError("No evaluation samples were processed.")

    overall_mse = overall_sum / overall_count

    # --- Generate GIFs AFTER the main eval loop ---
    gif_paths: list[Path] = []
    if gif_data and opts.max_gifs > 0:
        print(f"[INFO] Generating {len(gif_data)} GIFs...")
        for idx, (vis_frames, vis_clean, vis_pred) in enumerate(gif_data):
            recon_clean = decode_latents_to_frames(trainer.tokenizer, vis_clean.to(device))[0].cpu()
            recon_pred = decode_latents_to_frames(trainer.tokenizer, vis_pred.to(device))[0].cpu()
            combined = torch.cat([vis_frames[0], recon_clean, recon_pred], dim=3)
            gif_path = output_dir / f"step_{idx:03d}_gt_clean_pred.gif"
            save_video_gif(combined, gif_path, fps=opts.gif_fps)
            gif_paths.append(gif_path)
            mark_step()
        print(f"[INFO] GIFs done.")

    # ── Autoregressive Rollout Evaluation ──────────────────────────────
    K_inf = dynamics_cfg.K_inference
    num_context = opts.num_context_frames
    rollout_horizon = opts.rollout_horizon
    if rollout_horizon == 0:
        rollout_horizon = dynamics_cfg.seq_len - num_context

    rollout_mse_accum = torch.zeros(rollout_horizon, dtype=torch.float64, device=device)
    rollout_count = 0
    rollout_gif_count = 0
    rollout_gif_paths: list[Path] = []

    print(f"\n[INFO] Autoregressive rollout: {num_context} context frames, "
          f"{rollout_horizon} rollout steps, K={K_inf}")

    rollout_dataset = DatasetFactory.get_dataset(
        dataset_cfg,
        batch_size=opts.batch_size,
        steps_per_epoch=opts.rollout_batches,
        dataset_path=opts.dataset_path,
    )
    rollout_loader = DataLoader(
        rollout_dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Store rollout GIF data for post-loop generation
    rollout_gif_data: list[tuple[torch.Tensor, torch.Tensor]] = []

    with torch.no_grad():
        for step_idx, batch in enumerate(rollout_loader):
            if step_idx >= opts.rollout_batches:
                break

            frames, actions = batch[0].to(device), batch[1].to(device)
            if frames.dim() == 4:
                frames = frames.unsqueeze(1)
            B, T = frames.shape[:2]

            actual_horizon = min(rollout_horizon, T - num_context)
            if actual_horizon <= 0:
                print(f"[WARN] Sequence too short for rollout (T={T}, context={num_context})")
                continue

            z_clean = trainer.model.encode_frames(frames)

            z_rollout, mse_per_step = autoregressive_rollout(
                model=trainer.model,
                tokenizer=trainer.tokenizer,
                z_clean_full=z_clean,
                actions_full=actions,
                num_context=num_context,
                rollout_horizon=actual_horizon,
                K=K_inf,
                tau_ctx=dynamics_cfg.tau_ctx,
                context_length=dynamics_cfg.context_length,
                K_max=dynamics_cfg.K_max,
            )

            rollout_mse_accum[:actual_horizon] += mse_per_step[:actual_horizon].double()
            rollout_count += 1

            if len(rollout_gif_data) < opts.max_gifs and z_rollout.shape[1] > 0:
                n_vis = num_context + actual_horizon
                vis_gt = z_clean[:1, :n_vis].cpu()
                vis_rollout = torch.cat([z_clean[:1, :num_context], z_rollout[:1]], dim=1).cpu()
                rollout_gif_data.append((vis_gt, vis_rollout))

            mark_step()
            print(f"[INFO] Rollout batch {step_idx + 1}/{opts.rollout_batches} done")

    # Generate rollout GIFs after the loop
    if rollout_gif_data and opts.max_gifs > 0:
        print(f"[INFO] Generating {len(rollout_gif_data)} rollout GIFs...")
        for idx, (vis_gt, vis_rollout) in enumerate(rollout_gif_data):
            frames_gt = decode_latents_to_frames(trainer.tokenizer, vis_gt.to(device))[0].cpu()
            frames_pred = decode_latents_to_frames(trainer.tokenizer, vis_rollout.to(device))[0].cpu()
            combined = torch.cat([frames_gt, frames_pred], dim=3)
            gif_path = output_dir / f"rollout_{idx:03d}_gt_vs_pred.gif"
            save_video_gif(combined, gif_path, fps=opts.gif_fps)
            rollout_gif_paths.append(gif_path)
            mark_step()
        print(f"[INFO] Rollout GIFs done.")

    # ── Rollout Reporting ──────────────────────────────────────────────
    rollout_rows = []
    rollout_overall_mse = 0.0
    rollout_final_mse = 0.0

    if rollout_count > 0:
        avg_mse = rollout_mse_accum / rollout_count
        for h in range(rollout_horizon):
            rollout_rows.append([h + 1, float(avg_mse[h].item())])
        rollout_overall_mse = float(avg_mse.mean().item())
        rollout_final_mse = float(avg_mse[-1].item())

        write_csv(
            output_dir / "rollout_mse_per_step.csv",
            ["rollout_step", "latent_mse"],
            rollout_rows,
        )

        save_line_plot(
            path=output_dir / "rollout_mse_curve.png",
            title=f"Autoregressive Rollout MSE (K={K_inf}, ctx={num_context})",
            x_values=[h + 1 for h in range(rollout_horizon)],
            series=[("latent_mse", [r[1] for r in rollout_rows], (220, 20, 60))],
            y_label="Latent MSE",
        )

        print(f"[INFO] Rollout MSE: overall={rollout_overall_mse:.6f}, "
              f"final_step={rollout_final_mse:.6f}")

    tau_rows = []
    for i in range(opts.tau_bins):
        count_i = int(tau_count[i].item())
        mse_i = float((tau_sum[i] / max(count_i, 1)).item())
        tau_rows.append([float(tau_edges[i].item()), float(tau_edges[i + 1].item()), count_i, mse_i])

    d_rows = []
    for j, d_val in enumerate(d_values):
        count_j = int(d_count[j].item())
        mse_j = float((d_sum[j] / max(count_j, 1)).item())
        d_rows.append([float(d_val.item()), count_j, mse_j])

    joint_rows = []
    for i in range(opts.tau_bins):
        for j, d_val in enumerate(d_values):
            count_ij = int(joint_count[i, j].item())
            mse_ij = float((joint_sum[i, j] / max(count_ij, 1)).item())
            joint_rows.append(
                [
                    float(tau_edges[i].item()),
                    float(tau_edges[i + 1].item()),
                    float(d_val.item()),
                    count_ij,
                    mse_ij,
                ]
            )

    write_csv(output_dir / "tau_bucket_metrics.csv", ["tau_start", "tau_end", "count", "latent_mse"], tau_rows)
    write_csv(output_dir / "d_bucket_metrics.csv", ["d_value", "count", "latent_mse"], d_rows)
    write_csv(
        output_dir / "tau_d_joint_metrics.csv",
        ["tau_start", "tau_end", "d_value", "count", "latent_mse"],
        joint_rows,
    )

    summary = {
        "overall_latent_mse": overall_mse,
        "processed_batches": processed_steps,
        "batch_size": opts.batch_size,
        "seq_len": dynamics_cfg.seq_len,
        "gifs_written": len(gif_paths),
        "rollout_horizon": rollout_horizon,
        "rollout_K_inference": K_inf,
        "rollout_num_context": num_context,
        "rollout_overall_mse": rollout_overall_mse,
        "rollout_final_step_mse": rollout_final_mse,
        "rollout_batches_processed": rollout_count,
    }

    if action_batches > 0 and action_delta_sum is not None:
        avg_delta = (action_delta_sum / action_batches).detach().cpu().tolist()
        avg_true_mse = (action_true_mse_sum / action_batches).detach().cpu().tolist()
        avg_shuf_mse = (action_shuf_mse_sum / action_batches).detach().cpu().tolist()
        timesteps = list(range(len(avg_delta)))

        sensitivity_rows = []
        for t in timesteps:
            sensitivity_rows.append([t, avg_delta[t], avg_true_mse[t], avg_shuf_mse[t]])
        write_csv(
            output_dir / "action_shuffle_sensitivity.csv",
            ["timestep", "abs_latent_delta", "mse_true_actions", "mse_shuffled_actions"],
            sensitivity_rows,
        )

        save_line_plot(
            path=output_dir / "action_shuffle_delta.png",
            title="Action Shuffle Sensitivity (Latent Delta)",
            x_values=timesteps,
            series=[("abs_delta", avg_delta, (220, 20, 60))],
            y_label="|z_hat - z_hat_shuffled|",
        )
        save_line_plot(
            path=output_dir / "action_shuffle_mse.png",
            title="Prediction Error: True vs Shuffled Actions",
            x_values=timesteps,
            series=[
                ("true_actions_mse", avg_true_mse, (30, 144, 255)),
                ("shuffled_actions_mse", avg_shuf_mse, (255, 140, 0)),
            ],
            y_label="Latent MSE",
        )
        summary["action_shuffle_batches"] = action_batches

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    wandb.log(
        {
            "eval/overall_latent_mse": overall_mse,
            "eval/processed_batches": processed_steps,
            "eval/gifs_written": len(gif_paths),
            "eval/seq_len": dynamics_cfg.seq_len,
            "eval/batch_size": opts.batch_size,
        }
    )
    wandb.log(
        {
            "eval/tau_bucket_metrics": wandb.Table(
                columns=["tau_start", "tau_end", "count", "latent_mse"],
                data=tau_rows,
            ),
            "eval/d_bucket_metrics": wandb.Table(
                columns=["d_value", "count", "latent_mse"],
                data=d_rows,
            ),
            "eval/tau_d_joint_metrics": wandb.Table(
                columns=["tau_start", "tau_end", "d_value", "count", "latent_mse"],
                data=joint_rows,
            ),
        }
    )

    if action_batches > 0 and action_delta_sum is not None:
        wandb.log(
            {
                "eval/action_shuffle_sensitivity": wandb.Table(
                    columns=["timestep", "abs_latent_delta", "mse_true_actions", "mse_shuffled_actions"],
                    data=sensitivity_rows,
                ),
                "eval/action_shuffle_delta_plot": wandb.Image(
                    str(output_dir / "action_shuffle_delta.png")
                ),
                "eval/action_shuffle_mse_plot": wandb.Image(
                    str(output_dir / "action_shuffle_mse.png")
                ),
            }
        )

    if rollout_count > 0:
        rollout_wandb = {
            "eval/rollout_overall_mse": rollout_overall_mse,
            "eval/rollout_final_step_mse": rollout_final_mse,
            "eval/rollout_mse_per_step": wandb.Table(
                columns=["rollout_step", "latent_mse"],
                data=rollout_rows,
            ),
            "eval/rollout_mse_curve": wandb.Image(
                str(output_dir / "rollout_mse_curve.png")
            ),
        }
        wandb.log(rollout_wandb)
        for idx, gif_path in enumerate(rollout_gif_paths):
            wandb.log(
                {
                    f"eval/autoregressive_gif_{idx}": wandb.Video(
                        str(gif_path),
                        caption=f"Rollout {idx}: [GT | Context+Generated] (K={K_inf})",
                        fps=opts.gif_fps,
                        format="gif",
                    )
                }
            )

    for idx, gif_path in enumerate(gif_paths):
        wandb.log(
            {
                f"eval/denoise_gif_{idx}": wandb.Video(
                    str(gif_path),
                    caption=f"Step {idx}: [GT | CleanDecode | PredDecode]",
                    fps=opts.gif_fps,
                    format="gif",
                )
            }
        )
    wandb.finish()

    print("[DONE] Evaluation complete")
    print(f"  single-step latent_mse: {overall_mse:.6f}")
    if rollout_count > 0:
        print(f"  rollout latent_mse:     {rollout_overall_mse:.6f} (avg), "
              f"{rollout_final_mse:.6f} (final step)")
    print(f"  outputs: {output_dir}")


if __name__ == "__main__":
    main()
