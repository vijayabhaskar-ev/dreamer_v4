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
        --device cuda \
        --output-dir eval/dynamics
"""

from __future__ import annotations

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
from device_utils import get_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Dreamer V4 dynamics model")
    parser.add_argument("--dynamics-ckpt", type=str, required=True, help="Path to trained dynamics checkpoint")
    parser.add_argument("--tokenizer-ckpt", type=str, required=True, help="Path to tokenizer checkpoint")
    parser.add_argument("--dataset", type=str, default="dm_control")
    parser.add_argument("--task", type=str, default="cheetah_run")
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
    return parser


def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, TypeError):
        return torch.load(path, map_location=device, weights_only=False)


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
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=opts.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    tau_edges = torch.linspace(0.0, 1.0, steps=opts.tau_bins + 1, device=device)
    tau_sum = torch.zeros(opts.tau_bins, dtype=torch.float64, device=device)
    tau_count = torch.zeros(opts.tau_bins, dtype=torch.long, device=device)

    log2_kmax = int(math.log2(dynamics_cfg.K_max))
    d_values = torch.tensor([1.0 / (2 ** k) for k in range(log2_kmax + 1)], device=device)
    d_sum = torch.zeros(d_values.numel(), dtype=torch.float64, device=device)
    d_count = torch.zeros(d_values.numel(), dtype=torch.long, device=device)

    joint_sum = torch.zeros((opts.tau_bins, d_values.numel()), dtype=torch.float64, device=device)
    joint_count = torch.zeros((opts.tau_bins, d_values.numel()), dtype=torch.long, device=device)

    overall_sum = 0.0
    overall_count = 0

    action_delta_sum = None
    action_true_mse_sum = None
    action_shuf_mse_sum = None
    action_batches = 0

    gif_count = 0
    gif_paths: list[Path] = []
    processed_steps = 0

    with torch.no_grad():
        for step_idx, batch in enumerate(loader):
            if processed_steps >= opts.steps:
                break

            frames, actions = batch
            frames = frames.to(device)
            actions = actions.to(device)

            if frames.dim() == 4:
                frames = frames.unsqueeze(1)

            B, T = frames.shape[0:2]

            z_clean = trainer.model.encode_frames(frames)
            tau, d = sample_tau_and_d(B, T, K_max=dynamics_cfg.K_max, device=device)
            z_noised, _ = add_noise(z_clean, tau)
            z_hat = trainer.model(z_noised, actions, tau, d)

            mse_bt = ((z_hat - z_clean) ** 2).mean(dim=(-2, -1))
            overall_sum += mse_bt.sum().item()
            overall_count += mse_bt.numel()

            for i in range(opts.tau_bins):
                low = tau_edges[i]
                high = tau_edges[i + 1]
                if i == opts.tau_bins - 1:
                    mask_tau = (tau >= low) & (tau <= high)
                else:
                    mask_tau = (tau >= low) & (tau < high)
                if mask_tau.any():
                    tau_sum[i] += mse_bt[mask_tau].sum().double()
                    tau_count[i] += mask_tau.sum()

            for j, d_val in enumerate(d_values):
                mask_d = torch.isclose(d, d_val)
                if mask_d.any():
                    d_sum[j] += mse_bt[mask_d].sum().double()
                    d_count[j] += mask_d.sum()

            for i in range(opts.tau_bins):
                low = tau_edges[i]
                high = tau_edges[i + 1]
                if i == opts.tau_bins - 1:
                    mask_tau = (tau >= low) & (tau <= high)
                else:
                    mask_tau = (tau >= low) & (tau < high)
                if not mask_tau.any():
                    continue
                for j, d_val in enumerate(d_values):
                    mask = mask_tau & torch.isclose(d, d_val)
                    if mask.any():
                        joint_sum[i, j] += mse_bt[mask].sum().double()
                        joint_count[i, j] += mask.sum()

            if B > 1:
                perm = torch.randperm(B, device=device)
                shuffled_actions = actions[perm]
                z_hat_shuf = trainer.model(z_noised, shuffled_actions, tau, d)

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

            if gif_count < opts.max_gifs:
                vis_frames = frames[:1]
                vis_clean = z_clean[:1]
                vis_pred = z_hat[:1]

                recon_clean = decode_latents_to_frames(trainer.tokenizer, vis_clean)[0]
                recon_pred = decode_latents_to_frames(trainer.tokenizer, vis_pred)[0]

                combined = torch.cat([vis_frames[0], recon_clean, recon_pred], dim=3)
                gif_path = output_dir / f"step_{step_idx:03d}_gt_clean_pred.gif"
                save_video_gif(combined, gif_path, fps=opts.gif_fps)
                gif_paths.append(gif_path)
                gif_count += 1

            processed_steps += 1
            if processed_steps % 5 == 0 or processed_steps == opts.steps:
                print(f"[INFO] Processed {processed_steps}/{opts.steps} batches")

    if overall_count == 0:
        raise RuntimeError("No evaluation samples were processed.")

    overall_mse = overall_sum / overall_count

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
        "gifs_written": gif_count,
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
            "eval/gifs_written": gif_count,
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

    for idx, gif_path in enumerate(gif_paths):
        wandb.log(
            {
                f"eval/rollout_gif_{idx}": wandb.Video(
                    str(gif_path),
                    caption=f"Step {idx}: [GT | CleanDecode | PredDecode]",
                    fps=opts.gif_fps,
                    format="gif",
                )
            }
        )
    wandb.finish()

    print("[DONE] Evaluation complete")
    print(f"  overall_latent_mse: {overall_mse:.6f}")
    print(f"  outputs: {output_dir}")


if __name__ == "__main__":
    main()
