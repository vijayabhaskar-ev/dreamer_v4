"""
Evaluate a trained tokenizer's encode->decode roundtrip quality.

Diagnoses the question: "what does the tokenizer's bottleneck preserve, and on
which frames does it fail?" Uses encode_only() (no masking) so the resulting MSE
isolates pure reconstruction error from masked-fill-in error.

Example:
    python -m tokenizer.evaluate_tokenizer \\
        --tokenizer-ckpt checkpoints/tokenizer/final.pt \\
        --dataset offline --dataset-path cheetah_run.npz \\
        --batch-size 32 --steps 100 \\
        --max-gifs 4 --top-k-worst 8 \\
        --device tpu --output-dir evaluation/tokenizer
"""

from __future__ import annotations

import _env_setup  # noqa: F401  (side-effect import; must precede torch)

import argparse
import heapq
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader
import wandb

from device_utils import get_device, is_xla_device, mark_step
from dynamics.evaluate_dynamics import (
    decode_latents_to_frames,
    load_tokenizer_config_from_ckpt,
    safe_torch_load,
    save_line_plot,
    save_video_gif,
    write_csv,
)
from tokenizer.dataset import DatasetFactory
from tokenizer.tokenizer import MaskedAutoencoderTokenizer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Dreamer V4 tokenizer (encode->decode)")
    p.add_argument("--tokenizer-ckpt", type=str, required=True)
    p.add_argument("--dataset", type=str, default="offline")
    p.add_argument("--task", type=str, default="cheetah_run")
    p.add_argument("--dataset-path", type=str, default=None,
                   help="Path to .npz file (required when --dataset=offline)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=100, help="Eval batches to process")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="evaluation/tokenizer")
    p.add_argument("--seq-len", type=int, default=0,
                   help="Sequence length for eval (0 = use checkpoint's tokenizer_cfg.seq_len)")
    p.add_argument("--max-gifs", type=int, default=4, help="Number of GT||Recon GIFs to save")
    p.add_argument("--top-k-worst", type=int, default=8,
                   help="Number of worst-MSE frames to save as PNGs for inspection")
    p.add_argument("--gif-fps", type=int, default=8)
    p.add_argument("--lpips", action="store_true",
                   help="Also compute LPIPS per frame (slow; requires `lpips` package)")
    p.add_argument("--wandb-project", type=str, default="dreamer-v4-tokenizer-eval")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-offline", action="store_true")
    p.add_argument("--wandb-disabled", action="store_true")
    return p


def resolve_device(requested: str) -> torch.device:
    return get_device(requested)


def load_tokenizer_model(
    ckpt_path: str,
    tokenizer_cfg,
    device: torch.device,
) -> MaskedAutoencoderTokenizer:
    """Build tokenizer from cfg, load weights from ckpt, move to device, set eval."""
    state = safe_torch_load(ckpt_path, device)
    if "model" in state:
        state = state["model"]
    model = MaskedAutoencoderTokenizer(tokenizer_cfg).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # qk_norm params default-init to ones — safe to skip on older checkpoints.
    missing_real = [k for k in missing if "qk_norm" not in k]
    if missing_real:
        raise RuntimeError(f"Tokenizer checkpoint missing parameters: {missing_real}")
    if unexpected:
        raise RuntimeError(f"Tokenizer checkpoint has unexpected parameters: {unexpected}")
    model.eval()
    return model


def maybe_build_lpips(device: torch.device):
    """Construct an LPIPS module if --lpips passed and the package is installed."""
    try:
        import lpips as lpips_pkg
    except ImportError:
        print("[WARN] --lpips requested but `lpips` package not installed; skipping LPIPS.")
        return None
    net = lpips_pkg.LPIPS(net="alex").to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@torch.no_grad()
def reconstruct_via_bottleneck(
    tokenizer: MaskedAutoencoderTokenizer,
    frames: torch.Tensor,
) -> torch.Tensor:
    """frames (B,T,C,H,W) -> encode_only -> decode -> recon (B,T,C,H,W). No masking."""
    B, T, C, H, W = frames.shape
    z_3d = tokenizer.encode_only(frames)  # (B, T*L, D)
    L = tokenizer.config.num_latent_tokens
    D = z_3d.shape[-1]
    z_4d = z_3d.view(B, T, L, D)
    return decode_latents_to_frames(tokenizer, z_4d)


def stack_gt_recon_for_gif(gt: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """(T,C,H,W) + (T,C,H,W) -> (T,C,H,2W) horizontally concatenated."""
    return torch.cat([gt, recon], dim=-1)


def save_frame_png(gt_frame: torch.Tensor, recon_frame: torch.Tensor, path: Path) -> None:
    """Save a (C,H,W) GT and (C,H,W) recon stacked vertically as a PNG."""
    stacked = torch.cat([gt_frame, recon_frame], dim=-2).detach().cpu().clamp(0.0, 1.0)
    arr = (stacked.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
    Image.fromarray(arr).save(path)


def main(args: Optional[list[str]] = None) -> None:
    parser = build_parser()
    opts = parser.parse_args(args)

    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    worst_dir = output_dir / "worst_frames"
    worst_dir.mkdir(exist_ok=True)

    if opts.wandb_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"eval_tok_{opts.dataset}_{opts.task}_{timestamp}"
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
    eval_seq_len = opts.seq_len if opts.seq_len > 0 else tokenizer_cfg.seq_len
    print(f"[INFO] Eval seq_len: {eval_seq_len} (tokenizer_cfg default: {tokenizer_cfg.seq_len})")

    tokenizer = load_tokenizer_model(opts.tokenizer_ckpt, tokenizer_cfg, device)
    print(f"[INFO] Tokenizer loaded; image_size={tokenizer_cfg.image_size}, "
          f"patch_size={tokenizer_cfg.patch_size}, num_latent_tokens={tokenizer_cfg.num_latent_tokens}")

    lpips_net = maybe_build_lpips(device) if opts.lpips else None

    dataset_cfg = replace(
        tokenizer_cfg,
        dataset_name=opts.dataset,
        task_name=opts.task,
        seq_len=eval_seq_len,
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

    # Per-frame-index accumulators (length T, summed across batches).
    mse_per_index_sum = torch.zeros(eval_seq_len, dtype=torch.float64, device=device)
    lpips_per_index_sum = torch.zeros(eval_seq_len, dtype=torch.float64, device=device)
    frames_per_index = torch.zeros(eval_seq_len, dtype=torch.int64, device=device)

    # Top-K min-heap of (mse, counter, batch_step, b, t, gt_cpu, recon_cpu).
    # Counter prevents tensor-comparison ambiguity when MSEs tie.
    worst_heap: list[tuple[float, int, int, int, int, torch.Tensor, torch.Tensor]] = []
    worst_counter = 0

    gif_data: list[tuple[torch.Tensor, torch.Tensor]] = []

    processed = 0
    with torch.no_grad():
        for step_idx, batch in enumerate(loader):
            if processed >= opts.steps:
                break

            frames = batch[0] if isinstance(batch, (list, tuple)) else batch
            frames = frames.to(device)
            if frames.dim() == 4:
                frames = frames.unsqueeze(1)

            B, T = frames.shape[:2]
            if T != eval_seq_len:
                print(f"[WARN] Batch T={T} != eval_seq_len={eval_seq_len}; skipping")
                continue

            recon = reconstruct_via_bottleneck(tokenizer, frames)

            # Per-frame MSE: (B, T)
            mse_bt = ((recon - frames) ** 2).mean(dim=(2, 3, 4))
            mse_per_index_sum += mse_bt.sum(dim=0).double()
            frames_per_index += B

            if lpips_net is not None:
                # LPIPS expects [-1, 1] range; frames are [0, 1].
                f_flat = (frames * 2 - 1).reshape(B * T, *frames.shape[2:])
                r_flat = (recon * 2 - 1).reshape(B * T, *recon.shape[2:])
                lpips_flat = lpips_net(f_flat, r_flat).reshape(B, T)
                lpips_per_index_sum += lpips_flat.sum(dim=0).double()

            # Materialize once per batch for heap + GIFs (batched .cpu() is fine).
            mse_cpu = mse_bt.detach().cpu()
            frames_cpu = frames.detach().cpu()
            recon_cpu = recon.detach().cpu()

            # Update top-K worst frames heap.
            for b in range(B):
                for t in range(T):
                    val = float(mse_cpu[b, t])
                    entry = (val, worst_counter, processed, b, t,
                             frames_cpu[b, t].clone(), recon_cpu[b, t].clone())
                    if len(worst_heap) < opts.top_k_worst:
                        heapq.heappush(worst_heap, entry)
                        worst_counter += 1
                    elif val > worst_heap[0][0]:
                        heapq.heapreplace(worst_heap, entry)
                        worst_counter += 1

            # Stash a few GIFs from the first batches.
            if len(gif_data) < opts.max_gifs:
                gif_data.append((frames_cpu[0].clone(), recon_cpu[0].clone()))

            processed += 1
            if _is_xla:
                mark_step()
            if processed % 10 == 0 or processed == opts.steps:
                print(f"[INFO] Processed {processed}/{opts.steps} batches")

    if processed == 0:
        raise RuntimeError("No batches processed; check --dataset / --dataset-path")

    # Aggregate per-frame-index averages.
    counts = frames_per_index.clamp(min=1).double()
    mse_per_index = (mse_per_index_sum / counts).cpu().tolist()
    overall_mse = float(sum(mse_per_index_sum).item() / int(frames_per_index.sum().item()))
    print(f"[INFO] Overall recon MSE: {overall_mse:.6f}")

    overall_lpips: Optional[float] = None
    lpips_per_index: Optional[list[float]] = None
    if lpips_net is not None:
        lpips_per_index = (lpips_per_index_sum / counts).cpu().tolist()
        overall_lpips = float(sum(lpips_per_index_sum).item() / int(frames_per_index.sum().item()))
        print(f"[INFO] Overall recon LPIPS: {overall_lpips:.6f}")

    # Per-frame CSV + plot.
    write_csv(
        output_dir / "tokenizer_mse_per_frame.csv",
        ["frame_index", "mean_recon_mse"],
        [[t, mse_per_index[t]] for t in range(eval_seq_len)],
    )
    save_line_plot(
        path=output_dir / "tokenizer_mse_per_frame.png",
        title=f"Tokenizer Recon MSE per Frame Index (T={eval_seq_len})",
        x_values=list(range(eval_seq_len)),
        series=[("recon_mse", mse_per_index, (30, 144, 255))],
        y_label="MSE",
    )

    # GT||Recon GIFs.
    gif_paths: list[Path] = []
    print(f"[INFO] Generating {len(gif_data)} reconstruction GIFs...")
    for idx, (gt, rec) in enumerate(gif_data):
        side_by_side = stack_gt_recon_for_gif(gt, rec)
        gif_path = output_dir / f"recon_gif_{idx:03d}_gt_vs_recon.gif"
        save_video_gif(side_by_side, gif_path, fps=opts.gif_fps)
        gif_paths.append(gif_path)
    print("[INFO] GIFs done.")

    # Top-K worst frames as PNGs (sorted descending by MSE).
    worst_sorted = sorted(worst_heap, key=lambda x: -x[0])
    worst_png_paths: list[Path] = []
    for rank, (val, _ctr, batch_step, b, t, gt_frame, recon_frame) in enumerate(worst_sorted):
        png = worst_dir / f"worst_{rank:02d}_step{batch_step}_b{b}_t{t}_mse{val:.4f}.png"
        save_frame_png(gt_frame, recon_frame, png)
        worst_png_paths.append(png)
    print(f"[INFO] Wrote {len(worst_png_paths)} worst-frame PNGs to {worst_dir}")

    # Summary JSON.
    summary = {
        "tokenizer_mse_overall": overall_mse,
        "tokenizer_mse_per_frame_index": mse_per_index,
        "processed_batches": processed,
        "batch_size": opts.batch_size,
        "seq_len": eval_seq_len,
        "image_size": list(tokenizer_cfg.image_size),
        "patch_size": list(tokenizer_cfg.patch_size),
        "num_latent_tokens": tokenizer_cfg.num_latent_tokens,
        "gifs_written": len(gif_paths),
        "worst_frames_written": len(worst_png_paths),
    }
    if overall_lpips is not None:
        summary["tokenizer_lpips_overall"] = overall_lpips
        summary["tokenizer_lpips_per_frame_index"] = lpips_per_index

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # wandb logging.
    wandb.log({
        "eval_tokenizer/mse_overall": overall_mse,
        "eval_tokenizer/processed_batches": processed,
        "eval_tokenizer/seq_len": eval_seq_len,
        "eval_tokenizer/batch_size": opts.batch_size,
        "eval_tokenizer/mse_per_frame": wandb.Table(
            columns=["frame_index", "mean_recon_mse"],
            data=[[t, mse_per_index[t]] for t in range(eval_seq_len)],
        ),
        "eval_tokenizer/mse_per_frame_plot": wandb.Image(
            str(output_dir / "tokenizer_mse_per_frame.png")
        ),
    })
    if overall_lpips is not None:
        wandb.log({"eval_tokenizer/lpips_overall": overall_lpips})

    for idx, gif_path in enumerate(gif_paths):
        wandb.log({
            f"eval_tokenizer/recon_gif_{idx}":
                wandb.Video(str(gif_path), caption=f"Recon {idx}: [GT | Recon]", format="gif"),
        })
    for rank, png_path in enumerate(worst_png_paths):
        wandb.log({
            f"eval_tokenizer/worst_frame_{rank}":
                wandb.Image(str(png_path), caption=png_path.name),
        })

    print("[INFO] Tokenizer evaluation complete.")
    print(f"  recon_mse_overall: {overall_mse:.6f}")
    if overall_lpips is not None:
        print(f"  recon_lpips_overall: {overall_lpips:.6f}")
    print(f"  outputs in: {output_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()
