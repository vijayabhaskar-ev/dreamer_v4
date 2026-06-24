"""Convert Hansen's cup-catch PNG filmstrips + .pt metadata to project's OfflineDataset npz.

Pixel decoding follows nicklashansen/dreamer4's preprocess_dataset.py:
torchvision.io.read_image, view/permute split, F.interpolate bilinear.
Target resolution 128x128. Pairs decoded pixels with TensorDict actions/rewards
positionally (row index ↔ tile index).

Memory strategy for 32 GB RAM box with tmpfs /tmp:
  - Pre-allocate the final output arrays once at known shapes.
  - Decode each PNG in CHUNKS of 500 frames (not the full 4008-frame strip at once)
    to bound the F.interpolate float32 peak from ~2.4 GB → ~0.3 GB.
  - Save UNCOMPRESSED via np.savez (compressed save uses /tmp tmpfs which doubles RAM).
  - Caller can `gzip ball_in_cup_catch.npz` after — streaming compression is cheap.
"""
from pathlib import Path
import argparse
import gc

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_image

TILE = 224
TARGET = 128
ACTION_DIM = 2
T = 501
CHUNK = 500  # frames per F.interpolate call; tune for memory vs throughput

# Defaults (override with --hansen-root / --out-path). Expected input layout:
#   <hansen-root>/{expert,mixed-small,mixed-large}/  each with cup-catch-*.png + cup-catch.pt
# Source the cup-catch demos from https://github.com/nicklashansen/dreamer4
HANSEN_ROOT = Path("hansen_data")
OUT_PATH = Path("ball_in_cup_catch.npz")

TIERS = [
    ("expert", 20),
    ("mixed-small", 20),
    ("mixed-large", 200),
]


def _png_index(p: Path) -> int:
    return int(p.stem.rsplit("-", 1)[1])


@torch.no_grad()
def _decode_png_chunked(png_path: Path, out_flat: np.ndarray, pos: int) -> int:
    """Decode one PNG filmstrip into out_flat[pos:pos+N] in chunks. Returns new pos."""
    frames = read_image(str(png_path))                  # (3, 224, 224*N) uint8 CHW
    C, H, W_total = frames.shape
    assert H == TILE and W_total % TILE == 0, f"bad shape {tuple(frames.shape)} for {png_path}"
    N = W_total // TILE
    # Reshape to (N, 3, 224, 224) without copying the underlying storage
    frames = frames.view(C, TILE, N, TILE).permute(2, 0, 1, 3)

    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        chunk = frames[start:end].contiguous().to(torch.float32) / 255.0  # (k, 3, 224, 224) #TODO not needed to divide by 255. redudant. may be do more reseach on this
        chunk = F.interpolate(chunk, size=(TARGET, TARGET),
                              mode="bilinear", align_corners=False)
        chunk = (chunk.clamp_(0.0, 1.0) * 255.0).to(torch.uint8)          # (k, 3, 128, 128)
        # Permute to HWC and write directly into the pre-allocated slice
        out_flat[pos + start : pos + end] = chunk.permute(0, 2, 3, 1).contiguous().numpy()
        del chunk
    del frames
    return pos + N


def fill_tier_into_slice(tier_dir: Path, n_eps: int,
                          frames_slice, actions_slice, rewards_slice, dones_slice):
    expected_total = n_eps * T
    pngs = sorted(tier_dir.glob("cup-catch-*.png"), key=_png_index)

    out_flat = frames_slice.reshape(n_eps * T, TARGET, TARGET, 3)
    pos = 0
    for png in pngs:
        pos = _decode_png_chunked(png, out_flat, pos)
        gc.collect()
    assert pos == expected_total, f"wrote {pos} frames, expected {expected_total}"

    td = torch.load(tier_dir / "cup-catch.pt", map_location="cpu", weights_only=False)
    action = td["action"][:, :ACTION_DIM].numpy().astype(np.float32)
    reward = td["reward"].numpy().astype(np.float32)
    episode = td["episode"].numpy()
    del td

    action = np.nan_to_num(action, nan=0.0)
    reward = np.nan_to_num(reward, nan=0.0)
    assert episode.max() + 1 == n_eps and (episode == 0).sum() == T

    # TODO Hansen stores action/reward under convention B: row t = (frame_t,
    # action_that_led_TO_frame_t, reward_from_that_transition). Evidence:
    # action[0] and reward[0] of each episode are NaN (no action led to the
    # reset state). But dynamics/trainer.py:1909 expects convention A:
    # action[t] is taken AT frame[t] and causes frame[t+1]. This is a 1-step
    # misalignment that affects the dynamics + imagination phases (NOT the
    # tokenizer phase, which ignores actions). To fix: per episode, shift
    # action and reward LEFT by 1 and zero-fill the last step. Empirically
    # verify with a dm_control replay before applying. Re-run conversion
    # AFTER successful tokenizer training and BEFORE starting dynamics.

    actions_slice[:] = action.reshape(n_eps, T, ACTION_DIM)
    rewards_slice[:] = reward.reshape(n_eps, T)
    dones_slice[:] = 0.0
    dones_slice[:, -1] = 1.0


def main():
    parser = argparse.ArgumentParser(
        description="Convert Hansen cup-catch demos (PNG filmstrips + .pt) → OfflineDataset .npz")
    parser.add_argument("--hansen-root", type=Path, default=HANSEN_ROOT,
                        help="Dir containing {expert,mixed-small,mixed-large}/ subdirs of Hansen cup-catch demos")
    parser.add_argument("--out-path", type=Path, default=OUT_PATH, help="Output .npz path")
    args = parser.parse_args()

    total_eps = sum(n for _, n in TIERS)
    print(f"Pre-allocating arrays for {total_eps} episodes × {T} steps × {TARGET}×{TARGET}×3...")
    frames = np.empty((total_eps, T, TARGET, TARGET, 3), dtype=np.uint8)
    actions = np.empty((total_eps, T, ACTION_DIM), dtype=np.float32)
    rewards = np.empty((total_eps, T), dtype=np.float32)
    dones = np.empty((total_eps, T), dtype=np.float32)
    print(f"  frames pre-alloc: {frames.nbytes / 1024**3:.2f} GB")

    offset = 0
    for tier_name, n_eps in TIERS:
        print(f"\n=== {tier_name} ({n_eps} episodes) ===")
        sl = slice(offset, offset + n_eps)
        fill_tier_into_slice(
            args.hansen_root / tier_name, n_eps,
            frames[sl], actions[sl], rewards[sl], dones[sl],
        )
        tier_r = rewards[sl]
        print(f"  reward>0 frac = {(tier_r > 0).mean():.4f}, "
              f"per-episode return mean = {tier_r.sum(axis=1).mean():.1f}")
        offset += n_eps
        gc.collect()

    print(f"\nFinal:")
    print(f"  frames  {frames.shape} {frames.dtype}")
    print(f"  actions {actions.shape} {actions.dtype}")
    print(f"  rewards {rewards.shape} {rewards.dtype}, >0 frac = {(rewards > 0).mean():.4f}")
    print(f"  dones   {dones.shape} {dones.dtype}")

    print(f"\nSaving (uncompressed) to {args.out_path}...")
    print(f"  (Expected ~{frames.nbytes / 1024**3:.1f} GB on disk; gzip afterward for transport.)")
    np.savez(args.out_path, frames=frames, actions=actions, rewards=rewards, dones=dones)
    print("done.")


if __name__ == "__main__":
    main()
