"""Generate an offline dataset from dm_control for TPU-friendly training.

Usage:
    python generate_dataset.py --domain cheetah --task run --episodes 2000 --seq-len 50
    python generate_dataset.py --domain walker --task walk --episodes 1000 --seq-len 30 --output walker_walk.npz
"""

import argparse
import numpy as np
from dm_control import suite


def main():
    parser = argparse.ArgumentParser(description="Generate offline dataset from dm_control")
    parser.add_argument("--domain", type=str, default="cheetah")
    parser.add_argument("--task", type=str, default="run")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.domain}_{args.task}.npz"

    env = suite.load(args.domain, args.task)
    action_spec = env.action_spec()
    action_dim = action_spec.shape[0]

    all_frames = []
    all_actions = []

    for i in range(args.episodes):
        env.reset()
        ep_frames = []
        ep_actions = []

        # Sinusoidal policy (matches DMControlDataset for diversity)
        freq = np.random.uniform(0.5, 3.0)
        amplitude = np.random.uniform(0.3, 1.0)
        phase_offsets = np.array([j * np.pi / action_dim for j in range(action_dim)])

        for t in range(args.seq_len):
            frame = env.physics.render(
                height=args.image_size,
                width=args.image_size,
                camera_id=args.camera_id,
            )
            ep_frames.append(frame)

            t_norm = t / max(args.seq_len - 1, 1)
            action = amplitude * np.sin(t_norm * 2 * np.pi * freq + phase_offsets)
            action = np.clip(action, action_spec.minimum, action_spec.maximum)
            ep_actions.append(action)

            for _ in range(args.action_repeat):
                time_step = env.step(action)
                if time_step.last():
                    env.reset()
                    break

        all_frames.append(np.stack(ep_frames))    # (T, H, W, 3)
        all_actions.append(np.stack(ep_actions))   # (T, action_dim)

        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{args.episodes} episodes done")

    frames = np.stack(all_frames).astype(np.uint8)   # (N, T, H, W, 3)
    actions = np.stack(all_actions).astype(np.float32)  # (N, T, action_dim)

    print(f"frames:  {frames.shape}, {frames.nbytes / 1e9:.2f} GB")
    print(f"actions: {actions.shape}, {actions.nbytes / 1e6:.1f} MB")

    np.savez_compressed(args.output, frames=frames, actions=actions)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
