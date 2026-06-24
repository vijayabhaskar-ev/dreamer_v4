"""Dreamer V4 — REAL-ENVIRONMENT policy evaluation (closed-loop dm_control).

Every other evaluator in this repo (``evaluate_dynamics``, ``evaluate_agent``)
scores the agent on the offline dataset or *inside imagination* — i.e. the world
model grades its own student. This script closes the loop: it runs the trained
policy in the **live** ``dm_control`` environment and measures the only
ground-truth signal — real episode return — comparing the Phase-3 imagination
policy against the Phase-2 BC policy and a random-action floor.

Acting recipe (NO denoising): the environment supplies the real next frame, so
the world model is used as a *belief-state encoder*, not a predictor. Per step::

    z = encode_frames(real_frame)            # one frame; tokenizer is per-frame
    z = add_noise(z, tau=1-tau_ctx)          # corrupt to the deployment signal level (~0.9)
    h = model(<=16-frame window, use_agent_tokens=True).agent_out[:, -1]
    action = policy_head.act(h, readout)     # mean (default) | argmax | sample

Setup (one-time):
    pip install dm_control mujoco            # already present in this repo's conda env
    export MUJOCO_GL=egl                     # GPU headless render (osmesa for CPU-only hosts)

Examples:
    # CPU smoke (1 short episode, all 3 policies):
    python -m dynamics.evaluate_env \
        --phase2-ckpt checkpoints-phase2-rerun-optionA/dynamics/dynamics_epoch_040.pt \
        --phase3-ckpt checkpoints-phase3-imagination/dynamics/epoch_15.pt \
        --tokenizer-ckpt checkpoints-iter46-extended-550ep/tokenizer/tokenizer_epoch_500.pt \
        --num-episodes 1 --max-steps 20 --device cpu \
        --output-dir /tmp/env-smoke --wandb-disabled

    # Full run (20 seeded episodes, GPU):
    python -m dynamics.evaluate_env \
        --phase2-ckpt .../dynamics_epoch_040.pt --phase3-ckpt .../epoch_15.pt \
        --tokenizer-ckpt .../tokenizer_epoch_500.pt \
        --num-episodes 20 --device cuda --output-dir evaluation/env
"""

from __future__ import annotations

import _env_setup  # noqa: F401  (must precede torch; sets env vars / sys.path)

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb

from .flow_matching import add_noise
from .trainer import DynamicsTrainer, DynamicsTrainingConfig
from .evaluate_dynamics import (
    safe_torch_load,
    resolve_device,
    load_tokenizer_config_from_ckpt,
    load_dynamics_config_from_ckpt,
    save_video_gif,
    save_line_plot,
    write_csv,
)
from .evaluate_agent import peek_checkpoint, _pearson, _roc_auc, _clean
from tokenizer.dataset import _split_dmc_task_name
from wandb_utils import init_wandb, add_wandb_args  # noqa: F401  (kept for parity; init inlined)


# ─────────────────────────────────────────────────────────────────────────
# Dependency / rendering guard
# ─────────────────────────────────────────────────────────────────────────
def _require_dmc():
    """Import dm_control/mujoco with a headless GL backend. Returns the suite module.

    MUJOCO_GL must be set BEFORE mujoco is first imported. If unset we default to
    ``egl`` (GPU headless, preferred on this CUDA box); override by exporting
    ``MUJOCO_GL=osmesa`` for CPU-only hosts.
    """
    if not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "egl"
        print("[env] MUJOCO_GL was unset → defaulting to 'egl' (GPU headless). "
              "Export MUJOCO_GL=osmesa for CPU-only hosts.")
    try:
        import mujoco  # noqa: F401
        from dm_control import suite
    except ImportError as e:
        raise SystemExit(
            "dm_control/mujoco not importable. Install them:\n"
            "    pip install dm_control mujoco\n"
            "and set a headless GL backend:\n"
            "    export MUJOCO_GL=egl     # GPU (preferred), or osmesa for CPU-only\n"
            f"(original error: {e})")
    return suite


# ─────────────────────────────────────────────────────────────────────────
# Environment wrapper (mirrors generate_dataset.py render/preprocess exactly)
# ─────────────────────────────────────────────────────────────────────────
class DMCEnvWrapper:
    """Single dm_control env producing 128x128 RGB frames matching the dataset.

    Frame preprocessing reproduces ``tokenizer/dataset.py`` exactly
    (HWC uint8 → CHW float[0,1]) so the policy sees in-distribution inputs.
    """

    def __init__(self, task: str = "ball_in_cup_catch", image_size: int = 128,
                 camera_id: int = 0, action_repeat: int = 2, seed: int = 0):
        suite = _require_dmc()
        domain, task_name = _split_dmc_task_name(task)
        # dm_control seeds at construction → re-create per seed for reproducibility.
        self.env = suite.load(domain, task_name, task_kwargs={"random": int(seed)})
        self.action_spec = self.env.action_spec()
        self.action_dim = int(self.action_spec.shape[0])
        self.image_size = image_size
        self.camera_id = camera_id
        self.action_repeat = action_repeat

    def _render(self) -> np.ndarray:
        return self.env.physics.render(
            height=self.image_size, width=self.image_size, camera_id=self.camera_id)

    def reset(self) -> np.ndarray:
        self.env.reset()
        return self._render()  # (H, W, 3) uint8

    @staticmethod
    def preprocess(pixels: np.ndarray) -> torch.Tensor:
        # Per-frame equivalent of the dataset's frame normalization
        # (tokenizer/dataset.py:245 does the batched permute(0,3,1,2)/255):
        # HWC uint8 → CHW float[0,1].
        return torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 255.0

    def clip_action(self, a: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(a, dtype=np.float32),
                       self.action_spec.minimum, self.action_spec.maximum)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Apply ``action`` for ``action_repeat`` inner steps; render next frame.

        Returns (next_pixels HWC uint8, summed_reward, done).
        """
        action = self.clip_action(action)
        step_reward, done = 0.0, False
        for _ in range(self.action_repeat):
            ts = self.env.step(action)
            step_reward += (ts.reward or 0.0)  # ts.reward is None on the reset step
            if ts.last():
                done = True
                break
        return self._render(), float(step_reward), done


# ─────────────────────────────────────────────────────────────────────────
# Per-episode result + aggregation
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class EpisodeResult:
    seed: int
    return_: float
    length: int
    success: bool
    caught: bool
    pred_rewards: List[float] = field(default_factory=list)
    actual_rewards: List[float] = field(default_factory=list)
    pred_continue: List[float] = field(default_factory=list)
    dones: List[float] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────
# World-model + heads setup (Phase-2 base, optional Phase-3 policy override)
# ─────────────────────────────────────────────────────────────────────────
def setup_world_model(opts) -> Tuple[DynamicsTrainer, object, torch.device, int]:
    """Load the Phase-2 world model + reward/continue/policy heads (the BC base).

    The Phase-3 policy is layered on later via ``load_policy_head``; the WM and
    reward/continue heads are shared (frozen, Phase-2) across BC and Phase-3.
    """
    device = resolve_device(opts.device)
    tokenizer_cfg = load_tokenizer_config_from_ckpt(opts.tokenizer_ckpt, device)
    dynamics_cfg = load_dynamics_config_from_ckpt(opts.phase2_ckpt, device)
    dynamics_cfg.validate_against_tokenizer(tokenizer_cfg)

    inferred, has_heads = peek_checkpoint(opts.phase2_ckpt, device)
    if not has_heads:
        raise RuntimeError(
            f"{opts.phase2_ckpt} has no reward/continue/policy heads — it is not a "
            "Phase-2 (agent) checkpoint. Point --phase2-ckpt at a Phase-2 checkpoint.")
    num_tasks = opts.num_tasks
    if inferred is not None and inferred != opts.num_tasks:
        print(f"[WARN] --num-tasks={opts.num_tasks} but checkpoint has num_tasks={inferred}; "
              "using the checkpoint value.")
        num_tasks = inferred

    eval_train_cfg = DynamicsTrainingConfig(
        epochs=1, batch_size=1, amp=opts.amp, device=str(device),
        train_heads=True,  # keep eager (DynamicsTrainer only torch.compiles when False)
        log_model_stats=False, log_memory=False,
        log_interval=10_000, log_model_stats_interval=10_000,
    )
    trainer = DynamicsTrainer(
        dynamics_cfg=dynamics_cfg, tokenizer_cfg=tokenizer_cfg,
        training_cfg=eval_train_cfg, tokenizer_ckpt=opts.tokenizer_ckpt,
    )
    # CRITICAL: enable agent tokens BEFORE load_checkpoint, else agent_embedding.*
    # keys are stripped and agent_out is None on forward.
    trainer.model.enable_agent_tokens(num_tasks=num_tasks)
    trainer.model.agent_embedding.to(device)
    trainer.load_checkpoint(opts.phase2_ckpt, strict=False)

    trainer.model.to(device).eval()
    trainer.tokenizer.eval()
    trainer.reward_head.eval()
    trainer.continue_head.eval()
    trainer.policy_head.eval()
    trainer.tokenizer_cfg = tokenizer_cfg
    return trainer, dynamics_cfg, device, num_tasks


def snapshot_policy(trainer) -> Dict[str, torch.Tensor]:
    """Clone the current policy_head weights (so BC↔Phase-3 don't cross-contaminate)."""
    return {k: v.detach().clone() for k, v in trainer.policy_head.state_dict().items()}


def load_policy_head(trainer, phase3_ckpt: str, device: torch.device) -> None:
    state = safe_torch_load(phase3_ckpt, device)
    if "policy_head" not in state:
        raise RuntimeError(f"{phase3_ckpt} has no 'policy_head' state — not a Phase-3 checkpoint.")
    trainer.policy_head.load_state_dict(state["policy_head"], strict=False)
    trainer.policy_head.eval()


def restore_policy_head(trainer, sd: Dict[str, torch.Tensor]) -> None:
    trainer.policy_head.load_state_dict(sd, strict=False)
    trainer.policy_head.eval()


# ─────────────────────────────────────────────────────────────────────────
# Closed-loop rollout
# ─────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_episode(env: DMCEnvWrapper, trainer, dynamics_cfg, *, is_random: bool,
                readout: str, device: torch.device, max_steps: int,
                success_threshold: float, rng: np.random.Generator,
                collect_frames: bool) -> Tuple[EpisodeResult, Optional[List[np.ndarray]]]:
    """One closed-loop episode. Returns (EpisodeResult, optional rendered frames)."""
    tau_ctx_val = 1.0 - dynamics_cfg.tau_ctx          # ~0.9 deployment signal level
    d_val = 1.0 / dynamics_cfg.K_max
    cw = int(dynamics_cfg.context_length)             # 16

    pixels = env.reset()
    frames_rgb: List[np.ndarray] = [pixels] if collect_frames else []

    z_frames: List[torch.Tensor] = []   # each (1,1,S_z,D_lat)
    act_taken: List[np.ndarray] = []    # each (A,)
    pred_r_list, actual_r_list, pred_c_list, done_list = [], [], [], []
    total_return, length, done = 0.0, 0, False

    limit = max_steps if max_steps > 0 else 10_000
    for t in range(limit):
        if is_random:
            action = rng.uniform(env.action_spec.minimum,
                                 env.action_spec.maximum).astype(np.float32)
            pred_r = pred_c = None
        else:
            frame = env.preprocess(pixels).to(device)[None, None]      # (1,1,3,H,W)
            z_clean = trainer.model.encode_frames(frame)               # (1,1,S_z,D_lat)
            tau1 = torch.full((1, 1), tau_ctx_val, device=device)
            z_noi, _ = add_noise(z_clean, tau1)
            z_frames.append(z_noi)

            W = min(len(z_frames), cw)
            z_win = torch.cat(z_frames[-W:], dim=1)                    # (1,W,S_z,D_lat)
            tau_win = torch.full((1, W), tau_ctx_val, device=device)
            d_win = torch.full((1, W), d_val, device=device)
            if W > 1:
                a_hist = np.stack(act_taken[-(W - 1):])               # (W-1,A)
                act_win = torch.from_numpy(a_hist).float().to(device)[None]  # (1,W-1,A)
            else:
                act_win = None

            out = trainer.model(z_win, act_win, tau_win, d_win, use_agent_tokens=True)
            if out.agent_out is None:
                raise RuntimeError("agent_out is None — agent tokens not active "
                                   "(enable_agent_tokens must precede load_checkpoint).")
            h = out.agent_out[:, -1]                                   # (1, D_embed)
            pred_r = float(trainer.reward_head.predict(h).reshape(-1)[0].item())
            pred_c = float(trainer.continue_head.predict(h).reshape(-1)[0].item())
            act_t = trainer.policy_head.act(h, readout=readout)        # (1,A)
            action = act_t.reshape(-1).float().cpu().numpy()
            act_taken.append(env.clip_action(action))                 # store the action actually applied

        pixels, reward, done = env.step(action)
        total_return += reward
        length += 1
        if collect_frames:
            frames_rgb.append(pixels)
        if pred_r is not None:
            pred_r_list.append(pred_r)
            actual_r_list.append(reward)
            pred_c_list.append(pred_c)
            done_list.append(1.0 if done else 0.0)
        if done:
            break

    res = EpisodeResult(
        seed=-1, return_=total_return, length=length,
        success=(total_return >= success_threshold), caught=(total_return > 0.0),
        pred_rewards=pred_r_list, actual_rewards=actual_r_list,
        pred_continue=pred_c_list, dones=done_list,
    )
    return res, (frames_rgb if collect_frames else None)


def run_policy(name: str, trainer, dynamics_cfg, opts, device, *, is_random: bool):
    """Run ``--num-episodes`` seeded episodes for one policy. Returns (results, gif_frames)."""
    results: List[EpisodeResult] = []
    gif_frames: List[List[np.ndarray]] = []
    for i in range(opts.num_episodes):
        seed = opts.seed_base + i
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        env = DMCEnvWrapper(task=opts.task, image_size=opts.image_size,
                            camera_id=opts.camera_id, action_repeat=opts.action_repeat,
                            seed=seed)
        collect = i < opts.max_gifs
        res, frames = run_episode(
            env, trainer, dynamics_cfg, is_random=is_random, readout=opts.readout,
            device=device, max_steps=opts.max_steps, success_threshold=opts.success_threshold,
            rng=rng, collect_frames=collect)
        res.seed = seed
        results.append(res)
        if frames is not None:
            gif_frames.append(frames)
        print(f"  [{name}] ep {i + 1}/{opts.num_episodes} (seed={seed}): "
              f"return={res.return_:.1f} len={res.length} caught={res.caught}")
    return results, gif_frames


# ─────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────
def aggregate(results: List[EpisodeResult], is_random: bool) -> Dict:
    rets = np.array([r.return_ for r in results], dtype=np.float64)
    lens = np.array([r.length for r in results], dtype=np.float64)
    agg = {
        "num_episodes": len(results),
        "return_mean": float(rets.mean()) if len(rets) else float("nan"),
        "return_std": float(rets.std()) if len(rets) else float("nan"),
        "return_min": float(rets.min()) if len(rets) else float("nan"),
        "return_max": float(rets.max()) if len(rets) else float("nan"),
        "success_rate": float(np.mean([r.success for r in results])) if results else float("nan"),
        "caught_rate": float(np.mean([r.caught for r in results])) if results else float("nan"),
        "mean_length": float(lens.mean()) if len(lens) else float("nan"),
    }
    if not is_random:
        pr = torch.tensor([v for r in results for v in r.pred_rewards], dtype=torch.double)
        ar = torch.tensor([v for r in results for v in r.actual_rewards], dtype=torch.double)
        pc = torch.tensor([v for r in results for v in r.pred_continue], dtype=torch.double)
        dn = torch.tensor([v for r in results for v in r.dones], dtype=torch.double)
        if pr.numel() > 0:
            agg["reward_calibration"] = {
                "n": int(pr.numel()),
                "reward_mae": float((pr - ar).abs().mean().item()),
                "reward_mse": float(((pr - ar) ** 2).mean().item()),
                "reward_pearson": _pearson(pr, ar),
                "reward_event_auc": _roc_auc(pr, (ar > 0).double()),  # predicted-reward separates reward>0
            }
            cont_target = 1.0 - dn  # continue = 1 - done
            eps = 1e-6
            pc_c = pc.clamp(eps, 1 - eps)
            agg["continue_calibration"] = {
                "bce": float(-(cont_target * pc_c.log()
                               + (1 - cont_target) * (1 - pc_c).log()).mean().item()),
                "brier": float(((pc - cont_target) ** 2).mean().item()),
                "done_positive_count": int(dn.sum().item()),
            }
    return agg


# ─────────────────────────────────────────────────────────────────────────
# Outputs
# ─────────────────────────────────────────────────────────────────────────
_COLORS = {"phase3": (37, 99, 235), "bc": (220, 38, 38), "random": (107, 114, 128)}


def write_outputs(out_dir: Path, opts, dynamics_cfg, per_policy: Dict[str, Dict],
                  gif_paths: List[Path]) -> Dict:
    # Per-episode CSV
    rows = []
    for name, blob in per_policy.items():
        for r in blob["results"]:
            rows.append([name, r.seed, round(r.return_, 4), r.length, int(r.success), int(r.caught)])
    write_csv(out_dir / "episodes.csv",
              ["policy", "seed", "return", "length", "success", "caught"], rows)

    # Return-distribution plot (sorted per-episode returns per policy)
    series = []
    max_n = max((len(b["results"]) for b in per_policy.values()), default=0)
    for name, blob in per_policy.items():
        vals = sorted(r.return_ for r in blob["results"])
        if vals:
            series.append((name, vals, _COLORS.get(name, (0, 0, 0))))
    if series and max_n > 1:
        try:
            save_line_plot(out_dir / "return_distribution.png",
                           "Per-episode return (sorted)", list(range(max_n)), series, "return")
        except Exception as e:  # plotting is non-critical
            print(f"[WARN] return_distribution plot failed: {e}")

    # Comparison table + summary
    agg = {name: blob["agg"] for name, blob in per_policy.items()}
    deltas = {}
    if "phase3" in agg and "bc" in agg:
        deltas["phase3_minus_bc"] = agg["phase3"]["return_mean"] - agg["bc"]["return_mean"]
    if "bc" in agg and "random" in agg:
        deltas["bc_minus_random"] = agg["bc"]["return_mean"] - agg["random"]["return_mean"]
    if "phase3" in agg and "random" in agg:
        deltas["phase3_minus_random"] = agg["phase3"]["return_mean"] - agg["random"]["return_mean"]

    try:
        import dm_control as _dmc
        import mujoco as _mj
        dmc_ver, mj_ver = getattr(_dmc, "__version__", "?"), getattr(_mj, "__version__", "?")
    except Exception:
        dmc_ver = mj_ver = "?"

    summary = {
        "task": opts.task,
        "policies": {name: _clean(a) for name, a in agg.items()},
        "deltas_return_mean": deltas,
        "data_facts": {
            "num_episodes": opts.num_episodes,
            "seeds": [opts.seed_base + i for i in range(opts.num_episodes)],
            "max_steps": opts.max_steps,
            "image_size": opts.image_size,
            "camera_id": opts.camera_id,
            "action_repeat": opts.action_repeat,
            "readout": opts.readout,
            "success_threshold": opts.success_threshold,
            "tau_ctx_val": 1.0 - dynamics_cfg.tau_ctx,
            "K_max": dynamics_cfg.K_max,
            "context_length": dynamics_cfg.context_length,
            "phase2_ckpt": opts.phase2_ckpt,
            "phase3_ckpt": opts.phase3_ckpt,
            "tokenizer_ckpt": opts.tokenizer_ckpt,
            "dm_control_version": dmc_ver,
            "mujoco_version": mj_ver,
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
        },
        "opts": _clean(vars(opts)),
        "gifs": [str(p) for p in gif_paths],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def log_wandb(opts, out_dir: Path, summary: Dict, gif_paths: List[Path]):
    if opts.wandb_disabled:
        return
    flat = {}
    for name, agg in summary["policies"].items():
        for k, v in agg.items():
            if isinstance(v, (int, float)):
                flat[f"env/{name}/{k}"] = v
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, (int, float)):
                        flat[f"env/{name}/{k}/{kk}"] = vv
    for k, v in summary["deltas_return_mean"].items():
        flat[f"env/delta/{k}"] = v
    wandb.log(flat)
    images = {f"env/plot/{p.stem}": wandb.Image(str(p)) for p in sorted(out_dir.glob("*.png"))}
    if images:
        wandb.log(images)
    vids = {f"env/{p.stem}": wandb.Video(str(p), fps=opts.gif_fps, format="gif") for p in gif_paths}
    if vids:
        wandb.log(vids)
    # comparison table
    cols = ["policy", "return_mean", "return_std", "success_rate", "caught_rate", "mean_length"]
    table = wandb.Table(columns=cols)
    for name, agg in summary["policies"].items():
        table.add_data(name, agg.get("return_mean"), agg.get("return_std"),
                       agg.get("success_rate"), agg.get("caught_rate"), agg.get("mean_length"))
    wandb.log({"env/comparison": table})
    wandb.finish()


# ─────────────────────────────────────────────────────────────────────────
# CLI + main
# ─────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dreamer V4 real-environment policy evaluation")
    p.add_argument("--phase2-ckpt", required=True, help="Phase-2 (WM + heads, BC policy) checkpoint")
    p.add_argument("--phase3-ckpt", required=True, help="Phase-3 (imagination policy) checkpoint")
    p.add_argument("--tokenizer-ckpt", required=True)
    p.add_argument("--task", default="ball_in_cup_catch")
    p.add_argument("--action-dim", type=int, default=2, help="Sanity only; true dim from action_spec.")
    p.add_argument("--num-tasks", type=int, default=1, help="Auto-inferred from ckpt; warns on mismatch.")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--seed-base", type=int, default=0, help="seeds = range(seed_base, seed_base+N)")
    p.add_argument("--max-steps", type=int, default=0, help="0 = run to the env time limit.")
    p.add_argument("--success-threshold", type=float, default=1.0,
                   help="Episode 'success' if return >= this (sparse ball_in_cup).")
    p.add_argument("--readout", choices=["mean", "argmax", "sample"], default="sample",
                   help="Action decode at deployment. 'sample' (default): draw ~policy — "
                        "paper-faithful and the empirical winner on cup-catch (47%% caught vs "
                        "17%% for any deterministic readout); PMPO optimizes the SAMPLED policy, "
                        "so its return lives in the distribution, not the mode. 'mean': "
                        "probability-weighted bin centre — best DETERMINISTIC/reproducible "
                        "readout (lower teacher-forced MSE than argmax) but ~argmax in closed "
                        "loop. 'argmax': discrete mode (legacy baseline).")
    p.add_argument("--stochastic", action="store_true",
                   help="DEPRECATED alias for --readout sample (overrides --readout if set).")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--action-repeat", type=int, default=2)
    p.add_argument("--policies", default="phase3,bc,random",
                   help="Comma list subset of {phase3,bc,random}.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--output-dir", default="evaluation/env")
    p.add_argument("--max-gifs", type=int, default=3, help="Collect/render this many episodes as GIFs per policy.")
    p.add_argument("--gif-fps", type=int, default=8)
    add_wandb_args(p, default_project="dreamer-v4-env-eval")
    return p


def main(args: Optional[List[str]] = None) -> None:
    opts = build_parser().parse_args(args)
    if opts.stochastic:  # deprecated alias wins for back-compat with saved commands
        opts.readout = "sample"
    _require_dmc()  # set MUJOCO_GL + assert importable BEFORE any heavy work
    out_dir = Path(opts.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not opts.wandb_disabled:
        mode = "offline" if opts.wandb_offline else "online"
        wandb.init(project=opts.wandb_project, entity=opts.wandb_entity,
                   name=opts.wandb_name or f"env_eval_{opts.task}_{datetime.utcnow():%Y%m%d_%H%M%S}",
                   mode=mode, config=vars(opts))

    requested = [s.strip() for s in opts.policies.split(",") if s.strip()]
    needs_model = any(p in ("phase3", "bc") for p in requested)

    trainer = dynamics_cfg = device = None
    if needs_model:
        trainer, dynamics_cfg, device, num_tasks = setup_world_model(opts)
        print(f"[setup] device={device} num_tasks={num_tasks} "
              f"tau_ctx_val={1.0 - dynamics_cfg.tau_ctx:.3f} context_length={dynamics_cfg.context_length} "
              f"action_repeat={opts.action_repeat} readout={opts.readout}")
    else:
        device = resolve_device(opts.device)

    per_policy: Dict[str, Dict] = {}
    gif_paths: List[Path] = []
    bc_sd = snapshot_policy(trainer) if needs_model else None

    for name in requested:
        print(f"\n=== policy: {name} ===")
        is_random = (name == "random")
        if name == "bc":
            restore_policy_head(trainer, bc_sd)          # Phase-2 BC policy weights
        elif name == "phase3":
            load_policy_head(trainer, opts.phase3_ckpt, device)  # override with Phase-3
        # random: no model touch
        results, gif_frames = run_policy(name, trainer, dynamics_cfg, opts, device, is_random=is_random)
        per_policy[name] = {"results": results, "agg": aggregate(results, is_random)}
        # save GIFs (env frames already RGB → no tokenizer decode)
        for gi, frames in enumerate(gif_frames[: opts.max_gifs]):
            vid = torch.stack([DMCEnvWrapper.preprocess(f) for f in frames])  # (T,3,H,W)
            gp = out_dir / f"rollout_{name}_{gi}.gif"
            try:
                save_video_gif(vid, gp, fps=opts.gif_fps)
                gif_paths.append(gp)
            except Exception as e:
                print(f"[WARN] gif save failed ({gp.name}): {e}")

    summary = write_outputs(out_dir, opts, dynamics_cfg, per_policy, gif_paths)
    log_wandb(opts, out_dir, summary, gif_paths)

    # console digest
    print("\n=== Real-env evaluation (return mean ± std) ===")
    for name, agg in summary["policies"].items():
        print(f"  {name:>8}: {agg['return_mean']:7.2f} ± {agg['return_std']:5.2f}  "
              f"success={agg['success_rate']:.2f} caught={agg['caught_rate']:.2f} "
              f"len={agg['mean_length']:.0f}")
    for k, v in summary["deltas_return_mean"].items():
        print(f"  Δ {k}: {v:+.2f}")
    print(f"  outputs -> {out_dir}")


if __name__ == "__main__":
    main()
