"""ImaginationTrainer — Phase 3 training loop (Dreamer V4, Section 3.3).

Loads a Phase 2 checkpoint (dynamics + reward/continue/policy heads), freezes
the world model, deepcopies the policy into a frozen prior for the PMPO
reverse-KL term, and trains policy_head + a fresh value_head via AdamW on
imagined rollouts. Each training step encodes context frames into latents,
calls imagine_rollout() to produce a trajectory, and computes value_loss +
pmpo_policy_loss against TD(λ) targets.
"""

from __future__ import annotations

import copy
import gc
import math
import pickle
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

from dynamics.config import DynamicsConfig
from dynamics.dynamic_model import DynamicsModel
from tokenizer.config import TokenizerConfig
from tokenizer.tokenizer import MaskedAutoencoderTokenizer
from tokenizer.metrics import MetricsBuffer, ThroughputTracker
from heads import RewardHead, ContinueHead, PolicyHead
from device_utils import (
    get_device,
    make_grad_scaler,
    save_checkpoint as save_ckpt,
    is_master,
)

from .config import ImaginationConfig
from .algorithms import (
    compute_lambda_returns,
    compute_advantages,
    value_loss,
    pmpo_policy_loss,
)
from .rollout import imagine_rollout


# ════════════════════════════════════════════════════════════════════════
# Trainer
# ════════════════════════════════════════════════════════════════════════

class ImaginationTrainer:
    """Phase 3 trainer — RL on imagined rollouts.

    Unlike DynamicsTrainer, this class:
      - Freezes the world model (no dynamics loss)
      - Has only 2 trainable modules: policy_head and value_head
      - Uses a single AdamW param group per module (simpler than Phase 2)
      - Runs no curriculum, no alternating batch lengths, no MTP
    """

    def __init__(
        self,
        dynamics_cfg: DynamicsConfig,
        tokenizer_cfg: TokenizerConfig,
        imagination_cfg: ImaginationConfig,
        tokenizer_ckpt: str,
        phase2_ckpt: str,
        num_tasks: int = 1,
    ):
        self.dynamics_cfg = dynamics_cfg
        self.tokenizer_cfg = tokenizer_cfg
        self.cfg = imagination_cfg
        self.device = get_device(imagination_cfg.device)

        # ── Tokenizer (frozen) ──────────────────────────────────────
        self.tokenizer = MaskedAutoencoderTokenizer(tokenizer_cfg).to(self.device)
        self._load_tokenizer_checkpoint(tokenizer_ckpt)
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

        # ── Dynamics model (frozen, with agent tokens enabled) ─────
        self.model = DynamicsModel(dynamics_cfg, self.tokenizer).to(self.device)
        self.model.enable_agent_tokens(num_tasks=num_tasks)
        self.model.agent_embedding.to(self.device)

        head_input_dim = dynamics_cfg.embed_dim

        # ── Reward / continue heads (frozen, from Phase 2) ─────────
        self.reward_head = RewardHead(
            latent_dim=head_input_dim,
            hidden_dim=dynamics_cfg.head_hidden_dim,
            num_bins=dynamics_cfg.num_reward_bins,
            num_layers=dynamics_cfg.head_num_layers,
            mtp_length=0,  # Phase 3 uses only offset 0
        ).to(self.device)
        self.continue_head = ContinueHead(
            latent_dim=head_input_dim,
            hidden_dim=dynamics_cfg.head_hidden_dim,
            num_layers=dynamics_cfg.head_num_layers,
            mtp_length=0,
        ).to(self.device)

        # ── Policy head (trainable, loaded from Phase 2) ───────────
        self.policy_head = PolicyHead(
            latent_dim=head_input_dim,
            action_dim=dynamics_cfg.action_dim,
            hidden_dim=dynamics_cfg.head_hidden_dim,
            num_layers=dynamics_cfg.head_num_layers,
            mtp_length=0,
        ).to(self.device)

        # ── Value head (trainable, NEW — not in Phase 2 ckpt) ──────
        # Architecturally identical to RewardHead: symexp twohot over 255 bins.
        self.value_head = RewardHead( #TODO Rename it to a generic class name to avoid confusion
            latent_dim=head_input_dim,
            hidden_dim=imagination_cfg.value_hidden_dim,
            num_bins=imagination_cfg.value_num_bins,
            num_layers=imagination_cfg.value_num_layers,
            mtp_length=0,
        ).to(self.device)

        # ── Load Phase 2 checkpoint ────────────────────────────────
        self._load_phase2_checkpoint(phase2_ckpt)

        # ── Freeze everything except policy_head + value_head ──────
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.reward_head.eval()
        for p in self.reward_head.parameters():
            p.requires_grad_(False)
        self.continue_head.eval()
        for p in self.continue_head.parameters():
            p.requires_grad_(False)

        # ── Policy prior (frozen deepcopy of policy_head) ──────────
        # Used by PMPO for the reverse-KL regularization term.
        self.policy_prior = copy.deepcopy(self.policy_head).to(self.device)
        self.policy_prior.eval()
        for p in self.policy_prior.parameters():
            p.requires_grad_(False)

        # ── Optimizer (policy_head + value_head only) ──────────────
        self._build_optimizer()

        self.scaler = make_grad_scaler(self.device, enabled=imagination_cfg.amp)
        self.global_step = 0
        self.metrics_buffer = MetricsBuffer(window=imagination_cfg.log_smooth_window)
        self.throughput_tracker = ThroughputTracker()

    # ──────────────────────────────────────────────────────────────
    # Checkpoint I/O
    # ──────────────────────────────────────────────────────────────

    def _load_tokenizer_checkpoint(self, path: str) -> None:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            state = torch.load(path, map_location="cpu", weights_only=False)
        tok_state = state["model"] if "model" in state else state
        self.tokenizer.load_state_dict(tok_state, strict=False)
        if is_master():
            print(f"[INFO] Loaded tokenizer from {path}")

    def _load_phase2_checkpoint(self, path: str) -> None:
        """Load dynamics + reward/continue/policy heads from Phase 2 ckpt.

        Value head is NOT in the Phase 2 checkpoint — it starts from
        zero-initialized weights (inherited from RewardHead's zero init).
        """
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            state = torch.load(path, map_location="cpu", weights_only=False)

        # Dynamics model weights
        self.model.load_state_dict(state["model"], strict=False)

        # Heads (non-strict — Phase 2 may have different MTP length)
        for name, head in [
            ("reward_head", self.reward_head),
            ("continue_head", self.continue_head),
            ("policy_head", self.policy_head),
        ]:
            if name in state:
                try:
                    head.load_state_dict(state[name], strict=False)
                except RuntimeError as e:
                    if is_master():
                        print(f"[WARN] Could not load {name}: {e}")

        if is_master():
            print(f"[INFO] Loaded Phase 2 checkpoint from {path}")
            print(f"       value_head is fresh (zero-initialized)")

    def save_checkpoint(self, path: str, epoch: int) -> None:
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "imagination_cfg": self.cfg,
            "policy_head": self.policy_head.state_dict(),
            "value_head": self.value_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        save_ckpt(state, path, self.device)
        del state
        gc.collect()
        if is_master():
            print(f"[INFO] Saved imagination checkpoint to {path}")

    def load_checkpoint(self, path: str) -> int:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            state = torch.load(path, map_location="cpu", weights_only=False)
        self.policy_head.load_state_dict(state["policy_head"])
        self.value_head.load_state_dict(state["value_head"])
        try:
            self.optimizer.load_state_dict(state["optimizer"])
        except (ValueError, KeyError) as e:
            if is_master():
                print(f"[WARN] Could not load optimizer state: {e}")
        self.global_step = state.get("global_step", 0)
        return state.get("epoch", 0) + 1

    # ──────────────────────────────────────────────────────────────
    # Optimizer
    # ──────────────────────────────────────────────────────────────

    def _build_optimizer(self) -> None:
        """AdamW over policy_head + value_head only.

        Two param groups: weight-decay on matrices, no decay on biases/norms.
        Much simpler than DynamicsTrainer's groups because Phase 3 trains
        only ~2M parameters (two small MLP heads).
        """
        decay_params = []
        no_decay_params = []
        for head in (self.policy_head, self.value_head):
            for name, param in head.named_parameters():
                if not param.requires_grad:
                    continue
                if param.dim() <= 1 or "norm" in name.lower():
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(param_groups, lr=self.cfg.lr)

    def _set_lr_from_schedule(self, total_steps: int) -> None:
        """Linear warmup + cosine decay, quantized to a few buckets."""
        step = self.global_step
        warmup = self.cfg.warmup_steps
        if step < warmup:
            frac = step / max(warmup, 1)
        else:
            progress = (step - warmup) / max(total_steps - warmup, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            frac = max(self.cfg.min_lr / self.cfg.lr, cosine)

        # Quantize to 2 buckets
        bucket = round(frac * 2) / 2
        lr = self.cfg.lr * bucket
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    # ──────────────────────────────────────────────────────────────
    # Training step
    # ──────────────────────────────────────────────────────────────

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """One RL training step on imagined rollouts.

        Args:
            batch: dict with at least:
                'frames':  (B, C, C_img, H_img, W_img) real context frames
                'actions': (B, C-1, A) real context actions
            (rewards/dones from the dataset are NOT used — imagined rewards
            come from the reward_head)

        Returns:
            dict of scalar loss tensors (stays on device; no .item() calls).
        """
        frames = batch["frames"].to(self.device, non_blocking=True)
        actions_ctx = batch["actions"].to(self.device, non_blocking=True)

        # Encode real context frames → clean latents
        with torch.no_grad():
            z_context = self.model.encode_frames(frames)  # (B, C, S_z, D_lat)

        # ── Imagined rollout ────────────────────────────────────
        rollout = imagine_rollout(
            dynamics_model=self.model,
            policy_head=self.policy_head,
            reward_head=self.reward_head,
            continue_head=self.continue_head,
            value_head=self.value_head,
            z_context=z_context,
            actions_context=actions_ctx,
            horizon=self.cfg.imagination_horizon,
            K=self.cfg.K_imagination,
            tau_ctx=self.cfg.tau_ctx,
            K_max=self.cfg.K_max,
            context_window=self.cfg.context_window,
        )
        states    = rollout["states"]     # (B, H, D)
        im_actions= rollout["actions"]    # (B, H, A)
        rewards   = rollout["rewards"]    # (B, H)
        continues = rollout["continues"]  # (B, H)
        values    = rollout["values"]     # (B, H+1)

        # ── λ-returns + advantages ──────────────────────────────
        lambda_returns = compute_lambda_returns(
            rewards, values, continues, self.cfg.gamma, self.cfg.lambda_,
        )
        advantages = compute_advantages(lambda_returns, values)

        # ── Value loss ──────────────────────────────────────────
        v_loss = value_loss(
            value_head=self.value_head,
            states=states.detach(),
            lambda_returns=lambda_returns.detach(),
        )

        # ── Policy loss ─────────────────────────────────────────
        p_loss = pmpo_policy_loss(
            policy_head=self.policy_head,
            policy_prior=self.policy_prior,
            states=states.detach(),
            actions=im_actions.detach(),
            advantages=advantages.detach(),
            alpha=self.cfg.pmpo_alpha,
            beta=self.cfg.pmpo_beta,
        )

        total_loss = v_loss + p_loss

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_head.parameters()) + list(self.value_head.parameters()),
            self.cfg.grad_clip,
        )
        self.optimizer.step()

        return {
            "loss/total": total_loss.detach(),
            "loss/value": v_loss.detach(),
            "loss/policy": p_loss.detach(),
            "stats/mean_reward": rewards.mean().detach(),
            "stats/mean_value": values[:, :-1].mean().detach(),
            "stats/mean_advantage": advantages.mean().detach(),
        }

    # ──────────────────────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        checkpoint_dir: str,
        start_epoch: int = 1,
    ) -> None:
        """Run Phase 3 imagination training.

        Args:
            train_loader: yields batches with 'frames' and 'actions' keys
            checkpoint_dir: where to save intermediate checkpoints
            start_epoch: resume point (1-indexed)
        """
        ckpt_dir = Path(checkpoint_dir)
        if is_master():
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        total_steps = self.cfg.epochs * self.cfg.steps_per_epoch

        self.policy_head.train()
        self.value_head.train()

        for epoch in range(start_epoch, self.cfg.epochs + 1):
            epoch_metrics: Dict[str, float] = {}
            step_in_epoch = 0

            for batch in train_loader:
                if step_in_epoch >= self.cfg.steps_per_epoch:
                    break

                self._set_lr_from_schedule(total_steps)
                metrics = self.train_step(batch)
                self.global_step += 1
                step_in_epoch += 1

                # Batched logging — no .item() per step to avoid host sync.
                if self.global_step % self.cfg.log_interval == 0 and is_master():
                    log_payload = {k: v.item() for k, v in metrics.items()}
                    log_payload["epoch"] = epoch
                    log_payload["global_step"] = self.global_step
                    log_payload["lr"] = self.optimizer.param_groups[0]["lr"]
                    wandb.log(log_payload, step=self.global_step)
                    print(
                        f"[ep {epoch} step {self.global_step}] "
                        f"total={log_payload['loss/total']:.4f} "
                        f"value={log_payload['loss/value']:.4f} "
                        f"policy={log_payload['loss/policy']:.4f}"
                    )

            # Epoch-end checkpoint
            if is_master() and epoch % self.cfg.checkpoint_interval == 0:
                self.save_checkpoint(str(ckpt_dir / f"epoch_{epoch}.pt"), epoch)
