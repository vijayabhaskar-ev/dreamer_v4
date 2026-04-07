
from __future__ import annotations

from dataclasses import dataclass, field
import pickle
from typing import Dict, Optional, Tuple
import math

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .config import DynamicsConfig
from .dynamic_model import DynamicsModel, DynamicsOutput
from .flow_matching import add_noise, sample_tau_and_d
from heads import RewardHead, ContinueHead, PolicyHead
from tokenizer.config import TokenizerConfig
from tokenizer.tokenizer import MaskedAutoencoderTokenizer
from tokenizer.metrics import MetricsBuffer, ModelStatistics, GPUMemoryTracker, ThroughputTracker
from device_utils import get_device, get_device_type, make_grad_scaler, save_checkpoint as save_ckpt, is_master
import wandb


# ── MTP target preparation ──────────────────────────────────────────────

#TODO need to mask the padded values in the rewards and dones in the loss calculation for future minecraft and other envs
def build_mtp_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    mtp_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build future target tensors for multi-token prediction (MTP).

    For each timestep t, gathers targets at offsets 0..L so the heads
    can predict rewards/dones L steps into the future.

    Args:
        rewards: (B, T) per-timestep rewards
        dones:   (B, T) per-timestep done flags
        mtp_length: L (predict offsets 0..L → L+1 targets per position)

    Returns:
        rewards_future: (B, T, L+1) — rewards[b, t, n] = reward at t+n
        dones_future:   (B, T, L+1) — dones[b, t, n] = done at t+n
    """
    num_offsets = mtp_length + 1
    # Pad: rewards with 0 (no reward beyond sequence), dones with 1.0 (terminal)
    rewards_padded = F.pad(rewards, (0, mtp_length), value=0.0)
    dones_padded = F.pad(dones, (0, mtp_length), value=1.0)
    # unfold: fixed output shape for fixed (B, T, L) — no XLA recompilation
    rewards_future = rewards_padded.unfold(1, num_offsets, 1)  # (B, T, L+1)
    dones_future = dones_padded.unfold(1, num_offsets, 1)      # (B, T, L+1)
    return rewards_future, dones_future


def build_mtp_action_targets(
    actions: torch.Tensor,
    mtp_length: int,
) -> torch.Tensor:
    """Build future action targets for multi-token prediction (MTP).

    Args:
        actions: (B, T-1, action_dim) per-timestep actions
        mtp_length: L (predict offsets 0..L → L+1 targets per position)

    Returns:
        actions_future: (B, T-1, L+1, action_dim)
    """
    num_offsets = mtp_length + 1
    B, T_minus_1, A = actions.shape
    # Pad time dim with zeros (no action beyond sequence)
    actions_padded = F.pad(actions, (0, 0, 0, mtp_length), value=0.0)  # (B, T-1+L, A)
    # unfold on time dim → (B, T-1, L+1, A)
    return actions_padded.unfold(1, num_offsets, 1).transpose(-1, -2)


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
    weight_decay_heavy: float = 0.1   # attention + FF layers (drifted in charts)
    grad_clip: float = 5.0
    amp: bool = False
    checkpoint_interval: int = 1
    device: str = "cuda"
    # Curriculum: bootstrap ramp-up
    curriculum_warmup_steps: int = 2000
    curriculum_ramp_steps: int = 4000
    # Alternating batch lengths
    long_batch_ratio: float = 0.15  # fraction of training steps using T₂ (long sequences)
    # Logging
    steps_per_epoch: int = 100
    log_interval: int = 10
    log_smooth_window: int = 10
    log_model_stats: bool = True
    log_model_stats_interval: int = 50
    log_memory: bool = True
    # Phase 2: agent finetuning
    train_heads: bool = False           # False for Phase 1, True for Phase 2
    head_lr_multiplier: float = 3.0     # New heads + agent embedding get higher LR
    dynamics_lr_multiplier: float = 0.3 # Pretrained dynamics gets lower LR during finetuning


class DynamicsTrainer:
    """
    Training loop for the dynamics model using flow matching.

    Supports two training phases:
      Phase 1 (train_heads=False): World model pretraining — only dynamics loss.
      Phase 2 (train_heads=True):  Agent finetuning — dynamics loss + reward/continue
                                   heads reading from agent token transformer output.
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

        self.device = get_device(training_cfg.device)

        self.rms_flow = RMSNormalizer(decay=0.99)
        self.rms_bootstrap = RMSNormalizer(decay=0.99)

        self.tokenizer = MaskedAutoencoderTokenizer(tokenizer_cfg).to(self.device)
        self._load_tokenizer_checkpoint(tokenizer_ckpt)
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

        self.model = DynamicsModel(dynamics_cfg, self.tokenizer).to(self.device)

        # ── Prediction heads ────────────────────────────────────────
        # Phase 1: heads exist but are NOT in the optimizer (no gradients).
        # Phase 2: heads read from agent token output (embed_dim=512).
        head_input_dim = dynamics_cfg.embed_dim
        self.reward_head = RewardHead(
            latent_dim=head_input_dim,
            hidden_dim=dynamics_cfg.head_hidden_dim,
            num_bins=dynamics_cfg.num_reward_bins,
            num_layers=dynamics_cfg.head_num_layers,
            mtp_length=dynamics_cfg.mtp_length,
        ).to(self.device)
        self.continue_head = ContinueHead(
            latent_dim=head_input_dim,
            hidden_dim=dynamics_cfg.head_hidden_dim,
            num_layers=dynamics_cfg.head_num_layers,
            mtp_length=dynamics_cfg.mtp_length,
        ).to(self.device)
        self.policy_head = PolicyHead(
            latent_dim=head_input_dim,
            action_dim=dynamics_cfg.action_dim,
            hidden_dim=dynamics_cfg.head_hidden_dim,
            num_layers=dynamics_cfg.head_num_layers,
            mtp_length=dynamics_cfg.mtp_length,
        ).to(self.device)
        self.rms_reward = RMSNormalizer(decay=0.99)
        self.rms_continue = RMSNormalizer(decay=0.99)
        self.rms_bc = RMSNormalizer(decay=0.99)

        # ── Optimizer ───────────────────────────────────────────────
        self._build_optimizer()

        self.scaler = make_grad_scaler(self.device, enabled=training_cfg.amp)
        self.global_step = 0
        self._last_lr_bucket = -1
        self.metrics_buffer = MetricsBuffer(window=training_cfg.log_smooth_window)
        self.throughput_tracker = ThroughputTracker()

        # Cached device tensors — avoids creating new XLA graph nodes per step
        self._zero = torch.tensor(0.0, device=self.device)
        self._ramp_tensor = torch.tensor(0.0, device=self.device)

    # ── Optimizer construction ──────────────────────────────────────────

    def _build_optimizer(self) -> None:
        """Build AdamW optimizer with per-group weight decay.

        Phase 1 (train_heads=False): only dynamics model params.
        Phase 2 (train_heads=True):  dynamics (lower LR) + heads + agent embedding (higher LR).
        """
        cfg = self.training_cfg
        heavy_decay_params = []
        no_decay_params = []
        default_decay_params = []

        # Dynamics model params (skip agent_embedding — added to head group in Phase 2)
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('agent_embedding.'):
                continue
            name_lower = name.lower()
            if param.dim() <= 1 or 'norm' in name_lower:
                no_decay_params.append(param)
            elif any(p in name_lower for p in ['spatial_attn', 'temporal_attn', '.ff.']):
                heavy_decay_params.append(param)
            else:
                default_decay_params.append(param)

        param_groups = [
            {"params": heavy_decay_params, "weight_decay": cfg.weight_decay_heavy},
            {"params": no_decay_params, "weight_decay": 0.0},
            {"params": default_decay_params, "weight_decay": cfg.weight_decay},
        ]

        if cfg.train_heads:
            # Phase 2: add heads with separate LR group
            head_no_decay = []
            head_decay = []
            for head in (self.reward_head, self.continue_head, self.policy_head):
                for name, param in head.named_parameters():
                    if not param.requires_grad:
                        continue
                    if param.dim() <= 1 or 'norm' in name.lower():
                        head_no_decay.append(param)
                    else:
                        head_decay.append(param)

            # Agent embedding params (if enabled)
            if self.model.agent_embedding is not None:
                for param in self.model.agent_embedding.parameters():
                    if param.requires_grad:
                        head_decay.append(param)

            param_groups.extend([
                {"params": head_decay, "weight_decay": cfg.weight_decay, "lr_multiplier": "head"},
                {"params": head_no_decay, "weight_decay": 0.0, "lr_multiplier": "head"},
            ])

        self.optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr)

    # ── Phase 2 setup ───────────────────────────────────────────────────

    def setup_phase2(self, dynamics_ckpt: str, num_tasks: int = 1) -> None:
        """Transition from Phase 1 to Phase 2.

        1. Load Phase 1 dynamics checkpoint (model weights only).
        2. Enable agent tokens in the dynamics model.
        3. Rebuild optimizer with Phase 2 param groups.

        Args:
            dynamics_ckpt: Path to Phase 1 checkpoint.
            num_tasks: Number of tasks for agent token embedding.
        """
        # Load dynamics model weights (strict — all Phase 1 params must match)
        try:
            state = torch.load(dynamics_ckpt, map_location='cpu', weights_only=True)
        except pickle.UnpicklingError:
            state = torch.load(dynamics_ckpt, map_location='cpu', weights_only=False)

        # Filter out agent_embedding keys — they may be present if the
        # checkpoint came from a prior Phase 2 run.  We re-initialize
        # agent_embedding fresh via enable_agent_tokens() below.
        model_state = {k: v for k, v in state["model"].items()
                       if not k.startswith("agent_embedding.")}
        self.model.load_state_dict(model_state, strict=True)

        # Optionally restore flow/bootstrap RMS normalizers for continuity
        if "rms_normalizers" in state:
            rms_state = state["rms_normalizers"]
            if "flow" in rms_state:
                self.rms_flow.load_state_dict(rms_state["flow"])
            if "bootstrap" in rms_state:
                self.rms_bootstrap.load_state_dict(rms_state["bootstrap"])

        if "global_step" in state:
            self.global_step = state["global_step"]
        if "scheduler_bucket" in state:
            self._last_lr_bucket = state["scheduler_bucket"]

        # Enable agent tokens — adds new parameters
        self.model.enable_agent_tokens(num_tasks=num_tasks)
        self.model.agent_embedding.to(self.device)

        # Rebuild optimizer with Phase 2 groups (heads + agent embedding included)
        self._build_optimizer()

        if is_master():
            print(f"Phase 2 setup complete. Loaded dynamics from {dynamics_ckpt}")
            print(f"  Agent tokens enabled (num_tasks={num_tasks})")
            print(f"  Global step: {self.global_step}")

    # ── LR scheduling ───────────────────────────────────────────────────

    def _build_scheduler(self, total_steps: int) -> None:
        """Store schedule params. LR is set via _set_lr_from_schedule() each step.

        Using LambdaLR with XLA causes recompilation every step because the LR
        value is embedded as a compile-time constant in the XLA graph. Instead,
        we compute LR in Python and set it before optimizer.step() so the value
        is baked into each step's graph. Since we change LR infrequently
        (only at coarse intervals), this avoids most recompilations.
        """
        self._schedule_total_steps = total_steps
        self._schedule_warmup = self.training_cfg.warmup_steps
        self._schedule_min_lr = self.training_cfg.min_lr
        self._schedule_base_lr = self.training_cfg.lr
        self._last_lr_bucket = -1  # track when LR actually changes

    def _set_lr_from_schedule(self) -> None:
        """Set optimizer LR from schedule, quantized to reduce XLA recompilations.

        Each unique LR value creates a new XLA graph compilation (~50-100MB
        cache entry). With 4 code path variants (full/single × boot/no-boot),
        each LR bucket costs ~4 compilations. We use only 10 buckets → max
        ~40 compiled graphs total, keeping XLA cache under ~2-4GB.

        Phase 2: dynamics groups get lr * dynamics_lr_multiplier,
                 head groups get lr * head_lr_multiplier.
        """
        step = self.global_step
        warmup = self._schedule_warmup
        total = self._schedule_total_steps
        base_lr = self._schedule_base_lr
        min_lr = self._schedule_min_lr

        if step < warmup:
            frac = step / max(warmup, 1)
        else:
            progress = (step - warmup) / max(total - warmup, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            frac = max(min_lr / base_lr, cosine)

        # Quantize to 10 buckets → max ~40 XLA compilations (10 × 4 code paths)
        bucket = int(frac * 10)
        if bucket == self._last_lr_bucket:
            return  # Same bucket, no LR change, no recompilation
        self._last_lr_bucket = bucket

        lr = base_lr * (bucket / 10.0)
        cfg = self.training_cfg
        for group in self.optimizer.param_groups:
            if cfg.train_heads and group.get("lr_multiplier") == "head":
                group['lr'] = lr * cfg.head_lr_multiplier
            elif cfg.train_heads:
                group['lr'] = lr * cfg.dynamics_lr_multiplier
            else:
                group['lr'] = lr

    # ── Training loop ───────────────────────────────────────────────────

    def fit(
        self,
        train_loader_short: DataLoader,
        train_loader_long: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: Optional[str] = None,
        start_epoch: int = 1,
    ) -> None:

        steps_per_epoch = self.training_cfg.steps_per_epoch
        total_steps = steps_per_epoch * self.training_cfg.epochs
        self._build_scheduler(total_steps)


        for epoch in range(start_epoch, self.training_cfg.epochs + 1):
            train_metrics = self._run_epoch(
                train_loader_short, train_loader_long, epoch, training=True,
            )
            val_metrics = None
            if val_loader is not None:
                val_metrics = self._run_epoch(
                    val_loader, None, epoch, training=False,
                )

            if is_master():
                epoch_log = {
                    "epoch": epoch,
                    "epoch/train_loss": train_metrics["loss/dynamics_total"],
                    "epoch/train_flow": train_metrics["loss/dynamics_flow"],
                    "epoch/train_bootstrap": train_metrics["loss/dynamics_bootstrap"],
                    "epoch/train_reward": train_metrics["loss/dynamics_reward"],
                    "epoch/train_continue": train_metrics["loss/dynamics_continue"],
                    "global_step": self.global_step,
                }
                if val_metrics is not None:
                    epoch_log["epoch/val_loss"] = val_metrics["loss/dynamics_total"]
                    epoch_log["epoch/val_flow"] = val_metrics["loss/dynamics_flow"]
                    epoch_log["epoch/val_bootstrap"] = val_metrics["loss/dynamics_bootstrap"]
                    epoch_log["epoch/val_reward"] = val_metrics["loss/dynamics_reward"]
                    epoch_log["epoch/val_continue"] = val_metrics["loss/dynamics_continue"]
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

    # ── Checkpointing ───────────────────────────────────────────────────

    def save_checkpoint(self, path: str, epoch: int) -> None:
        state = {
                "epoch": epoch,
                "global_step": self.global_step,
                "dynamics_cfg": self.dynamics_cfg,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler_bucket": self._last_lr_bucket,
                "train_heads": self.training_cfg.train_heads,
                "rms_normalizers": {
                    "flow": self.rms_flow.state_dict(),
                    "bootstrap": self.rms_bootstrap.state_dict(),
                    "reward": self.rms_reward.state_dict(),
                    "continue": self.rms_continue.state_dict(),
                    "bc": self.rms_bc.state_dict(),
                },
            }
        # Only save head state when heads are being trained
        if self.training_cfg.train_heads:
            state["reward_head"] = self.reward_head.state_dict()
            state["continue_head"] = self.continue_head.state_dict()
            state["policy_head"] = self.policy_head.state_dict()
        save_ckpt(state, path, self.device)
        if is_master():
            print(f"Saved dynamics checkpoint to {path}")

    def load_checkpoint(self, path: str, strict: bool = True) -> int:
        try:
            state = torch.load(path, map_location='cpu', weights_only=True)
        except pickle.UnpicklingError:
            state = torch.load(path, map_location='cpu', weights_only=False)
        # Filter agent_embedding keys when the model doesn't have agent tokens
        # enabled (e.g. evaluating a Phase 2 checkpoint without agent tokens).
        model_state = state["model"]
        if self.model.agent_embedding is None:
            model_state = {k: v for k, v in model_state.items()
                           if not k.startswith("agent_embedding.")}
        missing, unexpected = self.model.load_state_dict(model_state, strict=False)
        missing_real = [k for k in missing if "qk_norm" not in k]
        if strict and missing_real:
            raise RuntimeError(f"Missing key(s) in state_dict: {missing_real}")
        if strict and unexpected:
            raise RuntimeError(f"Unexpected key(s) in state_dict: {unexpected}")
        # Load heads non-strictly: MTP length may differ between checkpoint
        # and current config (e.g. Phase 2 checkpoint loaded for evaluation
        # with mtp_length=0).  Missing/extra output_heads are harmless.
        for name, head in [("reward_head", self.reward_head),
                           ("continue_head", self.continue_head),
                           ("policy_head", self.policy_head)]:
            if name in state:
                try:
                    head.load_state_dict(state[name], strict=strict)
                except RuntimeError:
                    head.load_state_dict(state[name], strict=False)
                    if is_master():
                        print(f"[WARN] Loaded {name} non-strictly (MTP length mismatch)")
        if "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except (ValueError, KeyError) as e:
                if is_master():
                    print(f"[WARN] Could not load optimizer state ({e}). "
                          "Using fresh optimizer (expected when changing param groups).")
        if "rms_normalizers" in state:
            rms_state = state["rms_normalizers"]
            if "flow" in rms_state:
                self.rms_flow.load_state_dict(rms_state["flow"])
            if "bootstrap" in rms_state:
                self.rms_bootstrap.load_state_dict(rms_state["bootstrap"])
            if "reward" in rms_state:
                self.rms_reward.load_state_dict(rms_state["reward"])
            if "continue" in rms_state:
                self.rms_continue.load_state_dict(rms_state["continue"])
            if "bc" in rms_state:
                self.rms_bc.load_state_dict(rms_state["bc"])
        if "global_step" in state:
            self.global_step = state["global_step"]
        if "scheduler_bucket" in state:
            self._last_lr_bucket = state["scheduler_bucket"]

        return state.get("epoch", 0) + 1

    # ── Epoch runner ────────────────────────────────────────────────────

    def _run_epoch(
        self,
        loader_short: DataLoader,
        loader_long: Optional[DataLoader],
        epoch: int,
        training: bool,
    ) -> Dict[str, float]:

        self.model.train(training)
        if self.training_cfg.train_heads:
            self.reward_head.train(training)
            self.continue_head.train(training)
            self.policy_head.train(training)

        total_loss = torch.tensor(0.0, device=self.device)
        total_flow = torch.tensor(0.0, device=self.device)
        total_bootstrap = torch.tensor(0.0, device=self.device)
        total_reward = torch.tensor(0.0, device=self.device)
        total_continue = torch.tensor(0.0, device=self.device)
        total_bc = torch.tensor(0.0, device=self.device)
        total_steps = 0

        log_interval = self.training_cfg.log_interval
        model_stats_interval = self.training_cfg.log_model_stats_interval
        use_agent = self.training_cfg.train_heads and self.model.agent_embedding is not None

        # On-device accumulators for smoothed logging (reset every log_interval)
        _log_loss = torch.tensor(0.0, device=self.device)
        _log_flow = torch.tensor(0.0, device=self.device)
        _log_bootstrap = torch.tensor(0.0, device=self.device)
        _log_reward = torch.tensor(0.0, device=self.device)
        _log_continue = torch.tensor(0.0, device=self.device)
        _log_bc = torch.tensor(0.0, device=self.device)
        _log_grad_norm = torch.tensor(0.0, device=self.device)
        _log_tau = torch.tensor(0.0, device=self.device)
        _log_d = torch.tensor(0.0, device=self.device)
        _log_count = 0

        # Curriculum boundaries (Python ints — no XLA sync)   
        warmup_end = self.training_cfg.curriculum_warmup_steps
        ramp_end = self.training_cfg.curriculum_ramp_steps

        # Alternating batch lengths: iterate short loader, draw from long loader
        # with probability long_batch_ratio. During validation (loader_long=None),
        # only the short loader is used.
        long_ratio = self.training_cfg.long_batch_ratio if (training and loader_long is not None) else 0.0 #TODO Need to check the if we need to use loader_long in validation
        long_iter = iter(loader_long) if loader_long is not None else None

        for batch_short in loader_short:
            # Decide whether to use short or long batch this step
            # Python-level branch — both T values compile once, no per-step recompilation
            use_long = training and long_iter is not None and (torch.rand(1).item() < long_ratio)
            if use_long:
                try:
                    batch = next(long_iter)
                except StopIteration:
                    long_iter = iter(loader_long)
                    batch = next(long_iter)
            else:
                batch = batch_short

            frames, actions, rewards, dones = batch
            frames = frames.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)

            if frames.dim() == 4:
                frames = frames.unsqueeze(1)

            B, T = frames.shape[0:2]

            # Python-level gate: skip expensive bootstrap forward passes
            # during flow-only warmup (no XLA sync, no dynamic shapes)
            need_bootstrap = self.global_step >= warmup_end

            z_clean = self.model.encode_frames(frames)
            tau, d = sample_tau_and_d(B, T, K_max=self.dynamics_cfg.K_max, device=self.device)

            # Context frame: First frame always gets near-clean noise level
            tau[:, 0] = 1.0 - self.dynamics_cfg.tau_ctx

            # Curriculum d-override
            if self.global_step < warmup_end:
                d = torch.full_like(d, 1.0 / self.dynamics_cfg.K_max)
            elif self.global_step < ramp_end:
                ramp_frac = (self.global_step - warmup_end) / (ramp_end - warmup_end)
                self._ramp_tensor.fill_(int(ramp_frac * 10) / 10.0)
                mask = torch.rand_like(d) < self._ramp_tensor
                d_min = torch.full_like(d, 1.0 / self.dynamics_cfg.K_max)
                d = torch.where(mask, d, d_min)

            z_noised, _ = add_noise(z_clean, tau)
            tau_for_log, d_for_log = tau, d

            # Build MTP targets if needed
            rewards_future, dones_future, actions_future = None, None, None
            if use_agent and self.dynamics_cfg.mtp_length > 0:
                rewards_future, dones_future = build_mtp_targets(
                    rewards, dones, self.dynamics_cfg.mtp_length,
                )
                actions_future = build_mtp_action_targets(
                    actions, self.dynamics_cfg.mtp_length,
                )

            with torch.amp.autocast(device_type=get_device_type(self.device), enabled=self.training_cfg.amp):
                loss, metrics = self._compute_loss(
                    z_clean, z_noised, actions, tau, d,
                    training=training, compute_bootstrap=need_bootstrap,
                    rewards=rewards, dones=dones,
                    use_agent_tokens=use_agent,
                    rewards_future=rewards_future, dones_future=dones_future,
                    actions_future=actions_future,
                )

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                all_params = list(self.model.parameters())
                if self.training_cfg.train_heads:
                    all_params += list(self.reward_head.parameters()) + \
                        list(self.continue_head.parameters()) + \
                        list(self.policy_head.parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    all_params,
                    self.training_cfg.grad_clip,
                )

                # Compute LR manually as a tensor so XLA doesn't recompile
                # on every LR change (scheduler embeds LR as compile-time constant)
                self._set_lr_from_schedule()

                self.scaler.step(self.optimizer)  # xm.optimizer_step on TPU (includes mark_step)
                self.scaler.update()

                self.global_step += 1

                # Accumulate on-device — no .item() sync per step
                _log_loss += loss.detach()
                _log_flow += metrics["loss_flow"]
                _log_bootstrap += metrics["loss_bootstrap"]
                _log_reward += metrics["loss_reward"]
                _log_continue += metrics["loss_continue"]
                _log_bc += metrics["loss_bc"]
                _log_grad_norm += grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else grad_norm
                _log_tau += tau_for_log.mean().detach()
                _log_d += d_for_log.mean().detach()
                _log_count += 1

                if self.global_step % log_interval == 0:
                    # Master logs; ALL processes reset accumulators
                    if is_master():
                        n = max(_log_count, 1)
                        self.metrics_buffer.update({
                            "train/loss": (_log_loss / n).item(),
                            "train/flow": (_log_flow / n).item(),
                            "train/bootstrap": (_log_bootstrap / n).item(),
                            "train/reward": (_log_reward / n).item(),
                            "train/continue": (_log_continue / n).item(),
                            "train/bc": (_log_bc / n).item(),
                            "train/grad_norm": (_log_grad_norm / n).item() if isinstance(_log_grad_norm, torch.Tensor) else _log_grad_norm / n,
                            "train/tau_mean": (_log_tau / n).item(),
                            "train/d_mean": (_log_d / n).item(),
                            "train/seq_len": float(T),
                        })

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
                            smoothed_metrics.update(self._compute_wd_metrics())

                        wandb.log(smoothed_metrics, step=self.global_step)

                    # Reset on ALL processes — prevents unbounded accumulation
                    _log_loss.zero_()
                    _log_flow.zero_()
                    _log_bootstrap.zero_()
                    _log_reward.zero_()
                    _log_continue.zero_()
                    _log_bc.zero_()
                    _log_grad_norm.zero_()
                    _log_tau.zero_()
                    _log_d.zero_()
                    _log_count = 0

            total_loss += loss.detach()
            total_flow += metrics["loss_flow"]
            total_bootstrap += metrics["loss_bootstrap"]
            total_reward += metrics["loss_reward"]
            total_continue += metrics["loss_continue"]
            total_bc += metrics["loss_bc"]
            total_steps += 1

        # Single .item() at epoch end — one sync instead of N
        n = max(total_steps, 1)
        return {
            "loss/dynamics_total": (total_loss / n).item(),
            "loss/dynamics_flow": (total_flow / n).item(),
            "loss/dynamics_bootstrap": (total_bootstrap / n).item(),
            "loss/dynamics_reward": (total_reward / n).item(),
            "loss/dynamics_continue": (total_continue / n).item(),
            "loss/dynamics_bc": (total_bc / n).item(),
        }

    # ── Loss computation ────────────────────────────────────────────────

    def _compute_wd_metrics(self) -> Dict[str, float]:
        """Per-group effective weight decay: wd * sum(||p||^2).

        Shows how much regularization each group contributes.
        Only called at model_stats_interval so .item() syncs are acceptable.
        """
        names = ["attn_ff", "no_decay", "default"]
        if self.training_cfg.train_heads:
            names.extend(["head_decay", "head_no_decay"])
        metrics = {}
        for name, group in zip(names, self.optimizer.param_groups):
            wd = group["weight_decay"]
            if wd == 0.0:
                continue
            sq_sum = sum(p.data.norm(2) ** 2 for p in group["params"] if p.requires_grad)
            metrics[f"model/wd_{name}"] = (wd * sq_sum).item()
        return metrics

    def _compute_loss(self, z_clean, z_noised, actions, tau, d, training=True,
                      compute_bootstrap=True,
                      rewards=None, dones=None,
                      use_agent_tokens=False,
                      rewards_future=None, dones_future=None,
                      actions_future=None):
        """Compute flow matching + bootstrap + prediction head losses.

        Uses masked arithmetic instead of boolean indexing so all tensors
        keep fixed (B, T, ...) shapes — critical for XLA/TPU which
        recompiles the entire graph on shape changes. T can be T₁ or T₂
        (two fixed values from alternating batch lengths).

        Args:
            z_clean:  (B, T, S_z, D)  — target clean latents
            z_noised: (B, T, S_z, D)  — corrupted input
            actions:  (B, T-1, action_dim)
            tau:      (B, T)  — per-frame signal levels
            d:        (B, T)  — per-frame step sizes
            compute_bootstrap: Python-level gate to skip bootstrap forward passes
            rewards:  (B, T) or None — real-valued reward targets
            dones:    (B, T) or None — terminal flags (1.0 = done)
            use_agent_tokens: whether to use agent tokens (Phase 2)
            rewards_future: (B, T, L+1) or None — MTP reward targets
            dones_future:   (B, T, L+1) or None — MTP done targets
            actions_future: (B, T-1, L+1, action_dim) or None — MTP action targets

        Returns:
            loss, metrics_dict
        """
        d_min = 1 / self.dynamics_cfg.K_max

        output: DynamicsOutput = self.model(z_noised, actions, tau, d,
                                            use_agent_tokens=use_agent_tokens)
        z_hat = output.z_hat      # (B, T, S_z, D)
        agent_out = output.agent_out  # (B, T, D_embed) or None

        # Float masks — no boolean indexing, no dynamic shapes
        flow_mask = (d == d_min).float()   # (B, T)  1.0 for flow, 0.0 for bootstrap
        boot_mask = 1.0 - flow_mask        # (B, T)
        w = 0.9 * tau + 0.1               # (B, T)

        # ── Flow loss (masked) ──────────────────────────────────────
        per_sample_flow = ((z_hat - z_clean) ** 2).mean(dim=(-2, -1))  # (B, T)
        n_flow = flow_mask.sum().clamp(min=1.0)
        loss_flow = (flow_mask * w * per_sample_flow).sum() / n_flow
        loss_flow_normed = self.rms_flow.normalize(loss_flow, update=training)

        # ── Bootstrap loss (masked) ─────────────────────────────────
        if compute_bootstrap:
            tau_4d = tau[:, :, None, None]   # (B, T, 1, 1)
            d_4d = d[:, :, None, None]
            v_pred = (z_hat - z_noised) / (1 - tau_4d).clamp(min=1e-4)

            with torch.no_grad():
                out_half1 = self.model(z_noised, actions, tau, d / 2,
                                       use_agent_tokens=use_agent_tokens)
                z_hat_half1 = out_half1.z_hat
                v1 = (z_hat_half1 - z_noised) / (1 - tau_4d).clamp(min=1e-4)

                z_mid = z_noised + v1 * (d_4d / 2)

                out_half2 = self.model(z_mid, actions, tau + d / 2, d / 2,
                                       use_agent_tokens=use_agent_tokens)
                z_hat_half2 = out_half2.z_hat
                v2 = (z_hat_half2 - z_mid) / (1 - (tau_4d + d_4d / 2)).clamp(min=1e-4)

                v_target = (v1 + v2) / 2

            weight_tau_sq = (1 - tau[:, :, None, None]) ** 2  # (B, T, 1, 1)
            per_sample_boot = (
                weight_tau_sq * (v_pred - v_target) ** 2
            ).mean(dim=(-2, -1))  # (B, T)

            n_boot = boot_mask.sum().clamp(min=1.0)
            loss_bootstrap = (boot_mask * w * per_sample_boot).sum() / n_boot
            loss_bootstrap_normed = self.rms_bootstrap.normalize(loss_bootstrap, update=training)
        else:
            loss_bootstrap = self._zero
            loss_bootstrap_normed = self._zero
            n_boot = self._zero

        loss_total = loss_flow_normed + loss_bootstrap_normed

        # ── Prediction head losses ──────────────────────────────────
        loss_reward = self._zero
        loss_continue = self._zero
        loss_bc = self._zero

        if self.training_cfg.train_heads and agent_out is not None:
            # Phase 2: heads read from agent token transformer output
            h = agent_out.detach()  # (B, T, D_embed)

            if rewards is not None:
                if self.dynamics_cfg.mtp_length > 0 and rewards_future is not None:
                    loss_reward = self.reward_head.loss_mtp(h, rewards_future)
                else:
                    loss_reward = self.reward_head.loss(h, rewards)
                loss_reward_normed = self.rms_reward.normalize(loss_reward, update=training)
                loss_total = loss_total + self.dynamics_cfg.reward_loss_scale * loss_reward_normed #TODO Need to adjust the reward loss scale based on training result

            if dones is not None:
                if self.dynamics_cfg.mtp_length > 0 and dones_future is not None:
                    loss_continue = self.continue_head.loss_mtp(h, dones_future)
                else:
                    loss_continue = self.continue_head.loss(h, dones)
                loss_continue_normed = self.rms_continue.normalize(loss_continue, update=training)
                loss_total = loss_total + self.dynamics_cfg.continue_loss_scale * loss_continue_normed

            # ── Behavior cloning loss ─────────────────────────────
            # actions: (B, T-1, action_dim), agent_out: (B, T, D)
            # Agent at position t predicts the action taken after frame t
            if actions is not None:
                h_for_actions = h[:, :-1]  # (B, T-1, D_embed)
                if self.dynamics_cfg.mtp_length > 0 and actions_future is not None:
                    loss_bc = self.policy_head.loss_mtp(h_for_actions, actions_future)
                else:
                    loss_bc = self.policy_head.loss(h_for_actions, actions)
                loss_bc_normed = self.rms_bc.normalize(loss_bc, update=training)
                loss_total = loss_total + self.dynamics_cfg.bc_loss_scale * loss_bc_normed

        # Return detached tensors — .item() called only at log intervals
        metrics = {
            "loss_flow": loss_flow.detach(),
            "loss_bootstrap": loss_bootstrap.detach(),
            "loss_reward": loss_reward.detach() if isinstance(loss_reward, torch.Tensor) else loss_reward,
            "loss_continue": loss_continue.detach() if isinstance(loss_continue, torch.Tensor) else loss_continue,
            "loss_bc": loss_bc.detach() if isinstance(loss_bc, torch.Tensor) else loss_bc,
            "n_flow": n_flow.detach(),
            "n_bootstrap": n_boot.detach(),
        }

        return loss_total, metrics


    # ── Tokenizer loading ───────────────────────────────────────────────

    def _load_tokenizer_checkpoint(self, ckpt_path) -> None:
        if ckpt_path is None:
            raise ValueError(
                "DreamerAgent expects a pretrained tokenizer checkpoint. "
                "Set `tokenizer_ckpt` in the config to load weights before training the agent."
            )

        try:
            state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        except pickle.UnpicklingError:
            state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if "model" in state:
            state = state["model"]
        missing, unexpected = self.tokenizer.load_state_dict(state, strict=False)
        # QKNorm params (qk_norm.q_norm.weight, qk_norm.k_norm.weight) are
        # nn.RMSNorm weights that default-initialize to ones — safe to skip
        # when loading checkpoints trained before QKNorm was added.
        missing_real = [k for k in missing if "qk_norm" not in k]   #TODO Remove it after training new tokenizer
        if missing_real:
            raise RuntimeError(f"Tokenizer checkpoint missing parameters: {missing_real}")
        if unexpected:
            raise RuntimeError(f"Tokenizer checkpoint has unexpected parameters: {unexpected}")






class RMSNormalizer:

    def __init__(self, decay=0.99, epsilon=1e-8):
        self.decay = decay
        self.epsilon = epsilon
        self.rms_sq = torch.tensor(1.0)
        self._on_device = False

    def normalize(self, loss: torch.Tensor, update: bool = True) -> torch.Tensor:
        loss_sq = loss.detach() ** 2
        if not self._on_device:
            self.rms_sq = self.rms_sq.to(loss_sq.device)
            self._on_device = True

        if update:
            # In-place update — avoids creating new XLA graph nodes per step
            self.rms_sq.mul_(self.decay).add_(loss_sq, alpha=(1 - self.decay))

        rms = torch.sqrt(self.rms_sq + self.epsilon)
        return loss / rms


    def state_dict(self):
        return {"rms_sq": self.rms_sq}

    def load_state_dict(self, state):
        self.rms_sq = state["rms_sq"]
        self._on_device = False
