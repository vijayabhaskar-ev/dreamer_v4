
from __future__ import annotations

from dataclasses import dataclass, field
import pickle
from typing import Dict, Optional, Tuple
import math
import gc
import random

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
from device_utils import get_device, get_device_type, make_grad_scaler, save_checkpoint as save_ckpt, is_master, is_xla_device
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


def _xla_safe_clip_grad_norm(params, max_norm: float) -> torch.Tensor:
    """Clip gradients entirely on-device (no .item() host sync).

    PyTorch's clip_grad_norm_ internally calls .item() to compare the
    total gradient norm against max_norm on the CPU. On XLA/TPU this
    triggers a device→host sync every step, destroying pipelining.

    This version keeps the comparison on-device using torch.clamp.
    Returns the unclipped total_norm as a device tensor (for logging).
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    # torch.stack + single sum avoids Python's left-associative chain of
    # intermediate tensors that Python's built-in sum() would produce.
    total_norm_sq = torch.stack([g.detach().pow(2).sum() for g in grads]).sum()
    total_norm = total_norm_sq.sqrt()
    # clip_coef = max_norm / (total_norm + 1e-6), clamped to [0, 1]
    clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for g in grads:
        g.detach().mul_(clip_coef)
    return total_norm


class _DetachedParamView:
    """CPU snapshot of a model's parameters and grads for diagnostic stats.

    On XLA, running torch.stack/.norm/.item over live device parameters
    creates new compiled graphs every interval (one per group, per call,
    per Python iteration order).  Building a CPU copy once per interval
    and running the same stat ops on the CPU copy keeps the XLA graph
    untouched while preserving the wandb dashboards.

    Single host sync per build (the .to('cpu') triggers torch_xla.sync()
    implicitly).  Cost on a 47M-param model: ~200 ms per build.

    Compatible with ModelStatistics.compute_weight_stats /
    compute_gradient_stats — they only call .named_parameters() and
    read .data / .grad, all of which this view supports.

    Also captures the optimizer param-group structure (params + weight_decay)
    so _compute_wd_metrics can iterate it without touching device tensors.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        include_grads: bool = True,
    ):
        # Phase 1 — Collect device-side references. No data movement,
        # no compilation. Just walk the model once and hold refs to the
        # live device tensors so we can transfer them in a known order.
        named_params = list(model.named_parameters())
        device_params = [p.detach() for _, p in named_params]
        device_grads = [
            p.grad.detach() if (include_grads and p.grad is not None) else None
            for _, p in named_params
        ]

        # Phase 2 — Transfer to CPU.  The first .cpu() call on an XLA
        # tensor implicitly triggers torch_xla.sync() to materialize
        # pending ops, so we do NOT call torch_xla.sync() explicitly —
        # that explicit call was found to compile a new graph variant on
        # every invocation (because the "pending ops" snapshot differed
        # each time), contributing to host-RAM-driven silent hangs at
        # epoch 51 (Apr 18 debug).  The first .cpu() handles the sync;
        # subsequent .cpu() calls are cached D2H transfers.  Cost on a
        # 47M-param model: ~100-300 ms per build.
        cpu_params = [t.cpu() for t in device_params]
        cpu_grads = [
            t.cpu() if t is not None else None for t in device_grads
        ]

        # Phase 3 — Build the _items and id_to_cpu structures. Pure Python,
        # no tensor operations.
        self._items: list[tuple[str, torch.nn.Parameter]] = []
        id_to_cpu: dict[int, torch.nn.Parameter] = {}
        for (name, p), cpu_data, cpu_grad in zip(named_params, cpu_params, cpu_grads):
            cpu_p = torch.nn.Parameter(cpu_data, requires_grad=p.requires_grad)
            if include_grads and cpu_grad is not None:
                cpu_p.grad = cpu_grad
            self._items.append((name, cpu_p))
            id_to_cpu[id(p)] = cpu_p

        # Mirror the optimizer param groups using the CPU copies, preserving
        # the weight_decay attribute that _compute_wd_metrics needs.
        self.optimizer_groups: list[dict] = []
        if optimizer is not None:
            for group in optimizer.param_groups:
                cpu_params = [
                    id_to_cpu[id(p)] for p in group["params"]
                    if id(p) in id_to_cpu
                ]
                self.optimizer_groups.append({
                    "params": cpu_params,
                    "weight_decay": group.get("weight_decay", 0.0),
                })

    def named_parameters(self):
        return iter(self._items)

    def parameters(self):
        return (p for _, p in self._items)


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
    log_model_stats_interval: int = 100
    # On XLA, the CPU-offload model-stats path (_DetachedParamView) has a
    # non-trivial compile cost per invocation.  Fire it much less often
    # on XLA than on CUDA to keep the per-rank XLA program cache bounded
    # (each compiled program pins ~1 GB of host RAM × 4 ranks).  On v4-8
    # this is the difference between stable training at ~10 GB/rank and
    # a silent OOM kill at ~60 GB/rank around epoch 51.
    log_model_stats_interval_xla: int = 2000
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
        self._is_xla = is_xla_device(self.device)

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
        self._cached_clip_params: list[torch.nn.Parameter] | None = None
        self._compile_warning_emitted = False
        # Compile-count ceiling guard state. See the check in _run_epoch
        # around the `if self._is_xla:` metrics block. Locked in at step
        # ~100 after warmup; any further compile growth past the
        # tolerance fast-aborts training rather than silently wedging.
        self._expected_compile_count: int | None = None
        self._compile_tolerance: int = 2
        self._rss_warning_emitted = False
        self.metrics_buffer = MetricsBuffer(window=training_cfg.log_smooth_window)
        self.throughput_tracker = ThroughputTracker()

        # Cached device tensors — avoids creating new XLA graph nodes per step
        self._zero = torch.tensor(0.0, device=self.device)
        self._ramp_tensor = torch.tensor(0.0, device=self.device)

        # Device-side step counter.  Incremented in-place each training step
        # via .add_(1.0) — the constant 1.0 is stable, producing the same XLA
        # graph every step.  Used to compute curriculum ramp_frac without
        # baking a Python float (which would vary per step) into the IR.
        # Previous version used .fill_(python_float) which specialized the
        # compiled program on every unique float value → 4 ramp values × 2
        # shapes = 8 unique graphs during ramp, plus separate full-phase
        # graph = ~14 programs just for the curriculum.  Each compiled
        # program costs ~1 GB of host RSS; 100+ compiles → OOM kill.
        self._step_dev = torch.zeros((), device=self.device)

        # Precomputed schedule constants as device tensors.  Built ONCE at
        # init so their values never change after construction — they bake
        # as constants into the IR once, and every step reuses the same
        # compiled graph.  Same rationale as torch.tensor(3.0, device=...)
        # in the ramp quantization: stable constants, not Python-float-
        # per-step leaks.
        self._warmup_end_dev = torch.tensor(
            float(self.training_cfg.curriculum_warmup_steps),
            device=self.device,
        )
        self._ramp_range_dev = torch.tensor(
            float(
                self.training_cfg.curriculum_ramp_steps
                - self.training_cfg.curriculum_warmup_steps
            ),
            device=self.device,
        )
        # Quantization: floor(ramp_frac * 3) / 3 produces 4 discrete values
        # {0.0, 1/3, 2/3, 1.0}.  This matches the existing int(ramp_frac*3)/3.0
        # Python logic exactly, just computed on-device.
        self._ramp_levels_dev = torch.tensor(3.0, device=self.device)

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

        # NOTE: capturable=True + tensor-valued lr was tried as a way to avoid
        # baking the LR into the XLA graph, but it caused a NaN at ~step 50 on
        # this torch_xla build.  Reverted to the stock Python-float LR path;
        # compile count is still bounded because _LR_BUCKETS=2 quantizes the
        # schedule to 3 distinct values (and the warmup loop covers them).
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
            # Resync the device step counter to match the restored
            # Python step so the on-device ramp_frac computation
            # produces the correct curriculum value.  This .fill_()
            # call uses a Python float ONCE at load time (not per
            # step), specializing a single transfer program — an
            # acceptable one-time cost vs. a persistent per-step leak.
            self._step_dev.fill_(float(self.global_step))
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
   

    # Number of discrete LR values the schedule can take on.  Each unique
    # LR becomes a compile-time constant in the XLA graph, so the total
    # compilation count is LR_BUCKETS × num_curriculum_phases × num_shapes.
    # For a 47M-param model, each compiled train_step graph pins ~500MB–1GB
    # of host RAM (HLO IR + PJRT executable + param/optimizer layouts).
    # 10 × 3 × 2 = 60 graphs × ~1GB = ~60GB of host cache, which OOM-killed
    # prior v4-8 runs.  Reducing LR_BUCKETS is the highest-leverage lever.
    _LR_BUCKETS = 2

    def _set_lr_from_schedule(self) -> None:
        """Set optimizer LR from schedule, quantized to reduce XLA recompilations.

        LR is a compile-time constant in the XLA graph — every distinct value
        triggers a new compilation.  We therefore quantize the cosine + warmup
        schedule to ``_LR_BUCKETS + 1`` discrete values.

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

        # Quantize the schedule to few discrete LR values.  See the comment
        # on ``_LR_BUCKETS`` above for the compilation-count math.
        bucket = int(frac * self._LR_BUCKETS)
        if bucket == self._last_lr_bucket:
            return  # Same bucket, no LR change, no recompilation
        self._last_lr_bucket = bucket

        lr = base_lr * (bucket / float(self._LR_BUCKETS))
        cfg = self.training_cfg
        for group in self.optimizer.param_groups:
            if cfg.train_heads and group.get("lr_multiplier") == "head":
                group['lr'] = lr * cfg.head_lr_multiplier
            elif cfg.train_heads:
                group['lr'] = lr * cfg.dynamics_lr_multiplier
            else:
                group['lr'] = lr

    # ── XLA warmup ──────────────────────────────────────────────────────

    def _xla_warmup_compile(
        self,
        loader_short: DataLoader,
        loader_long: DataLoader,
    ) -> None:
        """Pre-compile the XLA graph variants the training loop will produce.

        Builds **monolithic** XLA graphs that match real training: each
        iteration includes encode_frames → loss → backward → optimizer in
        a single lazy graph, synced once at the end.  Earlier versions
        called encode_frames outside the inner loop, which materialized
        z_clean after the first sync and produced fragmented graphs that
        never matched the real training graphs — causing mid-run
        recompilations.  encode_frames is now inside the innermost body.

        LR is zeroed during warmup.  This means the warmup's compiled
        optimizer graph uses lr=0 (compile-time constant), and real
        training's first non-zero LR bucket will trigger a one-time
        recompile per shape.  We accept those 2-3 mid-training compiles
        as a DELIBERATE trade-off: pre-compiling every LR bucket doubles
        the resident XLA program cache on the TPU (each program pins
        ~500MB–1GB of HBM for its code + constant tensors + execution
        buffers), and a v4-8 with 32GB per chip cannot hold that cache
        plus the ~18GB long-batch training step.  Fewer resident programs
        ⇒ more HBM headroom for the active program.

        Variant dimensions covered:
          - sequence shape       : short, long (if long_batch_ratio > 0)
          - curriculum phase     : warmup / ramp / full
          - training mode        : train (grad + backward + optimizer)
                                   val   (no_grad + model.train(False))
          - use_agent_tokens     : False, True (Phase 2 only)

        Complete no-op on CUDA/CPU.
        """
        if not self._is_xla:
            return

        import torch_xla
        import torch_xla.debug.metrics as met

        if is_master():
            print("[XLA] Warming up compilation...")

        saved_step = self.global_step
        saved_bucket = self._last_lr_bucket
        # Clone the device step counter so we can restore it exactly
        # after the warmup advances it (each warmup iteration calls
        # _step_dev.add_(1.0) to match training's graph structure).
        # .clone() produces a new device buffer — no Python scalar
        # leaks into the IR.
        saved_step_dev = self._step_dev.clone()

        # Pre-populate clip params so the clip op is in the traced graph.
        if self._cached_clip_params is None:
            all_params = list(self.model.parameters())
            if self.training_cfg.train_heads:
                all_params += list(self.reward_head.parameters()) + \
                    list(self.continue_head.parameters()) + \
                    list(self.policy_head.parameters())
            self._cached_clip_params = all_params

        # Save original LR values, then zero them for the warmup.
        # The optimizer.step() will trace the full optimizer graph but
        # apply zero-magnitude updates (lr=0 → param -= 0).  This avoids
        # weight drift WITHOUT cloning/restoring parameters, which is
        # critical for preserving XLA tensor identity and throughput.
        saved_lrs = [group['lr'] for group in self.optimizer.param_groups]
        for group in self.optimizer.param_groups:
            group['lr'] = 0.0

        # Skip long loader warmup if long batches are never used
        use_long = self.training_cfg.long_batch_ratio > 0
        loaders = [("short", loader_short)]
        if use_long and loader_long is not None:
            loaders.append(("long", loader_long))

        warmup_end = self.training_cfg.curriculum_warmup_steps
        ramp_end = self.training_cfg.curriculum_ramp_steps
        K_max = self.dynamics_cfg.K_max
        mtp_length = self.dynamics_cfg.mtp_length

        use_agent_variants = [False]
        if (self.training_cfg.train_heads and
                self.model.agent_embedding is not None):
            use_agent_variants = [False, True]

        train_modes = [True, False]

        # Only TWO phases now.  Previously we traced warmup / ramp / full
        # separately, but the unified on-device ramp makes the ramp and
        # full graphs identical (ramp_frac saturates at 1.0 past
        # ramp_end).  One bootstrap variant compiled here covers all
        # steps ≥ warmup_end.
        phases = [
            ("flow_only", saved_step),          # step 0, need_bootstrap=False
            ("bootstrap", warmup_end + 1),      # step warmup_end+1, need_bootstrap=True
        ]

        for name, loader in loaders:
            if loader is None:
                continue

            batch = next(iter(loader))
            frames, actions, rewards, dones = batch
            frames = frames.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)

            if frames.dim() == 4:
                frames = frames.unsqueeze(1)

            B, T = frames.shape[0:2]

            for train_mode in train_modes:
                self.model.train(train_mode)
                if self.training_cfg.train_heads:
                    self.reward_head.train(train_mode)
                    self.continue_head.train(train_mode)
                    self.policy_head.train(train_mode)

                for use_agent in use_agent_variants:
                    for phase_name, phase_step in phases:
                        self.global_step = phase_step
                        self._last_lr_bucket = -1

                        # encode_frames MUST be inside the loop so it is
                        # part of the same lazy graph as loss+backward+
                        # optimizer.  Calling it outside materializes
                        # z_clean after the first sync, producing
                        # fragmented graphs that never match real
                        # training's monolithic graphs.
                        z_clean = self.model.encode_frames(frames)

                        tau, d = sample_tau_and_d(
                            B, T, K_max=K_max, device=self.device,
                        )
                        tau[:, 0] = 1.0 - self.dynamics_cfg.tau_ctx

                        # Mirror the training loop's device-step increment
                        # so the warmup's traced graph includes this op.
                        # Otherwise training's graph would have an extra
                        # add_(1.0) node that isn't in the cached warmup
                        # program → recompile on first real step.
                        self._step_dev.add_(1.0)

                        if self.global_step < warmup_end:
                            d = torch.full_like(d, 1.0 / K_max)
                        else:
                            # Must mirror the training-loop ramp
                            # computation exactly (same ops, same
                            # tensor operands) so the traced graph
                            # matches training's graph.
                            step_offset = self._step_dev - self._warmup_end_dev
                            ramp_frac_dev = step_offset / self._ramp_range_dev
                            quantized = (
                                torch.floor(ramp_frac_dev * self._ramp_levels_dev)
                                / self._ramp_levels_dev
                            )
                            self._ramp_tensor.copy_(quantized)
                            self._ramp_tensor.clamp_(0.0, 1.0)
                            mask = torch.rand_like(d) < self._ramp_tensor
                            d_min_t = torch.full_like(d, 1.0 / K_max)
                            d = torch.where(mask, d, d_min_t)

                        z_noised, _ = add_noise(z_clean, tau)
                        need_bootstrap = self.global_step >= warmup_end

                        rewards_future, dones_future, actions_future = None, None, None
                        if use_agent and mtp_length > 0:
                            rewards_future, dones_future = build_mtp_targets(
                                rewards, dones, mtp_length,
                            )
                            actions_future = build_mtp_action_targets(
                                actions, mtp_length,
                            )

                        grad_ctx = torch.enable_grad() if train_mode else torch.no_grad()
                        with grad_ctx, torch.amp.autocast(
                            device_type=get_device_type(self.device),
                            enabled=self.training_cfg.amp,
                        ):
                            loss, _ = self._compute_loss(
                                z_clean, z_noised, actions, tau, d,
                                training=train_mode, compute_bootstrap=need_bootstrap,
                                rewards=rewards, dones=dones,
                                use_agent_tokens=use_agent,
                                rewards_future=rewards_future,
                                dones_future=dones_future,
                                actions_future=actions_future,
                            )

                        if train_mode:
                            self.optimizer.zero_grad(set_to_none=True)
                            self.scaler.scale(loss).backward()
                            self.scaler.unscale_(self.optimizer)
                            _xla_safe_clip_grad_norm(
                                self._cached_clip_params, self.training_cfg.grad_clip,
                            )
                            # Exercise LR schedule path (will set LR to 0 since
                            # we zeroed it, but traces the bucket-change code).
                            self._set_lr_from_schedule()
                            self.scaler.step(self.optimizer)  # lr=0 → zero update
                            self.scaler.update()
                        torch_xla.sync()

                        if is_master():
                            mode_str = "train" if train_mode else "val"
                            print(
                                f"  [XLA] {name} (T={T}, phase={phase_name}, "
                                f"mode={mode_str}, agent={use_agent}, "
                                f"bootstrap={need_bootstrap}) compiled"
                            )

        # Restore LR and Python-int state. No tensor identity is changed.
        for group, lr in zip(self.optimizer.param_groups, saved_lrs):
            group['lr'] = lr
        self.global_step = saved_step
        self._last_lr_bucket = saved_bucket
        # Restore the device step counter to the pre-warmup value using
        # a device-to-device copy (no Python scalar leaks).  Training
        # step 0 will start with _step_dev matching global_step=0
        # (fresh run) or match the checkpoint value (resume).
        self._step_dev.copy_(saved_step_dev)
        # Put model back in training mode for real training.
        self.model.train(True)
        if self.training_cfg.train_heads:
            self.reward_head.train(True)
            self.continue_head.train(True)
            self.policy_head.train(True)

        torch_xla.sync()
        gc.collect()
        met.clear_all()

        if is_master():
            print("[XLA] Warmup compilation complete")

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
        self._xla_warmup_compile(train_loader_short, train_loader_long)

        # Hang watchdog — converts silent wedges into clean exits.  If
        # self.global_step does not advance for 5 minutes, hard-exit via
        # os._exit(1).  The TPU is released and the user can immediately
        # --resume-from the last checkpoint instead of manually killing
        # the process.  Previous silent hangs (Apr 18) left the process
        # alive but stuck at a collective with a dead peer rank.
        import threading, time as _time, os as _os
        def _hang_watchdog(trainer, idle_threshold_s: int = 900):
            # Grace period: the first N training steps after warmup can
            # trigger cold-cache compiles (e.g. lr-bucket recompiles — warmup
            # uses lr=0, real training uses lr=4e-4) that take minutes each.
            # Run 6 (Apr 19) tripped a 300s watchdog during exactly this
            # tail because the ~/xla_cache directory was empty.  Don't
            # engage the idle timer until training has actually moved past
            # the first compile-heavy window.
            grace_steps = 50
            start_step = trainer.global_step
            last_step = trainer.global_step
            last_change = _time.monotonic()
            while True:
                _time.sleep(60)
                if trainer.global_step != last_step:
                    last_step = trainer.global_step
                    last_change = _time.monotonic()
                    continue
                if trainer.global_step - start_step < grace_steps:
                    last_change = _time.monotonic()
                    continue
                idle_for = _time.monotonic() - last_change
                if idle_for >= idle_threshold_s:
                    print(
                        f"[WATCHDOG] No progress for {idle_for:.0f}s at "
                        f"step {trainer.global_step}. Hard-exiting so "
                        f"the TPU is released and --resume-from works.",
                        flush=True,
                    )
                    _os._exit(1)
        threading.Thread(
            target=_hang_watchdog, args=(self,), daemon=True,
        ).start()

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
                    "epoch/train_flow_normed": train_metrics["loss/dynamics_flow_normed"],
                    "epoch/train_bootstrap_normed": train_metrics["loss/dynamics_bootstrap_normed"],
                    "epoch/train_reward": train_metrics["loss/dynamics_reward"],
                    "epoch/train_continue": train_metrics["loss/dynamics_continue"],
                    "global_step": self.global_step,
                }
                if val_metrics is not None:
                    epoch_log["epoch/val_loss"] = val_metrics["loss/dynamics_total"]
                    epoch_log["epoch/val_flow"] = val_metrics["loss/dynamics_flow"]
                    epoch_log["epoch/val_bootstrap"] = val_metrics["loss/dynamics_bootstrap"]
                    epoch_log["epoch/val_flow_normed"] = val_metrics["loss/dynamics_flow_normed"]
                    epoch_log["epoch/val_bootstrap_normed"] = val_metrics["loss/dynamics_bootstrap_normed"]
                    epoch_log["epoch/val_reward"] = val_metrics["loss/dynamics_reward"]
                    epoch_log["epoch/val_continue"] = val_metrics["loss/dynamics_continue"]
                # commit=True forces wandb to flush its in-memory event buffer
                # to disk (offline mode) or to the server (online mode).  Without
                # this, offline-mode runs accumulate every logged dict in RAM
                # until wandb.finish() — over a 300-epoch run that's GBs of
                # serialized payloads pinned in the master process.
                wandb.log(epoch_log, step=self.global_step, commit=True)

                # Surface the XLA compile count in the epoch line so host-RAM
                # blow-ups from runaway recompilation are immediately visible
                # in the training terminal (not just in wandb).
                compile_str = ""
                if self._is_xla:
                    import torch_xla.debug.metrics as met
                    cd = met.metric_data('CompileTime')
                    if cd is not None:
                        compile_str = f" xla_compiles={int(cd[0])}"

                print(
                    f"Epoch {epoch}: train_loss={train_metrics['loss/dynamics_total']:.4f} "
                    f"flow={train_metrics['loss/dynamics_flow']:.4f} "
                    f"bootstrap={train_metrics['loss/dynamics_bootstrap']:.4f}"
                    + (f" val_loss={val_metrics['loss/dynamics_total']:.4f}" if val_metrics else "")
                    + compile_str
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
        del state
        gc.collect()
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
            # Resync the device step counter — see the matching comment
            # in the Phase-2 load path above.  One-time specialization
            # cost (single .fill_() with Python float) in exchange for
            # a stable ramp_frac computation across the entire resumed run.
            self._step_dev.fill_(float(self.global_step))
        if "scheduler_bucket" in state:
            self._last_lr_bucket = state["scheduler_bucket"]

        # Sanity-check the loaded weights — if the checkpoint was saved
        # during a divergent run, the NaN/Inf values will propagate forward.
        # Fail loudly here instead of training on poisoned state.
        if is_master():
            bad_params = []
            for name, param in self.model.named_parameters():
                if not torch.isfinite(param).all().item():
                    bad_params.append(name)
            if bad_params:
                raise RuntimeError(
                    f"[FATAL] Checkpoint contains non-finite values in {len(bad_params)} "
                    f"param(s). First few: {bad_params[:5]}. "
                    f"Roll back to an earlier checkpoint."
                )
            print(
                f"[INFO] Checkpoint loaded: global_step={self.global_step}, "
                f"lr_bucket={self._last_lr_bucket}, weights all finite."
            )

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
        total_flow_normed = torch.tensor(0.0, device=self.device)
        total_bootstrap_normed = torch.tensor(0.0, device=self.device)
        total_reward = torch.tensor(0.0, device=self.device)
        total_continue = torch.tensor(0.0, device=self.device)
        total_bc = torch.tensor(0.0, device=self.device)
        total_steps = 0

        log_interval = self.training_cfg.log_interval
        # On XLA, fire model stats at the larger `..._xla` interval to
        # keep the per-rank XLA program cache bounded.  On CUDA/CPU, the
        # original interval (no compile pressure) is used.
        model_stats_interval = (
            self.training_cfg.log_model_stats_interval_xla
            if self._is_xla
            else self.training_cfg.log_model_stats_interval
        )
        use_agent = self.training_cfg.train_heads and self.model.agent_embedding is not None

        # On-device accumulators for smoothed logging (reset every log_interval)
        _log_loss = torch.tensor(0.0, device=self.device)
        _log_flow = torch.tensor(0.0, device=self.device)
        _log_bootstrap = torch.tensor(0.0, device=self.device)
        _log_flow_normed = torch.tensor(0.0, device=self.device)
        _log_bootstrap_normed = torch.tensor(0.0, device=self.device)
        _log_reward = torch.tensor(0.0, device=self.device)
        _log_continue = torch.tensor(0.0, device=self.device)
        _log_bc = torch.tensor(0.0, device=self.device)
        _log_grad_norm = torch.tensor(0.0, device=self.device)
        _log_tau = torch.tensor(0.0, device=self.device)
        _log_d = torch.tensor(0.0, device=self.device)
        # On-device training-health accumulators.  Computed inside the
        # training graph (part of the one-time-compiled training step),
        # so they add ZERO new compile events but surface lightweight
        # weight/gradient health every log_interval (10 steps).  This
        # replaces the per-call _DetachedParamView compile pressure as
        # the primary training-health signal; _DetachedParamView now
        # fires only every 2000 steps for occasional per-layer detail.
        _log_weight_norm_sq = torch.tensor(0.0, device=self.device)  # Σ ||w||² across params
        _log_weight_max = torch.tensor(0.0, device=self.device)      # max |w_i| in window
        _log_grad_max = torch.tensor(0.0, device=self.device)        # max |g_i| in window
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
            use_long = training and long_iter is not None and (random.random() < long_ratio)
            if use_long:
                try:
                    batch = next(long_iter)
                except StopIteration:
                    # Explicitly drop the exhausted iterator before allocating
                    # a new one — _MultiProcessingDataLoaderIter holds refs to
                    # worker queues + prefetched batches that otherwise leak
                    # across long-loader epoch boundaries.
                    del long_iter
                    gc.collect()
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

            # Advance the device step counter FIRST so the ramp
            # computation below reads the current step as a device
            # tensor value (not a Python int).  .add_(1.0) with the
            # constant 1.0 produces the same IR every step; only the
            # underlying tensor value changes at runtime.
            self._step_dev.add_(1.0)

            # Unified curriculum d-override.  Two Python-level branches
            # (flow-only vs. bootstrap-with-ramp) — NOT three as before.
            # The former ramp and full branches are now a single graph
            # because the on-device ramp_frac computation naturally
            # saturates at 1.0 past ramp_end, making the mask all-True
            # and producing d = natural (the old full-phase behavior).
            if self.global_step < warmup_end:
                # Flow-only warmup: all d_min, no bootstrap.
                d = torch.full_like(d, 1.0 / self.dynamics_cfg.K_max)
            else:
                # Bootstrap path with on-device ramp.  Every tensor in
                # this block is either a persistent device tensor or
                # derived from one — no Python floats leak into the IR,
                # so this branch compiles to a SINGLE XLA program
                # regardless of how ramp_frac evolves across steps.
                step_offset = self._step_dev - self._warmup_end_dev
                ramp_frac_dev = step_offset / self._ramp_range_dev
                # Quantize to 4 discrete levels via torch.floor on-device.
                # torch.floor(x*3)/3 maps [0,1] → {0, 1/3, 2/3, 1.0}.
                # Past ramp_end this overshoots (values > 1.0); clamp_(0,1)
                # saturates it → mask all-True → d = natural (full-phase).
                quantized = (
                    torch.floor(ramp_frac_dev * self._ramp_levels_dev)
                    / self._ramp_levels_dev
                )
                self._ramp_tensor.copy_(quantized)
                self._ramp_tensor.clamp_(0.0, 1.0)
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

            # Validation: wrap in no_grad to avoid building unused gradient graphs
            grad_ctx = torch.no_grad() if not training else torch.enable_grad()
            with grad_ctx, torch.amp.autocast(device_type=get_device_type(self.device), enabled=self.training_cfg.amp):
                loss, metrics = self._compute_loss(
                    z_clean, z_noised, actions, tau, d,
                    training=training, compute_bootstrap=need_bootstrap,
                    rewards=rewards, dones=dones,
                    use_agent_tokens=use_agent,
                    rewards_future=rewards_future, dones_future=dones_future,
                    actions_future=actions_future,
                )

            # During validation there is no optimizer.step() (which calls
            # mark_step), so the forward-pass graph stays lazy.  MpDeviceLoader
            # inserts a mark_step when yielding the *next* batch, but the
            # *last* batch has no subsequent yield — its graph defers to the
            # epoch-end .item() burst.  An explicit sync here ensures every
            # validation step produces a clean, materialized result and
            # prevents unbounded lazy-graph growth.
            if not training and self._is_xla:
                import torch_xla
                torch_xla.sync()

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                if self._cached_clip_params is None:
                    all_params = list(self.model.parameters())
                    if self.training_cfg.train_heads:
                        all_params += list(self.reward_head.parameters()) + \
                            list(self.continue_head.parameters()) + \
                            list(self.policy_head.parameters())
                    self._cached_clip_params = all_params
                all_params = self._cached_clip_params
                if self._is_xla:
                    # XLA-safe: stays entirely on-device (no .item() host sync)
                    grad_norm = _xla_safe_clip_grad_norm(all_params, self.training_cfg.grad_clip)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        all_params, self.training_cfg.grad_clip,
                    )

                # Divergence guard — check grad_norm for NaN/Inf every 50 steps.
                # Costs one host sync per check, but only every 50 steps so the
                # amortized overhead is negligible.  On NaN, dumps the sub-losses
                # so we can pinpoint which term went bad without needing another run.
                if self.global_step % 50 == 0 and isinstance(grad_norm, torch.Tensor):
                    gn_val = grad_norm.item()
                    if not math.isfinite(gn_val):
                        loss_val = loss.item()
                        sub = {k: float(v.detach().item())
                               for k, v in metrics.items()
                               if isinstance(v, torch.Tensor) and v.numel() == 1}
                        raise RuntimeError(
                            f"[FATAL] Training diverged at step {self.global_step}: "
                            f"grad_norm={gn_val}, loss={loss_val}, "
                            f"sub_losses={sub}. "
                            f"Check loss scaling / LR / curriculum."
                        )

                # Compute LR manually as a tensor so XLA doesn't recompile
                # on every LR change (scheduler embeds LR as compile-time constant)
                self._set_lr_from_schedule()

                self.scaler.step(self.optimizer)  # xm.optimizer_step on TPU (includes mark_step)
                self.scaler.update()

                # Free large intermediate tensors whose device buffers are
                # still pinned by Python references even though mark_step
                # already executed the graph that consumed them.  On the
                # v4-8, HBM headroom after the training step is only ~320MB.
                # frames alone is ~126MB (B×T×C×H×W×4 bytes);  z_clean and
                # z_noised add another ~64MB.  Freeing these before the
                # log-interval .tolist() gives the XLA runtime enough room
                # to load the log program (~365MB).
                del frames, z_clean, z_noised

                self.global_step += 1

                # Accumulate on-device — no .item() sync per step
                # All tensors detached to prevent holding the loss computation graph
                # across accumulator ops (root cause of host memory leak).
                _log_loss.add_(loss.detach())
                _log_flow.add_(metrics["loss_flow"].detach())
                _log_bootstrap.add_(metrics["loss_bootstrap"].detach())
                _log_flow_normed.add_(metrics["loss_flow_normed"].detach())
                _log_bootstrap_normed.add_(metrics["loss_bootstrap_normed"].detach())
                _log_reward.add_(metrics["loss_reward"].detach())
                _log_continue.add_(metrics["loss_continue"].detach())
                _log_bc.add_(metrics["loss_bc"].detach())
                _log_grad_norm.add_(grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                _log_tau.add_(tau_for_log.mean().detach())
                _log_d.add_(d_for_log.mean().detach())

                # On-device training-health summary.  Runs every step as
                # part of the compiled training graph.  Cost: ~N_params
                # reductions (a few ms on TPU), compiled ONCE alongside
                # the training step — no new compile events.
                # `_cached_clip_params` already covers every trainable
                # param (dynamics model + heads in Phase 2).
                with torch.no_grad():
                    _w_sq = torch.stack([
                        p.detach().pow(2).sum()
                        for p in self._cached_clip_params
                    ]).sum()
                    _w_max = torch.stack([
                        p.detach().abs().max()
                        for p in self._cached_clip_params
                    ]).max()
                    # Grads may be None for params with requires_grad=False
                    # that slipped into clip_params (shouldn't happen, but
                    # be defensive).  Stack only the non-None grads.
                    _grad_maxes = [
                        p.grad.detach().abs().max()
                        for p in self._cached_clip_params
                        if p.grad is not None
                    ]
                    _g_max = (
                        torch.stack(_grad_maxes).max()
                        if _grad_maxes else self._zero
                    )
                _log_weight_norm_sq.add_(_w_sq)
                # max-over-window (not sum): use torch.maximum in-place
                _log_weight_max.copy_(torch.maximum(_log_weight_max, _w_max))
                _log_grad_max.copy_(torch.maximum(_log_grad_max, _g_max))

                _log_count += 1

                if self.global_step % log_interval == 0:
                    # All-rank RssAnon read (each rank reads its own
                    # /proc/self/status — no cross-rank collective, no
                    # deadlock risk).  Previous version did master-only
                    # reads; this missed one-rank-ahead memory growth
                    # and contributed to the silent hang at epoch 51
                    # when ONE rank's process was OOM-killed while the
                    # others kept looking healthy (Apr 18 debug).
                    # Master still pushes to wandb; all ranks print
                    # a warning to stderr once their RSS crosses the
                    # 55 GB threshold (early warning before kill).
                    _local_rss = -1.0
                    try:
                        with open('/proc/self/status') as _f:
                            for _line in _f:
                                if _line.startswith('RssAnon:'):
                                    _local_rss = int(_line.split()[1]) / (1024 ** 2)
                                    break
                    except OSError:
                        pass

                    # TODO(REMOVE-AFTER-DEBUG): resource-leak tracer.
                    # Runs 7 and 8 crashed at ~5h with Process CPU Threads
                    # climbing linearly from ~2300 → ~10000 (Run 8) and
                    # resource_tracker reporting leaked /dev/shm semaphores.
                    # These three reads identify WHICH resource axis is
                    # leaking (threads, FDs, or shm entries) so Phase B can
                    # apply a targeted fix.  All three reads are passive
                    # (no syncs, no device ops).  Remove this block + the
                    # matching smoothed_metrics writes once the leak source
                    # is fixed.
                    _local_threads = -1
                    _local_fds = -1
                    _local_shm = -1
                    try:
                        with open('/proc/self/status') as _f:
                            for _line in _f:
                                if _line.startswith('Threads:'):
                                    _local_threads = int(_line.split()[1])
                                    break
                    except OSError:
                        pass
                    try:
                        import os as _os_probe
                        _local_fds = len(_os_probe.listdir('/proc/self/fd'))
                    except OSError:
                        pass
                    try:
                        import os as _os_probe
                        _local_shm = len(_os_probe.listdir('/dev/shm'))
                    except OSError:
                        pass

                    if _local_rss > 55 and not self._rss_warning_emitted:
                        try:
                            import torch_xla.runtime as xr
                            rank_id = xr.global_ordinal() if self._is_xla else 0
                        except Exception:
                            rank_id = 0
                        print(
                            f"[WARN][rank{rank_id}] RSS {_local_rss:.1f} GB "
                            f"— near OOM-kill threshold.  If a silent hang "
                            f"follows, this rank is the likely victim.",
                            flush=True,
                        )
                        self._rss_warning_emitted = True

                    if is_master():
                        n = max(_log_count, 1)
                        # Batch all scalar syncs into ONE .tolist() call.
                        # On XLA, each .item() triggers a full cycle:
                        #   trace → compile → load program into HBM → execute → D2H.
                        # With 11 sequential .item() calls, that's 11 separate
                        # compiled programs loaded into TPU HBM. When HBM is
                        # near-full from model params + Adam state + cached
                        # programs, loading ~365MB for each program can OOM.
                        # torch.stack + single .tolist() produces ONE program.
                        _gn = _log_grad_norm if isinstance(_log_grad_norm, torch.Tensor) else _log_loss.new_tensor(_log_grad_norm)
                        # Sum-accumulators stack (divide-by-n → mean).
                        # weight_norm_sq is included here because the mean
                        # sum-of-squares is what we want to track.
                        _log_vals = (torch.stack([
                            _log_loss, _log_flow, _log_bootstrap,
                            _log_flow_normed, _log_bootstrap_normed,
                            _log_reward, _log_continue, _log_bc,
                            _gn, _log_tau, _log_d,
                            _log_weight_norm_sq,
                        ]) / n).tolist()
                        # Max-accumulators stack (NO divide-by-n — these
                        # are max-over-window values, already aggregated).
                        # Separate tiny stack keeps the divide-by-n pattern
                        # correct and compiles to one stable program.
                        _max_vals = torch.stack([
                            _log_weight_max, _log_grad_max,
                        ]).tolist()
                        # weight_norm_sq → global weight L2 norm via sqrt.
                        # Computed on host (Python math), no device sync.
                        _mean_wsq = _log_vals[11]
                        _global_weight_norm = math.sqrt(max(_mean_wsq, 0.0))
                        self.metrics_buffer.update({
                            "train/loss": _log_vals[0],
                            "train/flow": _log_vals[1],
                            "train/bootstrap": _log_vals[2],
                            "train/flow_normed": _log_vals[3],
                            "train/bootstrap_normed": _log_vals[4],
                            "train/reward": _log_vals[5],
                            "train/continue": _log_vals[6],
                            "train/bc": _log_vals[7],
                            "train/grad_norm": _log_vals[8],
                            "weights/global_norm": _global_weight_norm,
                            "weights/max_abs": _max_vals[0],
                            "grads/max_abs": _max_vals[1],
                            "train/tau_mean": _log_vals[9],
                            "train/d_mean": _log_vals[10],
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

                        if self._is_xla:
                            import torch_xla.debug.metrics as met
                            compile_data = met.metric_data('CompileTime')
                            if compile_data is not None:
                                n_compiles = compile_data[0]
                                smoothed_metrics['xla/total_compilations'] = n_compiles
                                smoothed_metrics['xla/total_compile_time_s'] = compile_data[1]

                                # Compile-count ceiling WARN (not a hard abort).
                                #
                                # Lock in the expected post-warmup compile
                                # count at step ~100. On every subsequent
                                # log_interval, if compile count has grown,
                                # print a warning with the step number so
                                # we can see in the logs WHEN a new variant
                                # leaked — but don't halt training.
                                # Training continues so the user retains
                                # control of when to stop.
                                if (self.global_step >= 100 and
                                        self._expected_compile_count is None):
                                    self._expected_compile_count = int(n_compiles)
                                    if is_master():
                                        print(
                                            f"[XLA] Locked expected compile "
                                            f"count at {int(n_compiles)} "
                                            f"(step {self.global_step})."
                                        )
                                if (self._expected_compile_count is not None and
                                        is_master()):
                                    growth = int(n_compiles) - self._expected_compile_count
                                    if growth > self._compile_tolerance:
                                        print(
                                            f"[WARN] XLA compile count grew "
                                            f"from {self._expected_compile_count} "
                                            f"→ {int(n_compiles)} at step "
                                            f"{self.global_step} (+{growth}). "
                                            f"A new graph variant leaked past "
                                            f"warmup. Check _xla_warmup_compile "
                                            f"for missing LR bucket / curriculum "
                                            f"phase / shape / agent variants. "
                                            f"See dmesg for gasket_page_table "
                                            f"warnings — any mid-training "
                                            f"compile can wedge libtpu."
                                        )
                                        # Bump the baseline so we only warn
                                        # once per new growth plateau
                                        # instead of every log_interval.
                                        self._expected_compile_count = int(n_compiles)

                                # Soft warning for early post-warmup growth
                                # (before the lock-in at step 100).
                                if n_compiles > 25 and not self._compile_warning_emitted:
                                    print(
                                        f"[WARN] XLA compilation count is "
                                        f"{n_compiles} — expected ≤25 after "
                                        f"warmup. New graph variants leaking."
                                    )
                                    self._compile_warning_emitted = True

                            # Host-RAM logging — master's own RSS to wandb.
                            # `_local_rss` was already read above at the
                            # top of the log-interval block (as an
                            # all-rank sweep that triggers per-rank
                            # stderr warnings near the OOM threshold).
                            if _local_rss >= 0:
                                smoothed_metrics['host/rss_anon_gb'] = _local_rss
                            # TODO(REMOVE-AFTER-DEBUG): resource-leak tracer
                            # metrics.  See matching /proc reads above.
                            # Remove together when Phase B lands.
                            if _local_threads >= 0:
                                smoothed_metrics['host/threads'] = _local_threads
                            if _local_fds >= 0:
                                smoothed_metrics['host/fd_count'] = _local_fds
                            if _local_shm >= 0:
                                smoothed_metrics['host/shm_entries'] = _local_shm

                        if self.training_cfg.log_memory:
                            smoothed_metrics.update(GPUMemoryTracker.get_memory_stats(self.device))

                        # Per-layer model stats: CUDA/CPU ONLY.
                        #
                        # On XLA, _DetachedParamView leaks ~3-5 compiled
                        # programs per invocation (each ~600 MB of TPU
                        # HBM).  Earlier fixes throttled its frequency
                        # 40× (every 2000 steps on XLA via
                        # log_model_stats_interval_xla) which delayed
                        # but did not eliminate the leak — a 29-epoch
                        # run in Run 5 (Apr 19) still accumulated to
                        # device OOM around step 14000 when the cached
                        # program cache reached ~29 GB.  Rate matters:
                        # even a slow leak is a leak, and training
                        # sessions can be arbitrarily long.
                        #
                        # On-device health stats above
                        # (weights/global_norm, weights/max_abs,
                        # grads/max_abs) run inside the compiled
                        # training graph → zero compile events.  They
                        # cover the same failure modes (weight
                        # explosion, grad vanishing/explosion) at the
                        # global scalar level, which is what matters
                        # for training health monitoring.
                        #
                        # If per-layer histograms are ever needed for
                        # XLA debugging, load a saved checkpoint on CPU
                        # and call ModelStatistics.compute_weight_stats
                        # offline — _DetachedParamView is still
                        # available at the top of this file for that.
                        if (self.training_cfg.log_model_stats
                                and not self._is_xla
                                and self.global_step % model_stats_interval == 0):
                            smoothed_metrics.update(ModelStatistics.compute_weight_stats(self.model))
                            smoothed_metrics.update(ModelStatistics.compute_gradient_stats(self.model))
                            smoothed_metrics.update(self._compute_wd_metrics())

                        wandb.log(smoothed_metrics, step=self.global_step)

                    # Reset on ALL processes — prevents unbounded accumulation
                    _log_loss.zero_()
                    _log_flow.zero_()
                    _log_bootstrap.zero_()
                    _log_flow_normed.zero_()
                    _log_bootstrap_normed.zero_()
                    _log_reward.zero_()
                    _log_continue.zero_()
                    _log_bc.zero_()
                    _log_grad_norm.zero_()
                    _log_tau.zero_()
                    _log_d.zero_()
                    # Reset on-device health accumulators too.
                    _log_weight_norm_sq.zero_()
                    _log_weight_max.zero_()
                    _log_grad_max.zero_()
                    _log_count = 0

            # All tensors detached — prevents the epoch-level accumulators from
            # holding every per-step loss computation graph alive until epoch end.
            total_loss.add_(loss.detach())
            total_flow.add_(metrics["loss_flow"].detach())
            total_bootstrap.add_(metrics["loss_bootstrap"].detach())
            total_flow_normed.add_(metrics["loss_flow_normed"].detach())
            total_bootstrap_normed.add_(metrics["loss_bootstrap_normed"].detach())
            total_reward.add_(metrics["loss_reward"].detach())
            total_continue.add_(metrics["loss_continue"].detach())
            total_bc.add_(metrics["loss_bc"].detach())
            total_steps += 1

            # Periodic GC to free Python-side refs to XLA tensors and numpy copies.
            # Without this, host memory grows linearly (~50MB/step from stale refs).
            # On XLA, run more often (every 20 steps) to keep stale wrappers from
            # holding device buffers alive across many compiled-program executions.
            gc_interval = 20 if self._is_xla else 50
            if total_steps % gc_interval == 0:
                gc.collect()

        # Batch all epoch-end scalar syncs into ONE .tolist() call.
        # Same rationale as the log-interval batch: each .item() on XLA
        # loads a separate compiled program into TPU HBM.
        n = max(total_steps, 1)
        _epoch_vals = (torch.stack([
            total_loss, total_flow, total_bootstrap,
            total_flow_normed, total_bootstrap_normed,
            total_reward, total_continue, total_bc,
        ]) / n).tolist()
        epoch_metrics = {
            "loss/dynamics_total": _epoch_vals[0],
            "loss/dynamics_flow": _epoch_vals[1],
            "loss/dynamics_bootstrap": _epoch_vals[2],
            "loss/dynamics_flow_normed": _epoch_vals[3],
            "loss/dynamics_bootstrap_normed": _epoch_vals[4],
            "loss/dynamics_reward": _epoch_vals[5],
            "loss/dynamics_continue": _epoch_vals[6],
            "loss/dynamics_bc": _epoch_vals[7],
        }
        # Explicitly drop on-device accumulators so any Python references to
        # the underlying graph subtrees are released before the next epoch.
        # Each is a scalar so the bytes saved are tiny, but the IR cleanup
        # prevents stale graph nodes from pinning compiled-program buffers.
        del total_loss, total_flow, total_bootstrap
        del total_flow_normed, total_bootstrap_normed
        del total_reward, total_continue, total_bc
        del _log_loss, _log_flow, _log_bootstrap
        del _log_flow_normed, _log_bootstrap_normed
        del _log_reward, _log_continue, _log_bc
        del _log_grad_norm, _log_tau, _log_d
        del _log_weight_norm_sq, _log_weight_max, _log_grad_max
        if self._is_xla:
            gc.collect()
        return epoch_metrics

    # ── Loss computation ────────────────────────────────────────────────

    def _compute_wd_metrics(self, cpu_view: "_DetachedParamView | None" = None) -> Dict[str, float]:
        """Per-group effective weight decay: wd * sum(||p||^2).

        Shows how much regularization each group contributes.
        Only called at model_stats_interval so .item() syncs are acceptable.

        On XLA, pass a pre-built ``cpu_view`` and the iteration runs over
        the CPU snapshot of the optimizer's param groups instead of touching
        the live device tensors — keeps the XLA graph clean.
        """
        names = ["attn_ff", "no_decay", "default"]
        if self.training_cfg.train_heads:
            names.extend(["head_decay", "head_no_decay"])
        groups_iter = (
            cpu_view.optimizer_groups if cpu_view is not None
            else self.optimizer.param_groups
        )
        metrics = {}
        for name, group in zip(names, groups_iter):
            wd = group["weight_decay"]
            if wd == 0.0:
                continue
            sq_list = [p.data.norm(2) ** 2 for p in group["params"] if p.requires_grad]
            if not sq_list:
                continue
            sq_sum = torch.stack(sq_list).sum()
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
            "loss_flow_normed": loss_flow_normed.detach(),
            "loss_bootstrap_normed": loss_bootstrap_normed.detach(),
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
