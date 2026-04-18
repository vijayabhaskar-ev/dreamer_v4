"""Environment setup that MUST run before the first `import torch`.

Some PyTorch/Inductor/XLA behaviors are configured at import time via
environment variables. Setting them after `import torch` is too late —
the relevant subsystems have already initialized.

Rules for this file:
- DO NOT import torch, torch_xla, numpy, or any heavy library here.
- Only use the stdlib (os, sys, etc.).
- Keep it tiny so importing it is nearly free.

Usage: every training/eval entry point should `import _env_setup` as its
FIRST import, before any torch-related imports.
"""

from __future__ import annotations

import os


def _setdefault(key: str, value: str) -> None:
    """Set an env var only if the user hasn't already set it."""
    if key not in os.environ:
        os.environ[key] = value


# ---------------------------------------------------------------------------
# torch._inductor worker pool suppression
# ---------------------------------------------------------------------------
# PyTorch's Inductor compiler (used by torch.compile) eagerly initializes a
# subprocess worker pool at torch import time. By default it spawns one worker
# per CPU core. On a TPU v4-8 VM with ~96 cores AND xmp.spawn(nprocs=4), that
# becomes 4 * 96 = 384 idle Python interpreter subprocesses, each holding
# ~500 MB - 2 GB of imported torch modules in RAM. Total footprint can exceed
# 300 GB of host memory, which triggers OOM kills.
#
# We never call torch.compile() on XLA (see tokenizer/layers.py where the
# flex_attention + torch.compile path is gated on `not _XLA_AVAILABLE`), so
# these workers are pure overhead. Cap them at 1 per process.
_setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
_setdefault('TORCH_COMPILE_DEBUG', '0')


# ---------------------------------------------------------------------------
# XLA compile cache location
# ---------------------------------------------------------------------------
# Default to ~/xla_cache instead of /tmp/xla_cache — on some TPU VMs /tmp is
# disk-backed and can silently fill the root filesystem. The cache dir is
# still overridable by the user setting XLA_CACHE_DIR explicitly.
_setdefault('XLA_CACHE_DIR', os.path.expanduser('~/xla_cache'))


# ---------------------------------------------------------------------------
# tempfile redirect — keep wandb tempdirs out of /tmp
# ---------------------------------------------------------------------------
# wandb stages media/artifacts via tempfile.mkdtemp() and frequently leaves
# orphans in /tmp on crash/SIGKILL. Over many failed iterations this fills
# /tmp with thousands of `tmp*wandb*` directories, slowing every subsequent
# tempfile.mkdtemp() call (collision retry loop) and eventually wedging
# anything that needs a tempdir — including pip and the training loop itself.
#
# Redirect TMPDIR/TMP/TEMP to a dedicated location under $HOME so wandb's
# tempdir churn is contained AND survives a future /tmp wipe. We don't auto-
# clean here because import-time cleanup would race with concurrent training
# runs; the cleanup happens explicitly at xmp.spawn time in train_dynamics.py.
_dreamer_tmp = os.path.expanduser('~/dreamer_tmp')
os.makedirs(_dreamer_tmp, exist_ok=True)
_setdefault('TMPDIR', _dreamer_tmp)
_setdefault('TMP', _dreamer_tmp)
_setdefault('TEMP', _dreamer_tmp)
# wandb honors WANDB_CACHE_DIR / WANDB_DATA_DIR for its on-disk staging area.
# Pointing both at the same dreamer_tmp keeps wandb fully off /tmp.
_setdefault('WANDB_CACHE_DIR', _dreamer_tmp)
_setdefault('WANDB_DATA_DIR', _dreamer_tmp)


# ---------------------------------------------------------------------------
# PJRT in-memory caps — REMOVED
# ---------------------------------------------------------------------------
# Attempted to cap PJRT compile cache and buffer pool via XLA_FLAGS with
# --xla_pjrt_compile_cache_max_entries and --xla_pjrt_buffer_pool_max_size_gb.
# These flag names are NOT recognized by the torch_xla version on this TPU VM
# and caused a ~15× per-step slowdown (likely by triggering a fallback path
# or constant recompilation).  Do NOT re-add without verifying the exact
# flag names against `python -c "import torch_xla; help(torch_xla)"` first.
