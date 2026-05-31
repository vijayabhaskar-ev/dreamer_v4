"""Environment setup that MUST run before the first `import torch`.

Some PyTorch/Inductor behaviors are configured at import time via
environment variables. Setting them after `import torch` is too late —
the relevant subsystems have already initialized.

Rules for this file:
- DO NOT import torch, numpy, or any heavy library here.
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
# per CPU core. On a many-core host that becomes a large pool of idle Python
# interpreter subprocesses, each holding ~500 MB - 2 GB of imported torch
# modules in RAM. The total footprint can grow large enough to trigger OOM
# kills, and these workers are pure overhead for our workload. Cap them at 1
# per process.
_setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
_setdefault('TORCH_COMPILE_DEBUG', '0')


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
# runs; the cleanup happens explicitly at process startup in train_dynamics.py.
_dreamer_tmp = os.path.expanduser('~/dreamer_tmp')
os.makedirs(_dreamer_tmp, exist_ok=True)
_setdefault('TMPDIR', _dreamer_tmp)
_setdefault('TMP', _dreamer_tmp)
_setdefault('TEMP', _dreamer_tmp)
# wandb honors WANDB_CACHE_DIR / WANDB_DATA_DIR for its on-disk staging area.
# Pointing both at the same dreamer_tmp keeps wandb fully off /tmp.
_setdefault('WANDB_CACHE_DIR', _dreamer_tmp)
_setdefault('WANDB_DATA_DIR', _dreamer_tmp)
