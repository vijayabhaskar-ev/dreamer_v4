"""Device abstraction for CUDA/CPU training.

Provides a unified interface so training code doesn't need
device-specific branches scattered throughout.

The distributed-training helpers (is_master / get_world_size / get_ordinal)
return single-process values today. They are kept as a seam so a future
multi-GPU (torch.distributed / NCCL) implementation can fill in the bodies
without touching any callsite.
"""

from __future__ import annotations

from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Multi-device helpers (single-process; seam for future torch.distributed)
# ---------------------------------------------------------------------------

def is_master() -> bool:
    """True on the master process. Single-process -> always True.

    Future multi-GPU: return torch.distributed.get_rank() == 0.
    """
    return True


def get_world_size() -> int:
    """Number of training processes. Single-process -> 1."""
    return 1


def get_ordinal() -> int:
    """This process's global rank. Single-process -> 0."""
    return 0


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def get_device(backend: str = "auto") -> torch.device:
    """Resolve a device string to a torch.device.

    Args:
        backend: One of "auto", "cuda", "cpu".
            - "auto": CUDA > CPU
            - "cuda": CUDA device (falls back to CPU if unavailable)
            - "cpu" : CPU
    """
    if backend == "cuda" or (backend == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_type(device: torch.device) -> str:
    """Return the device-type string used by torch.amp.autocast."""
    return device.type  # "cuda" or "cpu"


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str, device: Optional[torch.device] = None) -> None:
    """Save a checkpoint. 'device' is accepted for call-site compatibility."""
    torch.save(state, path)


# ---------------------------------------------------------------------------
# GradScaler abstraction
# ---------------------------------------------------------------------------

class NoOpGradScaler:
    """Drop-in replacement for torch.amp.GradScaler that does nothing.

    Used on CPU (where AMP loss-scaling is not used). The step method is also
    the seam where a future multi-GPU gradient all-reduce would be inserted.
    """

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        pass

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        pass

    def get_scale(self) -> float:
        return 1.0


def make_grad_scaler(device: torch.device, enabled: bool):
    """Create the appropriate GradScaler for the current device.

    - CUDA: torch.amp.GradScaler (real scaling when enabled)
    - CPU : NoOpGradScaler (scaling not needed)
    """
    if device.type == "cpu":
        return NoOpGradScaler()
    return torch.amp.GradScaler("cuda", enabled=enabled)
