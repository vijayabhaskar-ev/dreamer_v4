"""Device abstraction for TPU/CUDA/CPU training.

Provides a unified interface so training code doesn't need
device-specific branches scattered throughout.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# XLA detection (cached at import time)
# ---------------------------------------------------------------------------

def _has_xla() -> bool:
    try:
        import torch_xla  
        return True
    except ImportError:
        return False


_XLA_AVAILABLE = _has_xla()


# ---------------------------------------------------------------------------
# Multi-device helpers
# ---------------------------------------------------------------------------

def is_master() -> bool:
    """True on ordinal 0 (or non-XLA). Safe to call outside xmp.spawn."""
    if _XLA_AVAILABLE:
        import torch_xla.core.xla_model as xm
        return xm.get_ordinal() == 0
    return True


def get_world_size() -> int:
    if _XLA_AVAILABLE:
        import torch_xla.core.xla_model as xm
        return xm.xrt_world_size()
    return 1


def get_ordinal() -> int:
    if _XLA_AVAILABLE:
        import torch_xla.core.xla_model as xm
        return xm.get_ordinal()
    return 0


def wrap_loader(loader, device):
    """Wrap a DataLoader with MpDeviceLoader for async host→device transfer."""
    if is_xla_device(device):
        import torch_xla.distributed.parallel_loader as pl
        return pl.MpDeviceLoader(loader, device)
    return loader


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def get_device(backend: str = "auto") -> torch.device:
    """Resolve a device string to a torch.device.

    Args:
        backend: One of "auto", "tpu", "cuda", "cpu".
            - "auto": TPU > CUDA > CPU
            - "tpu" : XLA device (raises if torch_xla unavailable)
            - "cuda": CUDA device (falls back to CPU if unavailable)
            - "cpu" : CPU
    """
    if backend == "tpu" or (backend == "auto" and _XLA_AVAILABLE):
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    if backend == "cuda" or (backend == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if backend == "tpu":
        raise RuntimeError("TPU requested but torch_xla is not installed.")
    return torch.device("cpu")


def should_use_xla(backend: str = "auto") -> bool:
    """Check if we should use XLA *without* initializing the runtime.

    Safe to call before ``xmp.spawn()``.
    """
    if backend == "tpu":
        return _XLA_AVAILABLE
    if backend == "auto":
        return _XLA_AVAILABLE
    return False


def is_xla_device(device: torch.device) -> bool:
    """Check whether *device* is an XLA/TPU device."""
    return str(device).startswith("xla")


def get_device_type(device: torch.device) -> str:
    """Return the device-type string used by ``torch.amp.autocast``."""
    if is_xla_device(device):
        return "xla"
    return device.type  # "cuda" or "cpu"


# ---------------------------------------------------------------------------
# XLA mark_step (no-op on CUDA/CPU)
# ---------------------------------------------------------------------------

def mark_step() -> None:
    """Trigger XLA graph compilation + execution.

    Must be called after every optimizer step on TPU.
    No-op when running on CUDA or CPU.
    """
    if _XLA_AVAILABLE:
        import torch_xla.core.xla_model as xm
        xm.mark_step()


# ---------------------------------------------------------------------------
# Checkpoint saving (XLA-aware)
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str, device: torch.device) -> None:
    """Save a checkpoint, using ``xm.save`` on TPU for proper serialization."""
    if is_xla_device(device):
        import torch_xla.core.xla_model as xm
        xm.save(state, path)
    else:
        torch.save(state, path)


# ---------------------------------------------------------------------------
# GradScaler abstraction
# ---------------------------------------------------------------------------

class NoOpGradScaler:
    """Drop-in replacement for ``torch.amp.GradScaler`` that does nothing.

    Used on TPU (XLA handles scaling internally) and CPU.
    """

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None: 
        pass

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if _XLA_AVAILABLE:
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(optimizer)  # all-reduce + step + mark_step
        else:
            optimizer.step()

    def update(self) -> None:
        pass

    def get_scale(self) -> float:
        return 1.0


def make_grad_scaler(device: torch.device, enabled: bool):
    """Create the appropriate GradScaler for the current device.

    - CUDA: ``torch.amp.GradScaler`` (real scaling when *enabled*)
    - TPU/CPU: ``NoOpGradScaler`` (scaling not supported / not needed)
    """
    if is_xla_device(device) or device.type == "cpu":
        return NoOpGradScaler()
    return torch.amp.GradScaler("cuda", enabled=enabled)
