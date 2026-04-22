"""Device abstraction for TPU/CUDA/CPU training.

Provides a unified interface so training code doesn't need
device-specific branches scattered throughout.
"""

from __future__ import annotations

from typing import Optional

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
        try:
            import torch_xla.core.xla_model as xm
            return xm.get_ordinal()
        except RuntimeError:
            # DataLoader worker processes can't access TPU devices.
            # Fall back to 0; worker_info.id provides per-worker diversity.
            return 0
    return 0


def wrap_loader(loader, device, enable_mp_device_loader: bool = False):
    """Wrap a DataLoader for the target device.

    enable_mp_device_loader:
        False (default): return the loader unchanged. On XLA, the
            training loop's `.to(xla_device)` is lazy and fuses into the
            compiled step graph, so MpDeviceLoader's async prefetch
            provides no measurable benefit at typical batch sizes
            (~1.5 MB/batch → ~0.2 ms PCIe transfer, vs ~1s step time).
            Required on torch_xla 2.7: MpDeviceLoader's PerDeviceLoader
            spawned ~1 background thread per next() call that did not
            reliably join, accumulating to ~10k threads over 5 hours and
            tripping the hang watchdog (Apr 20-21 2026 debug).
        True: re-enable MpDeviceLoader. Flip only when scaling to v4-32+
            AND after verifying the torch_xla version has the thread-leak
            fix (2.8+). Measure TPU duty cycle before vs after for ≥15
            min; keep enabled only if it shows a measurable improvement.
    """
    if enable_mp_device_loader and is_xla_device(device):
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
# XLA compilation cache (persists compiled graphs across restarts)
# ---------------------------------------------------------------------------

def initialize_xla_cache(
    cache_dir: Optional[str] = None,
    max_size_gb: float = 15.0,
) -> None:
    """Enable persistent XLA compilation cache. No-op on CUDA/CPU.

    Design:
    - Single shared cache directory across all TPU processes (rank 0..N-1)
    - Only rank 0 writes; ranks 1..N-1 are read-only
      → avoids concurrent-write corruption AND the N-fold disk bloat that
      per-rank caches produce
    - Auto-clear on rank 0 if the cache exceeds ``max_size_gb``
      → safety net against stale entries accumulating over weeks

    Honors ``$XLA_CACHE_DIR`` if set, else ``$TMPDIR/xla_cache`` if set,
    else falls back to ``/tmp/xla_cache``. Set ``XLA_CACHE_DIR=~/xla_cache``
    to put the cache in your home directory where you can manage it.
    """
    import os
    if cache_dir is None:
        cache_dir = os.environ.get('XLA_CACHE_DIR')
        if cache_dir is None:
            tmpdir = os.environ.get('TMPDIR')
            cache_dir = f'{tmpdir}/xla_cache' if tmpdir else '/tmp/xla_cache'

    os.makedirs(cache_dir, exist_ok=True)

    if not _XLA_AVAILABLE:
        return

    import torch_xla.runtime as xr

    # Determine whether this process is the writer (rank 0) or a read-only
    # reader (any other rank). If ordinal lookup fails (e.g., PJRT not yet
    # initialized), default to writer — single-process fallback.
    try:
        ordinal = xr.global_ordinal()
    except Exception:
        ordinal = 0
    is_writer = (ordinal == 0)

    # Auto-clear the cache on rank 0 if it exceeds the size limit, AND
    # delete any zero-byte files left behind by a crash mid-write.
    # A zero-byte .xla_proto would cause torch_xla to die later with
    # "TfrtTpuExecutable proto deserialization failed" when it tries to
    # load it, which is unrecoverable inside the TPU runtime.
    if is_writer:
        try:
            size_bytes = 0
            file_count = 0
            zero_byte_files = []
            for dirpath, _, filenames in os.walk(cache_dir):
                for f in filenames:
                    fpath = os.path.join(dirpath, f)
                    try:
                        sz = os.path.getsize(fpath)
                    except OSError:
                        continue
                    size_bytes += sz
                    file_count += 1
                    if sz == 0:
                        zero_byte_files.append(fpath)

            size_gb = size_bytes / (1024 ** 3)

            if zero_byte_files:
                print(
                    f"[XLA] Found {len(zero_byte_files)} zero-byte cache "
                    f"file(s) (likely half-written from a prior crash). "
                    f"Deleting to prevent deserialize error."
                )
                for fpath in zero_byte_files:
                    try:
                        os.remove(fpath)
                    except OSError as e:
                        print(f"[XLA]   Could not delete {fpath}: {e}")

            import shutil

            if size_gb > max_size_gb:
                print(
                    f"[XLA] Cache at {cache_dir} is {size_gb:.1f} GB "
                    f"(> {max_size_gb} GB limit). Clearing..."
                )
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
                file_count = 0
                size_gb = 0.0

            # Disk-space guard: PJRT writes compiled executables to the cache
            # during training without bound, and an ENOSPC there silently wedges
            # the training loop (we lost a week debugging this on a 100%-full
            # /dev/root). If free space on the cache filesystem is below the
            # threshold, clear the cache preemptively even if it's under the
            # size limit. This is independent of and stricter than max_size_gb.
            free_gb = shutil.disk_usage(cache_dir).free / (1024 ** 3)
            min_free_gb = 20.0
            if free_gb < min_free_gb:
                print(
                    f"[XLA] Only {free_gb:.1f} GB free on cache filesystem "
                    f"(< {min_free_gb} GB headroom). Clearing cache "
                    f"preemptively to avoid mid-training ENOSPC wedge."
                )
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
                file_count = 0
                size_gb = 0.0
                free_gb = shutil.disk_usage(cache_dir).free / (1024 ** 3)
                if free_gb < min_free_gb:
                    print(
                        f"[XLA][WARN] Still only {free_gb:.1f} GB free after "
                        f"clearing the cache. Something else is filling the "
                        f"disk. Training is likely to ENOSPC-wedge during "
                        f"compilation. Investigate before continuing."
                    )

            print(
                f"[XLA] Cache at {cache_dir}: "
                f"{file_count} files, {size_gb:.2f} GB, "
                f"{free_gb:.1f} GB free on filesystem"
            )
        except Exception as e:
            print(f"[XLA] Cache size check failed: {e}")

    xr.initialize_cache(cache_dir, readonly=not is_writer)


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
