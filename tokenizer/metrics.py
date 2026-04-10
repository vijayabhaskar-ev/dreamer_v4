"""Training metrics utilities for enhanced logging and debugging.

Provides smoothed metrics, model statistics, GPU monitoring, and throughput tracking
for comprehensive training diagnostics suitable for technical documentation.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn


class MetricsBuffer:
    """Rolling average buffer for smoother WandB logging curves.
    
    Instead of logging noisy per-batch metrics, accumulate values and
    log their moving averages periodically for cleaner visualization.
    
    Example:
        >>> buffer = MetricsBuffer(window=10)
        >>> for batch in loader:
        ...     buffer.update({"loss": loss.item()})
        ...     if step % 10 == 0:
        ...         wandb.log(buffer.get_averages())
    """
    
    def __init__(self, window: int = 10):
        """Initialize metrics buffer.
        
        Args:
            window: Rolling average window size. Larger = smoother curves.
        """
        self.window = window
        self.buffers: Dict[str, deque] = {}
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Add new metric values to the buffer.
        
        Args:
            metrics: Dictionary of metric_name -> value pairs.
        """
        for key, value in metrics.items():
            if value is None:
                continue
            if key not in self.buffers:
                self.buffers[key] = deque(maxlen=self.window)
            self.buffers[key].append(value)
    
    def get_averages(self) -> Dict[str, float]:
        """Compute rolling averages for all buffered metrics.
        
        Returns:
            Dictionary of metric_name -> averaged_value pairs.
        """
        return {
            key: sum(values) / len(values) 
            for key, values in self.buffers.items() 
            if len(values) > 0
        }
    
    def reset(self) -> None:
        """Clear all buffered values."""
        self.buffers.clear()


class ModelStatistics:
    """Compute model weight and gradient statistics for debugging.
    
    Useful for detecting:
    - Weight explosion/collapse
    - Gradient vanishing/exploding by layer
    - Training dynamics over time
    """
    
    # Dynamics model parameter group patterns
    _DYNAMICS_GROUPS = {
        "proj": ["proj_in", "proj_out"],
        "attention": ["spatial_attn", "temporal_attn"],
        "embedding": ["action_embedding", "tau_d_embedding"],
        "ff": [".ff."],
        "norm": ["norm_spatial", "norm_temporal", "norm_ff", "norm."],
    }

    @classmethod
    def _classify_dynamics_param(cls, name_lower: str) -> str:
        """Return the dynamics group key if matched, else empty string."""
        for group, patterns in cls._DYNAMICS_GROUPS.items():
            if any(p in name_lower for p in patterns):
                return group
        return ""

    @staticmethod
    def compute_weight_stats(model: nn.Module) -> Dict[str, float]:
        """Compute weight norm statistics grouped by layer type.

        Accumulates squared norms on-device and calls .item() only once
        per group at the end — avoids per-parameter XLA syncs.

        Args:
            model: PyTorch model to analyze.

        Returns:
            Dictionary with weight norms for different layer groups.
        """
        # Build name→group mapping once
        groups: Dict[str, list] = {
            "encoder": [], "decoder": [], "attention": [],
            "embed": [], "other": [],
        }
        for g in ModelStatistics._DYNAMICS_GROUPS:
            groups[f"dyn_{g}"] = []

        total_params = 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            total_params += param.numel()
            name_lower = name.lower()

            dyn_group = ModelStatistics._classify_dynamics_param(name_lower)
            if dyn_group:
                groups[f"dyn_{dyn_group}"].append(param.data)
            elif name_lower.startswith('blocks.'):
                groups["encoder"].append(param.data)
            elif 'decoder' in name_lower:
                groups["decoder"].append(param.data)
            elif 'attn' in name_lower or 'attention' in name_lower:
                groups["attention"].append(param.data)
            elif 'embed' in name_lower or 'patch' in name_lower:
                groups["embed"].append(param.data)
            else:
                groups["other"].append(param.data)

        # Accumulate on-device, single .item() per group
        stats: Dict[str, float] = {}
        max_weight_t = None
        for key, params in groups.items():
            if not params:
                continue
            sq_sum = torch.stack([p.norm(2) ** 2 for p in params]).sum()
            stats[f"model/{key}_weight_norm"] = sq_sum.sqrt().item()
            group_max = torch.stack([p.abs().max() for p in params]).max()
            max_weight_t = group_max if max_weight_t is None else torch.maximum(max_weight_t, group_max)

        stats["model/max_weight"] = max_weight_t.item() if max_weight_t is not None else 0.0
        stats["model/total_params_millions"] = total_params / 1e6

        return stats

    @staticmethod
    def compute_gradient_stats(model: nn.Module) -> Dict[str, float]:
        """Compute gradient norm statistics grouped by layer type.

        Accumulates squared norms on-device and calls .item() only once
        per group at the end — avoids per-parameter XLA syncs.

        Args:
            model: PyTorch model with computed gradients.
        Returns:
            Dictionary with gradient norms for different layer groups.
        """
        groups: Dict[str, list] = {
            "encoder": [], "decoder": [], "attention": [],
        }
        for g in ModelStatistics._DYNAMICS_GROUPS:
            groups[f"dyn_{g}"] = []

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            name_lower = name.lower()

            dyn_group = ModelStatistics._classify_dynamics_param(name_lower)
            if dyn_group:
                groups[f"dyn_{dyn_group}"].append(param.grad)
            elif 'attn' in name_lower or 'attention' in name_lower:
                groups["attention"].append(param.grad)
            elif name_lower.startswith('blocks.'):
                groups["encoder"].append(param.grad)
            elif 'decoder' in name_lower:
                groups["decoder"].append(param.grad)

        stats: Dict[str, float] = {}
        for key, grads in groups.items():
            if not grads:
                continue
            sq_sum = torch.stack([g.norm(2) ** 2 for g in grads]).sum()
            stats[f"model/{key}_grad_norm"] = sq_sum.sqrt().item()

        return stats


class GPUMemoryTracker:
    """Track GPU memory utilization for debugging OOM issues."""
    
    @staticmethod
    def get_memory_stats(device: Optional[torch.device] = None) -> Dict[str, float]:
        """Get current GPU memory statistics.
        
        Args:
            device: CUDA device to query. Defaults to current device.
            
        Returns:
            Dictionary with memory stats in GB.
        """
        if not torch.cuda.is_available():
            return {}
        
        if device is None:
            device = torch.cuda.current_device()
        
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(device) / 1e9
        
        # Reset peak stats for next interval
        torch.cuda.reset_peak_memory_stats(device)
        
        return {
            "gpu/memory_allocated_gb": allocated,
            "gpu/memory_reserved_gb": reserved,
            "gpu/memory_peak_gb": max_allocated,
        }


class ThroughputTracker:
    """Track training throughput for performance monitoring.
    
    Measures samples per second and time per step to identify bottlenecks.
    """
    
    def __init__(self, warmup_steps: int = 5):
        """Initialize throughput tracker.
        
        Args:
            warmup_steps: Number of steps to skip for warm-up.
        """
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.last_time = time.perf_counter()
        self.samples_since_last = 0
        self.step_times: deque = deque(maxlen=100)
    
    def step(self, batch_size: int) -> Dict[str, float]:
        """Record a training step.
        
        Args:
            batch_size: Number of samples in current batch.
            
        Returns:
            Dictionary with throughput metrics.
        """
        current_time = time.perf_counter()
        step_time = current_time - self.last_time
        
        self.step_count += 1
        self.samples_since_last += batch_size
        self.step_times.append(step_time)
        
        # Skip warmup for accurate timing
        if self.step_count <= self.warmup_steps:
            self.last_time = current_time
            self.samples_since_last = 0
            return {}
        
        # Compute metrics
        avg_step_time = step_time
        samples_per_sec = batch_size / step_time if step_time > 0 else 0
        
        self.last_time = current_time
        
        return {
            "train/samples_per_sec": samples_per_sec,
            "train/step_time_ms": avg_step_time * 1000,
        }
    
    def reset(self) -> None:
        """Reset tracking state."""
        self.step_count = 0
        self.last_time = time.perf_counter()
        self.samples_since_last = 0
        self.step_times.clear()


__all__ = [
    "MetricsBuffer",
    "ModelStatistics", 
    "GPUMemoryTracker",
    "ThroughputTracker",
]
