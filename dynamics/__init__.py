"""Dynamics package for Dreamer V4 Phase 2 components.

This module exposes the dynamics model, configuration, and loss
used to predict future latent states via flow matching.
"""

from .config import DynamicsConfig
from .dynamic_model import DynamicsModel

__all__ = [
    "DynamicsConfig",
    "DynamicsModel",
]
