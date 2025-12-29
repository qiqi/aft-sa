"""
Numerical methods for 2D RANS solver.

This module provides:
- JST flux scheme for artificial compressibility formulation
- Time stepping utilities
"""

from .fluxes import (
    compute_fluxes,
    compute_time_step,
    compute_convective_flux,
    compute_spectral_radius,
    FluxConfig,
    GridMetrics,
)

__all__ = [
    'compute_fluxes',
    'compute_time_step',
    'compute_convective_flux',
    'compute_spectral_radius',
    'FluxConfig',
    'GridMetrics',
]

