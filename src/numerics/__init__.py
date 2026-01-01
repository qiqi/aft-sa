"""
Numerical methods for 2D RANS solver.

This module provides:
- JST flux scheme for artificial compressibility formulation
- Time stepping utilities
- Green-Gauss gradient reconstruction
- Viscous flux computation
"""

from .fluxes import (
    compute_fluxes,
    compute_time_step,
    compute_convective_flux,
    compute_spectral_radius,
    FluxConfig,
    GridMetrics,
)

from .gradients import (
    compute_gradients,
    compute_vorticity,
    compute_strain_rate,
    GradientMetrics,
)

from .viscous_fluxes import (
    compute_viscous_fluxes,
    add_viscous_fluxes,
    compute_nu_tilde_diffusion,
)

from .forces import (
    compute_aerodynamic_forces,
    compute_surface_distributions,
    create_surface_vtk_fields,
    AeroForces,
    SurfaceData,
)

from .smoothing import apply_residual_smoothing

from .diagnostics import (
    compute_total_pressure_loss,
    compute_solution_bounds,
    compute_residual_statistics,
)

from .multigrid import (
    restrict_state,
    restrict_residual,
    prolongate_correction,
    prolongate_injection,
    compute_integral,
    compute_residual_sum,
    create_coarse_arrays,
)

__all__ = [
    # Convective fluxes
    'compute_fluxes',
    'compute_time_step',
    'compute_convective_flux',
    'compute_spectral_radius',
    'FluxConfig',
    'GridMetrics',
    # Gradients
    'compute_gradients',
    'compute_vorticity',
    'compute_strain_rate',
    'GradientMetrics',
    # Viscous fluxes
    'compute_viscous_fluxes',
    'add_viscous_fluxes',
    'compute_nu_tilde_diffusion',
    # Forces and surface data
    'compute_aerodynamic_forces',
    'compute_surface_distributions',
    'create_surface_vtk_fields',
    'AeroForces',
    'SurfaceData',
    # Smoothing
    'apply_residual_smoothing',
    # Diagnostics
    'compute_total_pressure_loss',
    'compute_solution_bounds',
    'compute_residual_statistics',
    # Multigrid transfers
    'restrict_state',
    'restrict_residual',
    'prolongate_correction',
    'prolongate_injection',
    'compute_integral',
    'compute_residual_sum',
    'create_coarse_arrays',
]

