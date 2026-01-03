"""
Numerical methods for 2D RANS solver.
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
    compute_viscous_fluxes_with_sa_jax,
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

from .dissipation import (
    compute_sponge_sigma,
    compute_sponge_sigma_jax,
    compute_jst_blending_coefficients,
    compute_jst_blending_coefficients_jax,
)

from .sa_sources import (
    compute_sa_source_jax,
    compute_sa_production_only_jax,
    compute_sa_destruction_only_jax,
    compute_cb2_term_jax,
    compute_turbulent_viscosity_jax,
    compute_effective_viscosity_jax,
)

__all__ = [
    'compute_fluxes',
    'compute_time_step',
    'compute_convective_flux',
    'compute_spectral_radius',
    'FluxConfig',
    'GridMetrics',
    'compute_gradients',
    'compute_vorticity',
    'compute_strain_rate',
    'GradientMetrics',
    'compute_viscous_fluxes',
    'add_viscous_fluxes',
    'compute_nu_tilde_diffusion',
    'compute_viscous_fluxes_with_sa_jax',
    'compute_aerodynamic_forces',
    'compute_surface_distributions',
    'create_surface_vtk_fields',
    'AeroForces',
    'SurfaceData',
    'apply_residual_smoothing',
    'compute_total_pressure_loss',
    'compute_solution_bounds',
    'compute_residual_statistics',
    'compute_sponge_sigma',
    'compute_sponge_sigma_jax',
    'compute_jst_blending_coefficients',
    'compute_jst_blending_coefficients_jax',
    'compute_sa_source_jax',
    'compute_sa_production_only_jax',
    'compute_sa_destruction_only_jax',
    'compute_cb2_term_jax',
    'compute_turbulent_viscosity_jax',
    'compute_effective_viscosity_jax',
]
