"""
Solver components for 2D Incompressible RANS.

This package provides:
    - Time stepping schemes (explicit Euler, RK4 with local time stepping)
    - Boundary conditions for C-grid topology (airfoil, farfield, wake cut)
"""

from .time_stepping import (
    TimeStepConfig,
    SpectralRadius,
    compute_spectral_radii,
    compute_local_timestep,
    compute_global_timestep,
    ExplicitEuler,
    RungeKutta4,
)

from .boundary_conditions import (
    FreestreamConditions,
    BoundaryConditions,
    apply_boundary_conditions,
    initialize_state,
    InletOutletBC,
)

__all__ = [
    # Time stepping
    'TimeStepConfig',
    'SpectralRadius',
    'compute_spectral_radii',
    'compute_local_timestep', 
    'compute_global_timestep',
    'ExplicitEuler',
    'RungeKutta4',
    # Boundary conditions
    'FreestreamConditions',
    'BoundaryConditions',
    'apply_boundary_conditions',
    'initialize_state',
    'InletOutletBC',
]

