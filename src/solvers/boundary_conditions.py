"""
Boundary conditions for 2D incompressible RANS on C-grid.

State vector: Q = [p, u, v, ν̃]
Ghost cell convention: Q has shape (NI + 2*NGHOST, NJ + 2*NGHOST, 4)
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple, Any
from dataclasses import dataclass

from src.constants import NGHOST, N_VARS
from src.physics.jax_config import jax, jnp

NDArrayFloat = npt.NDArray[np.floating]


@dataclass
class FreestreamConditions:
    """Freestream flow conditions."""
    p_inf: float = 0.0
    u_inf: float = 1.0
    v_inf: float = 0.0
    nu_t_inf: float = 1e-9  # Default to negligible value
    
    @classmethod
    def from_alpha(cls, alpha_deg: float,
                   reynolds: float = 6e6,
                   chi_inf: float = 3.0) -> 'FreestreamConditions':
        """Create from angle of attack (degrees) and Reynolds number.

        Parameters
        ----------
        alpha_deg : float
            Angle of attack in degrees.
        reynolds : float
            Reynolds number.
        chi_inf : float
            Initial/farfield turbulent viscosity ratio χ = ν̃/ν.
            Typical values: 3-5 for external aerodynamics.
        """
        alpha: float = float(np.radians(alpha_deg))
        nu_laminar = 1.0 / reynolds if reynolds > 0 else 0.0
        return cls(
            p_inf=0.0,
            u_inf=float(np.cos(alpha)),
            v_inf=float(np.sin(alpha)),
            nu_t_inf=chi_inf * nu_laminar
        )


class BoundaryConditions:
    """Boundary condition handler for C-grid topology."""
    
    freestream: FreestreamConditions
    j_surface: int
    j_farfield: int
    i_wake_start: int
    i_wake_end: int
    n_wake_points: int
    farfield_normals: Optional[Tuple[NDArrayFloat, NDArrayFloat]]
    beta: float
    c_art: float
    
    def __init__(self, 
                 freestream: Optional[FreestreamConditions] = None,
                 j_surface: int = 0,
                 j_farfield: int = -1,
                 i_wake_start: int = 0,
                 i_wake_end: int = -1,
                 n_wake_points: int = 0,
                 farfield_normals: Optional[Tuple[NDArrayFloat, NDArrayFloat]] = None,
                 beta: float = 10.0) -> None:
        self.freestream = freestream or FreestreamConditions()
        self.j_surface = j_surface
        self.j_farfield = j_farfield
        self.i_wake_start = i_wake_start
        self.i_wake_end = i_wake_end
        self.n_wake_points = n_wake_points
        self.farfield_normals = farfield_normals
        self.beta = beta
        self.c_art = float(np.sqrt(beta))
    
    def apply(self, Q: NDArrayFloat) -> NDArrayFloat:
        """Apply all boundary conditions."""
        Q = Q.copy()
        Q = self.apply_surface(Q)
        Q = self.apply_farfield(Q)
        return Q
    
    def apply_surface(self, Q: NDArrayFloat) -> NDArrayFloat:
        """Apply airfoil surface and wake cut boundary conditions."""
        NI: int = Q.shape[0] - 2 * NGHOST
        
        n_wake: int = self.n_wake_points if self.n_wake_points > 0 else NI // 6
        i_wake_end_lower: int = n_wake
        i_wake_start_upper: int = NI - n_wake
        
        i_start: int = i_wake_end_lower + NGHOST
        i_end: int = i_wake_start_upper + NGHOST
        
        j_int_first: int = NGHOST
        
        # Airfoil surface (no-slip wall)
        # Q stores physical velocity. No-slip requires u_phys = 0 at wall.
        # Ghost cell: (u_ghost + u_interior) / 2 = 0, so u_ghost = -u_interior
        Q[i_start:i_end, 1, 0] = Q[i_start:i_end, j_int_first, 0]
        Q[i_start:i_end, 1, 1] = -Q[i_start:i_end, j_int_first, 1]
        Q[i_start:i_end, 1, 2] = -Q[i_start:i_end, j_int_first, 2]
        # nuHat: antisymmetric BC gives nuHat=0 at wall (correct physics)
        # Ghost = -interior, but clamp to prevent extreme negative values
        nuHat_interior: NDArrayFloat = np.maximum(Q[i_start:i_end, j_int_first, 3], 0.0)
        Q[i_start:i_end, 1, 3] = -nuHat_interior
        
        Q[i_start:i_end, 0, 0] = Q[i_start:i_end, 1, 0]
        Q[i_start:i_end, 0, 1] = 2*Q[i_start:i_end, 1, 1] - Q[i_start:i_end, j_int_first, 1]
        Q[i_start:i_end, 0, 2] = 2*Q[i_start:i_end, 1, 2] - Q[i_start:i_end, j_int_first, 2]
        # nuHat: extrapolate (more negative, but interior was clamped so bounded)
        Q[i_start:i_end, 0, 3] = 2*Q[i_start:i_end, 1, 3] - nuHat_interior
        
        # Wake cut (periodic)
        # Ghost cells from interior of opposite side
        lower_wake_int_j0: NDArrayFloat = Q[NGHOST:i_start, j_int_first, :]
        lower_wake_int_j1: NDArrayFloat = Q[NGHOST:i_start, j_int_first + 1, :]
        upper_wake_int_j0: NDArrayFloat = Q[i_end:NI + NGHOST, j_int_first, :]
        upper_wake_int_j1: NDArrayFloat = Q[i_end:NI + NGHOST, j_int_first + 1, :]
        
        Q[NGHOST:i_start, 1, :] = upper_wake_int_j0[::-1, :]
        Q[NGHOST:i_start, 0, :] = upper_wake_int_j1[::-1, :]
        
        Q[i_end:NI + NGHOST, 1, :] = lower_wake_int_j0[::-1, :]
        Q[i_end:NI + NGHOST, 0, :] = lower_wake_int_j1[::-1, :]
        
        return Q
    
    def apply_farfield(self, Q: NDArrayFloat) -> NDArrayFloat:
        """Apply Dirichlet farfield boundary conditions.
        
        Sets all farfield ghost cells to freestream values to stabilize the
        outer boundary and prevent oscillations. This is combined with a 
        sponge layer (2nd-order dissipation) in the flux computation to 
        absorb outgoing waves.
        
        J-Farfield (Outer Arc): Ghost cells set to freestream.
        I-Farfield (Downstream Outlets): Ghost cells set to freestream.
        
        Note: This overrides wake extrapolation at outlets - the wake is
        effectively clamped to freestream at the farfield boundary.
        """
        p_inf: float = self.freestream.p_inf
        u_inf: float = self.freestream.u_inf
        v_inf: float = self.freestream.v_inf
        nu_t_inf: float = self.freestream.nu_t_inf
        
        # J-direction farfield (outer arc) - Dirichlet to freestream
        for j_ghost in range(-NGHOST, 0):
            Q[:, j_ghost, 0] = p_inf
            Q[:, j_ghost, 1] = u_inf
            Q[:, j_ghost, 2] = v_inf
            Q[:, j_ghost, 3] = nu_t_inf
        
        # I-direction farfield (downstream outlet) - Dirichlet to freestream
        j_slice: slice = slice(0, -NGHOST)
        
        # Lower I boundary (i=0, i=1)
        Q[0, j_slice, 0] = p_inf
        Q[0, j_slice, 1] = u_inf
        Q[0, j_slice, 2] = v_inf
        Q[0, j_slice, 3] = nu_t_inf
        Q[1, j_slice, 0] = p_inf
        Q[1, j_slice, 1] = u_inf
        Q[1, j_slice, 2] = v_inf
        Q[1, j_slice, 3] = nu_t_inf
        
        # Upper I boundary (i=-2, i=-1)
        Q[-2, j_slice, 0] = p_inf
        Q[-2, j_slice, 1] = u_inf
        Q[-2, j_slice, 2] = v_inf
        Q[-2, j_slice, 3] = nu_t_inf
        Q[-1, j_slice, 0] = p_inf
        Q[-1, j_slice, 1] = u_inf
        Q[-1, j_slice, 2] = v_inf
        Q[-1, j_slice, 3] = nu_t_inf
        
        # Corner ghost cells - also set to freestream
        for i_g in range(NGHOST):
            for j_g in range(-NGHOST, 0):
                Q[i_g, j_g, 0] = p_inf
                Q[i_g, j_g, 1] = u_inf
                Q[i_g, j_g, 2] = v_inf
                Q[i_g, j_g, 3] = nu_t_inf
                Q[-NGHOST + i_g, j_g, 0] = p_inf
                Q[-NGHOST + i_g, j_g, 1] = u_inf
                Q[-NGHOST + i_g, j_g, 2] = v_inf
                Q[-NGHOST + i_g, j_g, 3] = nu_t_inf
        
        return Q


def apply_boundary_conditions(Q: NDArrayFloat, 
                              freestream: Optional[FreestreamConditions] = None) -> NDArrayFloat:
    """Convenience function to apply all boundary conditions."""
    bc: BoundaryConditions = BoundaryConditions(freestream=freestream)
    return bc.apply(Q)


def initialize_state(NI: int, NJ: int, 
                     freestream: Optional[FreestreamConditions] = None) -> NDArrayFloat:
    """Initialize state vector to freestream conditions."""
    if freestream is None:
        freestream = FreestreamConditions()
    
    Q: NDArrayFloat = np.zeros((NI + 2 * NGHOST, NJ + 2 * NGHOST, N_VARS))
    Q[:, :, 0] = freestream.p_inf
    Q[:, :, 1] = freestream.u_inf
    Q[:, :, 2] = freestream.v_inf
    Q[:, :, 3] = freestream.nu_t_inf
    
    return Q


def apply_initial_wall_damping(Q: NDArrayFloat, 
                                grid_metrics: Any,
                                decay_length: float = 0.1,
                                n_wake: int = 0) -> NDArrayFloat:
    """Apply velocity damping near wall for smooth cold start."""
    Q = Q.copy()
    
    wall_dist: NDArrayFloat
    if hasattr(grid_metrics, 'wall_distance'):
        wall_dist = grid_metrics.wall_distance
    else:
        raise ValueError("grid_metrics must have a 'wall_distance' attribute")
    
    _NI: int = Q.shape[0] - 2 * NGHOST
    
    damping_factor: NDArrayFloat = 1.0 - np.exp(-wall_dist / decay_length)
    
    if n_wake > 0:
        damping_factor[:n_wake, :] = 1.0
        damping_factor[-n_wake:, :] = 1.0
    
    int_slice: slice = slice(NGHOST, -NGHOST)
    Q[int_slice, int_slice, 1] *= damping_factor
    Q[int_slice, int_slice, 2] *= damping_factor
    
    return Q


class InletOutletBC:
    """BC for channel/duct flows (not C-grid)."""
    
    u_inlet: float
    v_inlet: float
    p_outlet: float
    
    def __init__(self, inlet_velocity: Tuple[float, float] = (1.0, 0.0), 
                 p_outlet: float = 0.0) -> None:
        self.u_inlet, self.v_inlet = inlet_velocity
        self.p_outlet = p_outlet
    
    def apply(self, Q: NDArrayFloat) -> NDArrayFloat:
        """Apply inlet/outlet/wall boundary conditions."""
        Q = Q.copy()
        
        # Inlet (i=0)
        Q[0, :, 0] = Q[1, :, 0]
        Q[0, :, 1] = 2 * self.u_inlet - Q[1, :, 1]
        Q[0, :, 2] = 2 * self.v_inlet - Q[1, :, 2]
        Q[0, :, 3] = Q[1, :, 3]
        
        # Outlet (i=-1)
        Q[-1, :, :] = Q[-2, :, :]
        
        # Bottom wall (j=0): antisymmetric velocity, nuHat with clamping
        Q[:, 0, 0] = Q[:, 1, 0]
        Q[:, 0, 1] = -Q[:, 1, 1]
        Q[:, 0, 2] = -Q[:, 1, 2]
        # nuHat: clamp interior to non-negative before antisymmetric BC
        Q[:, 0, 3] = -np.maximum(Q[:, 1, 3], 0.0)
        
        # Top wall (j=-1): antisymmetric velocity, nuHat with clamping
        Q[:, -1, 0] = Q[:, -2, 0]
        Q[:, -1, 1] = -Q[:, -2, 1]
        Q[:, -1, 2] = -Q[:, -2, 2]
        # nuHat: clamp interior to non-negative before antisymmetric BC
        Q[:, -1, 3] = -np.maximum(Q[:, -2, 3], 0.0)
        
        return Q


# =============================================================================
# JAX Implementations
# =============================================================================


def apply_surface_bc_jax(Q, n_wake_points, nghost=NGHOST):
    """
    JAX: Apply airfoil surface and wake cut boundary conditions.
    
    Parameters
    ----------
    Q : jnp.ndarray
        State array (NI+2*nghost, NJ+2*nghost, 4).
    n_wake_points : int
        Number of wake points on each side.
    nghost : int
        Number of ghost cells.
        
    Returns
    -------
    Q : jnp.ndarray
        State with surface/wake BCs applied.
    """
    NI = Q.shape[0] - 2 * nghost
    
    n_wake = n_wake_points if n_wake_points > 0 else NI // 6
    i_wake_end_lower = n_wake
    i_wake_start_upper = NI - n_wake
    
    i_start = i_wake_end_lower + nghost
    i_end = i_wake_start_upper + nghost
    j_int_first = nghost
    
    # Airfoil surface (no-slip wall)
    # Q stores physical velocity. No-slip requires u_phys = 0 at wall.
    # Ghost cell: (u_ghost + u_interior) / 2 = 0, so u_ghost = -u_interior
    # Ghost layer 1 (j=1)
    Q = Q.at[i_start:i_end, 1, 0].set(Q[i_start:i_end, j_int_first, 0])
    Q = Q.at[i_start:i_end, 1, 1].set(-Q[i_start:i_end, j_int_first, 1])
    Q = Q.at[i_start:i_end, 1, 2].set(-Q[i_start:i_end, j_int_first, 2])
    # nuHat: antisymmetric BC gives nuHat=0 at wall (correct physics)
    # Ghost = -interior, but clamp to prevent extreme negative values in JST stencil
    nuHat_interior = jnp.maximum(Q[i_start:i_end, j_int_first, 3], 0.0)
    Q = Q.at[i_start:i_end, 1, 3].set(-nuHat_interior)
    
    # Ghost layer 0 (j=0): extrapolate from j=1
    Q = Q.at[i_start:i_end, 0, 0].set(Q[i_start:i_end, 1, 0])
    Q = Q.at[i_start:i_end, 0, 1].set(2*Q[i_start:i_end, 1, 1] - Q[i_start:i_end, j_int_first, 1])
    Q = Q.at[i_start:i_end, 0, 2].set(2*Q[i_start:i_end, 1, 2] - Q[i_start:i_end, j_int_first, 2])
    # nuHat: extrapolate (more negative, but interior was clamped so bounded)
    Q = Q.at[i_start:i_end, 0, 3].set(2*Q[i_start:i_end, 1, 3] - nuHat_interior)
    
    # Wake cut (periodic)
    # Ghost cells from interior of opposite side
    lower_wake_int_j0 = Q[nghost:i_start, j_int_first, :]
    lower_wake_int_j1 = Q[nghost:i_start, j_int_first + 1, :]
    upper_wake_int_j0 = Q[i_end:NI + nghost, j_int_first, :]
    upper_wake_int_j1 = Q[i_end:NI + nghost, j_int_first + 1, :]
    
    # Set ghost cells from the opposite side's interior (periodic BC)
    # Ghost j=1 gets interior j=j_int_first from opposite side
    # Ghost j=0 gets interior j=j_int_first+1 from opposite side
    Q = Q.at[nghost:i_start, 1, :].set(upper_wake_int_j0[::-1, :])
    Q = Q.at[nghost:i_start, 0, :].set(upper_wake_int_j1[::-1, :])
    
    Q = Q.at[i_end:NI + nghost, 1, :].set(lower_wake_int_j0[::-1, :])
    Q = Q.at[i_end:NI + nghost, 0, :].set(lower_wake_int_j1[::-1, :])
    
    return Q

def apply_farfield_bc_jax(Q, farfield_normals, p_inf, u_inf, v_inf, nu_t_inf, 
                           n_wake_points, nghost=NGHOST):
    """
    JAX: Apply Dirichlet farfield boundary conditions.
    
    Sets all farfield ghost cells to freestream values to stabilize the
    outer boundary. Combined with sponge layer dissipation in flux computation.
    
    Parameters
    ----------
    Q : jnp.ndarray
        State array (NI+2*nghost, NJ+2*nghost, 4).
    farfield_normals : tuple of jnp.ndarray
        (nx, ny) outward unit normals at farfield (NI,) - not used with Dirichlet.
    p_inf, u_inf, v_inf, nu_t_inf : float
        Freestream conditions.
    n_wake_points : int
        Number of wake points - not used with Dirichlet.
    nghost : int
        Number of ghost cells.
        
    Returns
    -------
    Q : jnp.ndarray
        State with farfield BCs applied.
    """
    NJ_total = Q.shape[1]
    
    # J-direction farfield (outer arc) - Dirichlet to freestream
    # Set all J ghost cells (last nghost columns)
    Q = Q.at[:, -2, 0].set(p_inf)
    Q = Q.at[:, -2, 1].set(u_inf)
    Q = Q.at[:, -2, 2].set(v_inf)
    Q = Q.at[:, -2, 3].set(nu_t_inf)
    Q = Q.at[:, -1, 0].set(p_inf)
    Q = Q.at[:, -1, 1].set(u_inf)
    Q = Q.at[:, -1, 2].set(v_inf)
    Q = Q.at[:, -1, 3].set(nu_t_inf)
    
    # I-direction farfield (downstream outlets) - Dirichlet to freestream
    j_end = NJ_total - nghost
    
    # Lower I boundary (i=0, i=1)
    Q = Q.at[0, :j_end, 0].set(p_inf)
    Q = Q.at[0, :j_end, 1].set(u_inf)
    Q = Q.at[0, :j_end, 2].set(v_inf)
    Q = Q.at[0, :j_end, 3].set(nu_t_inf)
    Q = Q.at[1, :j_end, 0].set(p_inf)
    Q = Q.at[1, :j_end, 1].set(u_inf)
    Q = Q.at[1, :j_end, 2].set(v_inf)
    Q = Q.at[1, :j_end, 3].set(nu_t_inf)
    
    # Upper I boundary (i=-2, i=-1)
    Q = Q.at[-2, :j_end, 0].set(p_inf)
    Q = Q.at[-2, :j_end, 1].set(u_inf)
    Q = Q.at[-2, :j_end, 2].set(v_inf)
    Q = Q.at[-2, :j_end, 3].set(nu_t_inf)
    Q = Q.at[-1, :j_end, 0].set(p_inf)
    Q = Q.at[-1, :j_end, 1].set(u_inf)
    Q = Q.at[-1, :j_end, 2].set(v_inf)
    Q = Q.at[-1, :j_end, 3].set(nu_t_inf)
    
    # Corner ghost cells - also freestream
    for i_g in range(nghost):
        for j_g in range(-nghost, 0):
            Q = Q.at[i_g, j_g, 0].set(p_inf)
            Q = Q.at[i_g, j_g, 1].set(u_inf)
            Q = Q.at[i_g, j_g, 2].set(v_inf)
            Q = Q.at[i_g, j_g, 3].set(nu_t_inf)
            Q = Q.at[-nghost + i_g, j_g, 0].set(p_inf)
            Q = Q.at[-nghost + i_g, j_g, 1].set(u_inf)
            Q = Q.at[-nghost + i_g, j_g, 2].set(v_inf)
            Q = Q.at[-nghost + i_g, j_g, 3].set(nu_t_inf)
    
    return Q

def apply_bc_jax(Q, farfield_normals, freestream, n_wake_points, nghost=NGHOST):
    """
    JAX: Apply all boundary conditions.
    
    Parameters
    ----------
    Q : jnp.ndarray
        State array.
    farfield_normals : tuple
        (nx, ny) outward unit normals at farfield.
    freestream : FreestreamConditions
        Freestream conditions.
    n_wake_points : int
        Number of wake points.
    nghost : int
        Number of ghost cells.
        
    Returns
    -------
    Q : jnp.ndarray
        State with all BCs applied.
    """
    Q = apply_surface_bc_jax(Q, n_wake_points, nghost)
    
    if farfield_normals is not None:
        nx, ny = farfield_normals
        Q = apply_farfield_bc_jax(
            Q, (jnp.asarray(nx), jnp.asarray(ny)),
            freestream.p_inf, freestream.u_inf, freestream.v_inf, 
            freestream.nu_t_inf, n_wake_points, nghost
        )
    
    return Q

# =========================================================================
# JIT-compiled BC Factory Functions
# =========================================================================

def make_apply_bc_jit(NI: int, NJ: int, n_wake_points: int, 
                      nx: 'jnp.ndarray', ny: 'jnp.ndarray',
                      freestream: FreestreamConditions,
                      nghost: int = NGHOST):
    """
    Create a JIT-compiled BC function with Dirichlet farfield.
    
    Uses freestream values at all farfield boundaries to stabilize the
    outer boundary. Combined with sponge layer dissipation in flux computation.
    
    Parameters
    ----------
    NI, NJ : int
        Number of interior cells.
    n_wake_points : int
        Number of wake points on each side.
    nx, ny : jnp.ndarray
        Farfield outward unit normals (NI,) - not used with Dirichlet.
    freestream : FreestreamConditions
        Freestream conditions.
    nghost : int
        Number of ghost cells.
        
    Returns
    -------
    apply_bc : callable
        JIT-compiled function: Q_new = apply_bc(Q)
    """
    # Pre-compute all indices (these become constants in the JIT closure)
    n_wake = n_wake_points if n_wake_points > 0 else NI // 6
    i_wake_end_lower = n_wake
    i_wake_start_upper = NI - n_wake
    
    i_start = i_wake_end_lower + nghost  # airfoil start
    i_end = i_wake_start_upper + nghost  # airfoil end
    j_int_first = nghost
    _j_int_last = NJ + nghost - 1  # last interior J
    
    # Farfield values
    p_inf = freestream.p_inf
    u_inf = freestream.u_inf
    v_inf = freestream.v_inf
    nu_t_inf = freestream.nu_t_inf
    
    # Upper wake end index
    i_upper_end = NI + nghost
    
    # J boundary end (exclude J ghosts)
    j_end = NJ + nghost
    
    @jax.jit
    def apply_bc(Q):
        """Apply all boundary conditions (JIT-compiled) with Dirichlet farfield."""
        
        # === Surface BC: Airfoil wall (no-slip) ===
        # Q stores physical velocity. No-slip requires u_phys = 0 at wall.
        # Ghost cell: (u_ghost + u_interior) / 2 = 0, so u_ghost = -u_interior
        # Ghost layer 1 (j=1)
        Q = Q.at[i_start:i_end, 1, 0].set(Q[i_start:i_end, j_int_first, 0])
        Q = Q.at[i_start:i_end, 1, 1].set(-Q[i_start:i_end, j_int_first, 1])
        Q = Q.at[i_start:i_end, 1, 2].set(-Q[i_start:i_end, j_int_first, 2])
        Q = Q.at[i_start:i_end, 1, 3].set(-Q[i_start:i_end, j_int_first, 3])
        
        # Ghost layer 0 (j=0)
        Q = Q.at[i_start:i_end, 0, 0].set(Q[i_start:i_end, 1, 0])
        Q = Q.at[i_start:i_end, 0, 1].set(2*Q[i_start:i_end, 1, 1] - Q[i_start:i_end, j_int_first, 1])
        Q = Q.at[i_start:i_end, 0, 2].set(2*Q[i_start:i_end, 1, 2] - Q[i_start:i_end, j_int_first, 2])
        Q = Q.at[i_start:i_end, 0, 3].set(Q[i_start:i_end, 1, 3])
        
        # === Surface BC: Wake cut (periodic) ===
        # Ghost cells from interior of opposite side
        lower_wake_int_j0 = Q[nghost:i_start, j_int_first, :]
        lower_wake_int_j1 = Q[nghost:i_start, j_int_first + 1, :]
        upper_wake_int_j0 = Q[i_end:i_upper_end, j_int_first, :]
        upper_wake_int_j1 = Q[i_end:i_upper_end, j_int_first + 1, :]
        
        Q = Q.at[nghost:i_start, 1, :].set(upper_wake_int_j0[::-1, :])
        Q = Q.at[nghost:i_start, 0, :].set(upper_wake_int_j1[::-1, :])
        Q = Q.at[i_end:i_upper_end, 1, :].set(lower_wake_int_j0[::-1, :])
        Q = Q.at[i_end:i_upper_end, 0, :].set(lower_wake_int_j1[::-1, :])
        
        # === Farfield BC: J-direction (Dirichlet to freestream) ===
        Q = Q.at[:, -2, 0].set(p_inf)
        Q = Q.at[:, -2, 1].set(u_inf)
        Q = Q.at[:, -2, 2].set(v_inf)
        Q = Q.at[:, -2, 3].set(nu_t_inf)
        Q = Q.at[:, -1, 0].set(p_inf)
        Q = Q.at[:, -1, 1].set(u_inf)
        Q = Q.at[:, -1, 2].set(v_inf)
        Q = Q.at[:, -1, 3].set(nu_t_inf)
        
        # === Farfield BC: I-direction (Dirichlet to freestream) ===
        # Lower I boundary (i=0, i=1)
        Q = Q.at[0, :j_end, 0].set(p_inf)
        Q = Q.at[0, :j_end, 1].set(u_inf)
        Q = Q.at[0, :j_end, 2].set(v_inf)
        Q = Q.at[0, :j_end, 3].set(nu_t_inf)
        Q = Q.at[1, :j_end, 0].set(p_inf)
        Q = Q.at[1, :j_end, 1].set(u_inf)
        Q = Q.at[1, :j_end, 2].set(v_inf)
        Q = Q.at[1, :j_end, 3].set(nu_t_inf)
        
        # Upper I boundary (i=-2, i=-1)
        Q = Q.at[-2, :j_end, 0].set(p_inf)
        Q = Q.at[-2, :j_end, 1].set(u_inf)
        Q = Q.at[-2, :j_end, 2].set(v_inf)
        Q = Q.at[-2, :j_end, 3].set(nu_t_inf)
        Q = Q.at[-1, :j_end, 0].set(p_inf)
        Q = Q.at[-1, :j_end, 1].set(u_inf)
        Q = Q.at[-1, :j_end, 2].set(v_inf)
        Q = Q.at[-1, :j_end, 3].set(nu_t_inf)
        
        # Corner ghost cells - also freestream
        Q = Q.at[0, -2, 0].set(p_inf)
        Q = Q.at[0, -2, 1].set(u_inf)
        Q = Q.at[0, -2, 2].set(v_inf)
        Q = Q.at[0, -2, 3].set(nu_t_inf)
        Q = Q.at[0, -1, 0].set(p_inf)
        Q = Q.at[0, -1, 1].set(u_inf)
        Q = Q.at[0, -1, 2].set(v_inf)
        Q = Q.at[0, -1, 3].set(nu_t_inf)
        Q = Q.at[1, -2, 0].set(p_inf)
        Q = Q.at[1, -2, 1].set(u_inf)
        Q = Q.at[1, -2, 2].set(v_inf)
        Q = Q.at[1, -2, 3].set(nu_t_inf)
        Q = Q.at[1, -1, 0].set(p_inf)
        Q = Q.at[1, -1, 1].set(u_inf)
        Q = Q.at[1, -1, 2].set(v_inf)
        Q = Q.at[1, -1, 3].set(nu_t_inf)
        
        Q = Q.at[-2, -2, 0].set(p_inf)
        Q = Q.at[-2, -2, 1].set(u_inf)
        Q = Q.at[-2, -2, 2].set(v_inf)
        Q = Q.at[-2, -2, 3].set(nu_t_inf)
        Q = Q.at[-2, -1, 0].set(p_inf)
        Q = Q.at[-2, -1, 1].set(u_inf)
        Q = Q.at[-2, -1, 2].set(v_inf)
        Q = Q.at[-2, -1, 3].set(nu_t_inf)
        Q = Q.at[-1, -2, 0].set(p_inf)
        Q = Q.at[-1, -2, 1].set(u_inf)
        Q = Q.at[-1, -2, 2].set(v_inf)
        Q = Q.at[-1, -2, 3].set(nu_t_inf)
        Q = Q.at[-1, -1, 0].set(p_inf)
        Q = Q.at[-1, -1, 1].set(u_inf)
        Q = Q.at[-1, -1, 2].set(v_inf)
        Q = Q.at[-1, -1, 3].set(nu_t_inf)
        
        return Q
    
    return apply_bc