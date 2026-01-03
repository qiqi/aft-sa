"""
Boundary conditions for 2D incompressible RANS on C-grid.

State vector: Q = [p, u, v, ν̃]
Ghost cell convention: Q has shape (NI + 2*NGHOST, NJ + 2*NGHOST, 4)

Both NumPy and JAX implementations provided.
"""

import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple, Any
from dataclasses import dataclass

from src.constants import NGHOST, N_VARS

# JAX imports
try:
    from src.physics.jax_config import jax, jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

NDArrayFloat = npt.NDArray[np.floating]


@dataclass
class FreestreamConditions:
    """Freestream flow conditions."""
    p_inf: float = 0.0
    u_inf: float = 1.0
    v_inf: float = 0.0
    nu_t_inf: float = 1e-9  # Default to negligible value
    
    @classmethod
    def from_mach_alpha(cls, mach: float, alpha_deg: float, 
                        reynolds: float = 6e6) -> 'FreestreamConditions':
        """Create from angle of attack (degrees) and Reynolds number.
        
        Sets nu_t_inf to 0.001 * nu_laminar so turbulent viscosity is negligible.
        """
        alpha: float = float(np.radians(alpha_deg))
        nu_laminar = 1.0 / reynolds if reynolds > 0 else 0.0
        return cls(
            p_inf=0.0,
            u_inf=float(np.cos(alpha)),
            v_inf=float(np.sin(alpha)),
            nu_t_inf=0.001 * nu_laminar  # 0.1% of laminar viscosity
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
        
        # Airfoil surface (no-slip)
        Q[i_start:i_end, 1, 0] = Q[i_start:i_end, j_int_first, 0]
        Q[i_start:i_end, 1, 1] = -Q[i_start:i_end, j_int_first, 1]
        Q[i_start:i_end, 1, 2] = -Q[i_start:i_end, j_int_first, 2]
        Q[i_start:i_end, 1, 3] = -Q[i_start:i_end, j_int_first, 3]
        
        Q[i_start:i_end, 0, 0] = Q[i_start:i_end, 1, 0]
        Q[i_start:i_end, 0, 1] = 2*Q[i_start:i_end, 1, 1] - Q[i_start:i_end, j_int_first, 1]
        Q[i_start:i_end, 0, 2] = 2*Q[i_start:i_end, 1, 2] - Q[i_start:i_end, j_int_first, 2]
        Q[i_start:i_end, 0, 3] = Q[i_start:i_end, 1, 3]
        
        # Wake cut (periodic)
        # First, average j=0 interior rows to enforce exact periodicity
        # This fixes flux imbalance at wake cut where flux leaves but doesn't return
        lower_wake_int_j0: NDArrayFloat = Q[NGHOST:i_start, j_int_first, :].copy()
        upper_wake_int_j0: NDArrayFloat = Q[i_end:NI + NGHOST, j_int_first, :].copy()
        
        # Average corresponding cells (lower[i] matches upper[n_wake-1-i])
        avg_j0: NDArrayFloat = 0.5 * (lower_wake_int_j0 + upper_wake_int_j0[::-1, :])
        Q[NGHOST:i_start, j_int_first, :] = avg_j0
        Q[i_end:NI + NGHOST, j_int_first, :] = avg_j0[::-1, :]
        
        # Now set ghost cells from the averaged interior
        lower_wake_int_j0 = Q[NGHOST:i_start, j_int_first, :]
        lower_wake_int_j1: NDArrayFloat = Q[NGHOST:i_start, j_int_first + 1, :]
        upper_wake_int_j0 = Q[i_end:NI + NGHOST, j_int_first, :]
        upper_wake_int_j1: NDArrayFloat = Q[i_end:NI + NGHOST, j_int_first + 1, :]
        
        Q[NGHOST:i_start, 1, :] = upper_wake_int_j0[::-1, :]
        Q[NGHOST:i_start, 0, :] = upper_wake_int_j1[::-1, :]
        
        Q[i_end:NI + NGHOST, 1, :] = lower_wake_int_j0[::-1, :]
        Q[i_end:NI + NGHOST, 0, :] = lower_wake_int_j1[::-1, :]
        
        return Q
    
    def apply_farfield(self, Q: NDArrayFloat) -> NDArrayFloat:
        """Apply farfield boundary conditions."""
        NI: int = Q.shape[0] - 2 * NGHOST
        i_int: slice = slice(NGHOST, -NGHOST)
        j_int_last: int = -NGHOST - 1
        
        if self.farfield_normals is None:
            for j_ghost in range(-NGHOST, 0):
                Q[:, j_ghost, 0] = self.freestream.p_inf
                Q[:, j_ghost, 1] = self.freestream.u_inf
                Q[:, j_ghost, 2] = self.freestream.v_inf
                Q[:, j_ghost, 3] = self.freestream.nu_t_inf
            return Q
        
        nx: NDArrayFloat
        ny: NDArrayFloat
        nx, ny = self.farfield_normals
        
        p_int: NDArrayFloat = Q[i_int, j_int_last, 0]
        u_int: NDArrayFloat = Q[i_int, j_int_last, 1]
        v_int: NDArrayFloat = Q[i_int, j_int_last, 2]
        nu_t_int: NDArrayFloat = Q[i_int, j_int_last, 3]
        
        p_int2: NDArrayFloat = Q[i_int, j_int_last - 1, 0]
        u_int2: NDArrayFloat = Q[i_int, j_int_last - 1, 1]
        v_int2: NDArrayFloat = Q[i_int, j_int_last - 1, 2]
        nu_t_int2: NDArrayFloat = Q[i_int, j_int_last - 1, 3]
        
        if len(nx) != NI:
            raise ValueError(
                f"Farfield BC shape mismatch: farfield_normals has {len(nx)} values, "
                f"but Q has {NI} interior cells."
            )
        
        p_inf: float = self.freestream.p_inf
        u_inf: float = self.freestream.u_inf
        v_inf: float = self.freestream.v_inf
        nu_t_inf: float = self.freestream.nu_t_inf
        
        U_n: NDArrayFloat = u_int * nx + v_int * ny
        is_outflow: NDArrayFloat = U_n >= 0
        
        # Force outflow at I-direction outlet cells (wake region at farfield corners)
        # These cells exit in the x-direction, not through the J-farfield boundary
        # We mark them as a special outlet that needs zero-gradient (Neumann) for all variables
        is_i_outlet: NDArrayFloat = np.zeros_like(is_outflow)
        if self.n_wake_points > 0:
            # Lower wake outlet: i = 0 to n_wake_points-1
            is_i_outlet[:self.n_wake_points] = True
            # Upper wake outlet: i = NI - n_wake_points to NI-1
            is_i_outlet[-self.n_wake_points:] = True
            # These are outflow cells
            is_outflow[is_i_outlet] = True
        
        is_inflow: NDArrayFloat = ~is_outflow
        
        u_b: NDArrayFloat = np.zeros_like(u_int)
        v_b: NDArrayFloat = np.zeros_like(v_int)
        p_b: NDArrayFloat = np.zeros_like(p_int)
        nu_t_b: NDArrayFloat = np.zeros_like(nu_t_int)
        
        u_b[is_inflow] = u_inf
        v_b[is_inflow] = v_inf
        nu_t_b[is_inflow] = nu_t_inf
        p_b[is_inflow] = p_int[is_inflow]
        
        u_b[is_outflow] = u_int[is_outflow]
        v_b[is_outflow] = v_int[is_outflow]
        nu_t_b[is_outflow] = nu_t_int[is_outflow]
        # For normal outflow, use p_inf; for I-direction outlet cells, use zero-gradient
        p_b[is_outflow] = p_inf
        # Override with zero-gradient (extrapolation) for I-direction outlet cells
        p_b[is_i_outlet] = p_int[is_i_outlet]
        
        Q[i_int, -2, 0] = 2 * p_b - p_int
        Q[i_int, -2, 1] = 2 * u_b - u_int
        Q[i_int, -2, 2] = 2 * v_b - v_int
        Q[i_int, -2, 3] = 2 * nu_t_b - nu_t_int
        
        Q[i_int, -1, 0] = 2 * Q[i_int, -2, 0] - p_b
        Q[i_int, -1, 1] = 2 * Q[i_int, -2, 1] - u_b
        Q[i_int, -1, 2] = 2 * Q[i_int, -2, 2] - v_b
        Q[i_int, -1, 3] = 2 * Q[i_int, -2, 3] - nu_t_b
        
        # I-direction farfield (downstream outlet)
        j_slice: slice = slice(0, -NGHOST)
        
        Q[1, j_slice, :] = Q[NGHOST, j_slice, :]
        Q[0, j_slice, :] = 2 * Q[NGHOST, j_slice, :] - Q[NGHOST + 1, j_slice, :]
        
        Q[-2, j_slice, :] = Q[-NGHOST - 1, j_slice, :]
        Q[-1, j_slice, :] = 2 * Q[-NGHOST - 1, j_slice, :] - Q[-NGHOST - 2, j_slice, :]
        
        # Set corner ghost cells (intersection of I and J ghost regions)
        # These are at (i=0,1, j=-2,-1) and (i=-2,-1, j=-2,-1)
        # Use simple extrapolation from the nearest interior corner
        for i_g in range(NGHOST):
            for j_g in range(-NGHOST, 0):
                # Lower-left corner (i=0,1, j=-2,-1) - copy from i=NGHOST, j=-NGHOST-1
                Q[i_g, j_g, :] = Q[NGHOST, -NGHOST - 1, :]
                # Upper-right corner (i=-2,-1, j=-2,-1) - copy from i=-NGHOST-1, j=-NGHOST-1
                Q[-NGHOST + i_g, j_g, :] = Q[-NGHOST - 1, -NGHOST - 1, :]
        
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
    
    NI: int = Q.shape[0] - 2 * NGHOST
    
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
        
        Q[0, :, 0] = Q[1, :, 0]
        Q[0, :, 1] = 2 * self.u_inlet - Q[1, :, 1]
        Q[0, :, 2] = 2 * self.v_inlet - Q[1, :, 2]
        Q[0, :, 3] = Q[1, :, 3]
        
        Q[-1, :, :] = Q[-2, :, :]
        
        Q[:, 0, 0] = Q[:, 1, 0]
        Q[:, 0, 1] = -Q[:, 1, 1]
        Q[:, 0, 2] = -Q[:, 1, 2]
        Q[:, 0, 3] = -Q[:, 1, 3]
        
        Q[:, -1, 0] = Q[:, -2, 0]
        Q[:, -1, 1] = -Q[:, -2, 1]
        Q[:, -1, 2] = -Q[:, -2, 2]
        Q[:, -1, 3] = -Q[:, -2, 3]
        
        return Q


# =============================================================================
# JAX Implementations
# =============================================================================

if JAX_AVAILABLE:
    
    def apply_surface_bc_jax(Q, n_wake_points, nghost=NGHOST):
        """
        JAX: Apply airfoil surface and wake cut boundary conditions.
        
        Note: Not JIT-compiled due to dynamic slice indices.
        Still runs on GPU as vectorized JAX operations.
        
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
        # Ghost layer 1 (j=1): reflect velocity, copy pressure, reflect nu_t
        Q = Q.at[i_start:i_end, 1, 0].set(Q[i_start:i_end, j_int_first, 0])
        Q = Q.at[i_start:i_end, 1, 1].set(-Q[i_start:i_end, j_int_first, 1])
        Q = Q.at[i_start:i_end, 1, 2].set(-Q[i_start:i_end, j_int_first, 2])
        Q = Q.at[i_start:i_end, 1, 3].set(-Q[i_start:i_end, j_int_first, 3])
        
        # Ghost layer 0 (j=0): extrapolate
        Q = Q.at[i_start:i_end, 0, 0].set(Q[i_start:i_end, 1, 0])
        Q = Q.at[i_start:i_end, 0, 1].set(2*Q[i_start:i_end, 1, 1] - Q[i_start:i_end, j_int_first, 1])
        Q = Q.at[i_start:i_end, 0, 2].set(2*Q[i_start:i_end, 1, 2] - Q[i_start:i_end, j_int_first, 2])
        Q = Q.at[i_start:i_end, 0, 3].set(Q[i_start:i_end, 1, 3])
        
        # Wake cut (periodic)
        # Average interior j=0 rows for periodicity
        lower_wake_int_j0 = Q[nghost:i_start, j_int_first, :]
        upper_wake_int_j0 = Q[i_end:NI + nghost, j_int_first, :]
        
        avg_j0 = 0.5 * (lower_wake_int_j0 + upper_wake_int_j0[::-1, :])
        Q = Q.at[nghost:i_start, j_int_first, :].set(avg_j0)
        Q = Q.at[i_end:NI + nghost, j_int_first, :].set(avg_j0[::-1, :])
        
        # Now set ghost cells from the averaged interior
        lower_wake_int_j0 = Q[nghost:i_start, j_int_first, :]
        lower_wake_int_j1 = Q[nghost:i_start, j_int_first + 1, :]
        upper_wake_int_j0 = Q[i_end:NI + nghost, j_int_first, :]
        upper_wake_int_j1 = Q[i_end:NI + nghost, j_int_first + 1, :]
        
        Q = Q.at[nghost:i_start, 1, :].set(upper_wake_int_j0[::-1, :])
        Q = Q.at[nghost:i_start, 0, :].set(upper_wake_int_j1[::-1, :])
        
        Q = Q.at[i_end:NI + nghost, 1, :].set(lower_wake_int_j0[::-1, :])
        Q = Q.at[i_end:NI + nghost, 0, :].set(lower_wake_int_j1[::-1, :])
        
        return Q
    
    def apply_farfield_bc_jax(Q, farfield_normals, p_inf, u_inf, v_inf, nu_t_inf, 
                               n_wake_points, nghost=NGHOST):
        """
        JAX: Apply farfield boundary conditions.
        
        Note: Not JIT-compiled due to dynamic slice indices.
        Still runs on GPU as vectorized JAX operations.
        
        Parameters
        ----------
        Q : jnp.ndarray
            State array (NI+2*nghost, NJ+2*nghost, 4).
        farfield_normals : tuple of jnp.ndarray
            (nx, ny) outward unit normals at farfield (NI,).
        p_inf, u_inf, v_inf, nu_t_inf : float
            Freestream conditions.
        n_wake_points : int
            Number of wake points.
        nghost : int
            Number of ghost cells.
            
        Returns
        -------
        Q : jnp.ndarray
            State with farfield BCs applied.
        """
        NI = Q.shape[0] - 2 * nghost
        j_int_last = -nghost - 1
        
        nx, ny = farfield_normals
        
        p_int = Q[nghost:-nghost, j_int_last, 0]
        u_int = Q[nghost:-nghost, j_int_last, 1]
        v_int = Q[nghost:-nghost, j_int_last, 2]
        nu_t_int = Q[nghost:-nghost, j_int_last, 3]
        
        # Determine inflow/outflow
        U_n = u_int * nx + v_int * ny
        is_outflow = U_n >= 0
        
        # I-direction outlet cells (wake corners)
        is_i_outlet = jnp.zeros(NI, dtype=bool)
        if n_wake_points > 0:
            is_i_outlet = is_i_outlet.at[:n_wake_points].set(True)
            is_i_outlet = is_i_outlet.at[-n_wake_points:].set(True)
        
        is_outflow = jnp.logical_or(is_outflow, is_i_outlet)
        is_inflow = ~is_outflow
        
        # Boundary values
        u_b = jnp.where(is_inflow, u_inf, u_int)
        v_b = jnp.where(is_inflow, v_inf, v_int)
        nu_t_b = jnp.where(is_inflow, nu_t_inf, nu_t_int)
        
        # For pressure: inflow uses interior, outflow uses p_inf (except I-outlet)
        p_b = jnp.where(is_inflow, p_int, p_inf)
        p_b = jnp.where(is_i_outlet, p_int, p_b)  # I-outlet uses zero-gradient
        
        # Set ghost cells (linear extrapolation)
        Q = Q.at[nghost:-nghost, -2, 0].set(2 * p_b - p_int)
        Q = Q.at[nghost:-nghost, -2, 1].set(2 * u_b - u_int)
        Q = Q.at[nghost:-nghost, -2, 2].set(2 * v_b - v_int)
        Q = Q.at[nghost:-nghost, -2, 3].set(2 * nu_t_b - nu_t_int)
        
        Q = Q.at[nghost:-nghost, -1, 0].set(2 * Q[nghost:-nghost, -2, 0] - p_b)
        Q = Q.at[nghost:-nghost, -1, 1].set(2 * Q[nghost:-nghost, -2, 1] - u_b)
        Q = Q.at[nghost:-nghost, -1, 2].set(2 * Q[nghost:-nghost, -2, 2] - v_b)
        Q = Q.at[nghost:-nghost, -1, 3].set(2 * Q[nghost:-nghost, -2, 3] - nu_t_b)
        
        # I-direction farfield (downstream outlet)
        NJ_ghost = Q.shape[1]
        j_end = NJ_ghost - nghost
        
        Q = Q.at[1, :j_end, :].set(Q[nghost, :j_end, :])
        Q = Q.at[0, :j_end, :].set(2 * Q[nghost, :j_end, :] - Q[nghost + 1, :j_end, :])
        
        Q = Q.at[-2, :j_end, :].set(Q[-nghost - 1, :j_end, :])
        Q = Q.at[-1, :j_end, :].set(2 * Q[-nghost - 1, :j_end, :] - Q[-nghost - 2, :j_end, :])
        
        # Corner ghost cells (nghost=2)
        corner_val_ll = Q[nghost, -nghost - 1, :]
        Q = Q.at[0, -2, :].set(corner_val_ll)
        Q = Q.at[0, -1, :].set(corner_val_ll)
        Q = Q.at[1, -2, :].set(corner_val_ll)
        Q = Q.at[1, -1, :].set(corner_val_ll)
        
        corner_val_lr = Q[-nghost - 1, -nghost - 1, :]
        Q = Q.at[-2, -2, :].set(corner_val_lr)
        Q = Q.at[-2, -1, :].set(corner_val_lr)
        Q = Q.at[-1, -2, :].set(corner_val_lr)
        Q = Q.at[-1, -1, :].set(corner_val_lr)
        
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
