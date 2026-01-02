"""
Boundary conditions for 2D incompressible RANS on C-grid.

State vector: Q = [p, u, v, ν̃]
Ghost cell convention: Q has shape (NI + 2*NGHOST, NJ + 2*NGHOST, 4)
"""

import numpy as np
from typing import NamedTuple, Optional
from dataclasses import dataclass

from src.constants import NGHOST, N_VARS


@dataclass
class FreestreamConditions:
    """Freestream flow conditions."""
    p_inf: float = 0.0
    u_inf: float = 1.0
    v_inf: float = 0.0
    nu_t_inf: float = 3e-6
    
    @classmethod
    def from_mach_alpha(cls, mach: float, alpha_deg: float, 
                        nu_ratio: float = 3.0) -> 'FreestreamConditions':
        """Create from angle of attack (degrees)."""
        alpha = np.radians(alpha_deg)
        return cls(
            p_inf=0.0,
            u_inf=np.cos(alpha),
            v_inf=np.sin(alpha),
            nu_t_inf=nu_ratio * 1e-6
        )


class BoundaryConditions:
    """Boundary condition handler for C-grid topology."""
    
    def __init__(self, 
                 freestream: Optional[FreestreamConditions] = None,
                 j_surface: int = 0,
                 j_farfield: int = -1,
                 i_wake_start: int = 0,
                 i_wake_end: int = -1,
                 n_wake_points: int = 0,
                 farfield_normals: Optional[tuple] = None,
                 beta: float = 10.0):
        self.freestream = freestream or FreestreamConditions()
        self.j_surface = j_surface
        self.j_farfield = j_farfield
        self.i_wake_start = i_wake_start
        self.i_wake_end = i_wake_end
        self.n_wake_points = n_wake_points
        self.farfield_normals = farfield_normals
        self.beta = beta
        self.c_art = np.sqrt(beta)
    
    def apply(self, Q: np.ndarray) -> np.ndarray:
        """Apply all boundary conditions."""
        Q = Q.copy()
        Q = self.apply_surface(Q)
        Q = self.apply_farfield(Q)
        return Q
    
    def apply_surface(self, Q: np.ndarray) -> np.ndarray:
        """Apply airfoil surface and wake cut boundary conditions."""
        NI = Q.shape[0] - 2 * NGHOST
        
        n_wake = self.n_wake_points if self.n_wake_points > 0 else NI // 6
        i_wake_end_lower = n_wake
        i_wake_start_upper = NI - n_wake
        
        i_start = i_wake_end_lower + NGHOST
        i_end = i_wake_start_upper + NGHOST
        
        j_int_first = NGHOST
        
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
        lower_wake_int_j0 = Q[NGHOST:i_start, j_int_first, :]
        lower_wake_int_j1 = Q[NGHOST:i_start, j_int_first + 1, :]
        upper_wake_int_j0 = Q[i_end:NI + NGHOST, j_int_first, :]
        upper_wake_int_j1 = Q[i_end:NI + NGHOST, j_int_first + 1, :]
        
        Q[NGHOST:i_start, 1, :] = upper_wake_int_j0[::-1, :]
        Q[NGHOST:i_start, 0, :] = upper_wake_int_j1[::-1, :]
        
        Q[i_end:NI + NGHOST, 1, :] = lower_wake_int_j0[::-1, :]
        Q[i_end:NI + NGHOST, 0, :] = lower_wake_int_j1[::-1, :]
        
        return Q
    
    def apply_farfield(self, Q: np.ndarray) -> np.ndarray:
        """Apply farfield boundary conditions."""
        NI = Q.shape[0] - 2 * NGHOST
        i_int = slice(NGHOST, -NGHOST)
        j_int_last = -NGHOST - 1
        
        if self.farfield_normals is None:
            for j_ghost in range(-NGHOST, 0):
                Q[:, j_ghost, 0] = self.freestream.p_inf
                Q[:, j_ghost, 1] = self.freestream.u_inf
                Q[:, j_ghost, 2] = self.freestream.v_inf
                Q[:, j_ghost, 3] = self.freestream.nu_t_inf
            return Q
        
        nx, ny = self.farfield_normals
        
        p_int = Q[i_int, j_int_last, 0]
        u_int = Q[i_int, j_int_last, 1]
        v_int = Q[i_int, j_int_last, 2]
        nu_t_int = Q[i_int, j_int_last, 3]
        
        p_int2 = Q[i_int, j_int_last - 1, 0]
        u_int2 = Q[i_int, j_int_last - 1, 1]
        v_int2 = Q[i_int, j_int_last - 1, 2]
        nu_t_int2 = Q[i_int, j_int_last - 1, 3]
        
        if len(nx) != NI:
            raise ValueError(
                f"Farfield BC shape mismatch: farfield_normals has {len(nx)} values, "
                f"but Q has {NI} interior cells."
            )
        
        p_inf = self.freestream.p_inf
        u_inf = self.freestream.u_inf
        v_inf = self.freestream.v_inf
        nu_t_inf = self.freestream.nu_t_inf
        
        U_n = u_int * nx + v_int * ny
        is_outflow = U_n >= 0
        is_inflow = ~is_outflow
        
        u_b = np.zeros_like(u_int)
        v_b = np.zeros_like(v_int)
        p_b = np.zeros_like(p_int)
        nu_t_b = np.zeros_like(nu_t_int)
        
        u_b[is_inflow] = u_inf
        v_b[is_inflow] = v_inf
        nu_t_b[is_inflow] = nu_t_inf
        p_b[is_inflow] = p_int[is_inflow]
        
        u_b[is_outflow] = u_int[is_outflow]
        v_b[is_outflow] = v_int[is_outflow]
        nu_t_b[is_outflow] = nu_t_int[is_outflow]
        p_b[is_outflow] = p_inf
        
        Q[i_int, -2, 0] = 2 * p_b - p_int
        Q[i_int, -2, 1] = 2 * u_b - u_int
        Q[i_int, -2, 2] = 2 * v_b - v_int
        Q[i_int, -2, 3] = 2 * nu_t_b - nu_t_int
        
        Q[i_int, -1, 0] = 2 * Q[i_int, -2, 0] - p_b
        Q[i_int, -1, 1] = 2 * Q[i_int, -2, 1] - u_b
        Q[i_int, -1, 2] = 2 * Q[i_int, -2, 2] - v_b
        Q[i_int, -1, 3] = 2 * Q[i_int, -2, 3] - nu_t_b
        
        # I-direction farfield (downstream outlet)
        j_slice = slice(0, -NGHOST)
        
        Q[1, j_slice, :] = Q[NGHOST, j_slice, :]
        Q[0, j_slice, :] = 2 * Q[NGHOST, j_slice, :] - Q[NGHOST + 1, j_slice, :]
        
        Q[-2, j_slice, :] = Q[-NGHOST - 1, j_slice, :]
        Q[-1, j_slice, :] = 2 * Q[-NGHOST - 1, j_slice, :] - Q[-NGHOST - 2, j_slice, :]
        
        return Q


def apply_boundary_conditions(Q: np.ndarray, 
                              freestream: Optional[FreestreamConditions] = None) -> np.ndarray:
    """Convenience function to apply all boundary conditions."""
    bc = BoundaryConditions(freestream=freestream)
    return bc.apply(Q)


def initialize_state(NI: int, NJ: int, 
                     freestream: Optional[FreestreamConditions] = None) -> np.ndarray:
    """Initialize state vector to freestream conditions."""
    if freestream is None:
        freestream = FreestreamConditions()
    
    Q = np.zeros((NI + 2 * NGHOST, NJ + 2 * NGHOST, N_VARS))
    Q[:, :, 0] = freestream.p_inf
    Q[:, :, 1] = freestream.u_inf
    Q[:, :, 2] = freestream.v_inf
    Q[:, :, 3] = freestream.nu_t_inf
    
    return Q


def apply_initial_wall_damping(Q: np.ndarray, 
                                grid_metrics,
                                decay_length: float = 0.1,
                                n_wake: int = 0) -> np.ndarray:
    """Apply velocity damping near wall for smooth cold start."""
    Q = Q.copy()
    
    if hasattr(grid_metrics, 'wall_distance'):
        wall_dist = grid_metrics.wall_distance
    else:
        raise ValueError("grid_metrics must have a 'wall_distance' attribute")
    
    NI = Q.shape[0] - 2 * NGHOST
    
    damping_factor = 1.0 - np.exp(-wall_dist / decay_length)
    
    if n_wake > 0:
        damping_factor[:n_wake, :] = 1.0
        damping_factor[-n_wake:, :] = 1.0
    
    int_slice = slice(NGHOST, -NGHOST)
    Q[int_slice, int_slice, 1] *= damping_factor
    Q[int_slice, int_slice, 2] *= damping_factor
    
    return Q


class InletOutletBC:
    """BC for channel/duct flows (not C-grid)."""
    
    def __init__(self, inlet_velocity: tuple = (1.0, 0.0), 
                 p_outlet: float = 0.0):
        self.u_inlet, self.v_inlet = inlet_velocity
        self.p_outlet = p_outlet
    
    def apply(self, Q: np.ndarray) -> np.ndarray:
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
