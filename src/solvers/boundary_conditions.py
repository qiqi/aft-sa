"""
Boundary Conditions for 2D Incompressible RANS Solver on C-Grid.

This module implements boundary conditions for a structured C-grid topology
around an airfoil. The C-grid wraps around the airfoil from the lower wake
to the upper wake, with:
    - I-direction: Wraps around airfoil, from lower wake to upper wake
    - J-direction: From airfoil surface (j=0) to farfield (j=N_max)

Boundary Conditions:
    1. Airfoil Surface (j=0): No-slip, zero pressure gradient, ν̃=0
    2. Farfield (j=N_max): Characteristic non-reflecting BC
    3. Wake Cut (i=0 and i=N_max): Periodic connection

State Vector: Q = [p, u, v, ν̃] where
    - p: pseudo-pressure
    - u, v: velocity components
    - ν̃: SA turbulent viscosity

Ghost Cell Convention:
    - Q has shape (NI + 2*NGHOST, NJ + 2*NGHOST, 4) where NGHOST=2
    - Interior cells: Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :] = Q[2:-2, 2:-2, :]
    - I-direction ghosts: Q[0:2, :] (left), Q[-2:, :] (right) 
    - J-direction ghosts: Q[:, 0:2] (wall/wake), Q[:, -2:] (farfield)
    
The 2-layer ghost cells provide the necessary stencil for 4th-order JST dissipation.
"""

import numpy as np
from typing import NamedTuple, Optional
from dataclasses import dataclass

from src.constants import NGHOST, N_VARS


@dataclass
class FreestreamConditions:
    """Freestream flow conditions."""
    
    p_inf: float = 0.0      # Reference pressure (usually 0 for incompressible)
    u_inf: float = 1.0      # Freestream x-velocity
    v_inf: float = 0.0      # Freestream y-velocity (set by angle of attack)
    nu_t_inf: float = 3e-6  # Freestream eddy viscosity (typically ~3*nu_inf)
    
    @classmethod
    def from_mach_alpha(cls, mach: float, alpha_deg: float, 
                        nu_ratio: float = 3.0) -> 'FreestreamConditions':
        """
        Create freestream conditions from Mach number and angle of attack.
        
        For incompressible flow, we use unit velocity magnitude.
        
        Parameters
        ----------
        mach : float
            Reference Mach number (only affects initial guess scaling).
        alpha_deg : float
            Angle of attack in degrees.
        nu_ratio : float
            Ratio of freestream eddy viscosity to molecular viscosity.
        """
        alpha = np.radians(alpha_deg)
        return cls(
            p_inf=0.0,
            u_inf=np.cos(alpha),
            v_inf=np.sin(alpha),
            nu_t_inf=nu_ratio * 1e-6  # Assume nu_inf ~ 1e-6 for Re~10^6
        )


class BoundaryConditions:
    """
    Boundary condition handler for C-grid topology.
    
    Grid Layout (C-grid around airfoil):
    
        j = NJ+1  ─────────────────────────  (ghost farfield)
        j = NJ    ═════════════════════════  (farfield boundary)
                  │                       │
                  │      INTERIOR         │
                  │                       │
        j = 1     ─────┬─────┬─────┬─────┘  (first interior row)
        j = 0     ═════╧═════╧═════╧═════    (ghost surface/wake)
                  
        i=0,1    ↑               ↑   i=NI,NI+1
        (wake    (airfoil        (wake
         lower)   surface)        upper)
    
    The wake cut connects i=0 (lower wake) to i=NI (upper wake).
    """
    
    def __init__(self, 
                 freestream: Optional[FreestreamConditions] = None,
                 j_surface: int = 0,
                 j_farfield: int = -1,
                 i_wake_start: int = 0,
                 i_wake_end: int = -1,
                 n_wake_points: int = 0,
                 farfield_normals: Optional[tuple] = None,
                 beta: float = 10.0):
        """
        Initialize boundary conditions.
        
        Parameters
        ----------
        freestream : FreestreamConditions, optional
            Freestream conditions for farfield BC.
        j_surface : int
            J-index of surface boundary (default 0).
        j_farfield : int
            J-index of farfield boundary (default -1 = last).
        i_wake_start : int
            I-index where wake begins on lower side.
        i_wake_end : int  
            I-index where wake ends on upper side.
        n_wake_points : int
            Number of wake points (used to identify wake region).
        farfield_normals : tuple of (nx, ny), optional
            Outward unit normals at far-field face (j=NJ).
            If provided, enables characteristic-based BC.
        beta : float
            Artificial compressibility parameter.
        """
        self.freestream = freestream or FreestreamConditions()
        self.j_surface = j_surface
        self.j_farfield = j_farfield
        self.i_wake_start = i_wake_start
        self.i_wake_end = i_wake_end
        self.n_wake_points = n_wake_points
        self.farfield_normals = farfield_normals
        self.beta = beta
        self.c_art = np.sqrt(beta)  # Artificial sound speed
    
    def apply(self, Q: np.ndarray) -> np.ndarray:
        """
        Apply all boundary conditions to update ghost cells.
        
        Parameters
        ----------
        Q : ndarray, shape (NI+2, NJ+2, 4)
            State vector with ghost cells.
            
        Returns
        -------
        Q : ndarray
            State vector with updated ghost cells.
        """
        Q = Q.copy()
        
        # Apply in order: surface (J=0), then farfield (J=JMAX, I=0, I=IMAX)
        Q = self.apply_surface(Q)
        Q = self.apply_farfield(Q)
        
        return Q
    
    def apply_surface(self, Q: np.ndarray) -> np.ndarray:
        """
        Apply airfoil surface boundary conditions (j=0,1 ghost cells).
        
        For a C-grid, the J=0 boundary has two regions:
        1. Airfoil surface: No-slip wall BC
        2. Wake cut: Symmetry/continuity BC (not a physical wall)
        
        The wake region is identified by self.i_wake_start and i_wake_end,
        which are computed from the grid in __init__.
        
        No-slip wall (airfoil):
            - Velocity: u_ghost = -u_interior (so average at face = 0)
            - Pressure: p_ghost = p_interior (zero normal gradient)
            - Turbulence: ν̃_ghost = -ν̃_interior (so ν̃_face = 0)
        
        Wake cut (periodic/continuous):
            - All variables: ghost = interior (zero normal gradient)
            - This allows flow to pass through the wake cut freely
        
        Parameters
        ----------
        Q : ndarray, shape (NI + 2*NGHOST, NJ + 2*NGHOST, 4)
            State vector with NGHOST ghost layers on each side.
            
        Returns
        -------
        Q : ndarray
            Updated state with surface ghost cells set.
        """
        # With NGHOST=2 ghost layers:
        # Q[:, 0, :] is the outer ghost layer (farthest from wall)
        # Q[:, 1, :] is the inner ghost layer (adjacent to wall)
        # Q[:, 2, :] is the first interior layer
        
        NI = Q.shape[0] - 2 * NGHOST  # Number of interior cells in i
        
        # Determine airfoil vs wake regions using n_wake_points
        n_wake = self.n_wake_points if self.n_wake_points > 0 else NI // 6
        i_wake_end_lower = n_wake
        i_wake_start_upper = NI - n_wake
        
        # Q indices with NGHOST=2: ghosts at [0,1], interior at [2, NI+1], ghosts at [NI+2, NI+3]
        i_start = i_wake_end_lower + NGHOST      # Q index for first airfoil cell
        i_end = i_wake_start_upper + NGHOST      # Q index for last airfoil cell (exclusive)
        
        # First interior j-index
        j_int_first = NGHOST  # = 2
        
        # ===== AIRFOIL SURFACE (no-slip) =====
        # Inner ghost layer (j=1): anti-symmetric for velocity, copy for pressure
        Q[i_start:i_end, 1, 0] = Q[i_start:i_end, j_int_first, 0]       # p: zero gradient
        Q[i_start:i_end, 1, 1] = -Q[i_start:i_end, j_int_first, 1]      # u: no-slip
        Q[i_start:i_end, 1, 2] = -Q[i_start:i_end, j_int_first, 2]      # v: no-slip
        Q[i_start:i_end, 1, 3] = -Q[i_start:i_end, j_int_first, 3]      # nu_t: wall value = 0
        
        # Outer ghost layer (j=0): linear extrapolation Q[0] = 2*Q[1] - Q[2]
        Q[i_start:i_end, 0, 0] = Q[i_start:i_end, 1, 0]       # p: zero gradient
        Q[i_start:i_end, 0, 1] = 2*Q[i_start:i_end, 1, 1] - Q[i_start:i_end, j_int_first, 1]  # u: linear extrap
        Q[i_start:i_end, 0, 2] = 2*Q[i_start:i_end, 1, 2] - Q[i_start:i_end, j_int_first, 2]  # v: linear extrap
        Q[i_start:i_end, 0, 3] = Q[i_start:i_end, 1, 3]       # nu_t: zero gradient
        
        # ===== WAKE CUT (2-layer periodic boundary at J=0) =====
        # Lower wake: interior indices 0 to i_wake_end_lower-1 (Q indices NGHOST to i_start-1)
        # Upper wake: interior indices i_wake_start_upper to NI-1 (Q indices i_end to NI+NGHOST-1)
        #
        # The wake cut connects lower wake to upper wake with mirrored i-indices:
        # - Lower wake cell at i=0 is adjacent to upper wake cell at i=NI-1
        # - Lower wake cell at i=1 is adjacent to upper wake cell at i=NI-2
        # - etc.
        #
        # For periodic BC at J=0:
        # - Lower wake j-ghosts get values from upper wake j-interior (mirrored)
        # - Upper wake j-ghosts get values from lower wake j-interior (mirrored)
        
        # Lower wake Q-indices: NGHOST to i_start (exclusive) = NGHOST to NGHOST + n_wake
        # Upper wake Q-indices: i_end to NI + NGHOST (exclusive) = NI - n_wake + NGHOST to NI + NGHOST
        
        # Get the interior values from each side (reversed for mirroring)
        lower_wake_int_j0 = Q[NGHOST:i_start, j_int_first, :]      # Lower wake, first interior j
        lower_wake_int_j1 = Q[NGHOST:i_start, j_int_first + 1, :]  # Lower wake, second interior j
        upper_wake_int_j0 = Q[i_end:NI + NGHOST, j_int_first, :]   # Upper wake, first interior j
        upper_wake_int_j1 = Q[i_end:NI + NGHOST, j_int_first + 1, :]  # Upper wake, second interior j
        
        # Lower wake ghosts (j=0, j=1) get values from upper wake interior (mirrored i)
        # Flip the i-direction: [::-1]
        Q[NGHOST:i_start, 1, :] = upper_wake_int_j0[::-1, :]  # Inner ghost from first interior
        Q[NGHOST:i_start, 0, :] = upper_wake_int_j1[::-1, :]  # Outer ghost from second interior
        
        # Upper wake ghosts (j=0, j=1) get values from lower wake interior (mirrored i)
        Q[i_end:NI + NGHOST, 1, :] = lower_wake_int_j0[::-1, :]  # Inner ghost from first interior
        Q[i_end:NI + NGHOST, 0, :] = lower_wake_int_j1[::-1, :]  # Outer ghost from second interior
        
        return Q
    
    def apply_farfield(self, Q: np.ndarray) -> np.ndarray:
        """
        Apply farfield boundary conditions at all farfield boundaries.
        
        For a C-grid around an airfoil, the farfield boundaries are:
        - J=JMAX: Outer farfield (characteristic-based non-reflecting BC)
        - I=0 and I=IMAX: Downstream farfield/outlet (zero-gradient extrapolation)
        
        J-direction farfield (outer boundary):
        This BC determines whether each farfield face is inflow or outflow
        based on the normal velocity component, then:
        
        Inflow (U_n < 0):
            - Velocity: Set to freestream
            - Turbulence: Set to freestream (nu_t_inf)
            - Pressure: Extrapolate from interior (zero-order)
        
        Outflow (U_n >= 0):
            - Velocity: Extrapolate from interior
            - Turbulence: Extrapolate from interior
            - Pressure: Set to freestream (back-pressure)
        
        I-direction farfield (downstream outlet):
            - Zero-gradient (linear extrapolation) for subsonic outflow
        
        Parameters
        ----------
        Q : ndarray, shape (NI + 2*NGHOST, NJ + 2*NGHOST, 4)
            State vector with NGHOST ghost layers on each side.
            
        Returns
        -------
        Q : ndarray
            Updated state with farfield ghost cells set.
        """
        NI = Q.shape[0] - 2 * NGHOST  # Number of interior cells
        
        # Interior slice for i-direction (excludes I-ghost cells)
        i_int = slice(NGHOST, -NGHOST)
        
        # Last interior j-index
        j_int_last = -NGHOST - 1  # = -(NGHOST+1) = -3 for NGHOST=2
        
        if self.farfield_normals is None:
            # Fall back to simple Dirichlet for both ghost layers
            for j_ghost in range(-NGHOST, 0):  # j = -2, -1
                Q[:, j_ghost, 0] = self.freestream.p_inf
                Q[:, j_ghost, 1] = self.freestream.u_inf
                Q[:, j_ghost, 2] = self.freestream.v_inf
                Q[:, j_ghost, 3] = self.freestream.nu_t_inf
            return Q
        
        # Characteristic-based non-reflecting BC
        nx, ny = self.farfield_normals  # Shape: (NI,) - outward unit normals
        
        # Interior values (last interior row)
        p_int = Q[i_int, j_int_last, 0]
        u_int = Q[i_int, j_int_last, 1]
        v_int = Q[i_int, j_int_last, 2]
        nu_t_int = Q[i_int, j_int_last, 3]
        
        # Second-to-last interior for 2nd ghost layer extrapolation
        p_int2 = Q[i_int, j_int_last - 1, 0]
        u_int2 = Q[i_int, j_int_last - 1, 1]
        v_int2 = Q[i_int, j_int_last - 1, 2]
        nu_t_int2 = Q[i_int, j_int_last - 1, 3]
        
        # Validate shape consistency
        if len(nx) != NI:
            raise ValueError(
                f"Farfield BC shape mismatch: farfield_normals has {len(nx)} values, "
                f"but Q has {NI} interior cells (Q.shape={Q.shape}). "
                f"BoundaryConditions object may be from a different grid level."
            )
        
        # Freestream values
        p_inf = self.freestream.p_inf
        u_inf = self.freestream.u_inf
        v_inf = self.freestream.v_inf
        nu_t_inf = self.freestream.nu_t_inf
        
        # Calculate normal velocity (positive = outflow, negative = inflow)
        U_n = u_int * nx + v_int * ny
        
        # Determine inflow/outflow
        is_outflow = U_n >= 0
        is_inflow = ~is_outflow
        
        # Initialize boundary values (at the face)
        u_b = np.zeros_like(u_int)
        v_b = np.zeros_like(v_int)
        p_b = np.zeros_like(p_int)
        nu_t_b = np.zeros_like(nu_t_int)
        
        # Inflow (U_n < 0): velocity from freestream, pressure from interior
        u_b[is_inflow] = u_inf
        v_b[is_inflow] = v_inf
        nu_t_b[is_inflow] = nu_t_inf
        p_b[is_inflow] = p_int[is_inflow]  # Zero-order extrapolation
        
        # Outflow (U_n >= 0): velocity from interior, pressure from freestream
        u_b[is_outflow] = u_int[is_outflow]
        v_b[is_outflow] = v_int[is_outflow]
        nu_t_b[is_outflow] = nu_t_int[is_outflow]
        p_b[is_outflow] = p_inf  # Fix back-pressure
        
        # Set inner ghost layer (j=-2): extrapolate from boundary value
        # Ghost value = 2*boundary - interior
        Q[i_int, -2, 0] = 2 * p_b - p_int
        Q[i_int, -2, 1] = 2 * u_b - u_int
        Q[i_int, -2, 2] = 2 * v_b - v_int
        Q[i_int, -2, 3] = 2 * nu_t_b - nu_t_int
        
        # Set outer ghost layer (j=-1): linear extrapolation from inner ghost
        Q[i_int, -1, 0] = 2 * Q[i_int, -2, 0] - p_b
        Q[i_int, -1, 1] = 2 * Q[i_int, -2, 1] - u_b
        Q[i_int, -1, 2] = 2 * Q[i_int, -2, 2] - v_b
        Q[i_int, -1, 3] = 2 * Q[i_int, -2, 3] - nu_t_b
        
        # ===== I-DIRECTION FARFIELD (downstream outlet at I=0 and I=IMAX) =====
        # For a C-grid, I=0 and I=IMAX are the downstream farfield boundaries
        # Use zero-gradient (extrapolation) BC for subsonic outflow
        
        # All j except J-farfield ghosts (which were just set above)
        j_slice = slice(0, -NGHOST)
        
        # Left I-ghosts (i=0,1): extrapolate from left interior
        # Inner ghost (i=1) = first interior (i=NGHOST)
        # Outer ghost (i=0) = linear extrapolation: 2*Q[NGHOST] - Q[NGHOST+1]
        Q[1, j_slice, :] = Q[NGHOST, j_slice, :]
        Q[0, j_slice, :] = 2 * Q[NGHOST, j_slice, :] - Q[NGHOST + 1, j_slice, :]
        
        # Right I-ghosts (i=-2,-1): extrapolate from right interior
        # Inner ghost (i=-2) = last interior (i=-NGHOST-1)
        # Outer ghost (i=-1) = linear extrapolation: 2*Q[-NGHOST-1] - Q[-NGHOST-2]
        Q[-2, j_slice, :] = Q[-NGHOST - 1, j_slice, :]
        Q[-1, j_slice, :] = 2 * Q[-NGHOST - 1, j_slice, :] - Q[-NGHOST - 2, j_slice, :]
        
        return Q


def apply_boundary_conditions(Q: np.ndarray, 
                              freestream: Optional[FreestreamConditions] = None) -> np.ndarray:
    """
    Convenience function to apply all boundary conditions.
    
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+3, 4)
        State vector with ghost cells.
    freestream : FreestreamConditions, optional
        Freestream conditions.
        
    Returns
    -------
    Q : ndarray
        Updated state with ghost cells set.
    """
    bc = BoundaryConditions(freestream=freestream)
    return bc.apply(Q)


def initialize_state(NI: int, NJ: int, 
                     freestream: Optional[FreestreamConditions] = None) -> np.ndarray:
    """
    Initialize the state vector to freestream conditions.
    
    Parameters
    ----------
    NI, NJ : int
        Number of interior cells in I and J directions.
    freestream : FreestreamConditions, optional
        Freestream conditions.
        
    Returns
    -------
    Q : ndarray, shape (NI + 2*NGHOST, NJ + 2*NGHOST, N_VARS)
        Initialized state vector with NGHOST ghost layers on each side.
    """
    if freestream is None:
        freestream = FreestreamConditions()
    
    # NGHOST ghost layers on each side in both I and J directions
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
    """
    Apply initial wall damping to prevent impulsive start shockwaves.
    
    This function damps the velocity near the wall to zero before the first
    iteration, providing a smooth "cold start" for the simulation.
    
    The damping formula is:
        u_new = u_old * (1 - exp(-d / L))
        
    where:
        d = wall distance
        L = decay_length (characteristic length scale)
    
    As d → 0 (near wall): damping → 0 (velocity → 0)
    As d → ∞ (far from wall): damping → 1 (velocity unchanged)
    
    For C-grids, the wake region (first and last n_wake cells in i-direction)
    is NOT damped because there is no wall there.
    
    Parameters
    ----------
    Q : ndarray, shape (NI + 2*NGHOST, NJ + 2*NGHOST, 4)
        State vector with NGHOST ghost layers on each side.
    grid_metrics : object
        Grid metrics object containing wall_distance array of shape (NI, NJ).
    decay_length : float
        Characteristic decay length scale (default 0.1, in chord units).
        Larger values create a thicker damping region.
    n_wake : int
        Number of wake cells at each end of the i-direction. These cells
        are not damped because they are in the wake, not near the airfoil.
        
    Returns
    -------
    Q : ndarray
        State vector with damped velocities near the wall.
        
    Notes
    -----
    This function modifies only the velocity components (u, v) of interior cells.
    Ghost cells are not modified and should be updated via apply_boundary_conditions
    after this function is called.
    """
    Q = Q.copy()
    
    # Get wall distance from metrics
    # Handle both FVMMetrics (from metrics.py) and GridMetrics (from plot3d.py)
    if hasattr(grid_metrics, 'wall_distance'):
        wall_dist = grid_metrics.wall_distance
    else:
        raise ValueError("grid_metrics must have a 'wall_distance' attribute")
    
    NI = Q.shape[0] - 2 * NGHOST  # Number of interior cells in i
    
    # Compute damping factor: 1 - exp(-d/L)
    # Near wall (d=0): factor = 0 (fully damped)
    # Far from wall: factor = 1 (no damping)
    damping_factor = 1.0 - np.exp(-wall_dist / decay_length)
    
    # For C-grid: don't damp wake cells (they're not near a wall)
    # Wake cells are at i = 0 to n_wake-1 and i = NI-n_wake to NI-1
    if n_wake > 0:
        damping_factor[:n_wake, :] = 1.0  # Lower wake: no damping
        damping_factor[-n_wake:, :] = 1.0  # Upper wake: no damping
    
    # Apply damping to velocity components (indices 1 and 2 in state vector)
    # Only modify interior cells: Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
    int_slice = slice(NGHOST, -NGHOST)
    Q[int_slice, int_slice, 1] *= damping_factor  # u-velocity
    Q[int_slice, int_slice, 2] *= damping_factor  # v-velocity
    
    return Q


class InletOutletBC:
    """
    Alternative BC for channel/duct flows (not C-grid).
    
    This is provided for testing purposes or non-airfoil geometries.
    
    - Inlet (i=0): Dirichlet velocity, zero gradient pressure
    - Outlet (i=NI): Zero gradient all variables (convective outflow)
    - Walls (j=0, j=NJ): No-slip
    """
    
    def __init__(self, inlet_velocity: tuple = (1.0, 0.0), 
                 p_outlet: float = 0.0):
        """
        Initialize inlet/outlet BCs.
        
        Parameters
        ----------
        inlet_velocity : tuple
            (u, v) at inlet.
        p_outlet : float
            Pressure at outlet (for reference).
        """
        self.u_inlet, self.v_inlet = inlet_velocity
        self.p_outlet = p_outlet
    
    def apply(self, Q: np.ndarray) -> np.ndarray:
        """Apply inlet/outlet/wall boundary conditions."""
        Q = Q.copy()
        
        # Inlet (i=0): Dirichlet velocity, Neumann pressure
        Q[0, :, 0] = Q[1, :, 0]  # Zero gradient pressure
        Q[0, :, 1] = 2 * self.u_inlet - Q[1, :, 1]  # Dirichlet u
        Q[0, :, 2] = 2 * self.v_inlet - Q[1, :, 2]  # Dirichlet v
        Q[0, :, 3] = Q[1, :, 3]  # Zero gradient nu_t
        
        # Outlet (i=-1): Zero gradient (convective outflow)
        Q[-1, :, :] = Q[-2, :, :]
        
        # Bottom wall (j=0): No-slip
        Q[:, 0, 0] = Q[:, 1, 0]      # Neumann pressure
        Q[:, 0, 1] = -Q[:, 1, 1]     # No-slip u
        Q[:, 0, 2] = -Q[:, 1, 2]     # No-slip v
        Q[:, 0, 3] = -Q[:, 1, 3]     # Zero nu_t at wall
        
        # Top wall (j=-1): No-slip
        Q[:, -1, 0] = Q[:, -2, 0]    # Neumann pressure
        Q[:, -1, 1] = -Q[:, -2, 1]   # No-slip u
        Q[:, -1, 2] = -Q[:, -2, 2]   # No-slip v
        Q[:, -1, 3] = -Q[:, -2, 3]   # Zero nu_t at wall
        
        return Q

