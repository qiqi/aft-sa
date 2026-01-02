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
    - Q has shape (NI+2, NJ+2, 4)
    - Interior cells: Q[1:-1, 1:-1, :]
    - Ghost cells: Q[0, :], Q[-1, :], Q[:, 0], Q[:, -1]
"""

import numpy as np
from typing import NamedTuple, Optional
from dataclasses import dataclass


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
        
        # Apply in order: surface, farfield, then wake cut
        Q = self.apply_surface(Q)
        Q = self.apply_farfield(Q)
        Q = self.apply_wake_cut(Q)
        
        return Q
    
    def apply_surface(self, Q: np.ndarray) -> np.ndarray:
        """
        Apply airfoil surface boundary conditions (j=0 ghost cells).
        
        For a C-grid, j=0 has two regions:
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
        Q : ndarray, shape (NI+2, NJ+2, 4)
            State vector with ghost cells.
            
        Returns
        -------
        Q : ndarray
            Updated state with surface ghost cells set.
        """
        # Ghost cells at j=0, interior cells at j=1
        # Q[:, 0, :] is the ghost layer (surface)
        # Q[:, 1, :] is the first interior layer
        
        NI = Q.shape[0] - 2  # Number of interior cells in i
        
        # Determine airfoil vs wake regions using n_wake_points
        # For a C-grid with n_wake points: 
        #   - Lower wake: i = 1 to n_wake (in interior indexing)
        #   - Upper wake: i = NI-n_wake+1 to NI
        n_wake = self.n_wake_points if self.n_wake_points > 0 else NI // 6
        i_wake_end_lower = n_wake  # Last lower wake cell (interior index)
        i_wake_start_upper = NI - n_wake  # First upper wake cell (interior index)
        
        # Create mask for airfoil cells (interior indices 1 to NI in Q indexing)
        # Q indices: 0=ghost, 1=first interior, ..., NI=last interior, NI+1=ghost
        # So airfoil is Q[i_wake_end_lower+1 : i_wake_start_upper+1, 0, :]
        
        # ===== AIRFOIL SURFACE (no-slip) =====
        i_start = i_wake_end_lower + 1  # Q index for first airfoil cell
        i_end = i_wake_start_upper + 1   # Q index for last airfoil cell (inclusive)
        
        # Pressure: zero normal gradient
        Q[i_start:i_end+1, 0, 0] = Q[i_start:i_end+1, 1, 0]
        
        # Velocity: no-slip
        Q[i_start:i_end+1, 0, 1] = -Q[i_start:i_end+1, 1, 1]  # u
        Q[i_start:i_end+1, 0, 2] = -Q[i_start:i_end+1, 1, 2]  # v
        
        # Turbulence: ν̃ = 0 at wall
        Q[i_start:i_end+1, 0, 3] = -Q[i_start:i_end+1, 1, 3]
        
        # ===== WAKE CUT (periodic/continuous boundary) =====
        # The wake cut is NOT a symmetry plane - it's a periodic boundary where
        # upper and lower wake meet. All variables should be continuous.
        # The j=0 ghost cells in the wake should extrapolate from the interior
        # (zero normal gradient) to allow flow through.
        
        # Lower wake: indices 1 to i_wake_end_lower (Q indices 1 to i_wake_end_lower+1)
        Q[1:i_wake_end_lower+1, 0, 0] = Q[1:i_wake_end_lower+1, 1, 0]    # p continuous
        Q[1:i_wake_end_lower+1, 0, 1] = Q[1:i_wake_end_lower+1, 1, 1]    # u continuous  
        Q[1:i_wake_end_lower+1, 0, 2] = Q[1:i_wake_end_lower+1, 1, 2]    # v continuous
        Q[1:i_wake_end_lower+1, 0, 3] = Q[1:i_wake_end_lower+1, 1, 3]    # nu_t continuous
        
        # Upper wake: indices i_wake_start_upper to NI (Q indices i_wake_start_upper+1 to NI+1)
        Q[i_wake_start_upper+1:NI+1, 0, 0] = Q[i_wake_start_upper+1:NI+1, 1, 0]
        Q[i_wake_start_upper+1:NI+1, 0, 1] = Q[i_wake_start_upper+1:NI+1, 1, 1]
        Q[i_wake_start_upper+1:NI+1, 0, 2] = Q[i_wake_start_upper+1:NI+1, 1, 2]
        Q[i_wake_start_upper+1:NI+1, 0, 3] = Q[i_wake_start_upper+1:NI+1, 1, 3]
        
        return Q
    
    def apply_farfield(self, Q: np.ndarray) -> np.ndarray:
        """
        Apply characteristic-based non-reflecting farfield boundary conditions.
        
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
        
        Parameters
        ----------
        Q : ndarray, shape (NI+2, NJ+2, 4)
            State vector with ghost cells.
            
        Returns
        -------
        Q : ndarray
            Updated state with farfield ghost cells set.
        """
        if self.farfield_normals is None:
            # Fall back to simple Dirichlet
            Q[:, -1, 0] = self.freestream.p_inf
            Q[:, -1, 1] = self.freestream.u_inf
            Q[:, -1, 2] = self.freestream.v_inf
            Q[:, -1, 3] = self.freestream.nu_t_inf
            return Q
        
        # Characteristic-based non-reflecting BC
        nx, ny = self.farfield_normals  # Shape: (NI,) - outward unit normals
        
        # Interior values (last interior row, j = -2 with ghost cells)
        p_int = Q[1:-1, -2, 0]
        u_int = Q[1:-1, -2, 1]
        v_int = Q[1:-1, -2, 2]
        nu_t_int = Q[1:-1, -2, 3]
        
        # Validate shape consistency
        NI = Q.shape[0] - 2  # Number of interior cells
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
        
        # Initialize boundary values
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
        
        # Set ghost cells (extrapolate to get ghost value)
        # Ghost value = 2*boundary - interior
        Q[1:-1, -1, 0] = 2 * p_b - p_int
        Q[1:-1, -1, 1] = 2 * u_b - u_int
        Q[1:-1, -1, 2] = 2 * v_b - v_int
        Q[1:-1, -1, 3] = 2 * nu_t_b - nu_t_int
        
        # Handle corner ghost cells (i=0, i=-1)
        Q[0, -1, :] = Q[1, -1, :]
        Q[-1, -1, :] = Q[-2, -1, :]
        
        return Q
    
    def apply_wake_cut(self, Q: np.ndarray) -> np.ndarray:
        """
        Apply wake cut periodic boundary conditions.
        
        C-grid Topology:
            The C-grid wraps around the airfoil with a "cut" in the wake:
            
            - i=1 (first interior): adjacent to wake cut on lower side
            - i=NI (last interior): adjacent to wake cut on upper side  
            - These cells are NEIGHBORS across the wake cut
            
            The grid has reflection symmetry about the wake (Y=0):
            - X[0, :] == X[-1, :]  (same X coordinates)
            - Y[0, :] == -Y[-1, :] (opposite Y coordinates)
            
            However, this is handled by the GRID METRICS (face normals), not
            by negating velocities. The face normals satisfy:
            - Si_x[0, :] == -Si_x[-1, :]  (opposite, pointing into domain)
            - Si_y[0, :] == Si_y[-1, :]   (same)
            
            Therefore, simple copy is correct - the flux computation using
            these ghost values will automatically handle the geometry.
        
        Implementation:
            - Ghost at i=0: Copy from interior at i=NI (Q[-2, :, :])
            - Ghost at i=-1 (i=NI+1): Copy from interior at i=1 (Q[1, :, :])
            - IMPORTANT: Do NOT modify farfield row (j=-1) - that's set by apply_farfield
        
        Parameters
        ----------
        Q : ndarray, shape (NI+2, NJ+2, 4)
            State vector with ghost cells.
            
        Returns
        -------
        Q : ndarray
            Updated state with wake cut ghost cells set.
        """
        # The C-grid has a cut in the wake:
        #   - Left ghost (i=0) gets values from right interior (i=NI, which is Q[-2])
        #   - Right ghost (i=NI+1, which is Q[-1]) gets values from left interior (i=1)
        
        # Conservation-preserving wake cut BC:
        # Use AVERAGED ghost values for flux consistency at the wake cut.
        #
        # For a C-grid, the wake cut is where the grid wraps around. The cells on
        # either side (i=1 and i=NI) are physical neighbors. 
        #
        # By setting both ghost values to the SAME average, we ensure:
        # - Flux at face i=0.5 and face i=NI+0.5 are computed with identical stencils
        # - The JST artificial dissipation is consistent across the cut
        # - Mass and momentum are conserved
        #
        # IMPORTANT: Only apply to j=0:-1 (interior + surface ghost), 
        # NOT j=-1 (farfield ghost) which was set by apply_farfield
        
        # Average of the two interior cells adjacent to the wake cut
        avg = 0.5 * (Q[1, :-1, :] + Q[-2, :-1, :])
        Q[0, :-1, :] = avg    # Left ghost = average
        Q[-1, :-1, :] = avg   # Right ghost = average
        
        return Q


def apply_boundary_conditions(Q: np.ndarray, 
                              freestream: Optional[FreestreamConditions] = None) -> np.ndarray:
    """
    Convenience function to apply all boundary conditions.
    
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
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
    Q : ndarray, shape (NI+2, NJ+2, 4)
        Initialized state vector.
    """
    if freestream is None:
        freestream = FreestreamConditions()
    
    Q = np.zeros((NI + 2, NJ + 2, 4))
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
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector with ghost cells: [p, u, v, ν̃].
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
    
    NI = Q.shape[0] - 2  # Number of interior cells in i
    
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
    # Only modify interior cells (1:-1 in ghost-padded array)
    Q[1:-1, 1:-1, 1] *= damping_factor  # u-velocity
    Q[1:-1, 1:-1, 2] *= damping_factor  # v-velocity
    
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

