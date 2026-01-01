"""
Multigrid Hierarchy Manager for FAS Scheme.

This module provides the MultigridHierarchy class that manages all data
structures and operations for geometric multigrid with Full Approximation
Storage (FAS).

Design: GPU-ready with flat lists of arrays.
- All heavy operations use Numba-optimized kernels
- No recursion in kernels - Python driver handles level iteration
- Pre-allocated buffers at hierarchy build time
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..grid.metrics import MetricComputer, FVMMetrics
from ..grid.coarsening import Coarsener
from ..numerics.multigrid import restrict_state, restrict_residual, prolongate_correction, prolongate_injection

from .boundary_conditions import BoundaryConditions, FreestreamConditions


@dataclass
class MultigridLevel:
    """
    Data for a single multigrid level.
    
    All arrays are pre-allocated for efficiency.
    """
    
    # Grid dimensions
    NI: int
    NJ: int
    
    # Grid metrics
    metrics: FVMMetrics
    
    # State vector: [p, u, v, nu_tilde]
    # Shape: (NI+2, NJ+2, 4) with ghost cells
    Q: np.ndarray
    
    # Residual storage
    # Shape: (NI, NJ, 4) for interior cells only
    R: np.ndarray
    
    # FAS forcing term (coarse levels only)
    # Shape: (NI, NJ, 4) 
    forcing: np.ndarray
    
    # Local timestep
    # Shape: (NI, NJ)
    dt: np.ndarray
    
    # Boundary conditions handler
    bc: BoundaryConditions
    
    # Storage for pre-smoothing Q (needed for FAS correction)
    Q_old: Optional[np.ndarray] = None
    
    @property
    def nvar(self) -> int:
        return 4


class MultigridHierarchy:
    """
    Manages the multigrid hierarchy for FAS scheme.
    
    This class builds and stores all multigrid levels, and provides
    methods for:
    - Restriction of state and residuals
    - Prolongation of corrections
    - Boundary condition application at each level
    
    Example
    -------
    >>> hierarchy = MultigridHierarchy()
    >>> hierarchy.build(X, Y, Q_fine, freestream, config)
    >>> print(f"Built {hierarchy.num_levels} levels")
    >>> 
    >>> # Access level data
    >>> Q_coarse = hierarchy.levels[1].Q
    >>> R_coarse = hierarchy.levels[1].R
    """
    
    def __init__(self, min_size: int = 8, max_levels: int = 5):
        """
        Initialize multigrid hierarchy manager.
        
        Parameters
        ----------
        min_size : int
            Minimum cells in each direction on coarsest level.
        max_levels : int
            Maximum number of multigrid levels.
        """
        self.min_size = min_size
        self.max_levels = max_levels
        self.levels: List[MultigridLevel] = []
        self.num_levels = 0
    
    def build(self, 
              X: np.ndarray, 
              Y: np.ndarray,
              Q_fine: np.ndarray,
              freestream: FreestreamConditions,
              n_wake: int = 0,
              beta: float = 10.0) -> int:
        """
        Build the multigrid hierarchy.
        
        Parameters
        ----------
        X, Y : ndarray, shape (NI+1, NJ+1)
            Node coordinates for finest grid.
        Q_fine : ndarray, shape (NI+2, NJ+2, 4)
            Fine grid state vector with ghost cells.
        freestream : FreestreamConditions
            Freestream conditions for BCs.
        n_wake : int
            Number of wake points (scaled for coarse levels).
        beta : float
            Artificial compressibility parameter.
            
        Returns
        -------
        num_levels : int
            Number of levels built.
        """
        self.levels = []
        
        # Level 0: Finest grid
        NI = X.shape[0] - 1
        NJ = X.shape[1] - 1
        
        computer = MetricComputer(X, Y)
        metrics = computer.compute()
        
        # Compute farfield normals
        farfield_normals = self._compute_farfield_normals(X, Y)
        
        # Create BC handler for finest level
        bc = BoundaryConditions(
            freestream=freestream,
            n_wake_points=n_wake,
            farfield_normals=farfield_normals,
            beta=beta
        )
        
        # Create level 0
        level0 = MultigridLevel(
            NI=NI,
            NJ=NJ,
            metrics=metrics,
            Q=Q_fine.copy(),
            R=np.zeros((NI, NJ, 4)),
            forcing=np.zeros((NI, NJ, 4)),
            dt=np.zeros((NI, NJ)),
            bc=bc,
            Q_old=np.zeros((NI + 2, NJ + 2, 4))
        )
        self.levels.append(level0)
        
        # Build coarse levels
        current_X, current_Y = X, Y
        current_metrics = metrics
        current_n_wake = n_wake
        
        level_idx = 1
        while level_idx < self.max_levels:
            # Check if we can coarsen
            if not Coarsener.can_coarsen(current_metrics.NI, current_metrics.NJ, 
                                          self.min_size):
                break
            
            # Coarsen grid coordinates
            coarse_X = current_X[::2, ::2]
            coarse_Y = current_Y[::2, ::2]
            
            # Coarsen metrics
            coarse_metrics = Coarsener.coarsen(current_metrics)
            NI_c, NJ_c = coarse_metrics.NI, coarse_metrics.NJ
            
            # Scale BC indices
            coarse_n_wake = current_n_wake // 2
            
            # Compute coarse farfield normals
            coarse_farfield_normals = self._compute_farfield_normals(coarse_X, coarse_Y)
            
            # Create BC handler for coarse level
            bc_c = BoundaryConditions(
                freestream=freestream,
                n_wake_points=coarse_n_wake,
                farfield_normals=coarse_farfield_normals,
                beta=beta
            )
            
            # Create state and residual arrays
            Q_c = np.zeros((NI_c + 2, NJ_c + 2, 4))
            
            # Initialize coarse state by restriction from previous level
            prev_level = self.levels[-1]
            restrict_state(
                prev_level.Q[1:-1, 1:-1, :],  # Interior only
                prev_level.metrics.volume,
                Q_c[1:-1, 1:-1, :],
                coarse_metrics.volume
            )
            
            # Apply BCs to coarse state
            Q_c = bc_c.apply(Q_c)
            
            # Create level
            level = MultigridLevel(
                NI=NI_c,
                NJ=NJ_c,
                metrics=coarse_metrics,
                Q=Q_c,
                R=np.zeros((NI_c, NJ_c, 4)),
                forcing=np.zeros((NI_c, NJ_c, 4)),
                dt=np.zeros((NI_c, NJ_c)),
                bc=bc_c,
                Q_old=np.zeros((NI_c + 2, NJ_c + 2, 4))
            )
            self.levels.append(level)
            
            # Update for next iteration
            current_X, current_Y = coarse_X, coarse_Y
            current_metrics = coarse_metrics
            current_n_wake = coarse_n_wake
            level_idx += 1
        
        self.num_levels = len(self.levels)
        return self.num_levels
    
    def _compute_farfield_normals(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute outward-pointing unit normals at farfield (j=NJ).
        
        Parameters
        ----------
        X, Y : ndarray, shape (NI+1, NJ+1)
            Node coordinates.
            
        Returns
        -------
        nx, ny : ndarray, shape (NI,)
            Outward unit normals at farfield faces.
        """
        NI = X.shape[0] - 1
        NJ = X.shape[1] - 1
        
        # J-face at j=NJ connects nodes (i, NJ) and (i+1, NJ)
        dx = X[1:, -1] - X[:-1, -1]  # Shape: (NI,)
        dy = Y[1:, -1] - Y[:-1, -1]
        
        # Normal pointing in +j direction (outward): rotate 90° CCW
        # (dx, dy) -> (-dy, dx)
        nx_raw = -dy
        ny_raw = dx
        
        # Normalize
        mag = np.sqrt(nx_raw**2 + ny_raw**2) + 1e-30
        nx = nx_raw / mag
        ny = ny_raw / mag
        
        return nx, ny
    
    def restrict_to_coarse(self, fine_level: int) -> None:
        """
        Restrict state and residual from fine level to coarse level.
        
        Parameters
        ----------
        fine_level : int
            Index of fine level (coarse level is fine_level + 1).
        """
        if fine_level >= self.num_levels - 1:
            return
        
        fine = self.levels[fine_level]
        coarse = self.levels[fine_level + 1]
        
        # Store current coarse Q for correction computation
        coarse.Q_old[:] = coarse.Q
        
        # Restrict state (volume-weighted average)
        restrict_state(
            fine.Q[1:-1, 1:-1, :],
            fine.metrics.volume,
            coarse.Q[1:-1, 1:-1, :],
            coarse.metrics.volume
        )
        
        # Apply BCs to coarse state
        coarse.Q = coarse.bc.apply(coarse.Q)
        
        # Restrict residual (simple summation)
        restrict_residual(fine.R, coarse.R)
    
    def prolongate_correction(self, coarse_level: int, use_injection: bool = True) -> None:
        """
        Prolongate correction from coarse level to fine level.
        
        Parameters
        ----------
        coarse_level : int
            Index of coarse level (fine level is coarse_level - 1).
        use_injection : bool
            If True, use piecewise constant (injection) prolongation.
            If False, use bilinear interpolation.
        """
        if coarse_level <= 0:
            return
        
        fine = self.levels[coarse_level - 1]
        coarse = self.levels[coarse_level]
        
        # Prolongate correction dQ = Q_new - Q_old
        if use_injection:
            prolongate_injection(
                fine.Q[1:-1, 1:-1, :],  # Fine interior (modified in place)
                coarse.Q[1:-1, 1:-1, :],  # Coarse new
                coarse.Q_old[1:-1, 1:-1, :]  # Coarse old
            )
        else:
            prolongate_correction(
                fine.Q[1:-1, 1:-1, :],  # Fine interior (modified in place)
                coarse.Q[1:-1, 1:-1, :],  # Coarse new
                coarse.Q_old[1:-1, 1:-1, :]  # Coarse old
            )
        
        # Apply BCs to fine level after correction
        fine.Q = fine.bc.apply(fine.Q)
    
    def compute_fas_forcing(self, coarse_level: int) -> None:
        """
        Compute FAS forcing term for coarse level.
        
        P_c = R_c^inj - R(Q_c)
        
        where R_c^inj is the restricted residual and R(Q_c) is the
        residual computed on the coarse grid.
        
        Parameters
        ----------
        coarse_level : int
            Index of coarse level (must be > 0).
        """
        if coarse_level <= 0:
            return
        
        coarse = self.levels[coarse_level]
        
        # R_c^inj is already stored in coarse.R after restrict_to_coarse
        R_inj = coarse.R.copy()
        
        # R(Q_c) must be computed by the solver and stored in coarse.R
        # After that, the forcing is computed as:
        # coarse.forcing = R_inj - coarse.R
        # This method should be called after computing coarse residual
        coarse.forcing = R_inj - coarse.R
    
    def apply_bcs(self, level: int) -> None:
        """
        Apply boundary conditions at specified level.
        
        Parameters
        ----------
        level : int
            Level index.
        """
        lvl = self.levels[level]
        lvl.Q = lvl.bc.apply(lvl.Q)
    
    def get_level_info(self) -> str:
        """
        Get summary of all levels in hierarchy.
        
        Returns
        -------
        info : str
            Formatted string with level information.
        """
        lines = [f"Multigrid Hierarchy: {self.num_levels} levels"]
        lines.append("-" * 50)
        
        for i, level in enumerate(self.levels):
            cells = level.NI * level.NJ
            lines.append(
                f"  Level {i}: {level.NI:4d} x {level.NJ:4d} = {cells:6d} cells"
            )
        
        lines.append("-" * 50)
        total_cells = sum(lvl.NI * lvl.NJ for lvl in self.levels)
        lines.append(f"  Total cells: {total_cells}")
        
        return "\n".join(lines)


def build_multigrid_hierarchy(
    X: np.ndarray,
    Y: np.ndarray, 
    Q: np.ndarray,
    freestream: FreestreamConditions,
    n_wake: int = 0,
    beta: float = 10.0,
    min_size: int = 8,
    max_levels: int = 5
) -> MultigridHierarchy:
    """
    Convenience function to build multigrid hierarchy.
    
    Parameters
    ----------
    X, Y : ndarray, shape (NI+1, NJ+1)
        Node coordinates for finest grid.
    Q : ndarray, shape (NI+2, NJ+2, 4)
        Fine grid state vector with ghost cells.
    freestream : FreestreamConditions
        Freestream conditions for BCs.
    n_wake : int
        Number of wake points for C-grid.
    beta : float
        Artificial compressibility parameter.
    min_size : int
        Minimum cells per direction on coarsest grid.
    max_levels : int
        Maximum number of levels.
        
    Returns
    -------
    hierarchy : MultigridHierarchy
        Built hierarchy ready for V-cycle.
    """
    hierarchy = MultigridHierarchy(min_size=min_size, max_levels=max_levels)
    hierarchy.build(X, Y, Q, freestream, n_wake, beta)
    return hierarchy

