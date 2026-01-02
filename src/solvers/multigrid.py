"""
Multigrid Hierarchy Manager for FAS (Full Approximation Storage) Scheme.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..grid.metrics import MetricComputer, FVMMetrics
from ..grid.coarsening import Coarsener
from ..numerics.multigrid import restrict_state, restrict_residual, prolongate_correction, prolongate_injection

from .boundary_conditions import BoundaryConditions, FreestreamConditions
from ..constants import NGHOST


@dataclass
class MultigridLevel:
    """Data for a single multigrid level."""
    
    NI: int
    NJ: int
    metrics: FVMMetrics
    Q: np.ndarray
    R: np.ndarray
    forcing: np.ndarray
    dt: np.ndarray
    bc: BoundaryConditions
    k4: float = 0.04
    cfl_scale: float = 1.0
    Q_old: Optional[np.ndarray] = None
    
    @property
    def nvar(self) -> int:
        return 4


class MultigridHierarchy:
    """Manages the multigrid hierarchy for FAS scheme."""
    
    def __init__(self, min_size: int = 8, max_levels: int = 5):
        """Initialize multigrid hierarchy manager."""
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
              beta: float = 10.0,
        base_k4: float = 0.04,
        dissipation_scaling: float = 2.0,
        coarse_cfl_factor: float = 0.5) -> int:
        """Build the multigrid hierarchy."""
        self.levels = []
        
        NI = X.shape[0] - 1
        NJ = X.shape[1] - 1
        
        computer = MetricComputer(X, Y)
        metrics = computer.compute()
        
        farfield_normals = self._compute_farfield_normals(X, Y)
        
        nx, ny = farfield_normals
        if len(nx) != NI:
            raise ValueError(
                f"Level 0: Farfield normals length ({len(nx)}) != NI ({NI}). "
                f"X shape: {X.shape}, expected nodes: ({NI+1}, {NJ+1})"
            )
        
        bc = BoundaryConditions(
            freestream=freestream,
            n_wake_points=n_wake,
            farfield_normals=farfield_normals,
            beta=beta
        )
        
        level0 = MultigridLevel(
            NI=NI,
            NJ=NJ,
            metrics=metrics,
            Q=Q_fine.copy(),
            R=np.zeros((NI, NJ, 4)),
            forcing=np.zeros((NI, NJ, 4)),
            dt=np.zeros((NI, NJ)),
            bc=bc,
            k4=base_k4,
            cfl_scale=1.0,
            Q_old=np.zeros((NI + 2*NGHOST, NJ + 2*NGHOST, 4))
        )
        self.levels.append(level0)
        
        current_X, current_Y = X, Y
        current_metrics = metrics
        current_n_wake = n_wake
        
        level_idx = 1
        while level_idx < self.max_levels:
            if not Coarsener.can_coarsen(current_metrics.NI, current_metrics.NJ, 
                                          self.min_size):
                break
            
            coarse_X = current_X[::2, ::2]
            coarse_Y = current_Y[::2, ::2]
            
            coarse_metrics = Coarsener.coarsen(current_metrics)
            NI_c, NJ_c = coarse_metrics.NI, coarse_metrics.NJ
            coarse_n_wake = current_n_wake // 2
            coarse_farfield_normals = self._compute_farfield_normals(coarse_X, coarse_Y)
            
            nx_c, ny_c = coarse_farfield_normals
            if len(nx_c) != NI_c:
                raise ValueError(
                    f"Level {level_idx}: Farfield normals length ({len(nx_c)}) != NI_c ({NI_c}). "
                f"coarse_X shape: {coarse_X.shape}, expected nodes: ({NI_c+1}, {NJ_c+1})"
            )
            
            bc_c = BoundaryConditions(
                freestream=freestream,
                n_wake_points=coarse_n_wake,
                farfield_normals=coarse_farfield_normals,
                beta=beta
            )
            
            Q_c = np.zeros((NI_c + 2*NGHOST, NJ_c + 2*NGHOST, 4))
            
            prev_level = self.levels[-1]
            restrict_state(
                prev_level.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :],  # Interior only
                prev_level.metrics.volume,
                Q_c[NGHOST:-NGHOST, NGHOST:-NGHOST, :],
                coarse_metrics.volume
            )
            
            Q_c = bc_c.apply(Q_c)
            level_k4 = base_k4 * (dissipation_scaling ** level_idx)
            
            level = MultigridLevel(
                NI=NI_c,
                NJ=NJ_c,
                metrics=coarse_metrics,
                Q=Q_c,
                R=np.zeros((NI_c, NJ_c, 4)),
                forcing=np.zeros((NI_c, NJ_c, 4)),
                dt=np.zeros((NI_c, NJ_c)),
                bc=bc_c,
                k4=level_k4,
                cfl_scale=coarse_cfl_factor,
                Q_old=np.zeros((NI_c + 2*NGHOST, NJ_c + 2*NGHOST, 4))
            )
            self.levels.append(level)
            
            current_X, current_Y = coarse_X, coarse_Y
            current_metrics = coarse_metrics
            current_n_wake = coarse_n_wake
            level_idx += 1
        
        self.num_levels = len(self.levels)
        return self.num_levels
    
    def _compute_farfield_normals(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute outward-pointing unit normals at farfield (j=NJ)."""
        NI = X.shape[0] - 1
        
        dx = X[1:, -1] - X[:-1, -1]
        dy = Y[1:, -1] - Y[:-1, -1]
        
        nx_raw = -dy
        ny_raw = dx
        
        mag = np.sqrt(nx_raw**2 + ny_raw**2) + 1e-30
        nx = nx_raw / mag
        ny = ny_raw / mag
        
        return nx, ny
    
    def restrict_to_coarse(self, fine_level: int) -> None:
        """Restrict state and residual from fine level to coarse level."""
        if fine_level >= self.num_levels - 1:
            return
        
        fine = self.levels[fine_level]
        coarse = self.levels[fine_level + 1]
        
        coarse.Q_old[:] = coarse.Q
        
        restrict_state(
            fine.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :],
            fine.metrics.volume,
            coarse.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :],
            coarse.metrics.volume
        )
        
        coarse.Q = coarse.bc.apply(coarse.Q)
        restrict_residual(fine.R, coarse.R)
    
    def prolongate_correction(self, coarse_level: int, use_injection: bool = True) -> None:
        """Prolongate correction from coarse level to fine level."""
        if coarse_level <= 0:
            return
        
        fine = self.levels[coarse_level - 1]
        coarse = self.levels[coarse_level]
        
        if use_injection:
            prolongate_injection(
                fine.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :],
                coarse.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :],
                coarse.Q_old[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
            )
        else:
            prolongate_correction(
                fine.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :],
                coarse.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :],
                coarse.Q_old[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
            )
        
        fine.Q = fine.bc.apply(fine.Q)
    
    def compute_fas_forcing(self, coarse_level: int) -> None:
        """Compute FAS forcing term: P_c = R_c^inj - R(Q_c)."""
        if coarse_level <= 0:
            return
        
        coarse = self.levels[coarse_level]
        R_inj = coarse.R.copy()
        coarse.forcing = R_inj - coarse.R
    
    def apply_bcs(self, level: int) -> None:
        """Apply boundary conditions at specified level."""
        lvl = self.levels[level]
        lvl.Q = lvl.bc.apply(lvl.Q)
    
    def get_level_info(self) -> str:
        """Get summary of all levels in hierarchy."""
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
    max_levels: int = 5,
    base_k4: float = 0.04,
    dissipation_scaling: float = 2.0,
    coarse_cfl_factor: float = 0.5
) -> MultigridHierarchy:
    """Convenience function to build multigrid hierarchy."""
    hierarchy = MultigridHierarchy(min_size=min_size, max_levels=max_levels)
    hierarchy.build(X, Y, Q, freestream, n_wake, beta, base_k4, dissipation_scaling, coarse_cfl_factor)
    return hierarchy

