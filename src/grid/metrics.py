"""
Grid metrics for Finite Volume Method.

Coordinate system:
    i: wraps around the airfoil (streamwise for wake, surface-following for airfoil)
    j: wall-normal direction (j=0 at wall, j=NJ at farfield)
"""

import numpy as np
import numpy.typing as npt
from typing import NamedTuple, Tuple, Optional
from dataclasses import dataclass

NDArrayFloat = npt.NDArray[np.floating]


class FVMMetrics(NamedTuple):
    """Finite Volume Method metrics for 2D structured grid."""
    volume: NDArrayFloat
    xc: NDArrayFloat
    yc: NDArrayFloat
    Si_x: NDArrayFloat
    Si_y: NDArrayFloat
    Sj_x: NDArrayFloat
    Sj_y: NDArrayFloat
    wall_distance: NDArrayFloat
    
    @property
    def NI(self) -> int:
        return self.volume.shape[0]
    
    @property
    def NJ(self) -> int:
        return self.volume.shape[1]
    
    @property
    def Si_mag(self) -> NDArrayFloat:
        return np.sqrt(self.Si_x**2 + self.Si_y**2)
    
    @property
    def Sj_mag(self) -> NDArrayFloat:
        return np.sqrt(self.Sj_x**2 + self.Sj_y**2)


@dataclass
class GCLValidation:
    """Results of Geometric Conservation Law validation."""
    passed: bool
    max_x_residual: float
    max_y_residual: float
    mean_x_residual: float
    mean_y_residual: float
    message: str
    
    def __str__(self) -> str:
        status = "✓" if self.passed else "✗"
        return (f"{status} GCL: max residual ({self.max_x_residual:.2e}, {self.max_y_residual:.2e}), "
                f"mean ({self.mean_x_residual:.2e}, {self.mean_y_residual:.2e})")


class MetricComputer:
    """Computes FVM metrics from grid node coordinates.
    
    For C-grids, n_wake specifies how many points at each end of the i-direction
    are wake (not wall). The wall is only i = n_wake to NI - n_wake.
    """
    
    X: NDArrayFloat
    Y: NDArrayFloat
    wall_j: int
    n_wake: int
    NI: int
    NJ: int
    _metrics: Optional[FVMMetrics]
    
    def __init__(self, X: NDArrayFloat, Y: NDArrayFloat, wall_j: int = 0, n_wake: int = 0) -> None:
        self.X = X
        self.Y = Y
        self.wall_j = wall_j
        self.n_wake = n_wake  # Number of wake points on each end (not part of wall)
        self.NI = X.shape[0] - 1
        self.NJ = X.shape[1] - 1
        self._metrics = None
    
    def compute(self) -> FVMMetrics:
        """Compute all FVM metrics."""
        xc: NDArrayFloat
        yc: NDArrayFloat
        xc, yc = self._compute_cell_centers()
        volume: NDArrayFloat = self._compute_cell_volumes()
        Si_x: NDArrayFloat
        Si_y: NDArrayFloat
        Si_x, Si_y = self._compute_i_face_normals()
        Sj_x: NDArrayFloat
        Sj_y: NDArrayFloat
        Sj_x, Sj_y = self._compute_j_face_normals()
        wall_distance: NDArrayFloat = self._compute_wall_distance()
        
        self._metrics = FVMMetrics(
            volume=volume,
            xc=xc, yc=yc,
            Si_x=Si_x, Si_y=Si_y,
            Sj_x=Sj_x, Sj_y=Sj_y,
            wall_distance=wall_distance
        )
        
        return self._metrics
    
    def _compute_cell_centers(self) -> Tuple[NDArrayFloat, NDArrayFloat]:
        """Cell center = average of four corner nodes."""
        X: NDArrayFloat = self.X
        Y: NDArrayFloat = self.Y
        xc: NDArrayFloat = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[1:, 1:] + X[:-1, 1:])
        yc: NDArrayFloat = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[1:, 1:] + Y[:-1, 1:])
        return xc, yc
    
    def _compute_cell_volumes(self) -> NDArrayFloat:
        """Cell area = 0.5 * |diagonal1 × diagonal2|."""
        X: NDArrayFloat = self.X
        Y: NDArrayFloat = self.Y
        dx_ac: NDArrayFloat = X[1:, 1:] - X[:-1, :-1]
        dy_ac: NDArrayFloat = Y[1:, 1:] - Y[:-1, :-1]
        dx_bd: NDArrayFloat = X[:-1, 1:] - X[1:, :-1]
        dy_bd: NDArrayFloat = Y[:-1, 1:] - Y[1:, :-1]
        return 0.5 * np.abs(dx_ac * dy_bd - dy_ac * dx_bd)
    
    def _compute_i_face_normals(self) -> Tuple[NDArrayFloat, NDArrayFloat]:
        """I-face normal = 90° CW rotation of face vector, scaled by length."""
        X: NDArrayFloat = self.X
        Y: NDArrayFloat = self.Y
        dx: NDArrayFloat = X[:, 1:] - X[:, :-1]
        dy: NDArrayFloat = Y[:, 1:] - Y[:, :-1]
        return dy, -dx
    
    def _compute_j_face_normals(self) -> Tuple[NDArrayFloat, NDArrayFloat]:
        """J-face normal = 90° CCW rotation of face vector, scaled by length."""
        X: NDArrayFloat = self.X
        Y: NDArrayFloat = self.Y
        dx: NDArrayFloat = X[1:, :] - X[:-1, :]
        dy: NDArrayFloat = Y[1:, :] - Y[:-1, :]
        return -dy, dx
    
    @staticmethod
    def _point_to_segment_distance(px: float, py: float, 
                                    ax: float, ay: float, 
                                    bx: float, by: float) -> float:
        """Minimum distance from point P to line segment AB."""
        abx: float = bx - ax
        aby: float = by - ay
        apx: float = px - ax
        apy: float = py - ay
        ab_sq: float = abx * abx + aby * aby
        
        if ab_sq < 1e-30:
            return float(np.sqrt(apx * apx + apy * apy))
        
        t: float = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_sq))
        closest_x: float = ax + t * abx
        closest_y: float = ay + t * aby
        dx: float = px - closest_x
        dy: float = py - closest_y
        
        return float(np.sqrt(dx * dx + dy * dy))
    
    def _compute_wall_distance(self, search_radius: int = 20) -> NDArrayFloat:
        """Compute wall distance using point-to-segment distance.
        
        For C-grids, only the airfoil surface is considered wall (not the wake cut).
        The airfoil surface is at j=wall_j for i in [n_wake, NI - n_wake].
        """
        X: NDArrayFloat = self.X
        Y: NDArrayFloat = self.Y
        NI: int = self.NI
        NJ: int = self.NJ
        
        xc: NDArrayFloat
        yc: NDArrayFloat
        xc, yc = self._compute_cell_centers()
        
        # Only use airfoil surface, not wake cut
        # Wall nodes are from n_wake to NI - n_wake (inclusive) at j=wall_j
        wall_start: int = self.n_wake
        wall_end: int = NI + 1 - self.n_wake  # +1 because X has NI+1 nodes
        
        x_wall: NDArrayFloat = X[wall_start:wall_end, self.wall_j]
        y_wall: NDArrayFloat = Y[wall_start:wall_end, self.wall_j]
        n_wall: int = len(x_wall)
        
        wall_dist: NDArrayFloat = np.zeros((NI, NJ))
        
        for i in range(NI):
            for j in range(NJ):
                px: float = float(xc[i, j])
                py: float = float(yc[i, j])
                
                # For cells on the airfoil, search near the local position
                # For wake cells, search the entire airfoil
                if wall_start <= i < wall_end - 1:
                    # Cell is on the airfoil - use local search
                    local_i: int = i - wall_start  # Position in wall array
                    idx_min: int = max(0, local_i - search_radius)
                    idx_max: int = min(n_wall - 2, local_i + search_radius)
                else:
                    # Cell is in wake - search entire airfoil
                    idx_min = 0
                    idx_max = n_wall - 2
                
                min_dist: float = float('inf')
                for k in range(idx_min, idx_max + 1):
                    a_x: float = float(x_wall[k])
                    a_y: float = float(y_wall[k])
                    b_x: float = float(x_wall[k + 1])
                    b_y: float = float(y_wall[k + 1])
                    dist: float = self._point_to_segment_distance(px, py, a_x, a_y, b_x, b_y)
                    min_dist = min(min_dist, dist)
                
                wall_dist[i, j] = min_dist
        
        return wall_dist
    
    def validate_gcl(self, tol: float = 1e-10) -> GCLValidation:
        """Validate Geometric Conservation Law: sum of face normals = 0."""
        if self._metrics is None:
            self.compute()
        
        m: FVMMetrics = self._metrics  # type: ignore[assignment]
        residual_x: NDArrayFloat = (m.Si_x[1:, :] - m.Si_x[:-1, :] + 
                      m.Sj_x[:, 1:] - m.Sj_x[:, :-1])
        residual_y: NDArrayFloat = (m.Si_y[1:, :] - m.Si_y[:-1, :] + 
                      m.Sj_y[:, 1:] - m.Sj_y[:, :-1])
        
        perimeter: NDArrayFloat = (m.Si_mag[:-1, :] + m.Si_mag[1:, :] + 
                     m.Sj_mag[:, :-1] + m.Sj_mag[:, 1:])
        
        rel_residual_x: NDArrayFloat = np.abs(residual_x) / (perimeter + 1e-30)
        rel_residual_y: NDArrayFloat = np.abs(residual_y) / (perimeter + 1e-30)
        
        max_x: float = float(np.max(np.abs(residual_x)))
        max_y: float = float(np.max(np.abs(residual_y)))
        mean_x: float = float(np.mean(np.abs(residual_x)))
        mean_y: float = float(np.mean(np.abs(residual_y)))
        
        max_rel: float = float(max(np.max(rel_residual_x), np.max(rel_residual_y)))
        passed: bool = max_rel < tol
        
        message: str
        if passed:
            message = f"GCL satisfied (max relative residual: {max_rel:.2e})"
        else:
            message = f"GCL VIOLATED (max relative residual: {max_rel:.2e} > {tol:.2e})"
        
        return GCLValidation(
            passed=passed,
            max_x_residual=max_x,
            max_y_residual=max_y,
            mean_x_residual=mean_x,
            mean_y_residual=mean_y,
            message=message
        )


def compute_metrics(X: NDArrayFloat, Y: NDArrayFloat, wall_j: int = 0, n_wake: int = 0) -> FVMMetrics:
    """Convenience function to compute FVM metrics.
    
    Parameters
    ----------
    X, Y : ndarray
        Grid node coordinates, shape (NI+1, NJ+1).
    wall_j : int
        J-index of the wall boundary (default 0).
    n_wake : int
        Number of wake points on each end of i-direction (not part of physical wall).
        For C-grids, the airfoil surface is i = n_wake to NI - n_wake.
    """
    computer: MetricComputer = MetricComputer(X, Y, wall_j, n_wake)
    return computer.compute()
