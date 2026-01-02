"""
Grid metrics for Finite Volume Method.

Coordinate system:
    i: wraps around the airfoil (streamwise for wake, surface-following for airfoil)
    j: wall-normal direction (j=0 at wall, j=NJ at farfield)
"""

import numpy as np
from typing import NamedTuple, Tuple, Optional
from dataclasses import dataclass


class FVMMetrics(NamedTuple):
    """Finite Volume Method metrics for 2D structured grid."""
    volume: np.ndarray       # (NI, NJ) cell areas
    xc: np.ndarray           # (NI, NJ) cell center x
    yc: np.ndarray           # (NI, NJ) cell center y
    Si_x: np.ndarray         # (NI+1, NJ) I-face normal x * length
    Si_y: np.ndarray         # (NI+1, NJ) I-face normal y * length
    Sj_x: np.ndarray         # (NI, NJ+1) J-face normal x * length
    Sj_y: np.ndarray         # (NI, NJ+1) J-face normal y * length
    wall_distance: np.ndarray  # (NI, NJ)
    
    @property
    def NI(self) -> int:
        return self.volume.shape[0]
    
    @property
    def NJ(self) -> int:
        return self.volume.shape[1]
    
    @property
    def Si_mag(self) -> np.ndarray:
        return np.sqrt(self.Si_x**2 + self.Si_y**2)
    
    @property
    def Sj_mag(self) -> np.ndarray:
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
    """Computes FVM metrics from grid node coordinates."""
    
    def __init__(self, X: np.ndarray, Y: np.ndarray, wall_j: int = 0):
        self.X = X
        self.Y = Y
        self.wall_j = wall_j
        self.NI = X.shape[0] - 1
        self.NJ = X.shape[1] - 1
        self._metrics: Optional[FVMMetrics] = None
    
    def compute(self) -> FVMMetrics:
        """Compute all FVM metrics."""
        xc, yc = self._compute_cell_centers()
        volume = self._compute_cell_volumes()
        Si_x, Si_y = self._compute_i_face_normals()
        Sj_x, Sj_y = self._compute_j_face_normals()
        wall_distance = self._compute_wall_distance()
        
        self._metrics = FVMMetrics(
            volume=volume,
            xc=xc, yc=yc,
            Si_x=Si_x, Si_y=Si_y,
            Sj_x=Sj_x, Sj_y=Sj_y,
            wall_distance=wall_distance
        )
        
        return self._metrics
    
    def _compute_cell_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Cell center = average of four corner nodes."""
        X, Y = self.X, self.Y
        xc = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[1:, 1:] + X[:-1, 1:])
        yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[1:, 1:] + Y[:-1, 1:])
        return xc, yc
    
    def _compute_cell_volumes(self) -> np.ndarray:
        """Cell area = 0.5 * |diagonal1 × diagonal2|."""
        X, Y = self.X, self.Y
        dx_ac = X[1:, 1:] - X[:-1, :-1]
        dy_ac = Y[1:, 1:] - Y[:-1, :-1]
        dx_bd = X[:-1, 1:] - X[1:, :-1]
        dy_bd = Y[:-1, 1:] - Y[1:, :-1]
        return 0.5 * np.abs(dx_ac * dy_bd - dy_ac * dx_bd)
    
    def _compute_i_face_normals(self) -> Tuple[np.ndarray, np.ndarray]:
        """I-face normal = 90° CW rotation of face vector, scaled by length."""
        X, Y = self.X, self.Y
        dx = X[:, 1:] - X[:, :-1]
        dy = Y[:, 1:] - Y[:, :-1]
        return dy, -dx
    
    def _compute_j_face_normals(self) -> Tuple[np.ndarray, np.ndarray]:
        """J-face normal = 90° CCW rotation of face vector, scaled by length."""
        X, Y = self.X, self.Y
        dx = X[1:, :] - X[:-1, :]
        dy = Y[1:, :] - Y[:-1, :]
        return -dy, dx
    
    @staticmethod
    def _point_to_segment_distance(px: float, py: float, 
                                    ax: float, ay: float, 
                                    bx: float, by: float) -> float:
        """Minimum distance from point P to line segment AB."""
        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay
        ab_sq = abx * abx + aby * aby
        
        if ab_sq < 1e-30:
            return np.sqrt(apx * apx + apy * apy)
        
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_sq))
        closest_x = ax + t * abx
        closest_y = ay + t * aby
        dx = px - closest_x
        dy = py - closest_y
        
        return np.sqrt(dx * dx + dy * dy)
    
    def _compute_wall_distance(self, search_radius: int = 20) -> np.ndarray:
        """Compute wall distance using point-to-segment distance."""
        X, Y = self.X, self.Y
        NI, NJ = self.NI, self.NJ
        
        xc, yc = self._compute_cell_centers()
        x_wall = X[:, self.wall_j]
        y_wall = Y[:, self.wall_j]
        n_wall = len(x_wall)
        
        wall_dist = np.zeros((NI, NJ))
        
        for i in range(NI):
            for j in range(NJ):
                px = xc[i, j]
                py = yc[i, j]
                start_idx = i
                min_dist = float('inf')
                idx_min = max(0, start_idx - search_radius)
                idx_max = min(n_wall - 2, start_idx + search_radius)
                
                for k in range(idx_min, idx_max + 1):
                    ax, ay = x_wall[k], y_wall[k]
                    bx, by = x_wall[k + 1], y_wall[k + 1]
                    dist = self._point_to_segment_distance(px, py, ax, ay, bx, by)
                    min_dist = min(min_dist, dist)
                
                wall_dist[i, j] = min_dist
        
        return wall_dist
    
    def validate_gcl(self, tol: float = 1e-10) -> GCLValidation:
        """Validate Geometric Conservation Law: sum of face normals = 0."""
        if self._metrics is None:
            self.compute()
        
        m = self._metrics
        residual_x = (m.Si_x[1:, :] - m.Si_x[:-1, :] + 
                      m.Sj_x[:, 1:] - m.Sj_x[:, :-1])
        residual_y = (m.Si_y[1:, :] - m.Si_y[:-1, :] + 
                      m.Sj_y[:, 1:] - m.Sj_y[:, :-1])
        
        perimeter = (m.Si_mag[:-1, :] + m.Si_mag[1:, :] + 
                     m.Sj_mag[:, :-1] + m.Sj_mag[:, 1:])
        
        rel_residual_x = np.abs(residual_x) / (perimeter + 1e-30)
        rel_residual_y = np.abs(residual_y) / (perimeter + 1e-30)
        
        max_x = np.max(np.abs(residual_x))
        max_y = np.max(np.abs(residual_y))
        mean_x = np.mean(np.abs(residual_x))
        mean_y = np.mean(np.abs(residual_y))
        
        max_rel = max(np.max(rel_residual_x), np.max(rel_residual_y))
        passed = max_rel < tol
        
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


def compute_metrics(X: np.ndarray, Y: np.ndarray, wall_j: int = 0) -> FVMMetrics:
    """Convenience function to compute FVM metrics."""
    computer = MetricComputer(X, Y, wall_j)
    return computer.compute()
