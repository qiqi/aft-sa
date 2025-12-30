"""
Grid Metrics Computer for Finite Volume Method.

This module provides the MetricComputer class for computing all grid metrics
required by the FVM flux scheme, including validation of the Geometric
Conservation Law (GCL).

Coordinate System:
    - i: wraps around the airfoil (streamwise for wake, surface-following for airfoil)
    - j: wall-normal direction (j=0 at wall, j=NJ at farfield)

Grid Layout:
    - Node coordinates X, Y have shape (NI+1, NJ+1)
    - Cell (i,j) is bounded by nodes (i,j), (i+1,j), (i+1,j+1), (i,j+1)
    - There are NI×NJ cells total
"""

import numpy as np
from typing import NamedTuple, Tuple, Optional
from dataclasses import dataclass


class FVMMetrics(NamedTuple):
    """
    Finite Volume Method metrics for 2D structured grid.
    
    All arrays use the convention that:
    - Cell quantities have shape (NI, NJ)
    - I-face quantities have shape (NI+1, NJ) - faces between i and i+1
    - J-face quantities have shape (NI, NJ+1) - faces between j and j+1
    
    Face normals are scaled by face area (length in 2D), pointing in the
    positive coordinate direction.
    """
    
    # Cell properties
    volume: np.ndarray      # (NI, NJ) Cell areas
    xc: np.ndarray          # (NI, NJ) Cell center x
    yc: np.ndarray          # (NI, NJ) Cell center y
    
    # I-face normals (scaled by face length)
    # Normal points from cell (i-1,j) toward cell (i,j)
    Si_x: np.ndarray        # (NI+1, NJ) I-face normal x-component * length
    Si_y: np.ndarray        # (NI+1, NJ) I-face normal y-component * length
    
    # J-face normals (scaled by face length)
    # Normal points from cell (i,j-1) toward cell (i,j)
    Sj_x: np.ndarray        # (NI, NJ+1) J-face normal x-component * length
    Sj_y: np.ndarray        # (NI, NJ+1) J-face normal y-component * length
    
    # Wall distance (for turbulence models)
    wall_distance: np.ndarray  # (NI, NJ) Distance to nearest wall
    
    @property
    def NI(self) -> int:
        """Number of cells in i-direction."""
        return self.volume.shape[0]
    
    @property
    def NJ(self) -> int:
        """Number of cells in j-direction."""
        return self.volume.shape[1]
    
    @property
    def Si_mag(self) -> np.ndarray:
        """I-face area (length in 2D)."""
        return np.sqrt(self.Si_x**2 + self.Si_y**2)
    
    @property
    def Sj_mag(self) -> np.ndarray:
        """J-face area (length in 2D)."""
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
    """
    Computes Finite Volume Method metrics from grid coordinates.
    
    Given node coordinates X[i,j], Y[i,j], computes:
    - Cell volumes (areas in 2D)
    - Face normals scaled by area
    - Cell centers
    - Wall distance
    
    Also validates the Geometric Conservation Law (GCL).
    
    Example
    -------
    >>> computer = MetricComputer(X, Y)
    >>> metrics = computer.compute()
    >>> gcl = computer.validate_gcl()
    >>> print(gcl)
    ✓ GCL: max residual (1.23e-15, 4.56e-16), mean (2.34e-16, 1.23e-16)
    """
    
    def __init__(self, X: np.ndarray, Y: np.ndarray, wall_j: int = 0):
        """
        Initialize the metric computer.
        
        Parameters
        ----------
        X, Y : ndarray, shape (NI+1, NJ+1)
            Node coordinates.
        wall_j : int
            J-index of the wall boundary (default 0).
        """
        self.X = X
        self.Y = Y
        self.wall_j = wall_j
        
        # Grid dimensions
        self.NI = X.shape[0] - 1  # Number of cells in i
        self.NJ = X.shape[1] - 1  # Number of cells in j
        
        self._metrics: Optional[FVMMetrics] = None
    
    def compute(self) -> FVMMetrics:
        """
        Compute all FVM metrics.
        
        Returns
        -------
        metrics : FVMMetrics
            Named tuple containing all computed metrics.
        """
        # Cell centers
        xc, yc = self._compute_cell_centers()
        
        # Cell volumes
        volume = self._compute_cell_volumes()
        
        # Face normals
        Si_x, Si_y = self._compute_i_face_normals()
        Sj_x, Sj_y = self._compute_j_face_normals()
        
        # Wall distance
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
        """
        Compute cell centers as average of four corner nodes.
        
        Cell (i,j) has corners at (i,j), (i+1,j), (i+1,j+1), (i,j+1).
        """
        X, Y = self.X, self.Y
        
        xc = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[1:, 1:] + X[:-1, 1:])
        yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[1:, 1:] + Y[:-1, 1:])
        
        return xc, yc
    
    def _compute_cell_volumes(self) -> np.ndarray:
        """
        Compute cell volumes using cross-product of diagonals.
        
        Area = 0.5 * |d1 × d2|
        
        where d1 = AC and d2 = BD are the diagonals of the quadrilateral.
        A = (i,j), B = (i+1,j), C = (i+1,j+1), D = (i,j+1)
        """
        X, Y = self.X, self.Y
        
        # Diagonal AC: from (i,j) to (i+1,j+1)
        dx_ac = X[1:, 1:] - X[:-1, :-1]
        dy_ac = Y[1:, 1:] - Y[:-1, :-1]
        
        # Diagonal BD: from (i+1,j) to (i,j+1)
        dx_bd = X[:-1, 1:] - X[1:, :-1]
        dy_bd = Y[:-1, 1:] - Y[1:, :-1]
        
        # Cross product magnitude (z-component of 3D cross product)
        volume = 0.5 * np.abs(dx_ac * dy_bd - dy_ac * dx_bd)
        
        return volume
    
    def _compute_i_face_normals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute I-face normals (faces perpendicular to i-direction).
        
        Face at (i, j) connects nodes (i, j) and (i, j+1).
        Normal points in positive i-direction (from cell i-1 to cell i).
        
        For a face from node A to node B:
        - Face vector: (dx, dy) = B - A
        - Outward normal (90° CCW rotation): (-dy, dx)
        - But we want normal pointing in +i direction: (dy, -dx)
        
        Returns
        -------
        Si_x, Si_y : ndarray, shape (NI+1, NJ)
            Face normal components scaled by face length.
        """
        X, Y = self.X, self.Y
        
        # Face goes from node (i,j) to node (i,j+1)
        dx = X[:, 1:] - X[:, :-1]
        dy = Y[:, 1:] - Y[:, :-1]
        
        # Normal pointing in +i direction: rotate face vector 90° CW
        # (dx, dy) -> (dy, -dx)
        Si_x = dy
        Si_y = -dx
        
        return Si_x, Si_y
    
    def _compute_j_face_normals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute J-face normals (faces perpendicular to j-direction).
        
        Face at (i, j) connects nodes (i, j) and (i+1, j).
        Normal points in positive j-direction (from cell j-1 to cell j).
        
        For a face from node A to node B:
        - Face vector: (dx, dy) = B - A
        - Normal pointing in +j direction: rotate 90° CCW: (-dy, dx)
        
        Returns
        -------
        Sj_x, Sj_y : ndarray, shape (NI, NJ+1)
            Face normal components scaled by face length.
        """
        X, Y = self.X, self.Y
        
        # Face goes from node (i,j) to node (i+1,j)
        dx = X[1:, :] - X[:-1, :]
        dy = Y[1:, :] - Y[:-1, :]
        
        # Normal pointing in +j direction: rotate face vector 90° CCW
        # (dx, dy) -> (-dy, dx)
        Sj_x = -dy
        Sj_y = dx
        
        return Sj_x, Sj_y
    
    @staticmethod
    def _point_to_segment_distance(px: float, py: float, 
                                    ax: float, ay: float, 
                                    bx: float, by: float) -> float:
        """
        Compute the minimum distance from point P to line segment AB.
        
        The closest point on AB may be:
        - Point A (if projection falls before A)
        - Point B (if projection falls after B)  
        - A point on the segment (if projection falls between A and B)
        """
        # Vector from A to B
        abx = bx - ax
        aby = by - ay
        
        # Vector from A to P
        apx = px - ax
        apy = py - ay
        
        # Squared length of AB
        ab_sq = abx * abx + aby * aby
        
        if ab_sq < 1e-30:
            # Degenerate segment (A and B coincide)
            return np.sqrt(apx * apx + apy * apy)
        
        # Parameter t of projection of P onto line AB
        # t = (AP · AB) / |AB|²
        t = (apx * abx + apy * aby) / ab_sq
        
        # Clamp t to [0, 1] to stay within segment
        t = max(0.0, min(1.0, t))
        
        # Closest point on segment
        closest_x = ax + t * abx
        closest_y = ay + t * aby
        
        # Distance from P to closest point
        dx = px - closest_x
        dy = py - closest_y
        
        return np.sqrt(dx * dx + dy * dy)
    
    def _compute_wall_distance(self, search_radius: int = 20) -> np.ndarray:
        """
        Compute wall distance for each cell center using point-to-segment distance.
        
        This accurately computes the minimum distance from each cell center to the
        wall surface, properly handling curved walls by computing distance to wall
        segments (not just wall grid points).
        
        Algorithm:
        1. For each cell center, start at the wall point on the same i-line
        2. Search left and right along the wall to find local minimum
        3. Compute distance to wall segments, not just grid points
        
        Parameters
        ----------
        search_radius : int
            Number of wall segments to search in each direction.
        """
        X, Y = self.X, self.Y
        NI, NJ = self.NI, self.NJ
        
        # Compute cell center coordinates
        xc, yc = self._compute_cell_centers()
        
        # Wall node coordinates (at j = wall_j)
        x_wall = X[:, self.wall_j]  # Shape: (NI+1,)
        y_wall = Y[:, self.wall_j]
        n_wall = len(x_wall)
        
        wall_dist = np.zeros((NI, NJ))
        
        for i in range(NI):
            for j in range(NJ):
                px = xc[i, j]
                py = yc[i, j]
                
                # Start search from the wall point on the same i-line
                # Cell i is bounded by nodes i and i+1, so start at node i
                start_idx = i
                
                # Search in both directions along the wall
                min_dist = float('inf')
                
                # Search range: [start_idx - search_radius, start_idx + search_radius]
                # Clamp to valid segment indices (segments go from 0 to n_wall-2)
                idx_min = max(0, start_idx - search_radius)
                idx_max = min(n_wall - 2, start_idx + search_radius)
                
                # Check segments from idx_min to idx_max
                for k in range(idx_min, idx_max + 1):
                    # Segment from wall node k to k+1
                    ax, ay = x_wall[k], y_wall[k]
                    bx, by = x_wall[k + 1], y_wall[k + 1]
                    
                    dist = self._point_to_segment_distance(px, py, ax, ay, bx, by)
                    min_dist = min(min_dist, dist)
                
                wall_dist[i, j] = min_dist
        
        return wall_dist
    
    def validate_gcl(self, tol: float = 1e-10) -> GCLValidation:
        """
        Validate the Geometric Conservation Law.
        
        The GCL states that for a closed cell, the sum of all face normals
        (pointing outward) must be zero:
        
            ∑ S⃗ = 0
        
        This ensures that a uniform field has zero divergence.
        
        For cell (i,j), the four faces are:
        - Left (i): -Si[i, j]      (inward)
        - Right (i+1): Si[i+1, j]  (outward)
        - Bottom (j): -Sj[i, j]    (inward)
        - Top (j+1): Sj[i, j+1]    (outward)
        
        Parameters
        ----------
        tol : float
            Tolerance for GCL residual (relative to cell perimeter).
            
        Returns
        -------
        validation : GCLValidation
            Results of the GCL validation.
        """
        if self._metrics is None:
            self.compute()
        
        m = self._metrics
        
        # Compute GCL residual for each cell
        # Sum of outward-pointing normals should be zero
        residual_x = (m.Si_x[1:, :] - m.Si_x[:-1, :] + 
                      m.Sj_x[:, 1:] - m.Sj_x[:, :-1])
        residual_y = (m.Si_y[1:, :] - m.Si_y[:-1, :] + 
                      m.Sj_y[:, 1:] - m.Sj_y[:, :-1])
        
        # Normalize by cell perimeter for relative error
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
    """
    Convenience function to compute FVM metrics.
    
    Parameters
    ----------
    X, Y : ndarray, shape (NI+1, NJ+1)
        Node coordinates.
    wall_j : int
        J-index of the wall boundary.
        
    Returns
    -------
    metrics : FVMMetrics
        Computed grid metrics.
    """
    computer = MetricComputer(X, Y, wall_j)
    return computer.compute()

