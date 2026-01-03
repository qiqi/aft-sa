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

# JAX imports for vectorized LS weight computation
# Import from jax_config to ensure float64 is enabled before JAX operations
from src.physics.jax_config import jnp
import jax

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


class FaceGeometry(NamedTuple):
    """Face geometry for tight-stencil viscous flux computation.
    
    For each face, stores:
    - d_coord: distance between adjacent cell centers along coordinate line
    - e_coord: unit vector along coordinate line (from L cell to R cell)
    - e_ortho: unit vector orthogonal to coordinate line (90° CCW rotation of e_coord)
    
    I-faces are between cells (i,j) and (i+1,j), shape (NI+1, NJ).
    J-faces are between cells (i,j) and (i,j+1), shape (NI, NJ+1).
    """
    # I-face geometry (NI+1, NJ)
    d_coord_i: NDArrayFloat      # Distance between cell centers
    e_coord_i_x: NDArrayFloat    # Unit vector along coord line, x-component
    e_coord_i_y: NDArrayFloat    # Unit vector along coord line, y-component
    e_ortho_i_x: NDArrayFloat    # Unit vector orthogonal, x-component
    e_ortho_i_y: NDArrayFloat    # Unit vector orthogonal, y-component
    
    # J-face geometry (NI, NJ+1)
    d_coord_j: NDArrayFloat
    e_coord_j_x: NDArrayFloat
    e_coord_j_y: NDArrayFloat
    e_ortho_j_x: NDArrayFloat
    e_ortho_j_y: NDArrayFloat


class LSWeights(NamedTuple):
    """Least-squares weights for orthogonal derivative at each face.
    
    For tight-stencil viscous flux computation, we compute the orthogonal
    derivative using a weighted sum of 6 cell values:
        grad_ortho = sum(w_k * phi_k) for k=0..5
    
    I-faces use a 2x3 stencil: (i,j-1), (i,j), (i,j+1), (i+1,j-1), (i+1,j), (i+1,j+1)
    J-faces use a 3x2 stencil: (i-1,j), (i,j), (i+1,j), (i-1,j+1), (i,j+1), (i+1,j+1)
    
    Stencil ordering for I-faces (2x3):
        k=0: (i, j-1)     k=3: (i+1, j-1)
        k=1: (i, j)       k=4: (i+1, j)
        k=2: (i, j+1)     k=5: (i+1, j+1)
    
    Stencil ordering for J-faces (3x2):
        k=0: (i-1, j)     k=3: (i-1, j+1)
        k=1: (i, j)       k=4: (i, j+1)
        k=2: (i+1, j)     k=5: (i+1, j+1)
    """
    # I-face LS weights: (NI+1, NJ, 6)
    weights_i: NDArrayFloat
    
    # J-face LS weights: (NI, NJ+1, 6)
    weights_j: NDArrayFloat


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


def _compute_ls_weights_qr(s: NDArrayFloat, t: NDArrayFloat) -> NDArrayFloat:
    """Compute LS weights for orthogonal derivative using QR factorization.
    
    Single-face version for testing. Use _compute_ls_weights_batch_jax for production.

    Finds weights w_k that minimize sum(w_k^2) subject to:
        sum(w_k) = 0
        sum(w_k * s_k) = 1  (ortho derivative = 1)
        sum(w_k * t_k) = 0  (coord derivative = 0)

    Parameters
    ----------
    s : array of shape (6,)
        Positions in orthogonal direction relative to face center.
    t : array of shape (6,)
        Positions in coordinate direction relative to face center.

    Returns
    -------
    w : array of shape (6,)
        LS weights for computing grad_ortho = sum(w_k * phi_k).
    """
    # Build constraint matrix A (6x3): each row is [1, s_k, t_k]
    A = np.column_stack([np.ones(6), s, t])

    # QR factorization: A = Q @ R
    Q, R = np.linalg.qr(A, mode='reduced')  # Q: (6,3), R: (3,3)
    
    # Check for ill-conditioning via R diagonal (cheaper than cond())
    min_diag = np.min(np.abs(np.diag(R)))
    if min_diag < 1e-10:
        return np.zeros(6)

    # We want derivative in s-direction (ortho), so b = [0, 1, 0]
    b = np.array([0.0, 1.0, 0.0])

    # Solve R^T @ lambda = b for Lagrange multipliers
    # Then w = Q @ lambda
    try:
        lam = np.linalg.solve(R.T, b)
        w = Q @ lam
    except np.linalg.LinAlgError:
        # Singular matrix - return zero weights
        return np.zeros(6)
    
    # Additional safety: clip extreme weights
    max_weight = 1e3
    if np.max(np.abs(w)) > max_weight:
        return np.zeros(6)

    return w


@jax.jit
def _compute_ls_weights_batch_jax(s_all: jnp.ndarray, t_all: jnp.ndarray) -> jnp.ndarray:
    """Compute LS weights for all faces in a single vectorized operation.
    
    Uses explicit Gram-Schmidt QR factorization for 6x3 matrices, which is
    ~5000x faster than JAX's batched jnp.linalg.qr for this use case.
    
    Parameters
    ----------
    s_all : array of shape (N_faces, 6)
        Positions in orthogonal direction for all faces.
    t_all : array of shape (N_faces, 6)
        Positions in coordinate direction for all faces.
    
    Returns
    -------
    weights : array of shape (N_faces, 6)
        LS weights for all faces.
    """
    N = s_all.shape[0]
    
    # Build constraint matrices for all faces: columns are [1, s, t]
    ones = jnp.ones((N, 6))
    a0 = ones           # Column 0: all ones (N, 6)
    a1 = s_all          # Column 1: ortho positions (N, 6)
    a2 = t_all          # Column 2: coord positions (N, 6)
    
    # Modified Gram-Schmidt QR for 6x3 matrices (fully vectorized)
    # Q: (N, 6, 3), R: (N, 3, 3)
    
    # First column of Q
    r00 = jnp.sqrt(jnp.sum(a0**2, axis=1))  # (N,)
    r00_safe = jnp.maximum(r00, 1e-30)
    q0 = a0 / r00_safe[:, None]  # (N, 6)
    
    # Second column of Q
    r01 = jnp.sum(q0 * a1, axis=1)  # (N,)
    v1 = a1 - r01[:, None] * q0
    r11 = jnp.sqrt(jnp.sum(v1**2, axis=1))
    r11_safe = jnp.maximum(r11, 1e-30)
    q1 = v1 / r11_safe[:, None]  # (N, 6)
    
    # Third column of Q
    r02 = jnp.sum(q0 * a2, axis=1)  # (N,)
    r12 = jnp.sum(q1 * a2, axis=1)  # (N,)
    v2 = a2 - r02[:, None] * q0 - r12[:, None] * q1
    r22 = jnp.sqrt(jnp.sum(v2**2, axis=1))
    r22_safe = jnp.maximum(r22, 1e-30)
    q2 = v2 / r22_safe[:, None]  # (N, 6)
    
    # Check for ill-conditioning (small R diagonal elements)
    min_diag = jnp.minimum(jnp.minimum(r00, r11), r22)
    ill_cond_mask = min_diag < 1e-10  # (N,)
    
    # Solve R.T @ lambda = b where b = [0, 1, 0]
    # R.T is lower triangular:
    #   [r00,  0,   0 ] [l0]   [0]
    #   [r01, r11,  0 ] [l1] = [1]
    #   [r02, r12, r22] [l2]   [0]
    #
    # Forward substitution:
    #   l0 = 0 / r00 = 0
    #   l1 = (1 - r01*l0) / r11 = 1 / r11
    #   l2 = (0 - r02*l0 - r12*l1) / r22 = -r12 / (r11 * r22)
    
    l0 = jnp.zeros(N)
    l1 = 1.0 / r11_safe
    l2 = -r12 / (r11_safe * r22_safe)
    
    # Weights: w = Q @ lambda = l0*q0 + l1*q1 + l2*q2
    weights = l0[:, None] * q0 + l1[:, None] * q1 + l2[:, None] * q2  # (N, 6)
    
    # Check for extreme weights
    max_weight = jnp.max(jnp.abs(weights), axis=-1)  # (N,)
    too_large_mask = max_weight > 1e3  # (N,)
    
    # Zero out weights for ill-conditioned or too-large cases
    bad_mask = ill_cond_mask | too_large_mask  # (N,)
    weights = jnp.where(bad_mask[:, None], 0.0, weights)
    
    return weights


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
    
    def compute_face_geometry(self) -> FaceGeometry:
        """Compute face geometry for tight-stencil viscous flux computation.
        
        For each face, computes:
        - d_coord: distance between adjacent cell centers
        - e_coord: unit vector along coordinate line (from L to R cell)
        - e_ortho: unit vector orthogonal to coordinate line
        
        Returns
        -------
        FaceGeometry with arrays for I-faces (NI+1, NJ) and J-faces (NI, NJ+1).
        
        Note: Boundary faces (first and last in each direction) use linear 
        extrapolation from interior faces to get meaningful unit vectors.
        """
        # Ensure cell centers are computed
        xc, yc = self._compute_cell_centers()
        
        NI = self.NI
        NJ = self.NJ
        
        # === I-faces: between cells (i,j) and (i+1,j) ===
        # Shape: (NI+1, NJ) - includes boundary faces
        # Interior faces (1 to NI-1): difference between adjacent cell centers
        # Boundary faces (0 and NI): extrapolate using linear continuation
        
        # Use linear extrapolation for padding to get meaningful boundary vectors
        # xc_ghost[-1] = 2*xc[0] - xc[1] (extrapolate left)
        # xc_ghost[NI] = 2*xc[NI-1] - xc[NI-2] (extrapolate right)
        xc_left = 2 * xc[0:1, :] - xc[1:2, :]  # (1, NJ)
        xc_right = 2 * xc[-1:, :] - xc[-2:-1, :]  # (1, NJ)
        xc_padded_i = np.concatenate([xc_left, xc, xc_right], axis=0)  # (NI+2, NJ)
        
        yc_left = 2 * yc[0:1, :] - yc[1:2, :]
        yc_right = 2 * yc[-1:, :] - yc[-2:-1, :]
        yc_padded_i = np.concatenate([yc_left, yc, yc_right], axis=0)
        
        # Cell center differences across I-faces
        # Face i is between padded cells i and i+1
        dx_i = xc_padded_i[1:, :] - xc_padded_i[:-1, :]  # (NI+1, NJ)
        dy_i = yc_padded_i[1:, :] - yc_padded_i[:-1, :]
        
        d_coord_i = np.sqrt(dx_i**2 + dy_i**2)
        d_coord_i = np.maximum(d_coord_i, 1e-30)  # Avoid division by zero
        
        e_coord_i_x = dx_i / d_coord_i
        e_coord_i_y = dy_i / d_coord_i
        
        # Orthogonal vector: 90° CCW rotation of e_coord
        e_ortho_i_x = -e_coord_i_y
        e_ortho_i_y = e_coord_i_x
        
        # === J-faces: between cells (i,j) and (i,j+1) ===
        # Shape: (NI, NJ+1)
        
        xc_bot = 2 * xc[:, 0:1] - xc[:, 1:2]  # (NI, 1)
        xc_top = 2 * xc[:, -1:] - xc[:, -2:-1]
        xc_padded_j = np.concatenate([xc_bot, xc, xc_top], axis=1)  # (NI, NJ+2)
        
        yc_bot = 2 * yc[:, 0:1] - yc[:, 1:2]
        yc_top = 2 * yc[:, -1:] - yc[:, -2:-1]
        yc_padded_j = np.concatenate([yc_bot, yc, yc_top], axis=1)
        
        dx_j = xc_padded_j[:, 1:] - xc_padded_j[:, :-1]  # (NI, NJ+1)
        dy_j = yc_padded_j[:, 1:] - yc_padded_j[:, :-1]
        
        d_coord_j = np.sqrt(dx_j**2 + dy_j**2)
        d_coord_j = np.maximum(d_coord_j, 1e-30)
        
        e_coord_j_x = dx_j / d_coord_j
        e_coord_j_y = dy_j / d_coord_j
        
        e_ortho_j_x = -e_coord_j_y
        e_ortho_j_y = e_coord_j_x
        
        return FaceGeometry(
            d_coord_i=d_coord_i,
            e_coord_i_x=e_coord_i_x,
            e_coord_i_y=e_coord_i_y,
            e_ortho_i_x=e_ortho_i_x,
            e_ortho_i_y=e_ortho_i_y,
            d_coord_j=d_coord_j,
            e_coord_j_x=e_coord_j_x,
            e_coord_j_y=e_coord_j_y,
            e_ortho_j_x=e_ortho_j_x,
            e_ortho_j_y=e_ortho_j_y,
        )
    
    def compute_ls_weights(self, face_geom: Optional[FaceGeometry] = None) -> LSWeights:
        """Compute least-squares weights for orthogonal derivative at each face.
        
        Uses vectorized JAX operations for fast parallel computation on GPU.
        
        For each face, computes 6 weights such that:
            grad_ortho = sum(w_k * phi_k) for k=0..5
        
        The weights minimize sum(w_k^2) subject to:
            - sum(w_k) = 0 (constant -> zero derivative)
            - sum(w_k * s_k) = 1 (derivative in ortho direction = 1)
            - sum(w_k * t_k) = 0 (no contribution from coord direction)
        
        where s_k, t_k are positions in local ortho/coord coordinates.
        
        Parameters
        ----------
        face_geom : FaceGeometry, optional
            Pre-computed face geometry. If None, will be computed.
        
        Returns
        -------
        LSWeights with weights_i (NI+1, NJ, 6) and weights_j (NI, NJ+1, 6).
        """
        if face_geom is None:
            face_geom = self.compute_face_geometry()
        
        xc, yc = self._compute_cell_centers()
        NI, NJ = self.NI, self.NJ
        
        # Pad cell centers once (outside loops)
        xc_full = np.pad(xc, ((1, 1), (1, 1)), mode='edge')  # (NI+2, NJ+2)
        yc_full = np.pad(yc, ((1, 1), (1, 1)), mode='edge')
        
        # === I-face LS weights (2x3 stencil) ===
        # Extract all stencil positions for all I-faces at once
        # I-face (i_face, j) uses stencil cells in padded coords:
        #   k=0: (i_face, j), k=1: (i_face, j+1), k=2: (i_face, j+2)
        #   k=3: (i_face+1, j), k=4: (i_face+1, j+1), k=5: (i_face+1, j+2)
        
        # Create index arrays for vectorized extraction
        i_face_idx = np.arange(NI + 1)[:, None]  # (NI+1, 1)
        j_idx = np.arange(NJ)[None, :]  # (1, NJ)
        
        # Stencil cell positions for all I-faces: (NI+1, NJ, 6)
        # k=0,1,2: left column (i_face), k=3,4,5: right column (i_face+1)
        x_stencil_i = np.stack([
            xc_full[i_face_idx, j_idx],        # k=0: (i_face, j)
            xc_full[i_face_idx, j_idx + 1],    # k=1: (i_face, j+1)
            xc_full[i_face_idx, j_idx + 2],    # k=2: (i_face, j+2)
            xc_full[i_face_idx + 1, j_idx],    # k=3: (i_face+1, j)
            xc_full[i_face_idx + 1, j_idx + 1], # k=4: (i_face+1, j+1)
            xc_full[i_face_idx + 1, j_idx + 2], # k=5: (i_face+1, j+2)
        ], axis=-1)  # (NI+1, NJ, 6)
        
        y_stencil_i = np.stack([
            yc_full[i_face_idx, j_idx],
            yc_full[i_face_idx, j_idx + 1],
            yc_full[i_face_idx, j_idx + 2],
            yc_full[i_face_idx + 1, j_idx],
            yc_full[i_face_idx + 1, j_idx + 1],
            yc_full[i_face_idx + 1, j_idx + 2],
        ], axis=-1)  # (NI+1, NJ, 6)
        
        # Face centers (average of two adjacent cell centers)
        x_face_i = 0.5 * (xc_full[i_face_idx, j_idx + 1] + xc_full[i_face_idx + 1, j_idx + 1])  # (NI+1, NJ)
        y_face_i = 0.5 * (yc_full[i_face_idx, j_idx + 1] + yc_full[i_face_idx + 1, j_idx + 1])
        
        # Positions relative to face center: (NI+1, NJ, 6)
        dx_i = x_stencil_i - x_face_i[:, :, None]
        dy_i = y_stencil_i - y_face_i[:, :, None]
        
        # Transform to local (s, t) coordinates using e_ortho and e_coord
        # s = dx * e_ortho_x + dy * e_ortho_y (ortho direction)
        # t = dx * e_coord_x + dy * e_coord_y (coord direction)
        s_i = dx_i * face_geom.e_ortho_i_x[:, :, None] + dy_i * face_geom.e_ortho_i_y[:, :, None]
        t_i = dx_i * face_geom.e_coord_i_x[:, :, None] + dy_i * face_geom.e_coord_i_y[:, :, None]
        
        # Flatten for batched JAX computation
        n_i_faces = (NI + 1) * NJ
        s_i_flat = s_i.reshape(n_i_faces, 6)
        t_i_flat = t_i.reshape(n_i_faces, 6)
        
        # Compute all I-face weights in parallel using JAX
        weights_i_flat = _compute_ls_weights_batch_jax(jnp.asarray(s_i_flat), jnp.asarray(t_i_flat))
        weights_i = np.asarray(weights_i_flat).reshape(NI + 1, NJ, 6)
        
        # === J-face LS weights (3x2 stencil) ===
        # J-face (i, j_face) uses stencil cells in padded coords:
        #   k=0: (i, j_face), k=1: (i+1, j_face), k=2: (i+2, j_face)
        #   k=3: (i, j_face+1), k=4: (i+1, j_face+1), k=5: (i+2, j_face+1)
        
        i_idx = np.arange(NI)[:, None]  # (NI, 1)
        j_face_idx = np.arange(NJ + 1)[None, :]  # (1, NJ+1)
        
        x_stencil_j = np.stack([
            xc_full[i_idx, j_face_idx],        # k=0: (i, j_face)
            xc_full[i_idx + 1, j_face_idx],    # k=1: (i+1, j_face)
            xc_full[i_idx + 2, j_face_idx],    # k=2: (i+2, j_face)
            xc_full[i_idx, j_face_idx + 1],    # k=3: (i, j_face+1)
            xc_full[i_idx + 1, j_face_idx + 1], # k=4: (i+1, j_face+1)
            xc_full[i_idx + 2, j_face_idx + 1], # k=5: (i+2, j_face+1)
        ], axis=-1)  # (NI, NJ+1, 6)
        
        y_stencil_j = np.stack([
            yc_full[i_idx, j_face_idx],
            yc_full[i_idx + 1, j_face_idx],
            yc_full[i_idx + 2, j_face_idx],
            yc_full[i_idx, j_face_idx + 1],
            yc_full[i_idx + 1, j_face_idx + 1],
            yc_full[i_idx + 2, j_face_idx + 1],
        ], axis=-1)  # (NI, NJ+1, 6)
        
        # Face centers
        x_face_j = 0.5 * (xc_full[i_idx + 1, j_face_idx] + xc_full[i_idx + 1, j_face_idx + 1])  # (NI, NJ+1)
        y_face_j = 0.5 * (yc_full[i_idx + 1, j_face_idx] + yc_full[i_idx + 1, j_face_idx + 1])
        
        # Positions relative to face center
        dx_j = x_stencil_j - x_face_j[:, :, None]
        dy_j = y_stencil_j - y_face_j[:, :, None]
        
        # Transform to local coordinates
        s_j = dx_j * face_geom.e_ortho_j_x[:, :, None] + dy_j * face_geom.e_ortho_j_y[:, :, None]
        t_j = dx_j * face_geom.e_coord_j_x[:, :, None] + dy_j * face_geom.e_coord_j_y[:, :, None]
        
        # Flatten and compute
        n_j_faces = NI * (NJ + 1)
        s_j_flat = s_j.reshape(n_j_faces, 6)
        t_j_flat = t_j.reshape(n_j_faces, 6)
        
        weights_j_flat = _compute_ls_weights_batch_jax(jnp.asarray(s_j_flat), jnp.asarray(t_j_flat))
        weights_j = np.asarray(weights_j_flat).reshape(NI, NJ + 1, 6)
        
        return LSWeights(weights_i=weights_i, weights_j=weights_j)
    
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
