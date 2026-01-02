"""
Green-Gauss gradient reconstruction: ∇φ ≈ (1/V) Σ φ_face · S_face

Boundary handling relies on properly set ghost cells.
"""

import numpy as np
import numpy.typing as npt
from numba import njit
from typing import NamedTuple

from ..constants import NGHOST

# JAX imports
try:
    from src.physics.jax_config import jax, jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

NDArrayFloat = npt.NDArray[np.floating]


class GradientMetrics(NamedTuple):
    """Grid metrics for gradient computation (interior cells only)."""
    Si_x: NDArrayFloat
    Si_y: NDArrayFloat
    Sj_x: NDArrayFloat
    Sj_y: NDArrayFloat
    volume: NDArrayFloat


@njit(cache=True)
def _gradient_kernel(Q: np.ndarray,
                     Si_x: np.ndarray, Si_y: np.ndarray,
                     Sj_x: np.ndarray, Sj_y: np.ndarray,
                     volume: np.ndarray,
                     grad: np.ndarray,
                     nghost: int) -> None:
    """Numba kernel for Green-Gauss gradient computation."""
    NI = volume.shape[0]
    NJ = volume.shape[1]
    
    for i in range(NI):
        for j in range(NJ):
            for k in range(4):
                grad[i, j, k, 0] = 0.0
                grad[i, j, k, 1] = 0.0
    
    # I-face contributions
    for i in range(NI + 1):
        for j in range(NJ):
            nx = Si_x[i, j]
            ny = Si_y[i, j]
            
            i_L = i + nghost - 1
            i_R = i + nghost
            j_Q = j + nghost
            
            for k in range(4):
                phi_L = Q[i_L, j_Q, k]
                phi_R = Q[i_R, j_Q, k]
                phi_face = 0.5 * (phi_L + phi_R)
                contrib_x = phi_face * nx
                contrib_y = phi_face * ny
                
                if i > 0:
                    grad[i - 1, j, k, 0] += contrib_x
                    grad[i - 1, j, k, 1] += contrib_y
                if i < NI:
                    grad[i, j, k, 0] -= contrib_x
                    grad[i, j, k, 1] -= contrib_y
    
    # J-face contributions
    for i in range(NI):
        for j in range(NJ + 1):
            nx = Sj_x[i, j]
            ny = Sj_y[i, j]
            
            i_Q = i + nghost
            j_L = j + nghost - 1
            j_R = j + nghost
            
            for k in range(4):
                phi_L = Q[i_Q, j_L, k]
                phi_R = Q[i_Q, j_R, k]
                phi_face = 0.5 * (phi_L + phi_R)
                contrib_x = phi_face * nx
                contrib_y = phi_face * ny
                
                if j > 0:
                    grad[i, j - 1, k, 0] += contrib_x
                    grad[i, j - 1, k, 1] += contrib_y
                if j < NJ:
                    grad[i, j, k, 0] -= contrib_x
                    grad[i, j, k, 1] -= contrib_y
    
    # Divide by volume
    for i in range(NI):
        for j in range(NJ):
            inv_vol = 1.0 / volume[i, j]
            for k in range(4):
                grad[i, j, k, 0] *= inv_vol
                grad[i, j, k, 1] *= inv_vol


def compute_gradients(Q: NDArrayFloat, metrics: GradientMetrics) -> NDArrayFloat:
    """Compute cell-centered gradients using Green-Gauss theorem."""
    NI: int
    NJ: int
    NI, NJ = metrics.volume.shape
    grad: NDArrayFloat = np.zeros((NI, NJ, 4, 2), dtype=np.float64)
    
    _gradient_kernel(
        Q.astype(np.float64),
        metrics.Si_x.astype(np.float64),
        metrics.Si_y.astype(np.float64),
        metrics.Sj_x.astype(np.float64),
        metrics.Sj_y.astype(np.float64),
        metrics.volume.astype(np.float64),
        grad,
        NGHOST
    )
    
    return grad


@njit(cache=True)
def _compute_vorticity_kernel(dudx: np.ndarray, dudy: np.ndarray,
                               dvdx: np.ndarray, dvdy: np.ndarray,
                               omega: np.ndarray) -> None:
    """Compute vorticity magnitude ω = |∂v/∂x - ∂u/∂y|."""
    NI: int
    NJ: int
    NI, NJ = dudx.shape
    for i in range(NI):
        for j in range(NJ):
            omega[i, j] = abs(dvdx[i, j] - dudy[i, j])


def compute_vorticity(grad: NDArrayFloat) -> NDArrayFloat:
    """Compute vorticity magnitude from gradient array."""
    NI: int
    NJ: int
    NI, NJ = grad.shape[:2]
    dudy: NDArrayFloat = grad[:, :, 1, 1]
    dvdx: NDArrayFloat = grad[:, :, 2, 0]
    
    omega: NDArrayFloat = np.zeros((NI, NJ), dtype=np.float64)
    _compute_vorticity_kernel(
        grad[:, :, 1, 0].astype(np.float64),
        dudy.astype(np.float64),
        dvdx.astype(np.float64),
        grad[:, :, 2, 1].astype(np.float64),
        omega
    )
    
    return omega


def compute_strain_rate(grad: NDArrayFloat) -> NDArrayFloat:
    """Compute strain rate magnitude |S| = sqrt(2·S_ij·S_ij)."""
    dudx: NDArrayFloat = grad[:, :, 1, 0]
    dudy: NDArrayFloat = grad[:, :, 1, 1]
    dvdx: NDArrayFloat = grad[:, :, 2, 0]
    dvdy: NDArrayFloat = grad[:, :, 2, 1]
    
    S_xx: NDArrayFloat = dudx
    S_yy: NDArrayFloat = dvdy
    S_xy: NDArrayFloat = 0.5 * (dudy + dvdx)
    
    S_mag: NDArrayFloat = np.sqrt(2.0 * (S_xx**2 + S_yy**2 + 2.0 * S_xy**2))
    return S_mag


# =============================================================================
# JAX Implementations
# =============================================================================

if JAX_AVAILABLE:
    
    def compute_gradients_jax(Q, Si_x, Si_y, Sj_x, Sj_y, volume, nghost):
        """
        JAX: Compute cell-centered gradients using Green-Gauss theorem.
        
        ∇φ ≈ (1/V) Σ φ_face · S_face
        
        Parameters
        ----------
        Q : jnp.ndarray
            State array (NI+2*nghost, NJ+2*nghost, 4).
        Si_x, Si_y : jnp.ndarray
            I-face normal vectors (NI+1, NJ).
        Sj_x, Sj_y : jnp.ndarray
            J-face normal vectors (NI, NJ+1).
        volume : jnp.ndarray
            Cell volumes (NI, NJ).
        nghost : int
            Number of ghost cells.
            
        Returns
        -------
        grad : jnp.ndarray
            Gradients (NI, NJ, 4, 2) where last dim is (dx, dy).
        """
        NI, NJ = volume.shape
        
        # Extract Q slices using concrete nghost
        Q_L_i = Q[nghost-1:nghost+NI, nghost:nghost+NJ, :]   # (NI+1, NJ, 4)
        Q_R_i = Q[nghost:nghost+NI+1, nghost:nghost+NJ, :]   # (NI+1, NJ, 4)
        Q_L_j = Q[nghost:nghost+NI, nghost-1:nghost+NJ, :]   # (NI, NJ+1, 4)
        Q_R_j = Q[nghost:nghost+NI, nghost:nghost+NJ+1, :]   # (NI, NJ+1, 4)
        
        return _compute_gradients_jax_impl(Q_L_i, Q_R_i, Q_L_j, Q_R_j,
                                            Si_x, Si_y, Sj_x, Sj_y, volume)
    
    @jax.jit
    def _compute_gradients_jax_impl(Q_L_i, Q_R_i, Q_L_j, Q_R_j,
                                     Si_x, Si_y, Sj_x, Sj_y, volume):
        """JIT-compiled gradient computation kernel."""
        
        # I-face contributions
        # For cell m: receives +contrib from face m+1 (right), -contrib from face m (left)
        # So grad[m] = contrib[m+1] - contrib[m]
        phi_face_i = 0.5 * (Q_L_i + Q_R_i)  # (NI+1, NJ, 4)
        
        contrib_i_x = phi_face_i * Si_x[:, :, None]
        contrib_i_y = phi_face_i * Si_y[:, :, None]
        
        # contrib[m+1] - contrib[m] for each cell m
        grad_i_x = contrib_i_x[1:, :, :] - contrib_i_x[:-1, :, :]
        grad_i_y = contrib_i_y[1:, :, :] - contrib_i_y[:-1, :, :]
        
        # J-face contributions (same logic)
        phi_face_j = 0.5 * (Q_L_j + Q_R_j)  # (NI, NJ+1, 4)
        
        contrib_j_x = phi_face_j * Sj_x[:, :, None]
        contrib_j_y = phi_face_j * Sj_y[:, :, None]
        
        grad_j_x = contrib_j_x[:, 1:, :] - contrib_j_x[:, :-1, :]
        grad_j_y = contrib_j_y[:, 1:, :] - contrib_j_y[:, :-1, :]
        
        # Sum and divide by volume
        inv_vol = 1.0 / volume[:, :, None]
        grad_x = (grad_i_x + grad_j_x) * inv_vol
        grad_y = (grad_i_y + grad_j_y) * inv_vol
        
        return jnp.stack([grad_x, grad_y], axis=-1)

    @jax.jit
    def compute_vorticity_jax(grad):
        """JAX: Compute vorticity magnitude ω = |∂v/∂x - ∂u/∂y|."""
        dudy = grad[:, :, 1, 1]
        dvdx = grad[:, :, 2, 0]
        return jnp.abs(dvdx - dudy)

    @jax.jit
    def compute_strain_rate_jax(grad):
        """JAX: Compute strain rate magnitude |S| = sqrt(2·S_ij·S_ij)."""
        dudx = grad[:, :, 1, 0]
        dudy = grad[:, :, 1, 1]
        dvdx = grad[:, :, 2, 0]
        dvdy = grad[:, :, 2, 1]
        
        S_xx = dudx
        S_yy = dvdy
        S_xy = 0.5 * (dudy + dvdx)
        
        return jnp.sqrt(2.0 * (S_xx**2 + S_yy**2 + 2.0 * S_xy**2))
