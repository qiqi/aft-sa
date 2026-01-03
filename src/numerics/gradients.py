"""
Green-Gauss gradient reconstruction: ∇φ ≈ (1/V) Σ φ_face · S_face

Boundary handling relies on properly set ghost cells.
"""

import numpy as np
import numpy.typing as npt
from typing import NamedTuple

from ..constants import NGHOST
from src.physics.jax_config import jax, jnp

NDArrayFloat = npt.NDArray[np.floating]


class GradientMetrics(NamedTuple):
    """Grid metrics for gradient computation (interior cells only)."""
    Si_x: NDArrayFloat
    Si_y: NDArrayFloat
    Sj_x: NDArrayFloat
    Sj_y: NDArrayFloat
    volume: NDArrayFloat


def compute_gradients(Q: NDArrayFloat, metrics: GradientMetrics) -> NDArrayFloat:
    """Compute cell-centered gradients using Green-Gauss theorem.
    
    This function wraps the JAX implementation for compatibility with NumPy arrays.
    """
    # Convert to JAX arrays
    Q_jax = jnp.asarray(Q)
    Si_x = jnp.asarray(metrics.Si_x)
    Si_y = jnp.asarray(metrics.Si_y)
    Sj_x = jnp.asarray(metrics.Sj_x)
    Sj_y = jnp.asarray(metrics.Sj_y)
    volume = jnp.asarray(metrics.volume)
    
    # Call JAX implementation
    result = compute_gradients_jax(Q_jax, Si_x, Si_y, Sj_x, Sj_y, volume, NGHOST)
    
    # Convert back to NumPy
    return np.asarray(result)


def compute_vorticity(grad: NDArrayFloat) -> NDArrayFloat:
    """Compute vorticity magnitude from gradient array."""
    grad_jax = jnp.asarray(grad)
    result = compute_vorticity_jax(grad_jax)
    return np.asarray(result)


def compute_strain_rate(grad: NDArrayFloat) -> NDArrayFloat:
    """Compute strain rate magnitude |S| = sqrt(2·S_ij·S_ij)."""
    grad_jax = jnp.asarray(grad)
    result = compute_strain_rate_jax(grad_jax)
    return np.asarray(result)


# =============================================================================
# JAX Implementation
# =============================================================================

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
