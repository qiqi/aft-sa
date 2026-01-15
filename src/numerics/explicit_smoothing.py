"""
Multi-Stage Explicit Residual Smoothing for 2D structured grids.

Smoothing operator: R̄_i = R_i + ε(R_{i+1} - 2R_i + R_{i-1})
Applied 2 passes in ADI fashion (I-direction then J-direction).

For C-grid topology, J-direction smoothing at j=0 uses wake cut connectivity:
wake cells connect across the cut in reverse i-order.

Advantages over Implicit (tridiagonal) smoothing:
- Fully parallelizable on GPU
- No sequential dependencies along lines

Reference: Jameson, Schmidt, Turkel (1981). AIAA paper 81-1259.
"""

import numpy as np
import numpy.typing as npt

from src.physics.jax_config import jax, jnp

NDArrayFloat = npt.NDArray[np.floating]


# =============================================================================
# JAX Implementation
# =============================================================================

def _smooth_pass_i_jax(R, epsilon):
    """Apply one smoothing pass in I-direction (Neumann BC)."""
    # R has shape (NI, NJ, n_vars)
    
    # R_plus[i] = R[i+1], with R_plus[-1] = R[-1] (Neumann)
    R_plus = jnp.concatenate([R[1:, :, :], R[-1:, :, :]], axis=0)
    
    # R_minus[i] = R[i-1], with R_minus[0] = R[0] (Neumann)
    R_minus = jnp.concatenate([R[:1, :, :], R[:-1, :, :]], axis=0)
    
    # Apply smoothing operator: R̄ = R + ε(R+ - 2R + R-)
    return R + epsilon * (R_plus - 2.0 * R + R_minus)


def _smooth_pass_j_wake_jax(R, epsilon, n_wake):
    """Apply one smoothing pass in J-direction with wake cut connectivity.
    
    For C-grid topology at j=0:
    - Wake cells (i < n_wake or i >= NI - n_wake) connect across the cut
    - Wake cut connectivity: point (i, 0) connects to point (NI-1-i, 0) in reverse order
    - Airfoil surface cells use Neumann BC
    """
    NI = R.shape[0]
    
    # R_plus[j] = R[j+1], with R_plus[-1] = R[-1] (Neumann at farfield)
    R_plus = jnp.concatenate([R[:, 1:, :], R[:, -1:, :]], axis=1)
    
    # R_minus for j > 0: standard shift
    R_minus_interior = R[:, :-1, :]
    
    # R_minus at j=0: wake cut connectivity
    # For wake cells, R_minus comes from opposite side (reversed i)
    # R_minus(i, 0) = R(NI-1-i, 0) for wake cells
    R_at_j0 = R[:, 0, :]  # Shape: (NI, n_vars)
    R_at_j0_reversed = R_at_j0[::-1, :]  # Reverse i-order
    
    # Create mask for wake cells: i < n_wake OR i >= NI - n_wake
    i_indices = jnp.arange(NI)
    is_wake = (i_indices < n_wake) | (i_indices >= NI - n_wake)
    is_wake = is_wake[:, jnp.newaxis]  # Shape: (NI, 1) for broadcasting
    
    # At j=0: use wake connectivity for wake cells, Neumann for airfoil
    R_minus_j0 = jnp.where(is_wake, R_at_j0_reversed, R_at_j0)
    
    # Concatenate to form full R_minus
    R_minus = jnp.concatenate([R_minus_j0[:, jnp.newaxis, :], R_minus_interior], axis=1)
    
    # Apply smoothing operator
    return R + epsilon * (R_plus - 2.0 * R + R_minus)


def _smooth_2_passes_wake_jax(R, epsilon, n_wake):
    """2-pass ADI smoothing with wake cut connectivity."""
    R = _smooth_pass_i_jax(R, epsilon)
    R = _smooth_pass_j_wake_jax(R, epsilon, n_wake)
    R = _smooth_pass_i_jax(R, epsilon)
    R = _smooth_pass_j_wake_jax(R, epsilon, n_wake)
    return R


def smooth_explicit_jax(R, epsilon: float = 0.2, n_wake: int = 0, skip_nuhat: bool = True):
    """
    Apply 2-pass explicit residual smoothing (JAX).
    
    Parameters
    ----------
    R : jnp.ndarray
        Residual array (NI, NJ, n_vars).
    epsilon : float
        Smoothing coefficient (typically 0.1-0.3).
    n_wake : int
        Number of wake points on each side of the C-grid.
        Wake cut connectivity is used at j=0 for wake cells.
    skip_nuhat : bool
        If True, do NOT smooth the nuHat variable (index 3).
        This prevents smoothing from corrupting the SA turbulence residual
        near walls where sharp gradients exist.
        
    Returns
    -------
    R_smooth : jnp.ndarray
        Smoothed residual (NI, NJ, n_vars).
    """
    if epsilon <= 0.0:
        return R
    
    # Save nuHat residual if we need to skip it
    # Smoothing can corrupt the SA residual near walls by mixing with
    # neighboring cells that have very different physics
    if skip_nuhat and R.shape[-1] > 3:
        R_nuhat_original = R[:, :, 3]
    
    # Apply 2-pass smoothing with wake connectivity
    R_smooth = _smooth_2_passes_wake_jax(R, epsilon, n_wake)
    
    # Restore unsmoothed nuHat residual
    if skip_nuhat and R.shape[-1] > 3:
        R_smooth = R_smooth.at[:, :, 3].set(R_nuhat_original)
    
    return R_smooth


# =============================================================================
# Dispatch Interface
# =============================================================================

def apply_explicit_smoothing(R, epsilon: float = 0.2, n_wake: int = 0, skip_nuhat: bool = True):
    """
    Apply explicit residual smoothing.
    
    Parameters
    ----------
    R : array
        Residual array (NI, NJ, n_vars). Can be numpy or JAX array.
    epsilon : float
        Smoothing coefficient (typically 0.1-0.3).
    n_wake : int
        Number of wake points on each side of the C-grid.
    skip_nuhat : bool
        If True, do NOT smooth the nuHat variable (index 3).
        
    Returns
    -------
    R_smooth : array
        Smoothed residual.
    """
    # Convert to JAX if numpy
    if isinstance(R, np.ndarray):
        R = jnp.asarray(R)
        result = smooth_explicit_jax(R, epsilon, n_wake, skip_nuhat)
        return np.asarray(result)
    else:
        return smooth_explicit_jax(R, epsilon, n_wake, skip_nuhat)
