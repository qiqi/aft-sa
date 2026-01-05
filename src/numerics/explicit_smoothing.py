"""
Multi-Stage Explicit Residual Smoothing for 2D structured grids.

Smoothing operator: R̄_i = R_i + ε(R_{i+1} - 2R_i + R_{i-1})
Applied N passes in ADI fashion (I-direction then J-direction).

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

@jax.jit
def _smooth_pass_i_jax(R, epsilon):
    """Apply one smoothing pass in I-direction (Neumann BC)."""
    # R has shape (NI, NJ, n_vars)
    
    # Get shifted arrays using slicing (more efficient than roll for boundaries)
    # R_plus[i] = R[i+1], with R_plus[-1] = R[-1] (Neumann)
    R_plus = jnp.concatenate([R[1:, :, :], R[-1:, :, :]], axis=0)
    
    # R_minus[i] = R[i-1], with R_minus[0] = R[0] (Neumann)
    R_minus = jnp.concatenate([R[:1, :, :], R[:-1, :, :]], axis=0)
    
    # Apply smoothing operator: R̄ = R + ε(R+ - 2R + R-)
    R_smooth = R + epsilon * (R_plus - 2.0 * R + R_minus)
    
    return R_smooth


@jax.jit
def _smooth_pass_j_jax(R, epsilon):
    """Apply one smoothing pass in J-direction (Neumann BC)."""
    # R has shape (NI, NJ, n_vars)
    
    # R_plus[j] = R[j+1], with R_plus[-1] = R[-1] (Neumann)
    R_plus = jnp.concatenate([R[:, 1:, :], R[:, -1:, :]], axis=1)
    
    # R_minus[j] = R[j-1], with R_minus[0] = R[0] (Neumann)
    R_minus = jnp.concatenate([R[:, :1, :], R[:, :-1, :]], axis=1)
    
    # Apply smoothing operator
    R_smooth = R + epsilon * (R_plus - 2.0 * R + R_minus)
    
    return R_smooth


@jax.jit
def _smooth_1_pass_jax(R, epsilon):
    """Single ADI pass (I then J direction)."""
    R = _smooth_pass_i_jax(R, epsilon)
    R = _smooth_pass_j_jax(R, epsilon)
    return R


@jax.jit
def _smooth_2_passes_jax(R, epsilon):
    """Optimized 2-pass smoothing."""
    R = _smooth_pass_i_jax(R, epsilon)
    R = _smooth_pass_j_jax(R, epsilon)
    R = _smooth_pass_i_jax(R, epsilon)
    R = _smooth_pass_j_jax(R, epsilon)
    return R


@jax.jit
def _smooth_3_passes_jax(R, epsilon):
    """Optimized 3-pass smoothing."""
    R = _smooth_pass_i_jax(R, epsilon)
    R = _smooth_pass_j_jax(R, epsilon)
    R = _smooth_pass_i_jax(R, epsilon)
    R = _smooth_pass_j_jax(R, epsilon)
    R = _smooth_pass_i_jax(R, epsilon)
    R = _smooth_pass_j_jax(R, epsilon)
    return R


@jax.jit
def _smooth_4_passes_jax(R, epsilon):
    """Optimized 4-pass smoothing."""
    R = _smooth_pass_i_jax(R, epsilon)
    R = _smooth_pass_j_jax(R, epsilon)
    R = _smooth_pass_i_jax(R, epsilon)
    R = _smooth_pass_j_jax(R, epsilon)
    R = _smooth_pass_i_jax(R, epsilon)
    R = _smooth_pass_j_jax(R, epsilon)
    R = _smooth_pass_i_jax(R, epsilon)
    R = _smooth_pass_j_jax(R, epsilon)
    return R


def smooth_explicit_jax(R, epsilon: float = 0.2, n_passes: int = 2, skip_nuhat: bool = True):
    """
    Apply multi-pass explicit residual smoothing (JAX).
    
    Parameters
    ----------
    R : jnp.ndarray
        Residual array (NI, NJ, n_vars).
    epsilon : float
        Smoothing coefficient (typically 0.1-0.3).
    n_passes : int
        Number of smoothing passes.
    skip_nuhat : bool
        If True, do NOT smooth the nuHat variable (index 3).
        This prevents smoothing from corrupting the SA turbulence residual
        near walls where sharp gradients exist.
        
    Returns
    -------
    R_smooth : jnp.ndarray
        Smoothed residual (NI, NJ, n_vars).
    """
    if epsilon <= 0.0 or n_passes <= 0:
        return R
    
    # Save nuHat residual if we need to skip it
    # Smoothing can corrupt the SA residual near walls by mixing with
    # neighboring cells that have very different physics
    if skip_nuhat and R.shape[-1] > 3:
        R_nuhat_original = R[:, :, 3]
    
    # Use pre-compiled versions for common cases
    if n_passes == 1:
        R_smooth = _smooth_1_pass_jax(R, epsilon)
    elif n_passes == 2:
        R_smooth = _smooth_2_passes_jax(R, epsilon)
    elif n_passes == 3:
        R_smooth = _smooth_3_passes_jax(R, epsilon)
    elif n_passes == 4:
        R_smooth = _smooth_4_passes_jax(R, epsilon)
    else:
        # Fallback: manually unroll (less efficient for large n_passes)
        R_smooth = R
        for _ in range(n_passes):
            R_smooth = _smooth_1_pass_jax(R_smooth, epsilon)
    
    # Restore unsmoothed nuHat residual
    if skip_nuhat and R.shape[-1] > 3:
        R_smooth = R_smooth.at[:, :, 3].set(R_nuhat_original)
    
    return R_smooth


# =============================================================================
# Dispatch Interface
# =============================================================================

def apply_explicit_smoothing(R, epsilon: float = 0.2, n_passes: int = 2, skip_nuhat: bool = True):
    """
    Apply explicit residual smoothing.
    
    Parameters
    ----------
    R : array
        Residual array (NI, NJ, n_vars). Can be numpy or JAX array.
    epsilon : float
        Smoothing coefficient (typically 0.1-0.3).
    n_passes : int
        Number of smoothing passes.
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
        result = smooth_explicit_jax(R, epsilon, n_passes, skip_nuhat)
        return np.asarray(result)
    else:
        return smooth_explicit_jax(R, epsilon, n_passes, skip_nuhat)
