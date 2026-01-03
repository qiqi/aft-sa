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
from numba import njit, prange

# JAX imports
try:
    from src.physics.jax_config import jax, jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

NDArrayFloat = npt.NDArray[np.floating]


# =============================================================================
# Numba Implementation
# =============================================================================

@njit(cache=True, parallel=True)
def _smooth_pass_i_numba(R: np.ndarray, R_out: np.ndarray, epsilon: float) -> None:
    """Apply one smoothing pass in I-direction (Neumann BC)."""
    NI, NJ, n_vars = R.shape
    
    for j in prange(NJ):
        for k in range(n_vars):
            # Interior points
            for i in range(1, NI - 1):
                R_out[i, j, k] = R[i, j, k] + epsilon * (
                    R[i+1, j, k] - 2.0 * R[i, j, k] + R[i-1, j, k]
                )
            
            # Left boundary (i=0): Neumann, R[-1] = R[0]
            # Term: ε(R[1] - 2R[0] + R[0]) = ε(R[1] - R[0])
            R_out[0, j, k] = R[0, j, k] + epsilon * (R[1, j, k] - R[0, j, k])
            
            # Right boundary (i=NI-1): Neumann, R[NI] = R[NI-1]
            # Term: ε(R[NI-1] - 2R[NI-1] + R[NI-2]) = ε(R[NI-2] - R[NI-1])
            R_out[NI-1, j, k] = R[NI-1, j, k] + epsilon * (
                R[NI-2, j, k] - R[NI-1, j, k]
            )


@njit(cache=True, parallel=True)
def _smooth_pass_j_numba(R: np.ndarray, R_out: np.ndarray, epsilon: float) -> None:
    """Apply one smoothing pass in J-direction (Neumann BC)."""
    NI, NJ, n_vars = R.shape
    
    for i in prange(NI):
        for k in range(n_vars):
            # Interior points
            for j in range(1, NJ - 1):
                R_out[i, j, k] = R[i, j, k] + epsilon * (
                    R[i, j+1, k] - 2.0 * R[i, j, k] + R[i, j-1, k]
                )
            
            # Bottom boundary (j=0): Neumann
            R_out[i, 0, k] = R[i, 0, k] + epsilon * (R[i, 1, k] - R[i, 0, k])
            
            # Top boundary (j=NJ-1): Neumann
            R_out[i, NJ-1, k] = R[i, NJ-1, k] + epsilon * (
                R[i, NJ-2, k] - R[i, NJ-1, k]
            )


def smooth_explicit_numba(R: np.ndarray, epsilon: float = 0.2, 
                          n_passes: int = 2) -> np.ndarray:
    """
    Apply multi-pass explicit residual smoothing (Numba).
    
    Parameters
    ----------
    R : np.ndarray
        Residual array (NI, NJ, n_vars).
    epsilon : float
        Smoothing coefficient (typically 0.1-0.3).
    n_passes : int
        Number of smoothing passes.
        
    Returns
    -------
    R_smooth : np.ndarray
        Smoothed residual (NI, NJ, n_vars).
    """
    if epsilon <= 0.0 or n_passes <= 0:
        return R.copy()
    
    R_in = R.copy()
    R_out = np.empty_like(R)
    
    for _ in range(n_passes):
        # I-direction pass
        _smooth_pass_i_numba(R_in, R_out, epsilon)
        R_in, R_out = R_out, R_in  # Swap buffers
        
        # J-direction pass
        _smooth_pass_j_numba(R_in, R_out, epsilon)
        R_in, R_out = R_out, R_in  # Swap buffers
    
    return R_in


# =============================================================================
# JAX Implementation
# =============================================================================

if JAX_AVAILABLE:
    
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
    
    def smooth_explicit_jax(R, epsilon: float = 0.2, 
                            n_passes: int = 2):
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
            
        Returns
        -------
        R_smooth : jnp.ndarray
            Smoothed residual (NI, NJ, n_vars).
        """
        if epsilon <= 0.0 or n_passes <= 0:
            return R
        
        # Use pre-compiled versions for common cases
        if n_passes == 1:
            return _smooth_1_pass_jax(R, epsilon)
        elif n_passes == 2:
            return _smooth_2_passes_jax(R, epsilon)
        elif n_passes == 3:
            return _smooth_3_passes_jax(R, epsilon)
        elif n_passes == 4:
            return _smooth_4_passes_jax(R, epsilon)
        else:
            # Fallback: manually unroll (less efficient for large n_passes)
            for _ in range(n_passes):
                R = _smooth_1_pass_jax(R, epsilon)
            return R


# =============================================================================
# Dispatch Interface
# =============================================================================

def apply_explicit_smoothing(R, epsilon: float = 0.2, n_passes: int = 2):
    """
    Apply explicit residual smoothing with automatic backend dispatch.
    
    Chooses JAX or Numba implementation based on input array type.
    
    Parameters
    ----------
    R : array
        Residual array (NI, NJ, n_vars). Can be numpy or JAX array.
    epsilon : float
        Smoothing coefficient (typically 0.1-0.3).
    n_passes : int
        Number of smoothing passes.
        
    Returns
    -------
    R_smooth : array
        Smoothed residual (same type as input).
    """
    # Check if R is a JAX array
    if JAX_AVAILABLE:
        try:
            if hasattr(R, 'devices') or str(type(R).__module__).startswith('jax'):
                # Use optimized versions for common cases
                if n_passes == 2:
                    return _smooth_2_passes_jax(R, epsilon)
                elif n_passes == 3:
                    return _smooth_3_passes_jax(R, epsilon)
                else:
                    return smooth_explicit_jax(R, epsilon, n_passes)
        except:
            pass
    
    # Default to Numba
    return smooth_explicit_numba(np.asarray(R), epsilon, n_passes)

