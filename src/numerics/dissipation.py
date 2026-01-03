"""
Sponge layer dissipation for farfield boundary stabilization.

The sponge layer introduces localized 2nd-order dissipation (epsilon_2) near
the boundaries that absorbs outgoing waves, while maintaining pure 4th-order
dissipation in the interior.

References:
    Israeli & Orszag (1981). JCP 41.
    Colonius (2004). AIAA J.
"""

import numpy as np
import numpy.typing as npt

from src.physics.jax_config import jax, jnp

NDArrayFloat = npt.NDArray[np.floating]


def compute_sponge_sigma(NI: int, NJ: int, n_sponge: int = 15) -> NDArrayFloat:
    """
    Compute sponge layer coefficient field sigma.
    
    Creates a field that is:
    - 0.0 in the interior (no 2nd-order dissipation)
    - Ramps linearly from 0.0 to 1.0 at boundaries (max 2nd-order dissipation)
    
    The sponge is applied at:
    - J-direction: Last n_sponge cells (outer farfield arc)
    - I-direction: First and last n_sponge cells (downstream outlets)
    
    Parameters
    ----------
    NI : int
        Number of interior cells in I-direction.
    NJ : int
        Number of interior cells in J-direction.
    n_sponge : int
        Thickness of sponge layer in cells (default 15).
        
    Returns
    -------
    sigma : ndarray
        Sponge coefficient field (NI, NJ), values in [0, 1].
        sigma=0 in interior, sigma=1 at boundary.
    """
    sigma = np.zeros((NI, NJ), dtype=np.float64)
    
    # J-direction ramp (outer farfield arc)
    # For the last n_sponge cells in J, ramp from 0 to 1
    j_start = NJ - n_sponge
    for j in range(max(0, j_start), NJ):
        # Linear ramp: 0 at j_start, 1 at NJ-1
        ramp_val = (j - j_start + 1) / n_sponge
        sigma[:, j] = np.maximum(sigma[:, j], ramp_val)
    
    # I-direction ramp (downstream outlets)
    # Lower I boundary: first n_sponge cells
    for i in range(min(n_sponge, NI)):
        # Linear ramp: 1 at i=0, 0 at i=n_sponge
        ramp_val = (n_sponge - i) / n_sponge
        sigma[i, :] = np.maximum(sigma[i, :], ramp_val)
    
    # Upper I boundary: last n_sponge cells
    i_start = NI - n_sponge
    for i in range(max(0, i_start), NI):
        # Linear ramp: 0 at i_start, 1 at NI-1
        ramp_val = (i - i_start + 1) / n_sponge
        sigma[i, :] = np.maximum(sigma[i, :], ramp_val)
    
    # Clip to [0, 1]
    sigma = np.clip(sigma, 0.0, 1.0)
    
    return sigma


def compute_sponge_sigma_jax(NI: int, NJ: int, n_sponge: int = 15) -> jnp.ndarray:
    """
    JAX version of compute_sponge_sigma.
    
    Creates a sponge coefficient field for use in JAX flux computations.
    
    Parameters
    ----------
    NI : int
        Number of interior cells in I-direction.
    NJ : int
        Number of interior cells in J-direction.
    n_sponge : int
        Thickness of sponge layer in cells (default 15).
        
    Returns
    -------
    sigma : jnp.ndarray
        Sponge coefficient field (NI, NJ), values in [0, 1].
    """
    # Create coordinate arrays
    i_coords = jnp.arange(NI)
    j_coords = jnp.arange(NJ)
    I, J = jnp.meshgrid(i_coords, j_coords, indexing='ij')
    
    # J-direction ramp (outer farfield arc)
    # For j >= NJ - n_sponge: ramp from 0 to 1
    j_start = NJ - n_sponge
    sigma_j = jnp.where(
        J >= j_start,
        (J - j_start + 1) / n_sponge,
        0.0
    )
    
    # I-direction ramp (lower boundary)
    # For i < n_sponge: ramp from 1 to 0
    sigma_i_low = jnp.where(
        I < n_sponge,
        (n_sponge - I) / n_sponge,
        0.0
    )
    
    # I-direction ramp (upper boundary)
    # For i >= NI - n_sponge: ramp from 0 to 1
    i_start = NI - n_sponge
    sigma_i_high = jnp.where(
        I >= i_start,
        (I - i_start + 1) / n_sponge,
        0.0
    )
    
    # Combine: take maximum of all ramps
    sigma = jnp.maximum(sigma_j, jnp.maximum(sigma_i_low, sigma_i_high))
    
    # Clip to [0, 1]
    sigma = jnp.clip(sigma, 0.0, 1.0)
    
    return sigma


def compute_jst_blending_coefficients(sigma: NDArrayFloat, k4: float) -> tuple:
    """
    Compute JST blending coefficients epsilon_2 and epsilon_4.
    
    Standard JST blending: when 2nd-order dissipation is high (in sponge),
    reduce 4th-order dissipation to avoid over-dissipation.
    
    epsilon_2 = sigma (sponge sensor = 2nd order coefficient)
    epsilon_4 = max(0, k_4 - epsilon_2)
    
    In interior (sigma=0): pure 4th-order (eps2=0, eps4=k4)
    In sponge (sigma=1): pure 2nd-order (eps2=1, eps4=0)
    
    Parameters
    ----------
    sigma : ndarray
        Sponge coefficient field (NI, NJ), values in [0, 1].
    k4 : float
        Base 4th-order dissipation coefficient.
        
    Returns
    -------
    eps2 : ndarray
        2nd-order dissipation coefficient field (NI, NJ).
    eps4 : ndarray
        4th-order dissipation coefficient field (NI, NJ).
    """
    eps2 = sigma
    eps4 = np.maximum(0.0, k4 - eps2)
    return eps2, eps4


@jax.jit
def compute_jst_blending_coefficients_jax(sigma: jnp.ndarray, k4: float) -> tuple:
    """
    JAX version of JST blending coefficient computation.
    
    Parameters
    ----------
    sigma : jnp.ndarray
        Sponge coefficient field (NI, NJ), values in [0, 1].
    k4 : float
        Base 4th-order dissipation coefficient.
        
    Returns
    -------
    eps2 : jnp.ndarray
        2nd-order dissipation coefficient field (NI, NJ).
    eps4 : jnp.ndarray
        4th-order dissipation coefficient field (NI, NJ).
    """
    eps2 = sigma
    eps4 = jnp.maximum(0.0, k4 - eps2)
    return eps2, eps4
