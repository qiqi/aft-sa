"""
Implicit Residual Smoothing (IRS) for 2D structured grids.

Smoothing equation: -ε·u_{i-1} + (1+2ε)·u_i - ε·u_{i+1} = R_i

Reference: Jameson, Schmidt, Turkel (1981). AIAA paper 81-1259.
"""

import numpy as np
import numpy.typing as npt
from numba import njit

# JAX imports
try:
    from src.physics.jax_config import jax, jnp
    from jax.lax.linalg import tridiagonal_solve
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

NDArrayFloat = npt.NDArray[np.floating]


@njit(cache=True)
def _tdma_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, 
                d: np.ndarray, x: np.ndarray) -> None:
    """Tridiagonal solver (Thomas algorithm): a[i]·x[i-1] + b[i]·x[i] + c[i]·x[i+1] = d[i]"""
    n = len(d)
    
    if n == 0:
        return
    if n == 1:
        x[0] = d[0] / b[0]
        return
    
    c_star = np.empty(n)
    d_star = np.empty(n)
    
    c_star[0] = c[0] / b[0]
    d_star[0] = d[0] / b[0]
    
    for i in range(1, n):
        denom = b[i] - a[i] * c_star[i-1]
        if abs(denom) < 1e-30:
            denom = 1e-30
        c_star[i] = c[i] / denom
        d_star[i] = (d[i] - a[i] * d_star[i-1]) / denom
    
    x[n-1] = d_star[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i+1]


@njit(cache=True)
def _smooth_line_i(residual: np.ndarray, epsilon: float, 
                   j: int, var: int, work: np.ndarray) -> None:
    """Apply IRS along single I-line for one variable."""
    NI = residual.shape[0]
    diag = 1.0 + 2.0 * epsilon
    off_diag = -epsilon
    
    for i in range(NI):
        work[i, 0] = off_diag
        work[i, 1] = diag
        work[i, 2] = off_diag
        work[i, 3] = residual[i, j, var]
    
    work[0, 0] = 0.0
    work[NI-1, 2] = 0.0
    
    _tdma_solve(work[:, 0], work[:, 1], work[:, 2], work[:, 3], work[:, 4])
    
    for i in range(NI):
        residual[i, j, var] = work[i, 4]


@njit(cache=True)
def _smooth_line_j(residual: np.ndarray, epsilon: float,
                   i: int, var: int, work: np.ndarray) -> None:
    """Apply IRS along single J-line for one variable."""
    NJ = residual.shape[1]
    diag = 1.0 + 2.0 * epsilon
    off_diag = -epsilon
    
    for j in range(NJ):
        work[j, 0] = off_diag
        work[j, 1] = diag
        work[j, 2] = off_diag
        work[j, 3] = residual[i, j, var]
    
    work[0, 0] = 0.0
    work[NJ-1, 2] = 0.0
    
    _tdma_solve(work[:, 0], work[:, 1], work[:, 2], work[:, 3], work[:, 4])
    
    for j in range(NJ):
        residual[i, j, var] = work[j, 4]


@njit(cache=True)
def apply_residual_smoothing(residual: np.ndarray, epsilon: float = 0.5) -> None:
    """
    Apply ADI-style Implicit Residual Smoothing in-place.
    
    epsilon = 0: no smoothing
    epsilon = 0.5: moderate (allows CFL ~2x)
    epsilon = 1.0: strong (allows CFL ~3x)
    """
    if epsilon <= 0.0:
        return
    
    NI = residual.shape[0]
    NJ = residual.shape[1]
    n_vars = residual.shape[2]
    
    work_i = np.zeros((NI, 5))
    work_j = np.zeros((NJ, 5))
    
    for j in range(NJ):
        for var in range(n_vars):
            _smooth_line_i(residual, epsilon, j, var, work_i)
    
    for i in range(NI):
        for var in range(n_vars):
            _smooth_line_j(residual, epsilon, i, var, work_j)


# =============================================================================
# JAX Implementations
# =============================================================================

if JAX_AVAILABLE:
    
    def _build_tridiag_diagonals(n, epsilon):
        """
        Build diagonal arrays for tridiagonal_solve.
        
        The tridiagonal system is: -ε·x_{i-1} + (1+2ε)·x_i - ε·x_{i+1} = r_i
        
        JAX's tridiagonal_solve expects all diagonals to have shape (n,):
        - dl[i] := A[i, i-1], with dl[0] = 0 (ignored)
        - d[i]  := A[i, i]
        - du[i] := A[i, i+1], with du[n-1] = 0 (ignored)
        
        Returns:
            dl, d, du: all of shape (n,)
        """
        diag = 1.0 + 2.0 * epsilon
        off_diag = -epsilon
        
        dl = np.full(n, off_diag)   # Lower diagonal
        dl[0] = 0.0                  # dl[0] is ignored (no A[0, -1])
        
        d = np.full(n, diag)        # Main diagonal
        
        du = np.full(n, off_diag)   # Upper diagonal
        du[n-1] = 0.0                # du[n-1] is ignored (no A[n-1, n])
        
        return jnp.array(dl), jnp.array(d), jnp.array(du)
    
    def apply_residual_smoothing_jax(residual, epsilon):
        """
        JAX: Apply ADI-style Implicit Residual Smoothing.
        
        Uses jax.lax.linalg.tridiagonal_solve for efficient tridiagonal solves.
        The same matrix is reused for all lines and variables.
        
        Parameters
        ----------
        residual : jnp.ndarray
            Residual array (NI, NJ, n_vars).
        epsilon : float
            Smoothing parameter (0=none, 0.5=moderate, 1.0=strong).
            
        Returns
        -------
        smoothed : jnp.ndarray
            Smoothed residual (NI, NJ, n_vars).
        """
        NI, NJ, n_vars = residual.shape
        
        # Build diagonal arrays outside of JIT (uses concrete sizes)
        dl_i, d_i, du_i = _build_tridiag_diagonals(NI, epsilon)
        dl_j, d_j, du_j = _build_tridiag_diagonals(NJ, epsilon)
        
        return _apply_smoothing_impl(residual, dl_i, d_i, du_i, dl_j, d_j, du_j)
    
    @jax.jit
    def _apply_smoothing_impl(residual, dl_i, d_i, du_i, dl_j, d_j, du_j):
        """JIT-compiled smoothing implementation."""
        # residual: (NI, NJ, n_vars)
        
        # I-direction smoothing: solve along i for each (j, var)
        # Reshape to (NJ, n_vars, NI) for batch solving, then back
        # tridiagonal_solve expects: dl (n-1,), d (n,), du (n-1,), b (n,) or (n, k)
        
        # For I-smoothing: solve systems of size NI
        # residual has shape (NI, NJ, n_vars)
        # We need to solve for each column (j, var)
        # Reshape to (NI, NJ*n_vars)
        NI, NJ, n_vars = residual.shape
        rhs_i = residual.reshape(NI, NJ * n_vars)  # (NI, NJ*n_vars)
        
        # tridiagonal_solve(dl, d, du, b) where b can be (n, k)
        smoothed_i = tridiagonal_solve(dl_i, d_i, du_i, rhs_i)  # (NI, NJ*n_vars)
        smoothed_i = smoothed_i.reshape(NI, NJ, n_vars)
        
        # J-direction smoothing: solve along j for each (i, var)
        # Transpose to (NJ, NI, n_vars), reshape to (NJ, NI*n_vars)
        rhs_j = smoothed_i.transpose(1, 0, 2).reshape(NJ, NI * n_vars)  # (NJ, NI*n_vars)
        
        smoothed_j = tridiagonal_solve(dl_j, d_j, du_j, rhs_j)  # (NJ, NI*n_vars)
        smoothed_j = smoothed_j.reshape(NJ, NI, n_vars).transpose(1, 0, 2)
        
        return smoothed_j
