"""
Implicit Residual Smoothing (IRS) for 2D structured grids.

Smoothing equation: -ε·u_{i-1} + (1+2ε)·u_i - ε·u_{i+1} = R_i

Reference: Jameson, Schmidt, Turkel (1981). AIAA paper 81-1259.
"""

import numpy as np
from numba import njit


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
