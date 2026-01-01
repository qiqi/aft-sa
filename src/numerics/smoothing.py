"""
Implicit Residual Smoothing (IRS) for 2D Structured Grid Solvers.

This module implements IRS to stabilize explicit time-stepping schemes
and allow higher CFL numbers. IRS filters out high-frequency errors
by solving a tridiagonal system that effectively widens the stencil.

The smoothing equation at each point:
    -ε * u_{i-1} + (1 + 2ε) * u_i - ε * u_{i+1} = R_i

where:
    - ε (epsilon) is the smoothing coefficient (typically 0.5 to 2.0)
    - R_i is the original residual
    - u_i is the smoothed residual

Reference:
    Jameson, A., Schmidt, W., & Turkel, E. (1981). Numerical solution of
    the Euler equations by finite volume methods using Runge-Kutta time
    stepping schemes. AIAA paper 81-1259.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _tdma_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, 
                d: np.ndarray, x: np.ndarray) -> None:
    """
    Solve a tridiagonal system Ax = d using the Thomas Algorithm (TDMA).
    
    The system is:
        a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] = d[i]
    
    With boundary conditions:
        a[0] = 0 (no left neighbor for first point)
        c[n-1] = 0 (no right neighbor for last point)
    
    Parameters
    ----------
    a : ndarray, shape (n,)
        Sub-diagonal coefficients (a[0] is ignored).
    b : ndarray, shape (n,)
        Diagonal coefficients.
    c : ndarray, shape (n,)
        Super-diagonal coefficients (c[n-1] is ignored).
    d : ndarray, shape (n,)
        Right-hand side vector.
    x : ndarray, shape (n,)
        Solution vector (modified in-place).
        
    Notes
    -----
    This is an in-place algorithm. The input arrays a, b, c, d may be
    modified during the solve. The solution is stored in x.
    
    For IRS, the system is:
        -ε * u_{i-1} + (1 + 2ε) * u_i - ε * u_{i+1} = R_i
    So: a[i] = -ε, b[i] = 1 + 2ε, c[i] = -ε
    """
    n = len(d)
    
    if n == 0:
        return
    
    if n == 1:
        x[0] = d[0] / b[0]
        return
    
    # Forward elimination
    # Modify coefficients (create copies to avoid modifying input)
    c_star = np.empty(n)
    d_star = np.empty(n)
    
    c_star[0] = c[0] / b[0]
    d_star[0] = d[0] / b[0]
    
    for i in range(1, n):
        denom = b[i] - a[i] * c_star[i-1]
        if abs(denom) < 1e-30:
            denom = 1e-30  # Prevent division by zero
        c_star[i] = c[i] / denom
        d_star[i] = (d[i] - a[i] * d_star[i-1]) / denom
    
    # Back substitution
    x[n-1] = d_star[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i+1]


@njit(cache=True)
def _smooth_line_i(residual: np.ndarray, epsilon: float, 
                   j: int, var: int, work: np.ndarray) -> None:
    """
    Apply IRS along a single I-line for one variable.
    
    Parameters
    ----------
    residual : ndarray, shape (NI, NJ, 4)
        Residual array to smooth (modified in-place).
    epsilon : float
        Smoothing coefficient.
    j : int
        J-index of the line.
    var : int
        Variable index (0-3).
    work : ndarray, shape (NI, 5)
        Work array for TDMA [a, b, c, d, x].
    """
    NI = residual.shape[0]
    
    # Set up tridiagonal system coefficients
    # -ε * u_{i-1} + (1 + 2ε) * u_i - ε * u_{i+1} = R_i
    diag = 1.0 + 2.0 * epsilon
    off_diag = -epsilon
    
    for i in range(NI):
        work[i, 0] = off_diag  # a (sub-diagonal)
        work[i, 1] = diag      # b (diagonal)
        work[i, 2] = off_diag  # c (super-diagonal)
        work[i, 3] = residual[i, j, var]  # d (RHS)
    
    # Boundary conditions: Dirichlet-like (no smoothing across boundary)
    # First point: a[0] = 0 (no left neighbor)
    work[0, 0] = 0.0
    # Last point: c[n-1] = 0 (no right neighbor)
    work[NI-1, 2] = 0.0
    
    # Solve the tridiagonal system
    _tdma_solve(work[:, 0], work[:, 1], work[:, 2], 
                work[:, 3], work[:, 4])
    
    # Copy solution back to residual
    for i in range(NI):
        residual[i, j, var] = work[i, 4]


@njit(cache=True)
def _smooth_line_j(residual: np.ndarray, epsilon: float,
                   i: int, var: int, work: np.ndarray) -> None:
    """
    Apply IRS along a single J-line for one variable.
    
    Parameters
    ----------
    residual : ndarray, shape (NI, NJ, 4)
        Residual array to smooth (modified in-place).
    epsilon : float
        Smoothing coefficient.
    i : int
        I-index of the line.
    var : int
        Variable index (0-3).
    work : ndarray, shape (NJ, 5)
        Work array for TDMA [a, b, c, d, x].
    """
    NJ = residual.shape[1]
    
    # Set up tridiagonal system coefficients
    diag = 1.0 + 2.0 * epsilon
    off_diag = -epsilon
    
    for j in range(NJ):
        work[j, 0] = off_diag  # a (sub-diagonal)
        work[j, 1] = diag      # b (diagonal)
        work[j, 2] = off_diag  # c (super-diagonal)
        work[j, 3] = residual[i, j, var]  # d (RHS)
    
    # Boundary conditions: Dirichlet-like
    work[0, 0] = 0.0
    work[NJ-1, 2] = 0.0
    
    # Solve the tridiagonal system
    _tdma_solve(work[:, 0], work[:, 1], work[:, 2],
                work[:, 3], work[:, 4])
    
    # Copy solution back to residual
    for j in range(NJ):
        residual[i, j, var] = work[j, 4]


@njit(cache=True)
def apply_residual_smoothing(residual: np.ndarray, epsilon: float = 0.5) -> None:
    """
    Apply Implicit Residual Smoothing (IRS) to the residual array.
    
    This is an ADI-style (Alternating Direction Implicit) approach:
    1. First sweep along I-direction (for each J-line)
    2. Then sweep along J-direction (for each I-line)
    
    The smoothing equation is:
        -ε * u_{i-1} + (1 + 2ε) * u_i - ε * u_{i+1} = R_i
    
    Parameters
    ----------
    residual : ndarray, shape (NI, NJ, 4)
        Residual array to smooth. Modified in-place.
        Components: [p, u, v, nu_t] residuals.
    epsilon : float, optional
        Smoothing coefficient. Default is 0.5.
        - epsilon = 0: No smoothing
        - epsilon = 0.5: Moderate smoothing (allows CFL ~2x)
        - epsilon = 1.0: Strong smoothing (allows CFL ~3x)
        - epsilon = 2.0: Very strong smoothing
        
    Notes
    -----
    IRS extends the effective stencil of the scheme, filtering out
    high-frequency errors that cause odd-even decoupling. This allows
    the explicit scheme to remain stable at higher CFL numbers.
    
    For a C-grid around an airfoil:
    - I-direction wraps around the airfoil (with wake cut)
    - J-direction goes from wall to farfield
    
    The boundaries are treated as Dirichlet (residual = 0) which is
    appropriate for smoothing purposes.
    """
    if epsilon <= 0.0:
        return  # No smoothing
    
    NI = residual.shape[0]
    NJ = residual.shape[1]
    n_vars = residual.shape[2]
    
    # Allocate work arrays for TDMA
    work_i = np.zeros((NI, 5))  # [a, b, c, d, x]
    work_j = np.zeros((NJ, 5))
    
    # Step 1: I-sweep (for each J-line and each variable)
    for j in range(NJ):
        for var in range(n_vars):
            _smooth_line_i(residual, epsilon, j, var, work_i)
    
    # Step 2: J-sweep (for each I-line and each variable)
    for i in range(NI):
        for var in range(n_vars):
            _smooth_line_j(residual, epsilon, i, var, work_j)



