"""
Multigrid Transfer Operators for FAS Scheme.

This module provides Numba-optimized kernels for transferring data between
multigrid levels using Full Approximation Storage (FAS).

Transfer Operators:
- restrict_state: Volume-weighted average Q_c = (sum Q_f * V_f) / V_c
- restrict_residual: Simple summation R_c = sum R_f  
- prolongate_correction: Bilinear interpolation of correction dQ

Design: GPU-ready with flat arrays and @njit kernels.
"""

import numpy as np
from numba import njit, prange
from typing import Tuple


@njit(cache=True, parallel=True)
def restrict_state(Q_f: np.ndarray, vol_f: np.ndarray,
                   Q_c: np.ndarray, vol_c: np.ndarray) -> None:
    """
    Restrict state variables from fine to coarse grid using volume-weighted average.
    
    For FAS, the coarse grid state is the volume-weighted average of fine states:
        Q_c = (sum Q_f * V_f) / V_c
    
    This preserves conservation: integral of Q over domain is same.
    
    Parameters
    ----------
    Q_f : ndarray, shape (NI_f, NJ_f, nvar)
        Fine grid state variables.
    vol_f : ndarray, shape (NI_f, NJ_f)
        Fine grid cell volumes.
    Q_c : ndarray, shape (NI_c, NJ_c, nvar) [output]
        Coarse grid state variables (modified in-place).
    vol_c : ndarray, shape (NI_c, NJ_c)
        Coarse grid cell volumes.
    """
    NI_c, NJ_c, nvar = Q_c.shape
    
    for i_c in prange(NI_c):
        for j_c in range(NJ_c):
            i_f = 2 * i_c
            j_f = 2 * j_c
            
            v_total = vol_c[i_c, j_c]
            
            for k in range(nvar):
                # Volume-weighted sum of 4 fine cells
                Q_sum = (Q_f[i_f, j_f, k] * vol_f[i_f, j_f] +
                         Q_f[i_f+1, j_f, k] * vol_f[i_f+1, j_f] +
                         Q_f[i_f, j_f+1, k] * vol_f[i_f, j_f+1] +
                         Q_f[i_f+1, j_f+1, k] * vol_f[i_f+1, j_f+1])
                
                Q_c[i_c, j_c, k] = Q_sum / v_total


@njit(cache=True, parallel=True)
def restrict_residual(R_f: np.ndarray, R_c: np.ndarray) -> None:
    """
    Restrict residuals from fine to coarse grid using simple summation.
    
    For FAS, the coarse grid residual is the sum of fine residuals:
        R_c = sum R_f
    
    This is because residuals are already volume-integrated quantities.
    
    Parameters
    ----------
    R_f : ndarray, shape (NI_f, NJ_f, nvar)
        Fine grid residuals.
    R_c : ndarray, shape (NI_c, NJ_c, nvar) [output]
        Coarse grid residuals (modified in-place).
    """
    NI_c, NJ_c, nvar = R_c.shape
    
    for i_c in prange(NI_c):
        for j_c in range(NJ_c):
            i_f = 2 * i_c
            j_f = 2 * j_c
            
            for k in range(nvar):
                # Sum of 4 fine residuals
                R_c[i_c, j_c, k] = (R_f[i_f, j_f, k] +
                                    R_f[i_f+1, j_f, k] +
                                    R_f[i_f, j_f+1, k] +
                                    R_f[i_f+1, j_f+1, k])


@njit(cache=True, parallel=True)
def prolongate_correction(Q_f: np.ndarray, 
                           Q_c_new: np.ndarray, 
                           Q_c_old: np.ndarray) -> None:
    """
    Prolongate coarse grid correction to fine grid using bilinear interpolation.
    
    For FAS, the correction is:
        dQ_c = Q_c_new - Q_c_old
    
    This correction is bilinearly interpolated to the fine grid and added:
        Q_f = Q_f + interpolate(dQ_c)
    
    The interpolation uses the 4 neighboring coarse cells with proper weights.
    
    Parameters
    ----------
    Q_f : ndarray, shape (NI_f, NJ_f, nvar) [in/out]
        Fine grid state (modified in-place by adding interpolated correction).
    Q_c_new : ndarray, shape (NI_c, NJ_c, nvar)
        New coarse grid state after smoothing.
    Q_c_old : ndarray, shape (NI_c, NJ_c, nvar)
        Old coarse grid state before smoothing (restricted from fine).
    """
    NI_f, NJ_f, nvar = Q_f.shape
    NI_c, NJ_c = Q_c_new.shape[0], Q_c_new.shape[1]
    
    for i_f in prange(NI_f):
        for j_f in range(NJ_f):
            # Find the coarse cell that contains this fine cell
            i_c = i_f // 2
            j_c = j_f // 2
            
            # Local position within the 2x2 block (0 or 1)
            di = i_f % 2  # 0 = left half, 1 = right half
            dj = j_f % 2  # 0 = bottom half, 1 = top half
            
            # Bilinear interpolation weights
            # Fine cell center is at (i_c + 0.25 + 0.5*di, j_c + 0.25 + 0.5*dj) in coarse coords
            # We interpolate from coarse cell centers at (i_c + 0.5, j_c + 0.5)
            
            # For simplicity, use injection for the 4 fine cells in each coarse cell
            # This is equivalent to piecewise constant prolongation
            # For bilinear, we need to consider neighboring coarse cells
            
            # Bilinear interpolation from 4 nearest coarse cell centers
            # Position of fine cell center relative to coarse cell center at (i_c, j_c)
            xi = 0.25 + 0.5 * di  # 0.25 or 0.75
            eta = 0.25 + 0.5 * dj  # 0.25 or 0.75
            
            # Determine the 4 coarse cells for interpolation
            # If xi < 0.5, use i_c-1 and i_c; else use i_c and i_c+1
            if xi < 0.5:
                i0, i1 = max(0, i_c - 1), i_c
                alpha = xi + 0.5  # weight for i1
            else:
                i0, i1 = i_c, min(NI_c - 1, i_c + 1)
                alpha = xi - 0.5  # weight for i1
            
            if eta < 0.5:
                j0, j1 = max(0, j_c - 1), j_c
                beta = eta + 0.5  # weight for j1
            else:
                j0, j1 = j_c, min(NJ_c - 1, j_c + 1)
                beta = eta - 0.5  # weight for j1
            
            for k in range(nvar):
                # Coarse corrections at 4 interpolation points
                dQ_00 = Q_c_new[i0, j0, k] - Q_c_old[i0, j0, k]
                dQ_10 = Q_c_new[i1, j0, k] - Q_c_old[i1, j0, k]
                dQ_01 = Q_c_new[i0, j1, k] - Q_c_old[i0, j1, k]
                dQ_11 = Q_c_new[i1, j1, k] - Q_c_old[i1, j1, k]
                
                # Bilinear interpolation
                dQ_interp = ((1 - alpha) * (1 - beta) * dQ_00 +
                             alpha * (1 - beta) * dQ_10 +
                             (1 - alpha) * beta * dQ_01 +
                             alpha * beta * dQ_11)
                
                Q_f[i_f, j_f, k] += dQ_interp


@njit(cache=True, parallel=True)
def prolongate_injection(Q_f: np.ndarray,
                          Q_c_new: np.ndarray,
                          Q_c_old: np.ndarray) -> None:
    """
    Prolongate coarse grid correction to fine grid using injection (piecewise constant).
    
    Simpler than bilinear, just copies the coarse correction to all 4 fine children.
    
    Parameters
    ----------
    Q_f : ndarray, shape (NI_f, NJ_f, nvar) [in/out]
        Fine grid state (modified in-place by adding correction).
    Q_c_new : ndarray, shape (NI_c, NJ_c, nvar)
        New coarse grid state after smoothing.
    Q_c_old : ndarray, shape (NI_c, NJ_c, nvar)
        Old coarse grid state before smoothing.
    """
    NI_c, NJ_c, nvar = Q_c_new.shape
    
    for i_c in prange(NI_c):
        for j_c in range(NJ_c):
            i_f = 2 * i_c
            j_f = 2 * j_c
            
            for k in range(nvar):
                dQ = Q_c_new[i_c, j_c, k] - Q_c_old[i_c, j_c, k]
                
                # Apply same correction to all 4 fine children
                Q_f[i_f, j_f, k] += dQ
                Q_f[i_f+1, j_f, k] += dQ
                Q_f[i_f, j_f+1, k] += dQ
                Q_f[i_f+1, j_f+1, k] += dQ


@njit(cache=True)
def compute_integral(Q: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """
    Compute volume integral of state variables.
    
    Parameters
    ----------
    Q : ndarray, shape (NI, NJ, nvar)
        State variables.
    vol : ndarray, shape (NI, NJ)
        Cell volumes.
        
    Returns
    -------
    integral : ndarray, shape (nvar,)
        Volume integral of each variable.
    """
    NI, NJ, nvar = Q.shape
    result = np.zeros(nvar)
    
    for i in range(NI):
        for j in range(NJ):
            for k in range(nvar):
                result[k] += Q[i, j, k] * vol[i, j]
    
    return result


@njit(cache=True)
def compute_residual_sum(R: np.ndarray) -> np.ndarray:
    """
    Compute sum of residuals for each variable.
    
    Parameters
    ----------
    R : ndarray, shape (NI, NJ, nvar)
        Residuals.
        
    Returns
    -------
    total : ndarray, shape (nvar,)
        Sum of residuals for each variable.
    """
    NI, NJ, nvar = R.shape
    result = np.zeros(nvar)
    
    for i in range(NI):
        for j in range(NJ):
            for k in range(nvar):
                result[k] += R[i, j, k]
    
    return result


def create_coarse_arrays(NI_c: int, NJ_c: int, nvar: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create arrays for coarse grid state and residual.
    
    Parameters
    ----------
    NI_c, NJ_c : int
        Coarse grid dimensions.
    nvar : int
        Number of variables (default 4 for p, u, v, nu_tilde).
        
    Returns
    -------
    Q_c : ndarray, shape (NI_c, NJ_c, nvar)
        Coarse state array.
    R_c : ndarray, shape (NI_c, NJ_c, nvar)
        Coarse residual array.
    """
    Q_c = np.zeros((NI_c, NJ_c, nvar), dtype=np.float64)
    R_c = np.zeros((NI_c, NJ_c, nvar), dtype=np.float64)
    return Q_c, R_c

