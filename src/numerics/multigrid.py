"""
Multigrid transfer operators for FAS scheme.

Operations:
- restrict_state: Volume-weighted average Q_c = (sum Q_f * V_f) / V_c
- restrict_residual: Simple summation R_c = sum R_f
- prolongate_correction: Bilinear interpolation of correction dQ
"""

import numpy as np
from numba import njit, prange
from typing import Tuple


@njit(cache=True, parallel=True)
def restrict_timestep(dt_fine: np.ndarray, dt_coarse: np.ndarray) -> None:
    """Restrict timestep by averaging 4 fine cells."""
    NI_c, NJ_c = dt_coarse.shape
    NI_f, NJ_f = dt_fine.shape
    
    for i_c in prange(NI_c):
        for j_c in range(NJ_c):
            i_f = 2 * i_c
            j_f = 2 * j_c
            
            i_f1 = i_f + 1 if i_f + 1 < NI_f else NI_f - 1
            j_f1 = j_f + 1 if j_f + 1 < NJ_f else NJ_f - 1
            
            dt_coarse[i_c, j_c] = 0.25 * (
                dt_fine[i_f, j_f] + dt_fine[i_f1, j_f] +
                dt_fine[i_f, j_f1] + dt_fine[i_f1, j_f1]
            )


@njit(cache=True, parallel=True)
def restrict_state(Q_f: np.ndarray, vol_f: np.ndarray,
                   Q_c: np.ndarray, vol_c: np.ndarray) -> None:
    """Restrict state with volume-weighted averaging (preserves conservation)."""
    NI_c, NJ_c, nvar = Q_c.shape
    
    for i_c in prange(NI_c):
        for j_c in range(NJ_c):
            i_f = 2 * i_c
            j_f = 2 * j_c
            v_total = vol_c[i_c, j_c]
            
            for k in range(nvar):
                Q_sum = (Q_f[i_f, j_f, k] * vol_f[i_f, j_f] +
                         Q_f[i_f+1, j_f, k] * vol_f[i_f+1, j_f] +
                         Q_f[i_f, j_f+1, k] * vol_f[i_f, j_f+1] +
                         Q_f[i_f+1, j_f+1, k] * vol_f[i_f+1, j_f+1])
                Q_c[i_c, j_c, k] = Q_sum / v_total


@njit(cache=True, parallel=True)
def restrict_residual(R_f: np.ndarray, R_c: np.ndarray) -> None:
    """Restrict residuals by summation (residuals are volume-integrated)."""
    NI_c, NJ_c, nvar = R_c.shape
    
    for i_c in prange(NI_c):
        for j_c in range(NJ_c):
            i_f = 2 * i_c
            j_f = 2 * j_c
            
            for k in range(nvar):
                R_c[i_c, j_c, k] = (R_f[i_f, j_f, k] +
                                    R_f[i_f+1, j_f, k] +
                                    R_f[i_f, j_f+1, k] +
                                    R_f[i_f+1, j_f+1, k])


@njit(cache=True, parallel=True)
def prolongate_correction(Q_f: np.ndarray, 
                           Q_c_new: np.ndarray, 
                           Q_c_old: np.ndarray) -> None:
    """Prolongate coarse correction dQ = Q_c_new - Q_c_old using bilinear interpolation."""
    NI_f, NJ_f, nvar = Q_f.shape
    NI_c, NJ_c = Q_c_new.shape[0], Q_c_new.shape[1]
    
    for i_f in prange(NI_f):
        for j_f in range(NJ_f):
            i_c = i_f // 2
            j_c = j_f // 2
            
            di = i_f % 2
            dj = j_f % 2
            
            i_c_clamped = min(i_c, NI_c - 1)
            j_c_clamped = min(j_c, NJ_c - 1)
            
            xi = 0.25 + 0.5 * di
            eta = 0.25 + 0.5 * dj
            
            if xi < 0.5:
                i0, i1 = max(0, i_c_clamped - 1), i_c_clamped
                alpha = xi + 0.5
            else:
                i0, i1 = i_c_clamped, min(NI_c - 1, i_c_clamped + 1)
                alpha = xi - 0.5
            
            if eta < 0.5:
                j0, j1 = max(0, j_c_clamped - 1), j_c_clamped
                beta = eta + 0.5
            else:
                j0, j1 = j_c_clamped, min(NJ_c - 1, j_c_clamped + 1)
                beta = eta - 0.5
            
            for k in range(nvar):
                dQ_00 = Q_c_new[i0, j0, k] - Q_c_old[i0, j0, k]
                dQ_10 = Q_c_new[i1, j0, k] - Q_c_old[i1, j0, k]
                dQ_01 = Q_c_new[i0, j1, k] - Q_c_old[i0, j1, k]
                dQ_11 = Q_c_new[i1, j1, k] - Q_c_old[i1, j1, k]
                
                dQ_interp = ((1 - alpha) * (1 - beta) * dQ_00 +
                             alpha * (1 - beta) * dQ_10 +
                             (1 - alpha) * beta * dQ_01 +
                             alpha * beta * dQ_11)
                
                Q_f[i_f, j_f, k] += dQ_interp


@njit(cache=True, parallel=True)
def prolongate_injection(Q_f: np.ndarray,
                          Q_c_new: np.ndarray,
                          Q_c_old: np.ndarray) -> None:
    """Prolongate using injection (piecewise constant)."""
    NI_f, NJ_f, nvar = Q_f.shape
    NI_c, NJ_c = Q_c_new.shape[0], Q_c_new.shape[1]
    
    for i_f in prange(NI_f):
        for j_f in range(NJ_f):
            i_c = min(i_f // 2, NI_c - 1)
            j_c = min(j_f // 2, NJ_c - 1)
            
            for k in range(nvar):
                dQ = Q_c_new[i_c, j_c, k] - Q_c_old[i_c, j_c, k]
                Q_f[i_f, j_f, k] += dQ


@njit(cache=True)
def compute_integral(Q: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Compute volume integral of state variables."""
    NI, NJ, nvar = Q.shape
    result = np.zeros(nvar)
    
    for i in range(NI):
        for j in range(NJ):
            for k in range(nvar):
                result[k] += Q[i, j, k] * vol[i, j]
    
    return result


@njit(cache=True)
def compute_residual_sum(R: np.ndarray) -> np.ndarray:
    """Compute sum of residuals for each variable."""
    NI, NJ, nvar = R.shape
    result = np.zeros(nvar)
    
    for i in range(NI):
        for j in range(NJ):
            for k in range(nvar):
                result[k] += R[i, j, k]
    
    return result


def create_coarse_arrays(NI_c: int, NJ_c: int, nvar: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Create arrays for coarse grid state and residual."""
    Q_c = np.zeros((NI_c, NJ_c, nvar), dtype=np.float64)
    R_c = np.zeros((NI_c, NJ_c, nvar), dtype=np.float64)
    return Q_c, R_c
