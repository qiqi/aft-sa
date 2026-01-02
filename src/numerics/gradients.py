"""
Green-Gauss gradient reconstruction: ∇φ ≈ (1/V) Σ φ_face · S_face

Boundary handling relies on properly set ghost cells.
"""

import numpy as np
from numba import njit
from typing import NamedTuple

from ..constants import NGHOST


class GradientMetrics(NamedTuple):
    """Grid metrics for gradient computation (interior cells only)."""
    Si_x: np.ndarray  # (NI+1, NJ)
    Si_y: np.ndarray  # (NI+1, NJ)
    Sj_x: np.ndarray  # (NI, NJ+1)
    Sj_y: np.ndarray  # (NI, NJ+1)
    volume: np.ndarray  # (NI, NJ)


@njit(cache=True)
def _gradient_kernel(Q: np.ndarray,
                     Si_x: np.ndarray, Si_y: np.ndarray,
                     Sj_x: np.ndarray, Sj_y: np.ndarray,
                     volume: np.ndarray,
                     grad: np.ndarray,
                     nghost: int) -> None:
    """Numba kernel for Green-Gauss gradient computation."""
    NI = volume.shape[0]
    NJ = volume.shape[1]
    
    for i in range(NI):
        for j in range(NJ):
            for k in range(4):
                grad[i, j, k, 0] = 0.0
                grad[i, j, k, 1] = 0.0
    
    # I-face contributions
    for i in range(NI + 1):
        for j in range(NJ):
            nx = Si_x[i, j]
            ny = Si_y[i, j]
            
            i_L = i + nghost - 1
            i_R = i + nghost
            j_Q = j + nghost
            
            for k in range(4):
                phi_L = Q[i_L, j_Q, k]
                phi_R = Q[i_R, j_Q, k]
                phi_face = 0.5 * (phi_L + phi_R)
                contrib_x = phi_face * nx
                contrib_y = phi_face * ny
                
                if i > 0:
                    grad[i - 1, j, k, 0] += contrib_x
                    grad[i - 1, j, k, 1] += contrib_y
                if i < NI:
                    grad[i, j, k, 0] -= contrib_x
                    grad[i, j, k, 1] -= contrib_y
    
    # J-face contributions
    for i in range(NI):
        for j in range(NJ + 1):
            nx = Sj_x[i, j]
            ny = Sj_y[i, j]
            
            i_Q = i + nghost
            j_L = j + nghost - 1
            j_R = j + nghost
            
            for k in range(4):
                phi_L = Q[i_Q, j_L, k]
                phi_R = Q[i_Q, j_R, k]
                phi_face = 0.5 * (phi_L + phi_R)
                contrib_x = phi_face * nx
                contrib_y = phi_face * ny
                
                if j > 0:
                    grad[i, j - 1, k, 0] += contrib_x
                    grad[i, j - 1, k, 1] += contrib_y
                if j < NJ:
                    grad[i, j, k, 0] -= contrib_x
                    grad[i, j, k, 1] -= contrib_y
    
    # Divide by volume
    for i in range(NI):
        for j in range(NJ):
            inv_vol = 1.0 / volume[i, j]
            for k in range(4):
                grad[i, j, k, 0] *= inv_vol
                grad[i, j, k, 1] *= inv_vol


def compute_gradients(Q: np.ndarray, metrics: GradientMetrics) -> np.ndarray:
    """Compute cell-centered gradients using Green-Gauss theorem."""
    NI, NJ = metrics.volume.shape
    grad = np.zeros((NI, NJ, 4, 2), dtype=np.float64)
    
    _gradient_kernel(
        Q.astype(np.float64),
        metrics.Si_x.astype(np.float64),
        metrics.Si_y.astype(np.float64),
        metrics.Sj_x.astype(np.float64),
        metrics.Sj_y.astype(np.float64),
        metrics.volume.astype(np.float64),
        grad,
        NGHOST
    )
    
    return grad


@njit(cache=True)
def _compute_vorticity_kernel(dudx: np.ndarray, dudy: np.ndarray,
                               dvdx: np.ndarray, dvdy: np.ndarray,
                               omega: np.ndarray) -> None:
    """Compute vorticity magnitude ω = |∂v/∂x - ∂u/∂y|."""
    NI, NJ = dudx.shape
    for i in range(NI):
        for j in range(NJ):
            omega[i, j] = abs(dvdx[i, j] - dudy[i, j])


def compute_vorticity(grad: np.ndarray) -> np.ndarray:
    """Compute vorticity magnitude from gradient array."""
    NI, NJ = grad.shape[:2]
    dudy = grad[:, :, 1, 1]
    dvdx = grad[:, :, 2, 0]
    
    omega = np.zeros((NI, NJ), dtype=np.float64)
    _compute_vorticity_kernel(
        grad[:, :, 1, 0].astype(np.float64),
        dudy.astype(np.float64),
        dvdx.astype(np.float64),
        grad[:, :, 2, 1].astype(np.float64),
        omega
    )
    
    return omega


def compute_strain_rate(grad: np.ndarray) -> np.ndarray:
    """Compute strain rate magnitude |S| = sqrt(2·S_ij·S_ij)."""
    dudx = grad[:, :, 1, 0]
    dudy = grad[:, :, 1, 1]
    dvdx = grad[:, :, 2, 0]
    dvdy = grad[:, :, 2, 1]
    
    S_xx = dudx
    S_yy = dvdy
    S_xy = 0.5 * (dudy + dvdx)
    
    S_mag = np.sqrt(2.0 * (S_xx**2 + S_yy**2 + 2.0 * S_xy**2))
    return S_mag
