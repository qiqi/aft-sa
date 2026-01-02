"""
JST (Jameson-Schmidt-Turkel) flux scheme for 2D incompressible RANS.

State vector: Q = [p, u, v, ν̃]
Pseudo-compressibility: ∂p/∂τ + β∇·u = 0

References:
    Jameson, Schmidt, & Turkel (1981). AIAA Paper 81-1259.
    Turkel (1987). JCP 72.
"""

import numpy as np
import numpy.typing as npt
from typing import NamedTuple
from dataclasses import dataclass
from numba import njit

from src.constants import NGHOST

NDArrayFloat = npt.NDArray[np.floating]


@dataclass
class FluxConfig:
    """Configuration for JST flux computation."""
    k4: float = 0.016


class GridMetrics(NamedTuple):
    """Grid metrics for flux computation."""
    Si_x: NDArrayFloat
    Si_y: NDArrayFloat
    Sj_x: NDArrayFloat
    Sj_y: NDArrayFloat
    volume: NDArrayFloat


@njit(cache=True, fastmath=True)
def _flux_kernel(Q: np.ndarray, 
                 Si_x: np.ndarray, Si_y: np.ndarray,
                 Sj_x: np.ndarray, Sj_y: np.ndarray,
                 beta: float, k4: float,
                 residual: np.ndarray,
                 nghost: int) -> None:
    """Numba-optimized kernel for JST flux computation (4th-order dissipation only)."""
    NI_ghost = Q.shape[0]
    NJ_ghost = Q.shape[1]
    NI = NI_ghost - 2 * nghost
    NJ = NJ_ghost - 2 * nghost
    
    F_i = np.zeros((NI + 1, NJ, 4))
    F_j = np.zeros((NI, NJ + 1, 4))
    
    for i in range(NI + 1):
        for j in range(NJ):
            i_L = i + nghost - 1
            i_R = i + nghost
            j_cell = j + nghost
            
            nx = Si_x[i, j]
            ny = Si_y[i, j]
            S = np.sqrt(nx * nx + ny * ny)
            
            p_L, u_L, v_L, nu_L = Q[i_L, j_cell, :]
            p_R, u_R, v_R, nu_R = Q[i_R, j_cell, :]
            
            p_avg = 0.5 * (p_L + p_R)
            u_avg = 0.5 * (u_L + u_R)
            v_avg = 0.5 * (v_L + v_R)
            nu_avg = 0.5 * (nu_L + nu_R)
            
            U_n = u_avg * nx + v_avg * ny
            
            F_conv_0 = beta * U_n
            F_conv_1 = u_avg * U_n + p_avg * nx
            F_conv_2 = v_avg * U_n + p_avg * ny
            F_conv_3 = nu_avg * U_n
            
            c_art = np.sqrt(u_avg * u_avg + v_avg * v_avg + beta)
            nx_hat = nx / (S + 1e-12)
            ny_hat = ny / (S + 1e-12)
            U_n_unit = u_avg * nx_hat + v_avg * ny_hat
            eps4 = k4 * (np.abs(U_n_unit) + c_art) * S
            
            Q_Lm1 = Q[i_L - 1, j_cell, :]
            Q_Rp1 = Q[i_R + 1, j_cell, :]
            
            F_i[i, j, 0] = F_conv_0 + eps4 * (Q_Rp1[0] - 3.0 * p_R + 3.0 * p_L - Q_Lm1[0])
            F_i[i, j, 1] = F_conv_1 + eps4 * (Q_Rp1[1] - 3.0 * u_R + 3.0 * u_L - Q_Lm1[1])
            F_i[i, j, 2] = F_conv_2 + eps4 * (Q_Rp1[2] - 3.0 * v_R + 3.0 * v_L - Q_Lm1[2])
            F_i[i, j, 3] = F_conv_3 + eps4 * (Q_Rp1[3] - 3.0 * nu_R + 3.0 * nu_L - Q_Lm1[3])
    
    for i in range(NI):
        for j in range(NJ + 1):
            i_cell = i + nghost
            j_L = j + nghost - 1
            j_R = j + nghost
            
            nx = Sj_x[i, j]
            ny = Sj_y[i, j]
            S = np.sqrt(nx * nx + ny * ny)
            
            p_L, u_L, v_L, nu_L = Q[i_cell, j_L, :]
            p_R, u_R, v_R, nu_R = Q[i_cell, j_R, :]
            
            p_avg = 0.5 * (p_L + p_R)
            u_avg = 0.5 * (u_L + u_R)
            v_avg = 0.5 * (v_L + v_R)
            nu_avg = 0.5 * (nu_L + nu_R)
            
            U_n = u_avg * nx + v_avg * ny
            
            F_conv_0 = beta * U_n
            F_conv_1 = u_avg * U_n + p_avg * nx
            F_conv_2 = v_avg * U_n + p_avg * ny
            F_conv_3 = nu_avg * U_n
            
            c_art = np.sqrt(u_avg * u_avg + v_avg * v_avg + beta)
            nx_hat = nx / (S + 1e-12)
            ny_hat = ny / (S + 1e-12)
            U_n_unit = u_avg * nx_hat + v_avg * ny_hat
            eps4 = k4 * (np.abs(U_n_unit) + c_art) * S
            
            Q_Lm1 = Q[i_cell, j_L - 1, :]
            Q_Rp1 = Q[i_cell, j_R + 1, :]
            
            F_j[i, j, 0] = F_conv_0 + eps4 * (Q_Rp1[0] - 3.0 * p_R + 3.0 * p_L - Q_Lm1[0])
            F_j[i, j, 1] = F_conv_1 + eps4 * (Q_Rp1[1] - 3.0 * u_R + 3.0 * u_L - Q_Lm1[1])
            F_j[i, j, 2] = F_conv_2 + eps4 * (Q_Rp1[2] - 3.0 * v_R + 3.0 * v_L - Q_Lm1[2])
            F_j[i, j, 3] = F_conv_3 + eps4 * (Q_Rp1[3] - 3.0 * nu_R + 3.0 * nu_L - Q_Lm1[3])
    
    for i in range(NI):
        for j in range(NJ):
            for k in range(4):
                R_i = F_i[i, j, k] - F_i[i + 1, j, k]
                R_j = F_j[i, j, k] - F_j[i, j + 1, k]
                residual[i, j, k] = R_i + R_j


def compute_convective_flux(Q: NDArrayFloat, nx: NDArrayFloat, ny: NDArrayFloat, 
                            beta: float) -> NDArrayFloat:
    """Compute projected convective flux F·n."""
    p: NDArrayFloat = Q[..., 0]
    u: NDArrayFloat = Q[..., 1]
    v: NDArrayFloat = Q[..., 2]
    nu_t: NDArrayFloat = Q[..., 3]
    
    U_n: NDArrayFloat = u * nx + v * ny
    
    F: NDArrayFloat = np.zeros_like(Q)
    F[..., 0] = beta * U_n
    F[..., 1] = u * U_n + p * nx
    F[..., 2] = v * U_n + p * ny
    F[..., 3] = nu_t * U_n
    
    return F


def compute_spectral_radius(Q: NDArrayFloat, nx: NDArrayFloat, ny: NDArrayFloat,
                           beta: float) -> NDArrayFloat:
    """Compute spectral radius λ = |U_n| + c_art * |S|."""
    u: NDArrayFloat = Q[..., 1]
    v: NDArrayFloat = Q[..., 2]
    
    S: NDArrayFloat = np.sqrt(nx**2 + ny**2)
    nx_hat: NDArrayFloat = nx / (S + 1e-12)
    ny_hat: NDArrayFloat = ny / (S + 1e-12)
    U_n: NDArrayFloat = u * nx_hat + v * ny_hat
    c_art: NDArrayFloat = np.sqrt(u**2 + v**2 + beta)
    
    return (np.abs(U_n) + c_art) * S


def compute_fluxes(Q: NDArrayFloat, metrics: GridMetrics, beta: float,
                   cfg: FluxConfig) -> NDArrayFloat:
    """Compute flux residual using JST scheme with 4th-order dissipation."""
    NI_ghost: int = Q.shape[0]
    NJ_ghost: int = Q.shape[1]
    NI: int = NI_ghost - 2 * NGHOST
    NJ: int = NJ_ghost - 2 * NGHOST
    
    Q_c: NDArrayFloat = np.ascontiguousarray(Q)
    Si_x: NDArrayFloat = np.ascontiguousarray(metrics.Si_x)
    Si_y: NDArrayFloat = np.ascontiguousarray(metrics.Si_y)
    Sj_x: NDArrayFloat = np.ascontiguousarray(metrics.Sj_x)
    Sj_y: NDArrayFloat = np.ascontiguousarray(metrics.Sj_y)
    
    residual: NDArrayFloat = np.zeros((NI, NJ, 4), dtype=Q.dtype)
    
    _flux_kernel(Q_c, Si_x, Si_y, Sj_x, Sj_y, beta, cfg.k4, residual, NGHOST)
    
    return residual


def compute_time_step(Q: NDArrayFloat, metrics: GridMetrics, beta: float,
                      cfl: float = 0.8) -> NDArrayFloat:
    """Compute local time step based on CFL condition."""
    NI: int = Q.shape[0] - 2 * NGHOST
    NJ: int = Q.shape[1] - 2 * NGHOST
    
    int_slice: slice = slice(NGHOST, -NGHOST)
    Q_int: NDArrayFloat = Q[int_slice, int_slice, :]
    u: NDArrayFloat = Q_int[..., 1]
    v: NDArrayFloat = Q_int[..., 2]
    
    c_art: NDArrayFloat = np.sqrt(u**2 + v**2 + beta)
    
    Si_mag: NDArrayFloat = np.sqrt(metrics.Si_x**2 + metrics.Si_y**2)
    Si_avg: NDArrayFloat = 0.5 * (Si_mag[:-1, :] + Si_mag[1:, :])
    
    Sj_mag: NDArrayFloat = np.sqrt(metrics.Sj_x**2 + metrics.Sj_y**2)
    Sj_avg: NDArrayFloat = 0.5 * (Sj_mag[:, :-1] + Sj_mag[:, 1:])
    
    lambda_i: NDArrayFloat = (np.abs(u) + c_art) * Si_avg / metrics.volume
    lambda_j: NDArrayFloat = (np.abs(v) + c_art) * Sj_avg / metrics.volume
    
    dt: NDArrayFloat = cfl * metrics.volume / (lambda_i + lambda_j + 1e-12)
    
    return dt
