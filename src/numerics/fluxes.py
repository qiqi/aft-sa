"""
JST (Jameson-Schmidt-Turkel) Flux Scheme for 2D Incompressible RANS.

This module implements the central-difference flux scheme with scalar
artificial dissipation, as described by Jameson, Schmidt, and Turkel (1981).

Physics: Artificial Compressibility formulation for incompressible flow.
    - State vector: Q = [p, u, v, ν̃]  (pressure, velocities, SA variable)
    - Pseudo-compressibility: ∂p/∂τ + β∇·u = 0
    - β is the artificial compressibility parameter

Performance: Uses Numba JIT compilation for the inner flux kernel.

References:
    [1] Jameson, A., Schmidt, W., & Turkel, E. (1981). "Numerical solution of
        the Euler equations by finite volume methods using Runge-Kutta time
        stepping schemes." AIAA Paper 81-1259.
    [2] Turkel, E. (1987). "Preconditioned methods for solving the 
        incompressible and low speed compressible equations." JCP 72.
"""

import numpy as np
from typing import Tuple, NamedTuple, Optional
from dataclasses import dataclass
from numba import njit


@dataclass
class FluxConfig:
    """Configuration for JST flux computation.
    
    Note: For incompressible flow, k2 (2nd-order dissipation) should be 0.
    The 2nd-order dissipation is designed for shock capturing in compressible
    flow and is not needed (and can harm convergence) for incompressible flow.
    """
    
    # JST dissipation coefficients
    # ε(2) = k2 * ν   where ν is pressure sensor (0 to 1)
    # ε(4) = max(0, k4 - ε(2))
    # Note: k2=0 for incompressible flow (no shocks to capture)
    k2: float = 0.0      # 2nd-order dissipation (0 for incompressible)
    k4: float = 0.016    # 4th-order dissipation (background smoothing)
    
    # Limiter for pressure sensor
    eps_p: float = 1e-10  # Small number to avoid division by zero
    
    # Minimum sensor value: provides baseline 1st-order dissipation
    # For incompressible flows without shocks, the sensor is always ~0
    # Adding nu_min > 0 ensures stable dissipation in smooth regions
    # Typical values: 0.0 (standard JST), 0.1-0.5 (more stable)
    nu_min: float = 0.0
    
    # First-order mode: use constant maximum dissipation (ν = 1)
    # This is more stable but only 1st-order accurate
    first_order: bool = False


class GridMetrics(NamedTuple):
    """Grid metrics for flux computation.
    
    For a structured grid with indices (i, j):
    - I-faces are between cells (i, j) and (i+1, j)
    - J-faces are between cells (i, j) and (i, j+1)
    
    All arrays include ghost cells: shape (NI+2, NJ+2) for cell-centered,
    (NI+1, NJ+2) for I-faces, (NI+2, NJ+1) for J-faces.
    """
    # I-face normals and areas (faces at constant i)
    # Shape: (NI+1, NJ) - one more face than cells in i-direction
    Si_x: np.ndarray    # x-component of I-face normal (scaled by area)
    Si_y: np.ndarray    # y-component of I-face normal (scaled by area)
    
    # J-face normals and areas (faces at constant j)  
    # Shape: (NI, NJ+1) - one more face than cells in j-direction
    Sj_x: np.ndarray    # x-component of J-face normal (scaled by area)
    Sj_y: np.ndarray    # y-component of J-face normal (scaled by area)
    
    # Cell volumes
    # Shape: (NI, NJ)
    volume: np.ndarray


# =============================================================================
# Numba-optimized flux kernel
# =============================================================================

@njit(cache=True, fastmath=True)
def _flux_kernel(Q: np.ndarray, 
                 Si_x: np.ndarray, Si_y: np.ndarray,
                 Sj_x: np.ndarray, Sj_y: np.ndarray,
                 beta: float, k2: float, k4: float, eps_p: float,
                 nu_min: float, first_order: bool,
                 residual: np.ndarray) -> None:
    """
    Numba-optimized kernel for JST flux computation.
    
    Computes the flux residual for all interior cells using:
    - Central convective fluxes
    - JST artificial dissipation (2nd and 4th order)
    
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector with ghost cells [p, u, v, nu_t].
    Si_x, Si_y : ndarray, shape (NI+1, NJ)
        I-face normals scaled by area.
    Sj_x, Sj_y : ndarray, shape (NI, NJ+1)
        J-face normals scaled by area.
    beta : float
        Artificial compressibility parameter.
    k2 : float
        2nd-order dissipation coefficient.
    k4 : float
        4th-order dissipation coefficient.
    eps_p : float
        Small number for pressure sensor.
    first_order : bool
        If True, use maximum dissipation (nu_sensor = 1).
    residual : ndarray, shape (NI, NJ, 4)
        Output residual array (modified in-place).
    """
    NI_ghost = Q.shape[0]
    NJ_ghost = Q.shape[1]
    NI = NI_ghost - 2
    NJ = NJ_ghost - 2
    
    # Temporary arrays for face fluxes
    # I-face fluxes: shape (NI+1, NJ, 4)
    F_i = np.zeros((NI + 1, NJ, 4))
    
    # J-face fluxes: shape (NI, NJ+1, 4)
    F_j = np.zeros((NI, NJ + 1, 4))
    
    # =========================================================================
    # Compute I-face fluxes
    # =========================================================================
    for i in range(NI + 1):
        for j in range(NJ):
            # Ghost-array indices for cells left and right of face
            i_L = i      # Left cell (ghost index)
            i_R = i + 1  # Right cell (ghost index)
            j_cell = j + 1  # Interior j-index in ghost array
            
            # Get face normal (scaled by area)
            nx = Si_x[i, j]
            ny = Si_y[i, j]
            S = np.sqrt(nx * nx + ny * ny)
            
            # Left and right states
            p_L = Q[i_L, j_cell, 0]
            u_L = Q[i_L, j_cell, 1]
            v_L = Q[i_L, j_cell, 2]
            nu_L = Q[i_L, j_cell, 3]
            
            p_R = Q[i_R, j_cell, 0]
            u_R = Q[i_R, j_cell, 1]
            v_R = Q[i_R, j_cell, 2]
            nu_R = Q[i_R, j_cell, 3]
            
            # Average state at face
            p_avg = 0.5 * (p_L + p_R)
            u_avg = 0.5 * (u_L + u_R)
            v_avg = 0.5 * (v_L + v_R)
            nu_avg = 0.5 * (nu_L + nu_R)
            
            # Contravariant velocity
            U_n = u_avg * nx + v_avg * ny
            
            # Central convective flux
            F_conv_0 = beta * U_n
            F_conv_1 = u_avg * U_n + p_avg * nx
            F_conv_2 = v_avg * U_n + p_avg * ny
            F_conv_3 = nu_avg * U_n
            
            # Spectral radius at face
            c_art = np.sqrt(u_avg * u_avg + v_avg * v_avg + beta)
            nx_hat = nx / (S + 1e-12)
            ny_hat = ny / (S + 1e-12)
            U_n_unit = u_avg * nx_hat + v_avg * ny_hat
            lambda_f = (np.abs(U_n_unit) + c_art) * S
            
            # Pressure sensor (at cells)
            # For interior faces (1 to NI-1), compute from neighboring cells
            # For boundary faces, use simplified approach
            
            if i >= 1 and i <= NI - 1:
                # First-order mode: skip sensor calculation, use maximum dissipation
                if first_order:
                    nu_sensor = 1.0
                else:
                    # Full 4-point stencil available
                    # Sensor at left cell (i in ghost array = i in 1-based interior)
                    if i >= 2:
                        p_Lm1 = Q[i - 1, j_cell, 0]
                        p_Lc = Q[i, j_cell, 0]
                        p_Lp1 = Q[i + 1, j_cell, 0]
                        d2p_L = np.abs(p_Lp1 - 2.0 * p_Lc + p_Lm1)
                        sum_L = np.abs(p_Lp1) + 2.0 * np.abs(p_Lc) + np.abs(p_Lm1) + eps_p
                        nu_sensor_L = d2p_L / sum_L
                    else:
                        nu_sensor_L = 0.0
                    
                    # Sensor at right cell
                    if i <= NI - 2:
                        p_Rm1 = Q[i, j_cell, 0]
                        p_Rc = Q[i + 1, j_cell, 0]
                        p_Rp1 = Q[i + 2, j_cell, 0]
                        d2p_R = np.abs(p_Rp1 - 2.0 * p_Rc + p_Rm1)
                        sum_R = np.abs(p_Rp1) + 2.0 * np.abs(p_Rc) + np.abs(p_Rm1) + eps_p
                        nu_sensor_R = d2p_R / sum_R
                    else:
                        nu_sensor_R = 0.0
                    
                    nu_sensor = max(nu_sensor_L, nu_sensor_R, nu_min)
                
                # JST dissipation coefficients
                eps2 = k2 * nu_sensor * lambda_f
                eps4 = max(0.0, k4 - k2 * nu_sensor) * lambda_f
                
                # States for 4th-order dissipation
                # L-1, L, R, R+1
                Q_Lm1_0 = Q[i - 1, j_cell, 0]
                Q_Lm1_1 = Q[i - 1, j_cell, 1]
                Q_Lm1_2 = Q[i - 1, j_cell, 2]
                Q_Lm1_3 = Q[i - 1, j_cell, 3]
                
                Q_Rp1_0 = Q[i + 2, j_cell, 0]
                Q_Rp1_1 = Q[i + 2, j_cell, 1]
                Q_Rp1_2 = Q[i + 2, j_cell, 2]
                Q_Rp1_3 = Q[i + 2, j_cell, 3]
                
                # 2nd-order dissipation: eps2 * (Q_R - Q_L)
                D2_0 = eps2 * (p_R - p_L)
                D2_1 = eps2 * (u_R - u_L)
                D2_2 = eps2 * (v_R - v_L)
                D2_3 = eps2 * (nu_R - nu_L)
                
                # 4th-order dissipation: eps4 * (Q_R+1 - 3*Q_R + 3*Q_L - Q_L-1)
                D4_0 = eps4 * (Q_Rp1_0 - 3.0 * p_R + 3.0 * p_L - Q_Lm1_0)
                D4_1 = eps4 * (Q_Rp1_1 - 3.0 * u_R + 3.0 * u_L - Q_Lm1_1)
                D4_2 = eps4 * (Q_Rp1_2 - 3.0 * v_R + 3.0 * v_L - Q_Lm1_2)
                D4_3 = eps4 * (Q_Rp1_3 - 3.0 * nu_R + 3.0 * nu_L - Q_Lm1_3)
                
                # Total dissipation
                D_0 = D2_0 - D4_0
                D_1 = D2_1 - D4_1
                D_2 = D2_2 - D4_2
                D_3 = D2_3 - D4_3
            else:
                # Boundary faces: 2nd-order only
                nu_sensor = 0.0
                eps2 = k2 * nu_sensor * lambda_f
                
                D_0 = eps2 * (p_R - p_L)
                D_1 = eps2 * (u_R - u_L)
                D_2 = eps2 * (v_R - v_L)
                D_3 = eps2 * (nu_R - nu_L)
            
            # Total flux = convective - dissipation
            F_i[i, j, 0] = F_conv_0 - D_0
            F_i[i, j, 1] = F_conv_1 - D_1
            F_i[i, j, 2] = F_conv_2 - D_2
            F_i[i, j, 3] = F_conv_3 - D_3
    
    # =========================================================================
    # Compute J-face fluxes
    # =========================================================================
    for i in range(NI):
        for j in range(NJ + 1):
            # Ghost-array indices
            i_cell = i + 1  # Interior i-index in ghost array
            j_L = j      # Left (bottom) cell
            j_R = j + 1  # Right (top) cell
            
            # Get face normal
            nx = Sj_x[i, j]
            ny = Sj_y[i, j]
            S = np.sqrt(nx * nx + ny * ny)
            
            # Left and right states
            p_L = Q[i_cell, j_L, 0]
            u_L = Q[i_cell, j_L, 1]
            v_L = Q[i_cell, j_L, 2]
            nu_L = Q[i_cell, j_L, 3]
            
            p_R = Q[i_cell, j_R, 0]
            u_R = Q[i_cell, j_R, 1]
            v_R = Q[i_cell, j_R, 2]
            nu_R = Q[i_cell, j_R, 3]
            
            # Average state
            p_avg = 0.5 * (p_L + p_R)
            u_avg = 0.5 * (u_L + u_R)
            v_avg = 0.5 * (v_L + v_R)
            nu_avg = 0.5 * (nu_L + nu_R)
            
            # Contravariant velocity
            U_n = u_avg * nx + v_avg * ny
            
            # Central convective flux
            F_conv_0 = beta * U_n
            F_conv_1 = u_avg * U_n + p_avg * nx
            F_conv_2 = v_avg * U_n + p_avg * ny
            F_conv_3 = nu_avg * U_n
            
            # Spectral radius
            c_art = np.sqrt(u_avg * u_avg + v_avg * v_avg + beta)
            nx_hat = nx / (S + 1e-12)
            ny_hat = ny / (S + 1e-12)
            U_n_unit = u_avg * nx_hat + v_avg * ny_hat
            lambda_f = (np.abs(U_n_unit) + c_art) * S
            
            # Pressure sensor and dissipation
            if j >= 1 and j <= NJ - 1:
                # First-order mode: skip sensor calculation, use maximum dissipation
                if first_order:
                    nu_sensor = 1.0
                else:
                    # Full stencil available
                    if j >= 2:
                        p_Lm1 = Q[i_cell, j - 1, 0]
                        p_Lc = Q[i_cell, j, 0]
                        p_Lp1 = Q[i_cell, j + 1, 0]
                        d2p_L = np.abs(p_Lp1 - 2.0 * p_Lc + p_Lm1)
                        sum_L = np.abs(p_Lp1) + 2.0 * np.abs(p_Lc) + np.abs(p_Lm1) + eps_p
                        nu_sensor_L = d2p_L / sum_L
                    else:
                        nu_sensor_L = 0.0
                    
                    if j <= NJ - 2:
                        p_Rm1 = Q[i_cell, j, 0]
                        p_Rc = Q[i_cell, j + 1, 0]
                        p_Rp1 = Q[i_cell, j + 2, 0]
                        d2p_R = np.abs(p_Rp1 - 2.0 * p_Rc + p_Rm1)
                        sum_R = np.abs(p_Rp1) + 2.0 * np.abs(p_Rc) + np.abs(p_Rm1) + eps_p
                        nu_sensor_R = d2p_R / sum_R
                    else:
                        nu_sensor_R = 0.0
                    
                    nu_sensor = max(nu_sensor_L, nu_sensor_R, nu_min)
                
                eps2 = k2 * nu_sensor * lambda_f
                eps4 = max(0.0, k4 - k2 * nu_sensor) * lambda_f
                
                Q_Lm1_0 = Q[i_cell, j - 1, 0]
                Q_Lm1_1 = Q[i_cell, j - 1, 1]
                Q_Lm1_2 = Q[i_cell, j - 1, 2]
                Q_Lm1_3 = Q[i_cell, j - 1, 3]
                
                Q_Rp1_0 = Q[i_cell, j + 2, 0]
                Q_Rp1_1 = Q[i_cell, j + 2, 1]
                Q_Rp1_2 = Q[i_cell, j + 2, 2]
                Q_Rp1_3 = Q[i_cell, j + 2, 3]
                
                D2_0 = eps2 * (p_R - p_L)
                D2_1 = eps2 * (u_R - u_L)
                D2_2 = eps2 * (v_R - v_L)
                D2_3 = eps2 * (nu_R - nu_L)
                
                D4_0 = eps4 * (Q_Rp1_0 - 3.0 * p_R + 3.0 * p_L - Q_Lm1_0)
                D4_1 = eps4 * (Q_Rp1_1 - 3.0 * u_R + 3.0 * u_L - Q_Lm1_1)
                D4_2 = eps4 * (Q_Rp1_2 - 3.0 * v_R + 3.0 * v_L - Q_Lm1_2)
                D4_3 = eps4 * (Q_Rp1_3 - 3.0 * nu_R + 3.0 * nu_L - Q_Lm1_3)
                
                D_0 = D2_0 - D4_0
                D_1 = D2_1 - D4_1
                D_2 = D2_2 - D4_2
                D_3 = D2_3 - D4_3
            else:
                nu_sensor = 0.0
                eps2 = k2 * nu_sensor * lambda_f
                
                D_0 = eps2 * (p_R - p_L)
                D_1 = eps2 * (u_R - u_L)
                D_2 = eps2 * (v_R - v_L)
                D_3 = eps2 * (nu_R - nu_L)
            
            F_j[i, j, 0] = F_conv_0 - D_0
            F_j[i, j, 1] = F_conv_1 - D_1
            F_j[i, j, 2] = F_conv_2 - D_2
            F_j[i, j, 3] = F_conv_3 - D_3
    
    # =========================================================================
    # Compute residual from flux balance
    # =========================================================================
    for i in range(NI):
        for j in range(NJ):
            # Flux in from left minus flux out through right
            for k in range(4):
                R_i = F_i[i, j, k] - F_i[i + 1, j, k]
                R_j = F_j[i, j, k] - F_j[i, j + 1, k]
                residual[i, j, k] = R_i + R_j


# =============================================================================
# Public API functions
# =============================================================================

def compute_convective_flux(Q: np.ndarray, nx: np.ndarray, ny: np.ndarray, 
                            beta: float) -> np.ndarray:
    """
    Compute the projected convective flux F·n for given states.
    
    For artificial compressibility, the flux vector is:
        F_c = [β(u·n), u(u·n) + p*nx, v(u·n) + p*ny, ν̃(u·n)]
    
    where u·n = u*nx + v*ny is the normal velocity.
    
    Parameters
    ----------
    Q : ndarray, shape (..., 4)
        State vector [p, u, v, nu_tilde].
    nx, ny : ndarray, shape (...)
        Face normal components (already scaled by face area).
    beta : float
        Artificial compressibility parameter.
        
    Returns
    -------
    F : ndarray, shape (..., 4)
        Projected convective flux.
    """
    p = Q[..., 0]
    u = Q[..., 1]
    v = Q[..., 2]
    nu_t = Q[..., 3]
    
    # Contravariant velocity (normal velocity * area)
    U_n = u * nx + v * ny
    
    # Flux components
    F = np.zeros_like(Q)
    F[..., 0] = beta * U_n                  # Continuity (pseudo-compressibility)
    F[..., 1] = u * U_n + p * nx            # x-momentum
    F[..., 2] = v * U_n + p * ny            # y-momentum
    F[..., 3] = nu_t * U_n                  # SA transport
    
    return F


def compute_spectral_radius(Q: np.ndarray, nx: np.ndarray, ny: np.ndarray,
                           beta: float) -> np.ndarray:
    """
    Compute the spectral radius of the flux Jacobian.
    
    For artificial compressibility:
        λ = |U_n| + c_art * |S|
        
    where:
        U_n = u*nx + v*ny (normal velocity, already scaled by area)
        c_art = sqrt(u² + v² + β) (artificial speed of sound)
        |S| = sqrt(nx² + ny²) (face area)
    
    Parameters
    ----------
    Q : ndarray, shape (..., 4)
        State vector.
    nx, ny : ndarray
        Face normal components (scaled by area).
    beta : float
        Artificial compressibility parameter.
        
    Returns
    -------
    lambda_ : ndarray
        Spectral radius at each face.
    """
    u = Q[..., 1]
    v = Q[..., 2]
    
    # Face area
    S = np.sqrt(nx**2 + ny**2)
    
    # Unit normal
    nx_hat = nx / (S + 1e-12)
    ny_hat = ny / (S + 1e-12)
    
    # Normal velocity (not scaled by area)
    U_n = u * nx_hat + v * ny_hat
    
    # Artificial speed of sound
    c_art = np.sqrt(u**2 + v**2 + beta)
    
    # Spectral radius (scaled by face area)
    return (np.abs(U_n) + c_art) * S


def compute_fluxes(Q: np.ndarray, metrics: GridMetrics, beta: float,
                   cfg: FluxConfig) -> np.ndarray:
    """
    Compute the flux residual using the JST scheme.
    
    Uses Numba-optimized kernel for performance.
    
    The residual R for each cell is the sum of fluxes through all faces:
        R = -( F_{i+1/2} - F_{i-1/2} + F_{j+1/2} - F_{j-1/2} )
        
    where each flux consists of a central convective part and dissipation:
        F = F_conv - D
        
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector including ghost cells.
        Q[0, :] and Q[NI+1, :] are i-direction ghost cells.
        Q[:, 0] and Q[:, NJ+1] are j-direction ghost cells.
    metrics : GridMetrics
        Grid metrics containing face normals and cell volumes.
    beta : float
        Artificial compressibility parameter.
    cfg : FluxConfig
        Configuration with JST coefficients.
        
    Returns
    -------
    residual : ndarray, shape (NI, NJ, 4)
        Flux residual at interior cells (without ghost cells).
        
    Notes
    -----
    Sign convention: Positive residual means net flux INTO the cell.
    For steady state, we want R → 0.
    
    The JST scheme uses:
    - Central (averaged) convective flux
    - 2nd-order dissipation with pressure sensor (shock capturing)
    - 4th-order background dissipation (removes odd-even decoupling)
    """
    NI_ghost, NJ_ghost, n_vars = Q.shape
    NI = NI_ghost - 2
    NJ = NJ_ghost - 2
    
    # Ensure arrays are contiguous for Numba
    Q_c = np.ascontiguousarray(Q)
    Si_x = np.ascontiguousarray(metrics.Si_x)
    Si_y = np.ascontiguousarray(metrics.Si_y)
    Sj_x = np.ascontiguousarray(metrics.Sj_x)
    Sj_y = np.ascontiguousarray(metrics.Sj_y)
    
    # Allocate output array
    residual = np.zeros((NI, NJ, 4), dtype=Q.dtype)
    
    # Call Numba kernel
    _flux_kernel(Q_c, Si_x, Si_y, Sj_x, Sj_y,
                 beta, cfg.k2, cfg.k4, cfg.eps_p, cfg.nu_min, cfg.first_order, residual)
    
    return residual


def compute_time_step(Q: np.ndarray, metrics: GridMetrics, beta: float,
                      cfl: float = 0.8) -> np.ndarray:
    """
    Compute the local time step based on CFL condition.
    
    For artificial compressibility:
        Δt = CFL * Volume / (λ_i + λ_j)
        
    where λ is the spectral radius in each direction.
    
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector with ghost cells.
    metrics : GridMetrics
        Grid metrics.
    beta : float
        Artificial compressibility parameter.
    cfl : float
        CFL number (default 0.8).
        
    Returns
    -------
    dt : ndarray, shape (NI, NJ)
        Local time step for each cell.
    """
    NI = Q.shape[0] - 2
    NJ = Q.shape[1] - 2
    
    # Interior cell states
    Q_int = Q[1:-1, 1:-1, :]
    u = Q_int[..., 1]
    v = Q_int[..., 2]
    
    # Artificial speed of sound
    c_art = np.sqrt(u**2 + v**2 + beta)
    
    # Spectral radius in each direction (approximate using cell-averaged metrics)
    # I-direction: average of left and right face areas
    Si_mag = np.sqrt(metrics.Si_x**2 + metrics.Si_y**2)
    Si_avg = 0.5 * (Si_mag[:-1, :] + Si_mag[1:, :])  # Shape: (NI, NJ)
    
    Sj_mag = np.sqrt(metrics.Sj_x**2 + metrics.Sj_y**2)
    Sj_avg = 0.5 * (Sj_mag[:, :-1] + Sj_mag[:, 1:])  # Shape: (NI, NJ)
    
    # Spectral radius (simplified - assumes roughly aligned grid)
    lambda_i = (np.abs(u) + c_art) * Si_avg / metrics.volume
    lambda_j = (np.abs(v) + c_art) * Sj_avg / metrics.volume
    
    # Time step
    dt = cfl * metrics.volume / (lambda_i + lambda_j + 1e-12)
    
    return dt


# =============================================================================
# Legacy functions (kept for compatibility, but not used in optimized path)
# =============================================================================

def compute_pressure_sensor(p: np.ndarray, eps: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the pressure-based switching function for JST dissipation.
    
    Note: This function is kept for reference/testing. The optimized
    _flux_kernel computes the sensor inline.
    """
    d2p_i = np.abs(p[2:, :] - 2*p[1:-1, :] + p[:-2, :])
    sum_p_i = p[2:, :] + 2*p[1:-1, :] + p[:-2, :] + eps
    nu_cells_i = d2p_i / sum_p_i
    
    d2p_j = np.abs(p[:, 2:] - 2*p[:, 1:-1] + p[:, :-2])
    sum_p_j = p[:, 2:] + 2*p[:, 1:-1] + p[:, :-2] + eps
    nu_cells_j = d2p_j / sum_p_j
    
    return nu_cells_i, nu_cells_j
