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

from src.constants import NGHOST
from src.physics.jax_config import jax, jnp

NDArrayFloat = npt.NDArray[np.floating]


@dataclass
class FluxConfig:
    """Configuration for JST flux computation."""
    k4: float = 0.016
    martinelli_alpha: float = 0.667  # Exponent for aspect ratio scaling (2/3)
    martinelli_max: float = 3.0  # Maximum Martinelli scaling factor
    sponge_thickness: int = 15  # Sponge layer thickness in cells


class GridMetrics(NamedTuple):
    """Grid metrics for flux computation."""
    Si_x: NDArrayFloat
    Si_y: NDArrayFloat
    Sj_x: NDArrayFloat
    Sj_y: NDArrayFloat
    volume: NDArrayFloat


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
    """Compute flux residual using JST scheme with 4th-order dissipation and Martinelli scaling.
    
    This function wraps the JAX implementation for compatibility with NumPy arrays.
    """
    # Convert to JAX arrays
    Q_jax = jnp.asarray(Q)
    Si_x = jnp.asarray(metrics.Si_x)
    Si_y = jnp.asarray(metrics.Si_y)
    Sj_x = jnp.asarray(metrics.Sj_x)
    Sj_y = jnp.asarray(metrics.Sj_y)
    
    # Call JAX implementation
    result = compute_fluxes_jax(Q_jax, Si_x, Si_y, Sj_x, Sj_y, beta, cfg.k4, NGHOST,
                                cfg.martinelli_alpha)
    
    # Convert back to NumPy
    return np.asarray(result)


def compute_time_step(Q: NDArrayFloat, metrics: GridMetrics, beta: float,
                      cfl: float = 0.8) -> NDArrayFloat:
    """Compute local time step based on CFL condition."""
    _NI: int = Q.shape[0] - 2 * NGHOST
    _NJ: int = Q.shape[1] - 2 * NGHOST
    
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


# =============================================================================
# JAX Implementation
# =============================================================================

def compute_fluxes_jax(Q, Si_x, Si_y, Sj_x, Sj_y, beta, k4, nghost, 
                       martinelli_alpha=0.667, sigma=None):
    """
    JAX: Compute flux residual using JST scheme with Martinelli scaling and sponge layer.
    
    Parameters
    ----------
    Q : jnp.ndarray
        State array (NI+2*nghost, NJ+2*nghost, 4).
    Si_x, Si_y : jnp.ndarray
        I-face normal vectors (NI+1, NJ).
    Sj_x, Sj_y : jnp.ndarray
        J-face normal vectors (NI, NJ+1).
    beta : float
        Artificial compressibility parameter.
    k4 : float
        4th-order dissipation coefficient.
    nghost : int
        Number of ghost cells.
    martinelli_alpha : float
        Exponent for aspect ratio scaling (default 2/3).
    sigma : jnp.ndarray, optional
        Sponge coefficient field (NI, NJ). If None, pure 4th-order dissipation.
        Values in [0, 1]: 0=interior (pure 4th-order), 1=boundary (pure 2nd-order).
        
    Returns
    -------
    residual : jnp.ndarray
        Flux residual (NI, NJ, 4).
    """
    NI_p1, NJ = Si_x.shape  # (NI+1, NJ)
    NI = NI_p1 - 1
    
    # I-direction fluxes
    Q_L_i = Q[nghost-1:nghost+NI, nghost:nghost+NJ, :]     # (NI+1, NJ, 4)
    Q_R_i = Q[nghost:nghost+NI+1, nghost:nghost+NJ, :]     # (NI+1, NJ, 4)
    Q_Lm1_i = Q[nghost-2:nghost+NI-1, nghost:nghost+NJ, :] # (NI+1, NJ, 4)
    Q_Rp1_i = Q[nghost+1:nghost+NI+2, nghost:nghost+NJ, :] # (NI+1, NJ, 4)
    
    # Interior cell velocities for spectral radius
    Q_int = Q[nghost:nghost+NI, nghost:nghost+NJ, :]
    
    # If sigma not provided, use zero (pure 4th-order everywhere)
    if sigma is None:
        sigma = jnp.zeros((NI, NJ))
    
    return _compute_fluxes_jax_impl(
        Q_L_i, Q_R_i, Q_Lm1_i, Q_Rp1_i, Si_x, Si_y,
        Q[nghost:nghost+NI, nghost-1:nghost+NJ, :],       # Q_L_j
        Q[nghost:nghost+NI, nghost:nghost+NJ+1, :],       # Q_R_j  
        Q[nghost:nghost+NI, nghost-2:nghost+NJ-1, :],     # Q_Lm1_j
        Q[nghost:nghost+NI, nghost+1:nghost+NJ+2, :],     # Q_Rp1_j
        Sj_x, Sj_y, Q_int, beta, k4, martinelli_alpha, sigma
    )


@jax.jit
def _compute_fluxes_jax_impl(Q_L_i, Q_R_i, Q_Lm1_i, Q_Rp1_i, Si_x, Si_y,
                              Q_L_j, Q_R_j, Q_Lm1_j, Q_Rp1_j, Sj_x, Sj_y,
                              Q_int, beta, k4, martinelli_alpha, sigma):
    """JIT-compiled flux computation kernel with Martinelli scaling and sponge layer.
    
    The dissipation uses JST blending with sponge layer:
    - epsilon_2 = sigma (2nd-order from sponge, 0 in interior, 1 at boundary)
    - epsilon_4 = max(0, k_4 - epsilon_2) (4th-order with blending)
    
    Dissipation flux:
    F_diss = ε₂ * λ * (Q_{j+1} - Q_j) - ε₄ * λ * (Q_{j+2} - 3Q_{j+1} + 3Q_j - Q_{j-1})
    
    Interior (σ=0): pure 4th-order dissipation
    Boundary (σ=1): pure 2nd-order dissipation for wave absorption
    
    NOTE: nuHat (index 3) uses FIRST-ORDER UPWIND for stability instead of JST.
    This prevents oscillations near sharp gradients (wall, wake cut).
    """
    NI_p1, _NJ_i = Si_x.shape
    _NI = NI_p1 - 1
    NJ_j_p1 = Sj_y.shape[1]
    _NJ = NJ_j_p1 - 1
    
    # Extract interior velocities (NI, NJ)
    u_int = Q_int[:, :, 1]
    v_int = Q_int[:, :, 2]
    c_art_int = jnp.sqrt(u_int**2 + v_int**2 + beta)
    
    # Compute cell-centered spectral radii for Martinelli scaling
    # λ_i at cells: average I-face metrics
    Si_x_cell = 0.5 * (Si_x[:-1, :] + Si_x[1:, :])  # (NI, NJ)
    Si_y_cell = 0.5 * (Si_y[:-1, :] + Si_y[1:, :])
    Si_mag_cell = jnp.sqrt(Si_x_cell**2 + Si_y_cell**2)
    ni_x = Si_x_cell / (Si_mag_cell + 1e-12)
    ni_y = Si_y_cell / (Si_mag_cell + 1e-12)
    U_n_i_cell = u_int * ni_x + v_int * ni_y
    lambda_i_cell = (jnp.abs(U_n_i_cell) + c_art_int) * Si_mag_cell  # (NI, NJ)
    
    # λ_j at cells: average J-face metrics
    Sj_x_cell = 0.5 * (Sj_x[:, :-1] + Sj_x[:, 1:])  # (NI, NJ)
    Sj_y_cell = 0.5 * (Sj_y[:, :-1] + Sj_y[:, 1:])
    Sj_mag_cell = jnp.sqrt(Sj_x_cell**2 + Sj_y_cell**2)
    nj_x = Sj_x_cell / (Sj_mag_cell + 1e-12)
    nj_y = Sj_y_cell / (Sj_mag_cell + 1e-12)
    U_n_j_cell = u_int * nj_x + v_int * nj_y
    lambda_j_cell = (jnp.abs(U_n_j_cell) + c_art_int) * Sj_mag_cell  # (NI, NJ)
    
    # =================================================================
    # Sponge layer blending coefficients
    # epsilon_2 = sigma, epsilon_4 = max(0, k4 - sigma)
    # =================================================================
    eps2_cell = sigma  # (NI, NJ)
    eps4_base = jnp.maximum(0.0, k4 - eps2_cell)  # (NI, NJ)
    
    # =================================================================
    # I-direction fluxes
    # =================================================================
    S_i = jnp.sqrt(Si_x**2 + Si_y**2)
    
    Q_avg_i = 0.5 * (Q_L_i + Q_R_i)
    p_avg_i = Q_avg_i[:, :, 0]
    u_avg_i = Q_avg_i[:, :, 1]
    v_avg_i = Q_avg_i[:, :, 2]
    
    U_n_i = u_avg_i * Si_x + v_avg_i * Si_y
    
    # nuHat: first-order upwind based on sign of U_n (more stable than JST)
    nuHat_L_i = Q_L_i[:, :, 3]
    nuHat_R_i = Q_R_i[:, :, 3]
    nuHat_upwind_i = jnp.where(U_n_i >= 0, nuHat_L_i, nuHat_R_i)
    
    F_conv_i = jnp.stack([
        beta * U_n_i,
        u_avg_i * U_n_i + p_avg_i * Si_x,
        v_avg_i * U_n_i + p_avg_i * Si_y,
        nuHat_upwind_i * U_n_i  # First-order upwind for nuHat
    ], axis=-1)
    
    c_art_i = jnp.sqrt(u_avg_i**2 + v_avg_i**2 + beta)
    nx_hat_i = Si_x / (S_i + 1e-12)
    ny_hat_i = Si_y / (S_i + 1e-12)
    U_n_unit_i = u_avg_i * nx_hat_i + v_avg_i * ny_hat_i
    lambda_face_i = (jnp.abs(U_n_unit_i) + c_art_i) * S_i
    
    # Martinelli: f_i = 1 + (λ_j/λ_i)^α, λ_j at face from cell average
    # For face i (between cells i-1 and i), average cells i-1 and i
    # Pad lambda_j_cell for boundary faces
    lambda_j_padded = jnp.pad(lambda_j_cell, ((1, 1), (0, 0)), mode='edge')
    lambda_j_face_i = 0.5 * (lambda_j_padded[:-1, :] + lambda_j_padded[1:, :])  # (NI+1, NJ)
    
    ratio_i = lambda_j_face_i / (lambda_face_i + 1e-12)
    f_i = 1.0 + ratio_i ** martinelli_alpha
    f_i = jnp.minimum(f_i, 5.0)  # Cap to prevent excessive dissipation
    
    # Sponge blending for I-direction: average eps2/eps4 to faces
    eps2_i_padded = jnp.pad(eps2_cell, ((1, 1), (0, 0)), mode='edge')
    eps2_face_i = 0.5 * (eps2_i_padded[:-1, :] + eps2_i_padded[1:, :])  # (NI+1, NJ)
    
    eps4_i_padded = jnp.pad(eps4_base, ((1, 1), (0, 0)), mode='edge')
    eps4_face_i = 0.5 * (eps4_i_padded[:-1, :] + eps4_i_padded[1:, :])  # (NI+1, NJ)
    
    # 2nd-order dissipation: ε₂ * λ * (Q_R - Q_L)
    # Only apply to flow variables (p, u, v); nuHat uses upwind (no JST dissipation)
    diss2_i_flow = eps2_face_i[:, :, None] * lambda_face_i[:, :, None] * (Q_R_i[:, :, :3] - Q_L_i[:, :, :3])

    # 4th-order dissipation with Martinelli: ε₄ * λ * f * (Q_{+2} - 3Q_{+1} + 3Q_0 - Q_{-1})
    diss4_i_flow = eps4_face_i[:, :, None] * lambda_face_i[:, :, None] * f_i[:, :, None] * \
              (Q_Rp1_i[:, :, :3] - 3.0 * Q_R_i[:, :, :3] + 3.0 * Q_L_i[:, :, :3] - Q_Lm1_i[:, :, :3])

    # Combine: JST dissipation for flow, zero for nuHat (upwind already in convective flux)
    diss2_i = jnp.concatenate([diss2_i_flow, jnp.zeros_like(diss2_i_flow[:, :, :1])], axis=-1)
    diss4_i = jnp.concatenate([diss4_i_flow, jnp.zeros_like(diss4_i_flow[:, :, :1])], axis=-1)
    
    F_i = F_conv_i - diss2_i + diss4_i
    
    # =================================================================
    # J-direction fluxes
    # =================================================================
    S_j = jnp.sqrt(Sj_x**2 + Sj_y**2)
    
    Q_avg_j = 0.5 * (Q_L_j + Q_R_j)
    p_avg_j = Q_avg_j[:, :, 0]
    u_avg_j = Q_avg_j[:, :, 1]
    v_avg_j = Q_avg_j[:, :, 2]
    
    U_n_j = u_avg_j * Sj_x + v_avg_j * Sj_y
    
    # nuHat: first-order upwind based on sign of U_n (more stable than JST)
    nuHat_L_j = Q_L_j[:, :, 3]
    nuHat_R_j = Q_R_j[:, :, 3]
    nuHat_upwind_j = jnp.where(U_n_j >= 0, nuHat_L_j, nuHat_R_j)
    
    F_conv_j = jnp.stack([
        beta * U_n_j,
        u_avg_j * U_n_j + p_avg_j * Sj_x,
        v_avg_j * U_n_j + p_avg_j * Sj_y,
        nuHat_upwind_j * U_n_j  # First-order upwind for nuHat
    ], axis=-1)
    
    c_art_j = jnp.sqrt(u_avg_j**2 + v_avg_j**2 + beta)
    nx_hat_j = Sj_x / (S_j + 1e-12)
    ny_hat_j = Sj_y / (S_j + 1e-12)
    U_n_unit_j = u_avg_j * nx_hat_j + v_avg_j * ny_hat_j
    lambda_face_j = (jnp.abs(U_n_unit_j) + c_art_j) * S_j
    
    # Martinelli: f_j = 1 + (λ_i/λ_j)^α
    lambda_i_padded = jnp.pad(lambda_i_cell, ((0, 0), (1, 1)), mode='edge')
    lambda_i_face_j = 0.5 * (lambda_i_padded[:, :-1] + lambda_i_padded[:, 1:])  # (NI, NJ+1)
    
    ratio_j = lambda_i_face_j / (lambda_face_j + 1e-12)
    f_j = 1.0 + ratio_j ** martinelli_alpha
    f_j = jnp.minimum(f_j, 5.0)  # Cap to prevent excessive dissipation
    
    # Sponge blending for J-direction: average eps2/eps4 to faces
    eps2_j_padded = jnp.pad(eps2_cell, ((0, 0), (1, 1)), mode='edge')
    eps2_face_j = 0.5 * (eps2_j_padded[:, :-1] + eps2_j_padded[:, 1:])  # (NI, NJ+1)
    
    eps4_j_padded = jnp.pad(eps4_base, ((0, 0), (1, 1)), mode='edge')
    eps4_face_j = 0.5 * (eps4_j_padded[:, :-1] + eps4_j_padded[:, 1:])  # (NI, NJ+1)
    
    # 2nd-order dissipation: ε₂ * λ * (Q_R - Q_L)
    # Only apply to flow variables (p, u, v); nuHat uses upwind (no JST dissipation)
    diss2_j_flow = eps2_face_j[:, :, None] * lambda_face_j[:, :, None] * (Q_R_j[:, :, :3] - Q_L_j[:, :, :3])

    # 4th-order dissipation with Martinelli: ε₄ * λ * f * (Q_{+2} - 3Q_{+1} + 3Q_0 - Q_{-1})
    diss4_j_flow = eps4_face_j[:, :, None] * lambda_face_j[:, :, None] * f_j[:, :, None] * \
              (Q_Rp1_j[:, :, :3] - 3.0 * Q_R_j[:, :, :3] + 3.0 * Q_L_j[:, :, :3] - Q_Lm1_j[:, :, :3])

    # Combine: JST dissipation for flow, zero for nuHat (upwind already in convective flux)
    diss2_j = jnp.concatenate([diss2_j_flow, jnp.zeros_like(diss2_j_flow[:, :, :1])], axis=-1)
    diss4_j = jnp.concatenate([diss4_j_flow, jnp.zeros_like(diss4_j_flow[:, :, :1])], axis=-1)
    
    F_j = F_conv_j - diss2_j + diss4_j
    
    # =================================================================
    # Residual
    # =================================================================
    R_i = F_i[:-1, :, :] - F_i[1:, :, :]
    R_j = F_j[:, :-1, :] - F_j[:, 1:, :]
    
    return R_i + R_j
