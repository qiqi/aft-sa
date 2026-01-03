"""
Viscous flux computation for 2D incompressible RANS.

Stress tensor (incompressible): τ_xx = 2μ·∂u/∂x, τ_yy = 2μ·∂v/∂y, τ_xy = μ·(∂u/∂y + ∂v/∂x)
"""

import numpy as np
import numpy.typing as npt
from src.constants import NGHOST
from .gradients import GradientMetrics
from src.physics.jax_config import jax, jnp

NDArrayFloat = npt.NDArray[np.floating]


def compute_viscous_fluxes(
    Q: NDArrayFloat,
    gradients: NDArrayFloat,
    grid_metrics: GradientMetrics,
    mu_laminar: float,
    mu_turbulent: NDArrayFloat | None = None
) -> NDArrayFloat:
    """Compute viscous flux residuals.
    
    This function wraps the JAX implementation for compatibility with NumPy arrays.
    """
    NI, NJ = gradients.shape[:2]
    
    # Compute effective viscosity
    if mu_turbulent is None:
        mu_eff = jnp.full((NI, NJ), mu_laminar, dtype=jnp.float64)
    else:
        mu_eff = jnp.asarray(mu_laminar + mu_turbulent)
    
    # Convert to JAX arrays
    grad_jax = jnp.asarray(gradients)
    Si_x = jnp.asarray(grid_metrics.Si_x)
    Si_y = jnp.asarray(grid_metrics.Si_y)
    Sj_x = jnp.asarray(grid_metrics.Sj_x)
    Sj_y = jnp.asarray(grid_metrics.Sj_y)
    
    # Call JAX implementation
    result = compute_viscous_fluxes_jax(grad_jax, Si_x, Si_y, Sj_x, Sj_y, mu_eff)
    
    return np.asarray(result)


def add_viscous_fluxes(
    convective_residual: NDArrayFloat,
    Q: NDArrayFloat,
    gradients: NDArrayFloat,
    grid_metrics: GradientMetrics,
    mu_laminar: float,
    mu_turbulent: NDArrayFloat | None = None
) -> NDArrayFloat:
    """Add viscous fluxes to convective residual."""
    viscous_residual = compute_viscous_fluxes(
        Q, gradients, grid_metrics, mu_laminar, mu_turbulent
    )
    
    return convective_residual + viscous_residual


def compute_nu_tilde_diffusion(
    Q: NDArrayFloat,
    gradients: NDArrayFloat,
    grid_metrics: GradientMetrics,
    nu_laminar: float,
    sigma: float = 2.0/3.0
) -> NDArrayFloat:
    """Compute SA nu_tilde diffusion residual.
    
    This function wraps the JAX implementation for compatibility with NumPy arrays.
    """
    NI, NJ = gradients.shape[:2]
    
    nu_tilde = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, 3]
    nu_eff = jnp.asarray(nu_laminar + np.maximum(0.0, nu_tilde))
    grad_nu_tilde = jnp.asarray(gradients[:, :, 3, :])
    
    Si_x = jnp.asarray(grid_metrics.Si_x)
    Si_y = jnp.asarray(grid_metrics.Si_y)
    Sj_x = jnp.asarray(grid_metrics.Sj_x)
    Sj_y = jnp.asarray(grid_metrics.Sj_y)
    
    result = compute_nu_tilde_diffusion_jax(
        grad_nu_tilde, Si_x, Si_y, Sj_x, Sj_y, nu_eff, sigma
    )
    
    return np.asarray(result)


# =============================================================================
# JAX Implementation
# =============================================================================

@jax.jit
def compute_viscous_fluxes_jax(grad, Si_x, Si_y, Sj_x, Sj_y, mu_eff):
    """
    JAX: Compute viscous flux residuals.
    
    Parameters
    ----------
    grad : jnp.ndarray
        Velocity gradients (NI, NJ, 4, 2).
    Si_x, Si_y : jnp.ndarray
        I-face normal vectors (NI+1, NJ).
    Sj_x, Sj_y : jnp.ndarray
        J-face normal vectors (NI, NJ+1).
    mu_eff : jnp.ndarray
        Effective viscosity (NI, NJ).
        
    Returns
    -------
    residual : jnp.ndarray
        Viscous flux residual (NI, NJ, 4).
    """
    NI, NJ = mu_eff.shape
    
    # I-direction faces
    S_i = jnp.sqrt(Si_x**2 + Si_y**2)  # (NI+1, NJ)
    nx_i = Si_x / (S_i + 1e-12)
    ny_i = Si_y / (S_i + 1e-12)
    
    # Pad gradients for boundary handling
    grad_padded = jnp.pad(grad, ((1, 1), (0, 0), (0, 0), (0, 0)), mode='edge')
    mu_padded = jnp.pad(mu_eff, ((1, 1), (0, 0)), mode='edge')
    
    # Face-averaged gradients and viscosity (interior faces: 1 to NI-1)
    # For face i: average of cells i-1 and i (in padded coords: i and i+1)
    grad_face_i = 0.5 * (grad_padded[:-1, :, :, :] + grad_padded[1:, :, :, :])  # (NI+1, NJ, 4, 2)
    mu_face_i = 0.5 * (mu_padded[:-1, :] + mu_padded[1:, :])  # (NI+1, NJ)
    
    dudx_i = grad_face_i[:, :, 1, 0]
    dudy_i = grad_face_i[:, :, 1, 1]
    dvdx_i = grad_face_i[:, :, 2, 0]
    dvdy_i = grad_face_i[:, :, 2, 1]
    
    tau_xx_i = 2.0 * mu_face_i * dudx_i
    tau_yy_i = 2.0 * mu_face_i * dvdy_i
    tau_xy_i = mu_face_i * (dudy_i + dvdx_i)
    
    flux_u_i = (tau_xx_i * nx_i + tau_xy_i * ny_i) * S_i
    flux_v_i = (tau_xy_i * nx_i + tau_yy_i * ny_i) * S_i
    
    # J-direction faces
    S_j = jnp.sqrt(Sj_x**2 + Sj_y**2)  # (NI, NJ+1)
    nx_j = Sj_x / (S_j + 1e-12)
    ny_j = Sj_y / (S_j + 1e-12)
    
    grad_padded_j = jnp.pad(grad, ((0, 0), (1, 1), (0, 0), (0, 0)), mode='edge')
    mu_padded_j = jnp.pad(mu_eff, ((0, 0), (1, 1)), mode='edge')
    
    grad_face_j = 0.5 * (grad_padded_j[:, :-1, :, :] + grad_padded_j[:, 1:, :, :])
    mu_face_j = 0.5 * (mu_padded_j[:, :-1] + mu_padded_j[:, 1:])
    
    dudx_j = grad_face_j[:, :, 1, 0]
    dudy_j = grad_face_j[:, :, 1, 1]
    dvdx_j = grad_face_j[:, :, 2, 0]
    dvdy_j = grad_face_j[:, :, 2, 1]
    
    tau_xx_j = 2.0 * mu_face_j * dudx_j
    tau_yy_j = 2.0 * mu_face_j * dvdy_j
    tau_xy_j = mu_face_j * (dudy_j + dvdx_j)
    
    flux_u_j = (tau_xx_j * nx_j + tau_xy_j * ny_j) * S_j
    flux_v_j = (tau_xy_j * nx_j + tau_yy_j * ny_j) * S_j
    
    # Compute residual (flux difference)
    # For cell m: gets +flux from face m+1 (right), -flux from face m (left)
    # So res[m] = flux[m+1] - flux[m]
    res_u_i = flux_u_i[1:, :] - flux_u_i[:-1, :]
    res_v_i = flux_v_i[1:, :] - flux_v_i[:-1, :]
    res_u_j = flux_u_j[:, 1:] - flux_u_j[:, :-1]
    res_v_j = flux_v_j[:, 1:] - flux_v_j[:, :-1]
    
    # Assemble residual
    residual = jnp.zeros((NI, NJ, 4))
    residual = residual.at[:, :, 1].set(res_u_i + res_u_j)
    residual = residual.at[:, :, 2].set(res_v_i + res_v_j)
    
    return residual


@jax.jit
def compute_nu_tilde_diffusion_jax(grad_nu_tilde, Si_x, Si_y, Sj_x, Sj_y, 
                                    nu_eff, sigma=2.0/3.0):
    """
    JAX: Compute SA nu_tilde diffusion term.
    
    (1/σ)∇·[(ν + ν̃)∇ν̃]
    
    Parameters
    ----------
    grad_nu_tilde : jnp.ndarray
        Gradient of nu_tilde (NI, NJ, 2).
    Si_x, Si_y : jnp.ndarray
        I-face normals (NI+1, NJ).
    Sj_x, Sj_y : jnp.ndarray
        J-face normals (NI, NJ+1).
    nu_eff : jnp.ndarray
        Effective diffusivity (NI, NJ).
    sigma : float
        SA model constant.
        
    Returns
    -------
    residual : jnp.ndarray
        Diffusion residual (NI, NJ).
    """
    NI, NJ = nu_eff.shape
    coeff = 1.0 / sigma
    
    # I-direction
    S_i = jnp.sqrt(Si_x**2 + Si_y**2)
    nx_i = Si_x / (S_i + 1e-12)
    ny_i = Si_y / (S_i + 1e-12)
    
    grad_padded = jnp.pad(grad_nu_tilde, ((1, 1), (0, 0), (0, 0)), mode='edge')
    nu_padded = jnp.pad(nu_eff, ((1, 1), (0, 0)), mode='edge')
    
    grad_face_i = 0.5 * (grad_padded[:-1, :, :] + grad_padded[1:, :, :])
    nu_face_i = 0.5 * (nu_padded[:-1, :] + nu_padded[1:, :])
    
    flux_i = coeff * nu_face_i * (grad_face_i[:, :, 0] * nx_i + 
                                   grad_face_i[:, :, 1] * ny_i) * S_i
    
    # J-direction
    S_j = jnp.sqrt(Sj_x**2 + Sj_y**2)
    nx_j = Sj_x / (S_j + 1e-12)
    ny_j = Sj_y / (S_j + 1e-12)
    
    grad_padded_j = jnp.pad(grad_nu_tilde, ((0, 0), (1, 1), (0, 0)), mode='edge')
    nu_padded_j = jnp.pad(nu_eff, ((0, 0), (1, 1)), mode='edge')
    
    grad_face_j = 0.5 * (grad_padded_j[:, :-1, :] + grad_padded_j[:, 1:, :])
    nu_face_j = 0.5 * (nu_padded_j[:, :-1] + nu_padded_j[:, 1:])
    
    flux_j = coeff * nu_face_j * (grad_face_j[:, :, 0] * nx_j + 
                                   grad_face_j[:, :, 1] * ny_j) * S_j
    
    # Residual
    res_i = flux_i[:-1, :] - flux_i[1:, :]
    res_j = flux_j[:, :-1] - flux_j[:, 1:]
    
    return res_i + res_j
