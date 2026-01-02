"""
Viscous flux computation for 2D incompressible RANS.

Stress tensor (incompressible): τ_xx = 2μ·∂u/∂x, τ_yy = 2μ·∂v/∂y, τ_xy = μ·(∂u/∂y + ∂v/∂x)
"""

import numpy as np
import numpy.typing as npt
from src.constants import NGHOST
from numba import njit
from .gradients import GradientMetrics

# JAX imports
try:
    from src.physics.jax_config import jax, jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

NDArrayFloat = npt.NDArray[np.floating]


@njit(cache=True, fastmath=True)
def _viscous_flux_kernel(
    Q: np.ndarray,
    grad: np.ndarray,
    Si_x: np.ndarray, Si_y: np.ndarray,
    Sj_x: np.ndarray, Sj_y: np.ndarray,
    mu_eff: np.ndarray,
    residual: np.ndarray
) -> None:
    """Numba kernel for viscous flux computation."""
    NI_faces, NJ = Si_x.shape
    NI = NI_faces - 1
    _, NJ_faces = Sj_x.shape
    
    # I-direction faces
    for i in range(NI_faces):
        for j in range(NJ):
            Sx = Si_x[i, j]
            Sy = Si_y[i, j]
            area = np.sqrt(Sx * Sx + Sy * Sy)
            
            if area < 1e-20:
                continue
                
            nx = Sx / area
            ny = Sy / area
            
            iL = i - 1
            iR = i
            
            if iL < 0:
                dudx = grad[iR, j, 1, 0]
                dudy = grad[iR, j, 1, 1]
                dvdx = grad[iR, j, 2, 0]
                dvdy = grad[iR, j, 2, 1]
                mu = mu_eff[iR, j]
            elif iR >= NI:
                dudx = grad[iL, j, 1, 0]
                dudy = grad[iL, j, 1, 1]
                dvdx = grad[iL, j, 2, 0]
                dvdy = grad[iL, j, 2, 1]
                mu = mu_eff[iL, j]
            else:
                dudx = 0.5 * (grad[iL, j, 1, 0] + grad[iR, j, 1, 0])
                dudy = 0.5 * (grad[iL, j, 1, 1] + grad[iR, j, 1, 1])
                dvdx = 0.5 * (grad[iL, j, 2, 0] + grad[iR, j, 2, 0])
                dvdy = 0.5 * (grad[iL, j, 2, 1] + grad[iR, j, 2, 1])
                mu = 0.5 * (mu_eff[iL, j] + mu_eff[iR, j])
            
            tau_xx = 2.0 * mu * dudx
            tau_yy = 2.0 * mu * dvdy
            tau_xy = mu * (dudy + dvdx)
            
            flux_u = (tau_xx * nx + tau_xy * ny) * area
            flux_v = (tau_xy * nx + tau_yy * ny) * area
            
            if iL >= 0:
                residual[iL, j, 1] += flux_u
                residual[iL, j, 2] += flux_v
            if iR < NI:
                residual[iR, j, 1] -= flux_u
                residual[iR, j, 2] -= flux_v
    
    # J-direction faces
    for i in range(NI):
        for j in range(NJ_faces):
            Sx = Sj_x[i, j]
            Sy = Sj_y[i, j]
            area = np.sqrt(Sx * Sx + Sy * Sy)
            
            if area < 1e-20:
                continue
                
            nx = Sx / area
            ny = Sy / area
            
            jL = j - 1
            jR = j
            
            if jL < 0:
                dudx = grad[i, jR, 1, 0]
                dudy = grad[i, jR, 1, 1]
                dvdx = grad[i, jR, 2, 0]
                dvdy = grad[i, jR, 2, 1]
                mu = mu_eff[i, jR]
            elif jR >= NJ:
                dudx = grad[i, jL, 1, 0]
                dudy = grad[i, jL, 1, 1]
                dvdx = grad[i, jL, 2, 0]
                dvdy = grad[i, jL, 2, 1]
                mu = mu_eff[i, jL]
            else:
                dudx = 0.5 * (grad[i, jL, 1, 0] + grad[i, jR, 1, 0])
                dudy = 0.5 * (grad[i, jL, 1, 1] + grad[i, jR, 1, 1])
                dvdx = 0.5 * (grad[i, jL, 2, 0] + grad[i, jR, 2, 0])
                dvdy = 0.5 * (grad[i, jL, 2, 1] + grad[i, jR, 2, 1])
                mu = 0.5 * (mu_eff[i, jL] + mu_eff[i, jR])
            
            tau_xx = 2.0 * mu * dudx
            tau_yy = 2.0 * mu * dvdy
            tau_xy = mu * (dudy + dvdx)
            
            flux_u = (tau_xx * nx + tau_xy * ny) * area
            flux_v = (tau_xy * nx + tau_yy * ny) * area
            
            if jL >= 0:
                residual[i, jL, 1] += flux_u
                residual[i, jL, 2] += flux_v
            if jR < NJ:
                residual[i, jR, 1] -= flux_u
                residual[i, jR, 2] -= flux_v


def compute_viscous_fluxes(
    Q: NDArrayFloat,
    gradients: NDArrayFloat,
    grid_metrics: GradientMetrics,
    mu_laminar: float,
    mu_turbulent: NDArrayFloat | None = None
) -> NDArrayFloat:
    """Compute viscous flux residuals."""
    NI: int
    NJ: int
    NI, NJ = gradients.shape[:2]
    residual: NDArrayFloat = np.zeros((NI, NJ, 4), dtype=np.float64)
    
    mu_eff: NDArrayFloat
    if mu_turbulent is None:
        mu_eff = np.full((NI, NJ), mu_laminar, dtype=np.float64)
    else:
        mu_eff = mu_laminar + mu_turbulent
    
    _viscous_flux_kernel(
        Q, gradients,
        grid_metrics.Si_x, grid_metrics.Si_y,
        grid_metrics.Sj_x, grid_metrics.Sj_y,
        mu_eff,
        residual
    )
    
    return residual


def add_viscous_fluxes(
    convective_residual: NDArrayFloat,
    Q: NDArrayFloat,
    gradients: NDArrayFloat,
    grid_metrics: GradientMetrics,
    mu_laminar: float,
    mu_turbulent: NDArrayFloat | None = None
) -> NDArrayFloat:
    """Add viscous fluxes to convective residual."""
    viscous_residual: NDArrayFloat = compute_viscous_fluxes(
        Q, gradients, grid_metrics, mu_laminar, mu_turbulent
    )
    
    return convective_residual + viscous_residual


@njit(cache=True, fastmath=True)
def compute_nu_tilde_diffusion_kernel(
    nu_tilde: np.ndarray,
    grad_nu_tilde: np.ndarray,
    Si_x: np.ndarray, Si_y: np.ndarray,
    Sj_x: np.ndarray, Sj_y: np.ndarray,
    nu_eff: np.ndarray,
    sigma: float,
    residual: np.ndarray
) -> None:
    """Compute SA diffusion term: (1/σ)∇·[(ν + ν̃)∇ν̃]"""
    NI_faces, NJ = Si_x.shape
    NI = NI_faces - 1
    _, NJ_faces = Sj_x.shape
    
    coeff = 1.0 / sigma
    
    for i in range(NI_faces):
        for j in range(NJ):
            Sx = Si_x[i, j]
            Sy = Si_y[i, j]
            area = np.sqrt(Sx * Sx + Sy * Sy)
            
            if area < 1e-20:
                continue
                
            nx = Sx / area
            ny = Sy / area
            
            iL = i - 1
            iR = i
            
            if iL < 0:
                dnu_dx = grad_nu_tilde[iR, j, 0]
                dnu_dy = grad_nu_tilde[iR, j, 1]
                nu = nu_eff[iR, j]
            elif iR >= NI:
                dnu_dx = grad_nu_tilde[iL, j, 0]
                dnu_dy = grad_nu_tilde[iL, j, 1]
                nu = nu_eff[iL, j]
            else:
                dnu_dx = 0.5 * (grad_nu_tilde[iL, j, 0] + grad_nu_tilde[iR, j, 0])
                dnu_dy = 0.5 * (grad_nu_tilde[iL, j, 1] + grad_nu_tilde[iR, j, 1])
                nu = 0.5 * (nu_eff[iL, j] + nu_eff[iR, j])
            
            flux = coeff * nu * (dnu_dx * nx + dnu_dy * ny) * area
            
            if iL >= 0:
                residual[iL, j] += flux
            if iR < NI:
                residual[iR, j] -= flux
    
    for i in range(NI):
        for j in range(NJ_faces):
            Sx = Sj_x[i, j]
            Sy = Sj_y[i, j]
            area = np.sqrt(Sx * Sx + Sy * Sy)
            
            if area < 1e-20:
                continue
                
            nx = Sx / area
            ny = Sy / area
            
            jL = j - 1
            jR = j
            
            if jL < 0:
                dnu_dx = grad_nu_tilde[i, jR, 0]
                dnu_dy = grad_nu_tilde[i, jR, 1]
                nu = nu_eff[i, jR]
            elif jR >= NJ:
                dnu_dx = grad_nu_tilde[i, jL, 0]
                dnu_dy = grad_nu_tilde[i, jL, 1]
                nu = nu_eff[i, jL]
            else:
                dnu_dx = 0.5 * (grad_nu_tilde[i, jL, 0] + grad_nu_tilde[i, jR, 0])
                dnu_dy = 0.5 * (grad_nu_tilde[i, jL, 1] + grad_nu_tilde[i, jR, 1])
                nu = 0.5 * (nu_eff[i, jL] + nu_eff[i, jR])
            
            flux = coeff * nu * (dnu_dx * nx + dnu_dy * ny) * area
            
            if jL >= 0:
                residual[i, jL] += flux
            if jR < NJ:
                residual[i, jR] -= flux


def compute_nu_tilde_diffusion(
    Q: NDArrayFloat,
    gradients: NDArrayFloat,
    grid_metrics: GradientMetrics,
    nu_laminar: float,
    sigma: float = 2.0/3.0
) -> NDArrayFloat:
    """Compute SA nu_tilde diffusion residual."""
    NI: int
    NJ: int
    NI, NJ = gradients.shape[:2]
    
    nu_tilde: NDArrayFloat = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, 3]
    nu_eff: NDArrayFloat = nu_laminar + np.maximum(0.0, nu_tilde)
    grad_nu_tilde: NDArrayFloat = gradients[:, :, 3, :]
    
    residual: NDArrayFloat = np.zeros((NI, NJ), dtype=np.float64)
    
    compute_nu_tilde_diffusion_kernel(
        Q[:, :, 3],
        grad_nu_tilde,
        grid_metrics.Si_x, grid_metrics.Si_y,
        grid_metrics.Sj_x, grid_metrics.Sj_y,
        nu_eff,
        sigma,
        residual
    )
    
    return residual


# =============================================================================
# JAX Implementations
# =============================================================================

if JAX_AVAILABLE:
    
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
