"""
Viscous flux computation for 2D incompressible RANS.

Stress tensor (incompressible): τ_xx = 2μ·∂u/∂x, τ_yy = 2μ·∂v/∂y, τ_xy = μ·(∂u/∂y + ∂v/∂x)
"""

import numpy as np
from src.constants import NGHOST
from numba import njit
from .gradients import GradientMetrics


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
    Q: np.ndarray,
    gradients: np.ndarray,
    grid_metrics: GradientMetrics,
    mu_laminar: float,
    mu_turbulent: np.ndarray = None
) -> np.ndarray:
    """Compute viscous flux residuals."""
    NI, NJ = gradients.shape[:2]
    residual = np.zeros((NI, NJ, 4), dtype=np.float64)
    
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
    convective_residual: np.ndarray,
    Q: np.ndarray,
    gradients: np.ndarray,
    grid_metrics: GradientMetrics,
    mu_laminar: float,
    mu_turbulent: np.ndarray = None
) -> np.ndarray:
    """Add viscous fluxes to convective residual."""
    viscous_residual = compute_viscous_fluxes(
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
    Q: np.ndarray,
    gradients: np.ndarray,
    grid_metrics: GradientMetrics,
    nu_laminar: float,
    sigma: float = 2.0/3.0
) -> np.ndarray:
    """Compute SA nu_tilde diffusion residual."""
    NI, NJ = gradients.shape[:2]
    
    nu_tilde = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, 3]
    nu_eff = nu_laminar + np.maximum(0.0, nu_tilde)
    grad_nu_tilde = gradients[:, :, 3, :]
    
    residual = np.zeros((NI, NJ), dtype=np.float64)
    
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
