"""
Viscous Flux Computation for 2D Incompressible RANS.

This module computes the diffusive (viscous) flux contributions to the
momentum equations using a cell-centered finite volume formulation.

For incompressible flow:
    Viscous stress tensor:
        τ_xx = 2μ ∂u/∂x
        τ_yy = 2μ ∂v/∂y
        τ_xy = μ (∂u/∂y + ∂v/∂x)

    The viscous flux through a face with normal (nx, ny) and area S:
        F_visc_u = (τ_xx * nx + τ_xy * ny) * S
        F_visc_v = (τ_xy * nx + τ_yy * ny) * S

Note: For incompressible flow, div(u) = 0, so no bulk viscosity term.
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
    """
    Numba-optimized kernel for viscous flux computation.
    
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector [p, u, v, nu_t] with ghost cells.
    grad : ndarray, shape (NI, NJ, 4, 2)
        Gradients [dp/d*, du/d*, dv/d*, dnu_t/d*] where * = x, y.
    Si_x, Si_y : ndarray, shape (NI+1, NJ)
        I-face area-weighted normals.
    Sj_x, Sj_y : ndarray, shape (NI, NJ+1)
        J-face area-weighted normals.
    mu_eff : ndarray, shape (NI, NJ)
        Effective (total) viscosity at cell centers: mu_lam + mu_turb.
    residual : ndarray, shape (NI, NJ, 4)
        Output: viscous flux residual (accumulated).
        
    Notes
    -----
    The residual is ADDED to (not replaced), so caller can accumulate
    with convective fluxes.
    
    Sign convention: Positive residual = rate of increase.
    Viscous fluxes are diffusive, so they reduce gradients (dissipate).
    """
    NI_faces, NJ = Si_x.shape
    NI = NI_faces - 1
    _, NJ_faces = Sj_x.shape
    
    # =========================================================================
    # I-direction faces (vertical faces between i and i+1)
    # =========================================================================
    for i in range(NI_faces):
        for j in range(NJ):
            # Face area-weighted normal
            Sx = Si_x[i, j]
            Sy = Si_y[i, j]
            area = np.sqrt(Sx * Sx + Sy * Sy)
            
            if area < 1e-20:
                continue
                
            # Unit normal
            nx = Sx / area
            ny = Sy / area
            
            # Left and right cell indices (in gradient array, 0-indexed)
            iL = i - 1  # Left cell
            iR = i      # Right cell
            
            # Average gradients to face
            # Handle boundary faces
            if iL < 0:
                # Left boundary: use right cell gradients only
                dudx = grad[iR, j, 1, 0]
                dudy = grad[iR, j, 1, 1]
                dvdx = grad[iR, j, 2, 0]
                dvdy = grad[iR, j, 2, 1]
                mu = mu_eff[iR, j]
            elif iR >= NI:
                # Right boundary: use left cell gradients only
                dudx = grad[iL, j, 1, 0]
                dudy = grad[iL, j, 1, 1]
                dvdx = grad[iL, j, 2, 0]
                dvdy = grad[iL, j, 2, 1]
                mu = mu_eff[iL, j]
            else:
                # Interior face: average
                dudx = 0.5 * (grad[iL, j, 1, 0] + grad[iR, j, 1, 0])
                dudy = 0.5 * (grad[iL, j, 1, 1] + grad[iR, j, 1, 1])
                dvdx = 0.5 * (grad[iL, j, 2, 0] + grad[iR, j, 2, 0])
                dvdy = 0.5 * (grad[iL, j, 2, 1] + grad[iR, j, 2, 1])
                mu = 0.5 * (mu_eff[iL, j] + mu_eff[iR, j])
            
            # Stress tensor components (incompressible, no bulk viscosity)
            tau_xx = 2.0 * mu * dudx
            tau_yy = 2.0 * mu * dvdy
            tau_xy = mu * (dudy + dvdx)
            
            # Viscous flux projected onto face normal (times area)
            # F_visc = τ · n * S
            flux_u = (tau_xx * nx + tau_xy * ny) * area
            flux_v = (tau_xy * nx + tau_yy * ny) * area
            
            # Add to residuals
            # Viscous flux is a diffusive term: it appears with POSITIVE sign
            # in the momentum equation (reduces velocity gradients)
            # dQ/dt = ... + div(τ)
            # In FVM: Res = ∮ τ·n dS
            # Left cell gains, right cell loses (flux goes from L to R)
            if iL >= 0:
                residual[iL, j, 1] += flux_u
                residual[iL, j, 2] += flux_v
            if iR < NI:
                residual[iR, j, 1] -= flux_u
                residual[iR, j, 2] -= flux_v
    
    # =========================================================================
    # J-direction faces (horizontal faces between j and j+1)
    # =========================================================================
    for i in range(NI):
        for j in range(NJ_faces):
            # Face area-weighted normal
            Sx = Sj_x[i, j]
            Sy = Sj_y[i, j]
            area = np.sqrt(Sx * Sx + Sy * Sy)
            
            if area < 1e-20:
                continue
                
            # Unit normal
            nx = Sx / area
            ny = Sy / area
            
            # Left and right cell indices (in gradient array)
            jL = j - 1  # "Left" = lower j
            jR = j      # "Right" = higher j
            
            # Average gradients to face
            if jL < 0:
                # Bottom boundary
                dudx = grad[i, jR, 1, 0]
                dudy = grad[i, jR, 1, 1]
                dvdx = grad[i, jR, 2, 0]
                dvdy = grad[i, jR, 2, 1]
                mu = mu_eff[i, jR]
            elif jR >= NJ:
                # Top boundary
                dudx = grad[i, jL, 1, 0]
                dudy = grad[i, jL, 1, 1]
                dvdx = grad[i, jL, 2, 0]
                dvdy = grad[i, jL, 2, 1]
                mu = mu_eff[i, jL]
            else:
                # Interior face
                dudx = 0.5 * (grad[i, jL, 1, 0] + grad[i, jR, 1, 0])
                dudy = 0.5 * (grad[i, jL, 1, 1] + grad[i, jR, 1, 1])
                dvdx = 0.5 * (grad[i, jL, 2, 0] + grad[i, jR, 2, 0])
                dvdy = 0.5 * (grad[i, jL, 2, 1] + grad[i, jR, 2, 1])
                mu = 0.5 * (mu_eff[i, jL] + mu_eff[i, jR])
            
            # Stress tensor
            tau_xx = 2.0 * mu * dudx
            tau_yy = 2.0 * mu * dvdy
            tau_xy = mu * (dudy + dvdx)
            
            # Viscous flux
            flux_u = (tau_xx * nx + tau_xy * ny) * area
            flux_v = (tau_xy * nx + tau_yy * ny) * area
            
            # Add to residuals
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
    """
    Compute viscous flux residuals for 2D incompressible RANS.
    
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector [p, u, v, nu_t] with ghost cells.
    gradients : ndarray, shape (NI, NJ, 4, 2)
        Gradients from compute_gradients().
    grid_metrics : GradientMetrics
        Grid metrics containing Si, Sj, volume.
    mu_laminar : float
        Laminar (molecular) dynamic viscosity.
    mu_turbulent : ndarray, shape (NI, NJ), optional
        Turbulent eddy viscosity at cell centers.
        If None, pure laminar flow is assumed.
        
    Returns
    -------
    residual : ndarray, shape (NI, NJ, 4)
        Viscous flux contribution to residual.
        
    Notes
    -----
    The returned residual should be ADDED to the convective residual:
        total_residual = convective_residual + viscous_residual
        
    The pressure equation (index 0) has no viscous flux.
    The nu_t equation (index 3) has its own diffusion handled separately
    by the turbulence model.
    """
    NI, NJ = gradients.shape[:2]
    
    # Initialize residual array
    residual = np.zeros((NI, NJ, 4), dtype=np.float64)
    
    # Compute effective viscosity
    if mu_turbulent is None:
        mu_eff = np.full((NI, NJ), mu_laminar, dtype=np.float64)
    else:
        mu_eff = mu_laminar + mu_turbulent
    
    # Call the Numba kernel
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
    """
    Convenience function to add viscous fluxes to convective residual.
    
    Parameters
    ----------
    convective_residual : ndarray, shape (NI, NJ, 4)
        Residual from convective flux computation.
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector with ghost cells.
    gradients : ndarray, shape (NI, NJ, 4, 2)
        Velocity gradients.
    grid_metrics : GradientMetrics
        Grid metrics.
    mu_laminar : float
        Laminar viscosity.
    mu_turbulent : ndarray, optional
        Turbulent viscosity field.
        
    Returns
    -------
    total_residual : ndarray, shape (NI, NJ, 4)
        Combined convective + viscous residual.
    """
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
    """
    Compute diffusion term for nu_tilde (SA turbulence variable).
    
    The SA diffusion term is:
        (1/σ) ∇·[(ν + ν̃) ∇ν̃]
        
    Parameters
    ----------
    nu_tilde : ndarray, shape (NI+2, NJ+2)
        SA working variable with ghost cells.
    grad_nu_tilde : ndarray, shape (NI, NJ, 2)
        Gradient of nu_tilde: [d/dx, d/dy].
    Si_x, Si_y : ndarray
        I-face normals.
    Sj_x, Sj_y : ndarray
        J-face normals.
    nu_eff : ndarray, shape (NI, NJ)
        Effective diffusivity: ν + max(0, ν̃).
    sigma : float
        SA model constant (typically 2/3).
    residual : ndarray, shape (NI, NJ)
        Output: diffusion contribution to nu_tilde residual.
    """
    NI_faces, NJ = Si_x.shape
    NI = NI_faces - 1
    _, NJ_faces = Sj_x.shape
    
    coeff = 1.0 / sigma
    
    # I-faces
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
            
            # Diffusion flux: (ν_eff / σ) * ∇ν̃ · n * S
            flux = coeff * nu * (dnu_dx * nx + dnu_dy * ny) * area
            
            if iL >= 0:
                residual[iL, j] += flux
            if iR < NI:
                residual[iR, j] -= flux
    
    # J-faces
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
    """
    Compute diffusion residual for SA nu_tilde equation.
    
    Parameters
    ----------
    Q : ndarray, shape (NI+2, NJ+2, 4)
        State vector with ghost cells.
    gradients : ndarray, shape (NI, NJ, 4, 2)
        State gradients.
    grid_metrics : GradientMetrics
        Grid metrics.
    nu_laminar : float
        Laminar kinematic viscosity.
    sigma : float
        SA diffusion coefficient (default 2/3).
        
    Returns
    -------
    residual : ndarray, shape (NI, NJ)
        Diffusion contribution to nu_tilde equation.
    """
    NI, NJ = gradients.shape[:2]
    
    # Extract nu_tilde from Q (interior cells with 2 J-ghosts at wall)
    nu_tilde = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, 3]
    
    # Effective diffusivity: ν + max(0, ν̃)
    # This ensures non-negative diffusivity even if nu_tilde < 0
    nu_eff = nu_laminar + np.maximum(0.0, nu_tilde)
    
    # Extract gradient of nu_tilde
    grad_nu_tilde = gradients[:, :, 3, :]
    
    # Initialize residual
    residual = np.zeros((NI, NJ), dtype=np.float64)
    
    # Compute diffusion
    compute_nu_tilde_diffusion_kernel(
        Q[:, :, 3],  # Full nu_tilde field with ghosts
        grad_nu_tilde,
        grid_metrics.Si_x, grid_metrics.Si_y,
        grid_metrics.Sj_x, grid_metrics.Sj_y,
        nu_eff,
        sigma,
        residual
    )
    
    return residual

