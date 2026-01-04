"""
Viscous flux computation for 2D incompressible RANS.

================================================================================
FINITE VOLUME FORMULATION
================================================================================

Both momentum and SA diffusion terms are implemented with viscosity INSIDE
the divergence, OUTSIDE the gradient. This is the standard FV formulation.

MOMENTUM EQUATIONS:
    ∇·(μ_eff ∇u) = ∫_faces μ_face · (∇u·n) · dS
    
    where μ_eff = μ + μ_t (laminar + turbulent)
    
    Stress tensor (incompressible):
        τ_xx = 2μ·∂u/∂x
        τ_yy = 2μ·∂v/∂y  
        τ_xy = μ·(∂u/∂y + ∂v/∂x)
    
    Flux through face: F = (τ·n) · S

SA DIFFUSION:
    (1/σ)∇·[(ν + ν̃)∇ν̃] = (1/σ) ∫_faces (ν + ν̃)_face · (∇ν̃·n) · dS
    
    Note: The cb2 term (cb2/σ)|∇ν̃|² is added separately as a source term.
    
    When (ν + ν̃) is inside the divergence, the product rule gives:
        ∇·[(ν + ν̃)∇ν̃] = (ν + ν̃)∇²ν̃ + |∇ν̃|²
    
    So the FV formulation naturally produces a |∇ν̃|² contribution.
    The cb2 term adds an additional cb2·|∇ν̃|² for a total of (1+cb2)|∇ν̃|².

================================================================================
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
    JAX: Compute SA nu_tilde diffusion term using finite volume formulation.
    
    FORMULATION:
        (1/σ) ∇·[(ν + ν̃)∇ν̃]
    
    This is implemented as:
        (1/σ) ∑_faces [(ν + ν̃)_face · (∇ν̃·n) · S]
    
    The (ν + ν̃) is INSIDE the divergence, evaluated at face centers.
    This is the standard SA formulation and naturally produces a |∇ν̃|²
    contribution when the product rule is applied.
    
    Note: The cb2 term is NOT included here - it's added separately as
    a source term in compute_sa_source_jax.
    
    Parameters
    ----------
    grad_nu_tilde : jnp.ndarray
        Gradient of nu_tilde (NI, NJ, 2).
    Si_x, Si_y : jnp.ndarray
        I-face normals (NI+1, NJ).
    Sj_x, Sj_y : jnp.ndarray
        J-face normals (NI, NJ+1).
    nu_eff : jnp.ndarray
        Effective diffusivity (ν + ν̃), NOT divided by σ (NI, NJ).
    sigma : float
        SA model constant (default 2/3).
        
    Returns
    -------
    residual : jnp.ndarray
        Diffusion residual (NI, NJ), representing ∫∇·[(ν + ν̃)/σ ∇ν̃] dV.
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
    
    # Residual (flux into cell = flux[m+1] - flux[m])
    res_i = flux_i[1:, :] - flux_i[:-1, :]
    res_j = flux_j[:, 1:] - flux_j[:, :-1]
    
    return res_i + res_j


@jax.jit
def compute_viscous_fluxes_with_sa_jax(grad, Si_x, Si_y, Sj_x, Sj_y, 
                                        mu_eff, nu_laminar, nuHat, sigma=2.0/3.0):
    """
    JAX: Compute viscous flux residuals including SA diffusion.
    
    Computes:
    - Momentum viscous fluxes (indices 1, 2) using mu_eff
    - SA diffusion (index 3): (1/σ)∇·[(ν + ν̃)∇ν̃]
    
    Parameters
    ----------
    grad : jnp.ndarray
        Gradients of Q (NI, NJ, 4, 2).
    Si_x, Si_y : jnp.ndarray
        I-face normal vectors (NI+1, NJ).
    Sj_x, Sj_y : jnp.ndarray
        J-face normal vectors (NI, NJ+1).
    mu_eff : jnp.ndarray
        Effective viscosity for momentum (NI, NJ).
    nu_laminar : float
        Laminar kinematic viscosity.
    nuHat : jnp.ndarray
        SA working variable (NI, NJ).
    sigma : float
        SA model constant (default 2/3).
        
    Returns
    -------
    residual : jnp.ndarray
        Viscous flux residual (NI, NJ, 4).
    """
    # Compute momentum viscous fluxes (indices 1, 2)
    residual = compute_viscous_fluxes_jax(grad, Si_x, Si_y, Sj_x, Sj_y, mu_eff)
    
    # SA diffusion coefficient: (ν + max(0, ν̃)) / σ
    # Note: the 1/σ factor is applied inside compute_nu_tilde_diffusion_jax
    nuHat_safe = jnp.maximum(nuHat, 0.0)
    nu_eff_sa = nu_laminar + nuHat_safe
    
    # Compute SA diffusion
    grad_nuHat = grad[:, :, 3, :]  # (NI, NJ, 2)
    sa_diff = compute_nu_tilde_diffusion_jax(
        grad_nuHat, Si_x, Si_y, Sj_x, Sj_y, nu_eff_sa, sigma
    )
    
    # Add SA diffusion to index 3
    residual = residual.at[:, :, 3].set(sa_diff)
    
    return residual


# =============================================================================
# Tight-Stencil Viscous Flux Implementation
# =============================================================================

@jax.jit
def compute_viscous_fluxes_tight_jax(
    Q,           # (NI, NJ, 4) cell-centered state
    Si_x, Si_y,  # (NI+1, NJ) I-face normal components
    Sj_x, Sj_y,  # (NI, NJ+1) J-face normal components
    d_coord_i, e_coord_i_x, e_coord_i_y, e_ortho_i_x, e_ortho_i_y,  # I-face geometry
    d_coord_j, e_coord_j_x, e_coord_j_y, e_ortho_j_x, e_ortho_j_y,  # J-face geometry
    ls_weights_i,  # (NI+1, NJ, 6) LS weights for I-faces
    ls_weights_j,  # (NI, NJ+1, 6) LS weights for J-faces
    mu_eff,        # (NI, NJ) effective viscosity for momentum
    nu_laminar,    # scalar laminar kinematic viscosity
    nuHat,         # (NI, NJ) SA working variable
    sigma=2.0/3.0  # SA model constant
):
    """
    Unified tight-stencil viscous/diffusion flux for all 4 variables.
    
    FORMULATION:
    -----------
    All diffusion terms have viscosity INSIDE the divergence:
    
    Momentum: ∇·(μ_eff ∇u) where μ_eff = μ + μ_t
    SA:       (1/σ) ∇·[(ν + ν̃)∇ν̃]
    
    Viscosity is evaluated at face centers (averaged from cells) and
    multiplied by the face-normal gradient to compute the flux.
    
    NUMERICAL METHOD:
    ----------------
    Uses two-point difference along coordinate line (tight stencil for diagonal
    dominance) plus least-squares correction for non-orthogonality.
    
    For each face:
    1. Two-point difference: grad_coord = (phi_R - phi_L) / d_coord
    2. LS orthogonal correction: grad_ortho = sum(w_k * phi_k)
    3. Combine: grad_x = grad_coord * e_coord_x + grad_ortho * e_ortho_x
                grad_y = grad_coord * e_coord_y + grad_ortho * e_ortho_y
    4. Compute flux: F = μ_face · (∇φ·n) · S
    
    VARIABLES:
    - Index 0 (p): no diffusion
    - Index 1,2 (u,v): stress tensor with μ_eff INSIDE divergence
    - Index 3 (nuHat): SA diffusion with (ν + ν̃)/σ INSIDE divergence
    
    Note: The cb2 term for SA is NOT included here - it's added as a source.
    
    Parameters
    ----------
    Q : (NI, NJ, 4) interior cell values
    Si_x, Si_y : I-face normals (NI+1, NJ)
    Sj_x, Sj_y : J-face normals (NI, NJ+1)
    d_coord_*, e_coord_*, e_ortho_* : face geometry from MetricComputer
    ls_weights_i, ls_weights_j : LS weights from MetricComputer
    mu_eff : effective viscosity (μ + μ_t) for momentum
    nu_laminar : laminar kinematic viscosity ν
    nuHat : SA working variable ν̃
    sigma : SA model constant (default 2/3)
    
    Returns
    -------
    residual : (NI, NJ, 4) viscous flux residual
    """
    NI, NJ = Q.shape[:2]
    
    # SA diffusion coefficient: (nu + max(0, nuHat)) / sigma
    nuHat_safe = jnp.maximum(nuHat, 0.0)
    nu_eff_sa = (nu_laminar + nuHat_safe) / sigma
    
    # CRITICAL: For wall boundaries to work correctly, Q should include ghost cells
    # that were set by the boundary conditions. However, for backward compatibility,
    # this function still accepts interior-only Q and pads with edge values.
    # 
    # BUG: This edge padding gives ZERO gradient at walls (Q[j=0] - Q[j=0] = 0),
    # instead of the correct gradient from the no-slip BC ghost cells.
    # 
    # The fix is to call compute_viscous_fluxes_tight_with_ghosts_jax instead,
    # passing Q that includes one layer of BC ghost cells.
    Q_padded = jnp.pad(Q, ((1, 1), (1, 1), (0, 0)), mode='edge')  # (NI+2, NJ+2, 4)
    
    # Pad diffusion coefficients (these don't have BCs, so edge padding is fine)
    mu_padded = jnp.pad(mu_eff, ((1, 1), (1, 1)), mode='edge')  # (NI+2, NJ+2)
    nu_sa_padded = jnp.pad(nu_eff_sa, ((1, 1), (1, 1)), mode='edge')
    
    # ==========================================================================
    # I-faces: between cells (i-1,j) and (i,j) in original coords
    # In padded coords: between (i, j+1) and (i+1, j+1)
    # Shape: (NI+1, NJ)
    # Stencil is 2x3: cells at i, i+1 in padded-i and j, j+1, j+2 in padded-j
    # ==========================================================================
    
    # Extract left and right cells for tight 2-point difference
    # I-face i is between padded cells (i, j+1) and (i+1, j+1)
    Q_L_i = Q_padded[:-1, 1:-1, :]  # (NI+1, NJ, 4) - left cell of each I-face
    Q_R_i = Q_padded[1:, 1:-1, :]   # (NI+1, NJ, 4) - right cell of each I-face
    
    # Tight 2-point gradient along coordinate line
    grad_coord_i = (Q_R_i - Q_L_i) / d_coord_i[:, :, None]  # (NI+1, NJ, 4)
    
    # Extract 6-cell stencil for LS orthogonal derivative
    # Stencil order: k=0:(i,j), k=1:(i,j+1), k=2:(i,j+2), k=3:(i+1,j), k=4:(i+1,j+1), k=5:(i+1,j+2)
    # In padded coords for I-face at position (i_face, j):
    #   left column (i_face in padded-i): j, j+1, j+2 in padded-j
    #   right column (i_face+1 in padded-i): j, j+1, j+2 in padded-j
    
    # Build stencil values efficiently using slicing
    # phi_stencil_i shape: (NI+1, NJ, 6, 4)
    phi_stencil_i = jnp.stack([
        Q_padded[:-1, :-2, :],   # k=0: (i, j)
        Q_padded[:-1, 1:-1, :],  # k=1: (i, j+1) = left cell
        Q_padded[:-1, 2:, :],    # k=2: (i, j+2)
        Q_padded[1:, :-2, :],    # k=3: (i+1, j)
        Q_padded[1:, 1:-1, :],   # k=4: (i+1, j+1) = right cell
        Q_padded[1:, 2:, :],     # k=5: (i+1, j+2)
    ], axis=2)  # (NI+1, NJ, 6, 4)
    
    # LS orthogonal gradient: grad_ortho = sum_k(w_k * phi_k)
    # ls_weights_i: (NI+1, NJ, 6), phi_stencil_i: (NI+1, NJ, 6, 4)
    grad_ortho_i = jnp.einsum('ijk,ijkv->ijv', ls_weights_i, phi_stencil_i)  # (NI+1, NJ, 4)
    
    # Full face gradient
    # grad_x = grad_coord * e_coord_x + grad_ortho * e_ortho_x
    # grad_y = grad_coord * e_coord_y + grad_ortho * e_ortho_y
    grad_x_i = grad_coord_i * e_coord_i_x[:, :, None] + grad_ortho_i * e_ortho_i_x[:, :, None]
    grad_y_i = grad_coord_i * e_coord_i_y[:, :, None] + grad_ortho_i * e_ortho_i_y[:, :, None]
    
    # Face-averaged diffusion coefficients
    mu_face_i = 0.5 * (mu_padded[:-1, 1:-1] + mu_padded[1:, 1:-1])  # (NI+1, NJ)
    nu_sa_face_i = 0.5 * (nu_sa_padded[:-1, 1:-1] + nu_sa_padded[1:, 1:-1])
    
    # Face normal direction
    S_i = jnp.sqrt(Si_x**2 + Si_y**2)  # (NI+1, NJ)
    nx_i = Si_x / (S_i + 1e-12)
    ny_i = Si_y / (S_i + 1e-12)
    
    # Momentum stress tensor: τ_xx = 2μ∂u/∂x, τ_yy = 2μ∂v/∂y, τ_xy = μ(∂u/∂y + ∂v/∂x)
    dudx_i = grad_x_i[:, :, 1]
    dudy_i = grad_y_i[:, :, 1]
    dvdx_i = grad_x_i[:, :, 2]
    dvdy_i = grad_y_i[:, :, 2]
    
    tau_xx_i = 2.0 * mu_face_i * dudx_i
    tau_yy_i = 2.0 * mu_face_i * dvdy_i
    tau_xy_i = mu_face_i * (dudy_i + dvdx_i)
    
    # Momentum flux through face
    flux_u_i = (tau_xx_i * nx_i + tau_xy_i * ny_i) * S_i
    flux_v_i = (tau_xy_i * nx_i + tau_yy_i * ny_i) * S_i
    
    # SA diffusion flux: (nu_eff/sigma) * (grad · n) * S
    dnuHat_dx_i = grad_x_i[:, :, 3]
    dnuHat_dy_i = grad_y_i[:, :, 3]
    flux_nuHat_i = nu_sa_face_i * (dnuHat_dx_i * nx_i + dnuHat_dy_i * ny_i) * S_i
    
    # ==========================================================================
    # J-faces: between cells (i,j-1) and (i,j) in original coords
    # In padded coords: between (i+1, j) and (i+1, j+1)
    # Shape: (NI, NJ+1)
    # Stencil is 3x2: cells at i, i+1, i+2 in padded-i and j, j+1 in padded-j
    # ==========================================================================
    
    Q_L_j = Q_padded[1:-1, :-1, :]  # (NI, NJ+1, 4) - bottom cell
    Q_R_j = Q_padded[1:-1, 1:, :]   # (NI, NJ+1, 4) - top cell
    
    grad_coord_j = (Q_R_j - Q_L_j) / d_coord_j[:, :, None]  # (NI, NJ+1, 4)
    
    # 6-cell stencil for J-faces
    # k=0:(i-1,j-1), k=1:(i,j-1), k=2:(i+1,j-1), k=3:(i-1,j), k=4:(i,j), k=5:(i+1,j)
    # In padded coords for J-face at (i, j_face):
    #   bottom row (j_face in padded-j): i, i+1, i+2 in padded-i
    #   top row (j_face+1 in padded-j): i, i+1, i+2 in padded-i
    
    phi_stencil_j = jnp.stack([
        Q_padded[:-2, :-1, :],   # k=0: (i-1, j-1) -> padded (i, j_face)
        Q_padded[1:-1, :-1, :],  # k=1: (i, j-1) = bottom cell
        Q_padded[2:, :-1, :],    # k=2: (i+1, j-1)
        Q_padded[:-2, 1:, :],    # k=3: (i-1, j)
        Q_padded[1:-1, 1:, :],   # k=4: (i, j) = top cell
        Q_padded[2:, 1:, :],     # k=5: (i+1, j)
    ], axis=2)  # (NI, NJ+1, 6, 4)
    
    grad_ortho_j = jnp.einsum('ijk,ijkv->ijv', ls_weights_j, phi_stencil_j)  # (NI, NJ+1, 4)
    
    grad_x_j = grad_coord_j * e_coord_j_x[:, :, None] + grad_ortho_j * e_ortho_j_x[:, :, None]
    grad_y_j = grad_coord_j * e_coord_j_y[:, :, None] + grad_ortho_j * e_ortho_j_y[:, :, None]
    
    mu_face_j = 0.5 * (mu_padded[1:-1, :-1] + mu_padded[1:-1, 1:])  # (NI, NJ+1)
    nu_sa_face_j = 0.5 * (nu_sa_padded[1:-1, :-1] + nu_sa_padded[1:-1, 1:])
    
    S_j = jnp.sqrt(Sj_x**2 + Sj_y**2)
    nx_j = Sj_x / (S_j + 1e-12)
    ny_j = Sj_y / (S_j + 1e-12)
    
    dudx_j = grad_x_j[:, :, 1]
    dudy_j = grad_y_j[:, :, 1]
    dvdx_j = grad_x_j[:, :, 2]
    dvdy_j = grad_y_j[:, :, 2]
    
    tau_xx_j = 2.0 * mu_face_j * dudx_j
    tau_yy_j = 2.0 * mu_face_j * dvdy_j
    tau_xy_j = mu_face_j * (dudy_j + dvdx_j)
    
    flux_u_j = (tau_xx_j * nx_j + tau_xy_j * ny_j) * S_j
    flux_v_j = (tau_xy_j * nx_j + tau_yy_j * ny_j) * S_j
    
    dnuHat_dx_j = grad_x_j[:, :, 3]
    dnuHat_dy_j = grad_y_j[:, :, 3]
    flux_nuHat_j = nu_sa_face_j * (dnuHat_dx_j * nx_j + dnuHat_dy_j * ny_j) * S_j
    
    # ==========================================================================
    # Assemble residual: res = flux[i+1] - flux[i]
    # ==========================================================================
    res_u_i = flux_u_i[1:, :] - flux_u_i[:-1, :]  # (NI, NJ)
    res_v_i = flux_v_i[1:, :] - flux_v_i[:-1, :]
    res_nuHat_i = flux_nuHat_i[1:, :] - flux_nuHat_i[:-1, :]
    
    res_u_j = flux_u_j[:, 1:] - flux_u_j[:, :-1]
    res_v_j = flux_v_j[:, 1:] - flux_v_j[:, :-1]
    res_nuHat_j = flux_nuHat_j[:, 1:] - flux_nuHat_j[:, :-1]
    
    residual = jnp.zeros((NI, NJ, 4))
    residual = residual.at[:, :, 1].set(res_u_i + res_u_j)
    residual = residual.at[:, :, 2].set(res_v_i + res_v_j)
    residual = residual.at[:, :, 3].set(res_nuHat_i + res_nuHat_j)
    
    return residual


@jax.jit
def compute_viscous_fluxes_tight_with_ghosts_jax(
    Q_with_ghosts,  # (NI+2, NJ+2, 4) - interior + one layer of BC ghost cells
    Si_x, Si_y,     # (NI+1, NJ) I-face normal components  
    Sj_x, Sj_y,     # (NI, NJ+1) J-face normal components
    d_coord_i, e_coord_i_x, e_coord_i_y, e_ortho_i_x, e_ortho_i_y,
    d_coord_j, e_coord_j_x, e_coord_j_y, e_ortho_j_x, e_ortho_j_y,
    ls_weights_i,   # (NI+1, NJ, 6) LS weights for I-faces
    ls_weights_j,   # (NI, NJ+1, 6) LS weights for J-faces
    mu_eff,         # (NI, NJ) effective viscosity for momentum
    nu_laminar,     # scalar laminar kinematic viscosity
    nuHat,          # (NI, NJ) SA working variable
    sigma=2.0/3.0
):
    """
    Viscous flux computation using Q that already includes BC ghost cells.
    
    CRITICAL FIX: This version uses the ghost cell values from the boundary
    conditions instead of padding with edge values. This is essential for
    correct wall shear stress computation.
    
    For no-slip wall, the BC sets ghost velocity to -interior_velocity,
    giving du/dy = 2*u_int/dy at the wall (correct for wall shear).
    
    The old version (compute_viscous_fluxes_tight_jax) padded interior-only Q
    with edge values, giving du/dy = 0 at the wall (WRONG!).
    
    Parameters
    ----------
    Q_with_ghosts : (NI+2, NJ+2, 4) 
        State with one layer of ghost cells from BC (nghost=1 equivalent).
        Extract from full Q[nghost-1:-nghost+1, nghost-1:-nghost+1, :] or similar.
    Si_x, Si_y, Sj_x, Sj_y : face normals
    d_coord_*, e_coord_*, e_ortho_*, ls_weights_* : geometry from MetricComputer
    mu_eff : (NI, NJ) effective viscosity
    nu_laminar : laminar viscosity
    nuHat : (NI, NJ) SA working variable
    sigma : SA constant (default 2/3)
    
    Returns
    -------
    residual : (NI, NJ, 4) viscous flux residual
    """
    # Q_with_ghosts already has shape (NI+2, NJ+2, 4) with ghost cells
    Q_padded = Q_with_ghosts  # Use directly, no padding needed!
    
    # Interior dimensions
    NI = Q_padded.shape[0] - 2
    NJ = Q_padded.shape[1] - 2
    
    # SA diffusion coefficient: (nu + max(0, nuHat)) / sigma
    nuHat_safe = jnp.maximum(nuHat, 0.0)
    nu_eff_sa = (nu_laminar + nuHat_safe) / sigma
    
    # Pad diffusion coefficients (these don't have BCs, so edge padding is OK)
    mu_padded = jnp.pad(mu_eff, ((1, 1), (1, 1)), mode='edge')
    nu_sa_padded = jnp.pad(nu_eff_sa, ((1, 1), (1, 1)), mode='edge')
    
    # ==========================================================================
    # I-faces computation (same as before, but using properly ghosted Q_padded)
    # ==========================================================================
    Q_L_i = Q_padded[:-1, 1:-1, :]
    Q_R_i = Q_padded[1:, 1:-1, :]
    
    grad_coord_i = (Q_R_i - Q_L_i) / d_coord_i[:, :, None]
    
    phi_stencil_i = jnp.stack([
        Q_padded[:-1, :-2, :],
        Q_padded[:-1, 1:-1, :],
        Q_padded[:-1, 2:, :],
        Q_padded[1:, :-2, :],
        Q_padded[1:, 1:-1, :],
        Q_padded[1:, 2:, :],
    ], axis=2)
    
    grad_ortho_i = jnp.einsum('ijk,ijkv->ijv', ls_weights_i, phi_stencil_i)
    
    grad_x_i = grad_coord_i * e_coord_i_x[:, :, None] + grad_ortho_i * e_ortho_i_x[:, :, None]
    grad_y_i = grad_coord_i * e_coord_i_y[:, :, None] + grad_ortho_i * e_ortho_i_y[:, :, None]
    
    mu_face_i = 0.5 * (mu_padded[:-1, 1:-1] + mu_padded[1:, 1:-1])
    nu_sa_face_i = 0.5 * (nu_sa_padded[:-1, 1:-1] + nu_sa_padded[1:, 1:-1])
    
    S_i = jnp.sqrt(Si_x**2 + Si_y**2)
    nx_i = Si_x / (S_i + 1e-12)
    ny_i = Si_y / (S_i + 1e-12)
    
    dudx_i = grad_x_i[:, :, 1]
    dudy_i = grad_y_i[:, :, 1]
    dvdx_i = grad_x_i[:, :, 2]
    dvdy_i = grad_y_i[:, :, 2]
    
    tau_xx_i = 2.0 * mu_face_i * dudx_i
    tau_yy_i = 2.0 * mu_face_i * dvdy_i
    tau_xy_i = mu_face_i * (dudy_i + dvdx_i)
    
    flux_u_i = (tau_xx_i * nx_i + tau_xy_i * ny_i) * S_i
    flux_v_i = (tau_xy_i * nx_i + tau_yy_i * ny_i) * S_i
    
    dnuHat_dx_i = grad_x_i[:, :, 3]
    dnuHat_dy_i = grad_y_i[:, :, 3]
    flux_nuHat_i = nu_sa_face_i * (dnuHat_dx_i * nx_i + dnuHat_dy_i * ny_i) * S_i
    
    # ==========================================================================
    # J-faces computation (same as before, but using properly ghosted Q_padded)
    # ==========================================================================
    Q_L_j = Q_padded[1:-1, :-1, :]
    Q_R_j = Q_padded[1:-1, 1:, :]
    
    grad_coord_j = (Q_R_j - Q_L_j) / d_coord_j[:, :, None]
    
    phi_stencil_j = jnp.stack([
        Q_padded[:-2, :-1, :],
        Q_padded[1:-1, :-1, :],
        Q_padded[2:, :-1, :],
        Q_padded[:-2, 1:, :],
        Q_padded[1:-1, 1:, :],
        Q_padded[2:, 1:, :],
    ], axis=2)
    
    grad_ortho_j = jnp.einsum('ijk,ijkv->ijv', ls_weights_j, phi_stencil_j)
    
    grad_x_j = grad_coord_j * e_coord_j_x[:, :, None] + grad_ortho_j * e_ortho_j_x[:, :, None]
    grad_y_j = grad_coord_j * e_coord_j_y[:, :, None] + grad_ortho_j * e_ortho_j_y[:, :, None]
    
    mu_face_j = 0.5 * (mu_padded[1:-1, :-1] + mu_padded[1:-1, 1:])
    nu_sa_face_j = 0.5 * (nu_sa_padded[1:-1, :-1] + nu_sa_padded[1:-1, 1:])
    
    S_j = jnp.sqrt(Sj_x**2 + Sj_y**2)
    nx_j = Sj_x / (S_j + 1e-12)
    ny_j = Sj_y / (S_j + 1e-12)
    
    dudx_j = grad_x_j[:, :, 1]
    dudy_j = grad_y_j[:, :, 1]
    dvdx_j = grad_x_j[:, :, 2]
    dvdy_j = grad_y_j[:, :, 2]
    
    tau_xx_j = 2.0 * mu_face_j * dudx_j
    tau_yy_j = 2.0 * mu_face_j * dvdy_j
    tau_xy_j = mu_face_j * (dudy_j + dvdx_j)
    
    flux_u_j = (tau_xx_j * nx_j + tau_xy_j * ny_j) * S_j
    flux_v_j = (tau_xy_j * nx_j + tau_yy_j * ny_j) * S_j
    
    dnuHat_dx_j = grad_x_j[:, :, 3]
    dnuHat_dy_j = grad_y_j[:, :, 3]
    flux_nuHat_j = nu_sa_face_j * (dnuHat_dx_j * nx_j + dnuHat_dy_j * ny_j) * S_j
    
    # ==========================================================================
    # Assemble residual
    # ==========================================================================
    res_u_i = flux_u_i[1:, :] - flux_u_i[:-1, :]
    res_v_i = flux_v_i[1:, :] - flux_v_i[:-1, :]
    res_nuHat_i = flux_nuHat_i[1:, :] - flux_nuHat_i[:-1, :]
    
    res_u_j = flux_u_j[:, 1:] - flux_u_j[:, :-1]
    res_v_j = flux_v_j[:, 1:] - flux_v_j[:, :-1]
    res_nuHat_j = flux_nuHat_j[:, 1:] - flux_nuHat_j[:, :-1]
    
    residual = jnp.zeros((NI, NJ, 4))
    residual = residual.at[:, :, 1].set(res_u_i + res_u_j)
    residual = residual.at[:, :, 2].set(res_v_i + res_v_j)
    residual = residual.at[:, :, 3].set(res_nuHat_i + res_nuHat_j)
    
    return residual
