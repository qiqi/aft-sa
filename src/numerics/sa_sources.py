"""
Spalart-Allmaras source term computation for 2D FVM.

Source terms: P - D + cb2_term
  - P: Production = cb1 * S_tilde * nuHat
  - D: Destruction = cw1 * fw * (nuHat/d)^2
  - cb2_term: (cb2/sigma) * |∇nuHat|^2
"""

from src.physics.jax_config import jax, jnp
from src.physics.spalart_allmaras import compute_sa_production, compute_sa_destruction
from .gradients import compute_vorticity_jax


# SA model constants
CB1 = 0.1355
CB2 = 0.622
SIGMA = 2.0 / 3.0
KAPPA = 0.41
CW1 = CB1 / (KAPPA ** 2) + (1.0 + CB2) / SIGMA


@jax.jit
def compute_sa_source_jax(nuHat, grad, wall_dist, nu_laminar):
    """
    Compute SA turbulence model source terms for 2D FVM.
    
    Returns Production, Destruction, and cb2_term separately for point-implicit treatment.
    
    Source = Production - Destruction + cb2_term
    
    Parameters
    ----------
    nuHat : jnp.ndarray (NI, NJ)
        SA working variable from interior cells.
    grad : jnp.ndarray (NI, NJ, 4, 2)
        Cell-centered gradients of Q = [p, u, v, nuHat].
        Last dimension is (d/dx, d/dy).
    wall_dist : jnp.ndarray (NI, NJ)
        Distance to nearest wall.
    nu_laminar : float
        Kinematic viscosity (1/Re).
    
    Returns
    -------
    P : jnp.ndarray (NI, NJ)
        Production term (positive, treat explicitly).
    D : jnp.ndarray (NI, NJ)
        Destruction term (positive, treat point-implicitly).
    cb2_term : jnp.ndarray (NI, NJ)
        cb2 gradient term (treat explicitly).
    """
    # Vorticity magnitude from velocity gradients
    omega = compute_vorticity_jax(grad)
    
    # SA production & destruction (with safe handling of negative nuHat)
    P = compute_sa_production(omega, nuHat, wall_dist)
    D = compute_sa_destruction(omega, nuHat, wall_dist)
    
    # cb2 gradient term: (cb2/σ)(∇ν̃)·(∇ν̃)
    grad_nuHat = grad[:, :, 3, :]  # (NI, NJ, 2) - gradients of nuHat
    grad_nuHat_sq = jnp.sum(grad_nuHat**2, axis=-1)  # (NI, NJ)
    cb2_term = (CB2 / SIGMA) * grad_nuHat_sq
    
    return P, D, cb2_term


@jax.jit
def compute_sa_production_only_jax(nuHat, grad, wall_dist):
    """
    Compute SA production term only.
    
    P = cb1 * S_tilde * nuHat
    
    Parameters
    ----------
    nuHat : jnp.ndarray (NI, NJ)
        SA working variable.
    grad : jnp.ndarray (NI, NJ, 4, 2)
        Cell-centered gradients.
    wall_dist : jnp.ndarray (NI, NJ)
        Distance to nearest wall.
    
    Returns
    -------
    P : jnp.ndarray (NI, NJ)
        Production term.
    """
    omega = compute_vorticity_jax(grad)
    return compute_sa_production(omega, nuHat, wall_dist)


@jax.jit
def compute_sa_destruction_only_jax(nuHat, grad, wall_dist):
    """
    Compute SA destruction term only.
    
    D = cw1 * fw * (nuHat/d)^2
    
    Parameters
    ----------
    nuHat : jnp.ndarray (NI, NJ)
        SA working variable.
    grad : jnp.ndarray (NI, NJ, 4, 2)
        Cell-centered gradients.
    wall_dist : jnp.ndarray (NI, NJ)
        Distance to nearest wall.
    
    Returns
    -------
    D : jnp.ndarray (NI, NJ)
        Destruction term.
    """
    omega = compute_vorticity_jax(grad)
    return compute_sa_destruction(omega, nuHat, wall_dist)


@jax.jit
def compute_cb2_term_jax(grad):
    """
    Compute cb2 gradient term: (cb2/σ)(∇ν̃)·(∇ν̃).
    
    This term enhances diffusion where nuHat gradients are large.
    
    Parameters
    ----------
    grad : jnp.ndarray (NI, NJ, 4, 2)
        Cell-centered gradients.
    
    Returns
    -------
    cb2_term : jnp.ndarray (NI, NJ)
        The cb2 gradient-squared term.
    """
    grad_nuHat = grad[:, :, 3, :]  # (NI, NJ, 2)
    grad_nuHat_sq = jnp.sum(grad_nuHat**2, axis=-1)
    return (CB2 / SIGMA) * grad_nuHat_sq


@jax.jit  
def compute_turbulent_viscosity_jax(nuHat, nu_laminar):
    """
    Compute turbulent viscosity from SA working variable.
    
    mu_t = nuHat * fv1(chi), where chi = nuHat / nu_laminar
    
    Parameters
    ----------
    nuHat : jnp.ndarray
        SA working variable (any shape).
    nu_laminar : float
        Laminar kinematic viscosity.
    
    Returns
    -------
    mu_t : jnp.ndarray
        Turbulent viscosity (same shape as nuHat).
    """
    cv1 = 7.1
    nuHat_safe = jnp.maximum(nuHat, 0.0)
    chi = nuHat_safe / nu_laminar
    chi3 = chi ** 3
    fv1_val = chi3 / (chi3 + cv1 ** 3)
    return nuHat_safe * fv1_val


@jax.jit
def compute_effective_viscosity_jax(nuHat, nu_laminar):
    """
    Compute effective viscosity for momentum equations.
    
    mu_eff = nu_laminar + mu_t
    
    Parameters
    ----------
    nuHat : jnp.ndarray
        SA working variable.
    nu_laminar : float
        Laminar kinematic viscosity.
    
    Returns
    -------
    mu_eff : jnp.ndarray
        Effective viscosity.
    """
    mu_t = compute_turbulent_viscosity_jax(nuHat, nu_laminar)
    return nu_laminar + mu_t


# =============================================================================
# SA cb2 Term as Advection with JST Dissipation
# =============================================================================

@jax.jit
def compute_sa_cb2_advection_jax(nuHat, grad_nuHat, Si_x, Si_y, Sj_x, Sj_y, k2=None, k4=None):
    """
    SA cb2 term implemented as advection with FIRST-ORDER UPWIND: (cb2/σ)(∇ν̃)·(∇ν̃)
    
    The gradient ∇ν̃ (computed via Green-Gauss) acts as "advection velocity".
    Uses first-order upwind to avoid dispersive oscillations.
    
    Parameters
    ----------
    nuHat : jnp.ndarray (NI, NJ)
        Cell-centered SA working variable.
    grad_nuHat : jnp.ndarray (NI, NJ, 2)
        Green-Gauss gradient of nuHat at cell centers.
    Si_x, Si_y : jnp.ndarray (NI+1, NJ)
        I-face normal components.
    Sj_x, Sj_y : jnp.ndarray (NI, NJ+1)
        J-face normal components.
    k2, k4 : float, optional
        DEPRECATED - ignored. Kept for API compatibility.
    
    Returns
    -------
    cb2_residual : jnp.ndarray (NI, NJ)
        Contribution to ν̃ equation from cb2 term.
    """
    NI, NJ = nuHat.shape
    coeff = CB2 / SIGMA
    
    # "Advection velocity" = ∇ν̃
    vel_x = grad_nuHat[:, :, 0]  # (NI, NJ)
    vel_y = grad_nuHat[:, :, 1]  # (NI, NJ)
    
    # Pad for face averaging (only 1-cell padding needed for first-order upwind)
    nuHat_padded = jnp.pad(nuHat, ((1, 1), (1, 1)), mode='edge')  # (NI+2, NJ+2)
    vel_x_padded = jnp.pad(vel_x, ((1, 1), (1, 1)), mode='edge')  # (NI+2, NJ+2)
    vel_y_padded = jnp.pad(vel_y, ((1, 1), (1, 1)), mode='edge')
    
    # ==========================================================================
    # I-direction (NI+1 faces at constant i)
    # I-face i is between original cells (i-1) and (i)
    # ==========================================================================
    
    # Face-averaged velocity
    vel_x_face_i = 0.5 * (vel_x_padded[:-1, 1:-1] + vel_x_padded[1:, 1:-1])  # (NI+1, NJ)
    vel_y_face_i = 0.5 * (vel_y_padded[:-1, 1:-1] + vel_y_padded[1:, 1:-1])
    
    # Face area and normal
    S_i = jnp.sqrt(Si_x**2 + Si_y**2)  # (NI+1, NJ)
    nx_i = Si_x / (S_i + 1e-12)
    ny_i = Si_y / (S_i + 1e-12)
    
    # Normal velocity at face (using grad_nuHat as "velocity")
    vel_n_i = vel_x_face_i * nx_i + vel_y_face_i * ny_i  # (NI+1, NJ)
    
    # Left and right cell values
    nuHat_L_i = nuHat_padded[:-1, 1:-1]  # (NI+1, NJ)
    nuHat_R_i = nuHat_padded[1:, 1:-1]   # (NI+1, NJ)
    
    # FIRST-ORDER UPWIND: nuHat_upwind = nuHat_L if vel_n > 0 else nuHat_R
    nuHat_upwind_i = jnp.where(vel_n_i >= 0, nuHat_L_i, nuHat_R_i)
    
    # Convective flux: coeff * (vel · n) * nuHat_upwind * S
    F_i = coeff * vel_n_i * nuHat_upwind_i * S_i
    
    # ==========================================================================
    # J-direction (NI, NJ+1 faces at constant j)
    # J-face j is between original cells (j-1) and (j)
    # ==========================================================================
    
    vel_x_face_j = 0.5 * (vel_x_padded[1:-1, :-1] + vel_x_padded[1:-1, 1:])  # (NI, NJ+1)
    vel_y_face_j = 0.5 * (vel_y_padded[1:-1, :-1] + vel_y_padded[1:-1, 1:])
    
    S_j = jnp.sqrt(Sj_x**2 + Sj_y**2)  # (NI, NJ+1)
    nx_j = Sj_x / (S_j + 1e-12)
    ny_j = Sj_y / (S_j + 1e-12)
    
    vel_n_j = vel_x_face_j * nx_j + vel_y_face_j * ny_j
    
    nuHat_L_j = nuHat_padded[1:-1, :-1]  # (NI, NJ+1)
    nuHat_R_j = nuHat_padded[1:-1, 1:]   # (NI, NJ+1)
    
    # FIRST-ORDER UPWIND
    nuHat_upwind_j = jnp.where(vel_n_j >= 0, nuHat_L_j, nuHat_R_j)
    
    F_j = coeff * vel_n_j * nuHat_upwind_j * S_j
    
    # ==========================================================================
    # Residual: res = F[i] - F[i+1] (flux INTO cell)
    # ==========================================================================
    res_i = F_i[:-1, :] - F_i[1:, :]  # (NI, NJ)
    res_j = F_j[:, :-1] - F_j[:, 1:]  # (NI, NJ)
    
    return res_i + res_j
