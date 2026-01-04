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
def compute_sa_cb2_advection_jax(nuHat, grad_nuHat, Si_x, Si_y, Sj_x, Sj_y, k2, k4):
    """
    SA cb2 term implemented as advection: (cb2/σ)(∇ν̃)·(∇ν̃)
    
    The gradient ∇ν̃ (computed via Green-Gauss) acts as "advection velocity".
    JST dissipation uses this velocity for spectral radius, NOT physical (u,v).
    
    This discretization provides stability through JST artificial dissipation
    while treating the cb2 term in a consistent manner.
    
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
    k2 : float
        2nd-order JST dissipation coefficient (for non-smooth regions).
    k4 : float
        4th-order JST dissipation coefficient (for smooth regions).
    
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
    
    # Pad for face averaging and stencil extraction
    # 2-cell padding in each direction for 4th-order stencil
    nuHat_padded = jnp.pad(nuHat, ((2, 2), (2, 2)), mode='edge')  # (NI+4, NJ+4)
    vel_x_padded = jnp.pad(vel_x, ((1, 1), (1, 1)), mode='edge')  # (NI+2, NJ+2)
    vel_y_padded = jnp.pad(vel_y, ((1, 1), (1, 1)), mode='edge')
    
    # ==========================================================================
    # I-direction (NI+1 faces at constant i)
    # I-face i is between original cells (i-1) and (i)
    # With padding offset of 2: between padded cells (i+1) and (i+2)
    # ==========================================================================
    
    # Face-averaged velocity
    # vel_padded has 1-cell padding, so face i uses padded indices i and i+1
    vel_x_face_i = 0.5 * (vel_x_padded[:-1, 1:-1] + vel_x_padded[1:, 1:-1])  # (NI+1, NJ)
    vel_y_face_i = 0.5 * (vel_y_padded[:-1, 1:-1] + vel_y_padded[1:, 1:-1])
    
    # Face area and normal
    S_i = jnp.sqrt(Si_x**2 + Si_y**2)  # (NI+1, NJ)
    nx_i = Si_x / (S_i + 1e-12)
    ny_i = Si_y / (S_i + 1e-12)
    
    # Normal velocity at face (using grad_nuHat as "velocity")
    vel_n_i = vel_x_face_i * nx_i + vel_y_face_i * ny_i  # (NI+1, NJ)
    
    # Spectral radius based on grad_nuHat velocity (NOT physical u,v!)
    # CRITICAL: Cap spectral radius to prevent explosive feedback when grad_nuHat is large
    lambda_i_raw = jnp.abs(vel_n_i) * S_i  # (NI+1, NJ)
    lambda_i = jnp.minimum(lambda_i_raw, 10.0)  # Cap to prevent runaway
    
    # Face-averaged nuHat for convective flux
    # nuHat_padded has 2-cell padding, shape (NI+4, NJ+4)
    # Original cell i -> padded index i+2
    # I-face index if (0 to NI): left cell (if-1) -> padded (if+1), right cell (if) -> padded (if+2)
    # For NI+1 faces: left = padded[1:NI+2], right = padded[2:NI+3]
    # With array of size NI+4: [1:-2] gives NI+1 elements, [2:-1] gives NI+1 elements
    nuHat_L_i = nuHat_padded[1:-2, 2:-2]  # (NI+1, NJ)
    nuHat_R_i = nuHat_padded[2:-1, 2:-2]  # (NI+1, NJ)
    nuHat_face_i = 0.5 * (nuHat_L_i + nuHat_R_i)
    
    # Convective flux: coeff * (vel · n) * nuHat_face * S
    F_conv_i = coeff * vel_n_i * nuHat_face_i * S_i
    
    # JST dissipation with spectral radius from grad_nuHat velocity
    # 2nd-order: k2 * λ * (nuHat_R - nuHat_L)
    diss2_i = k2 * lambda_i * (nuHat_R_i - nuHat_L_i)
    
    # 4th-order: k4 * λ * (nuHat_{i+1} - 3*nuHat_i + 3*nuHat_{i-1} - nuHat_{i-2})
    # For I-face if: Lm1 = cell (if-2) -> padded (if), Rp1 = cell (if+1) -> padded (if+3)
    # [:-3] gives (NI+4)-3 = NI+1 elements, [3:] also gives NI+1 elements
    nuHat_Lm1_i = nuHat_padded[:-3, 2:-2]  # (NI+1, NJ)
    nuHat_Rp1_i = nuHat_padded[3:, 2:-2]   # (NI+1, NJ)
    diss4_i = k4 * lambda_i * (nuHat_Rp1_i - 3.0 * nuHat_R_i + 3.0 * nuHat_L_i - nuHat_Lm1_i)
    
    # Combined flux: F = F_conv - diss2 + diss4
    F_i = F_conv_i - diss2_i + diss4_i
    
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
    # Cap spectral radius to prevent explosive feedback
    lambda_j_raw = jnp.abs(vel_n_j) * S_j
    lambda_j = jnp.minimum(lambda_j_raw, 10.0)
    
    # J-face jf: left cell (jf-1) -> padded (jf+1), right cell (jf) -> padded (jf+2)
    # For NJ+1 faces: left = padded[:, 1:-2], right = padded[:, 2:-1]
    nuHat_L_j = nuHat_padded[2:-2, 1:-2]  # (NI, NJ+1)
    nuHat_R_j = nuHat_padded[2:-2, 2:-1]  # (NI, NJ+1)
    nuHat_face_j = 0.5 * (nuHat_L_j + nuHat_R_j)
    
    F_conv_j = coeff * vel_n_j * nuHat_face_j * S_j
    
    diss2_j = k2 * lambda_j * (nuHat_R_j - nuHat_L_j)
    
    # [:-3] gives (NJ+4)-3 = NJ+1 elements, [3:] also gives NJ+1 elements
    nuHat_Lm1_j = nuHat_padded[2:-2, :-3]  # (NI, NJ+1)
    nuHat_Rp1_j = nuHat_padded[2:-2, 3:]   # (NI, NJ+1)
    diss4_j = k4 * lambda_j * (nuHat_Rp1_j - 3.0 * nuHat_R_j + 3.0 * nuHat_L_j - nuHat_Lm1_j)
    
    F_j = F_conv_j - diss2_j + diss4_j
    
    # ==========================================================================
    # Residual: res = F[i] - F[i+1] (flux INTO cell)
    # ==========================================================================
    res_i = F_i[:-1, :] - F_i[1:, :]  # (NI, NJ)
    res_j = F_j[:, :-1] - F_j[:, 1:]  # (NI, NJ)
    
    return res_i + res_j
