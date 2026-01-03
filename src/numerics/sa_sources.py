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
def compute_sa_source_jax(nuHat, grad, wall_dist, nu_laminar, max_source_factor=10.0):
    """
    Compute SA turbulence model source terms for 2D FVM.
    
    Source = Production - Destruction + cb2_term
    
    Includes limiting to prevent stiff destruction from destabilizing.
    
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
    max_source_factor : float
        Maximum allowed |source| relative to nuHat/tau_turb, where
        tau_turb = d^2 / nu_laminar is a characteristic turbulent time.
    
    Returns
    -------
    source : jnp.ndarray (NI, NJ)
        Net source term: P - D + cb2_term (limited for stability).
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
    
    # Raw source
    source = P - D + cb2_term
    
    # Limit destruction to prevent instability near wall
    # Only limit negative (destruction-dominated) source
    # Limit: |D_eff| <= max_source_factor * nuHat * omega (using vorticity as reference)
    omega_safe = jnp.maximum(omega, 1e-10)
    max_destruction = max_source_factor * jnp.maximum(nuHat, 1e-20) * omega_safe
    
    # Only limit negative source (destruction), allow positive (production)
    source_limited = jnp.where(
        source < 0,
        jnp.maximum(source, -max_destruction),  # Limit destruction
        source  # Don't limit production
    )
    
    return source_limited


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
