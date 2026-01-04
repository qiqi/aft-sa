"""
Spalart-Allmaras Turbulence Model with analytical gradients.

Dimension Agnostic: Works with any tensor shape (1D profiles, 2D/3D fields).

Notation:
    omega = vorticity magnitude (|du/dy| for boundary layers, |curl(v)| for 2D/3D)
    d = wall distance (y for boundary layers)

Robustness:
    "Safe" versions handle negative nuHat by zeroing source terms,
    allowing the Maximum Principle to fill negative regions via diffusion.
"""

from typing import Tuple, Union
from .jax_config import jax, jnp

ArrayLike = Union[jnp.ndarray, float]


@jax.jit
def fv1(nuHat: ArrayLike) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute fv1 = chi³/(chi³ + cv1³) and d(fv1)/d(nuHat)."""
    cv1: float = 7.1
    chi: jnp.ndarray = nuHat
    chi3: jnp.ndarray = chi ** 3
    denom: jnp.ndarray = chi3 + cv1 ** 3
    val: jnp.ndarray = chi3 / denom
    grad: jnp.ndarray = (3 * chi**2 * cv1**3) / (denom ** 2)
    return val, grad


@jax.jit
def fv1_value(nuHat: ArrayLike) -> jnp.ndarray:
    """Compute fv1 value only (for autograd)."""
    cv1: float = 7.1
    chi3: jnp.ndarray = nuHat ** 3
    return chi3 / (chi3 + cv1 ** 3)


@jax.jit
def fv2(nuHat):
    """
    Compute fv2 and d(fv2)/d(nuHat).
    
    Parameters
    ----------
    nuHat : jnp.ndarray
        SA working variable (any shape).
        
    Returns
    -------
    val : jnp.ndarray
        fv2 value.
    grad : jnp.ndarray
        d(fv2)/d(nuHat).
    """
    chi = nuHat
    fv1_val, fv1_grad = fv1(nuHat)
    denom = 1.0 + chi * fv1_val
    val = 1.0 - chi / denom
    term_grad = (1.0 - chi**2 * fv1_grad) / (denom ** 2)
    grad = -term_grad
    return val, grad


@jax.jit
def _S_tilde(dudy, nuHat, y):
    """
    Compute modified vorticity S̃ = Ω + (ν̃/κ²d²)·fv2 and d(S̃)/d(nuHat).
    
    Parameters
    ----------
    dudy : jnp.ndarray
        Vorticity magnitude |ω|.
    nuHat : jnp.ndarray
        SA working variable.
    y : jnp.ndarray
        Wall distance d.
        
    Returns
    -------
    val : jnp.ndarray
        S_tilde value.
    grad : jnp.ndarray
        d(S_tilde)/d(nuHat).
    """
    kappa = 0.41
    omega = jnp.abs(dudy)
    fv2_val, fv2_grad = fv2(nuHat)
    inv_k2y2 = 1.0 / (kappa ** 2 * y ** 2)
    s_tilde_term = nuHat * inv_k2y2 * fv2_val
    S_tilde_raw = omega + s_tilde_term
    grad_raw = inv_k2y2 * (fv2_val + nuHat * fv2_grad)
    val = jnp.clip(S_tilde_raw, min=1e-16)
    grad = jnp.where(S_tilde_raw < 1e-16, jnp.zeros_like(grad_raw), grad_raw)
    return val, grad


@jax.jit
def r(dudy, nuHat, y):
    """
    Compute r = nuHat/(S̃·κ²d²) and d(r)/d(nuHat).
    
    Parameters
    ----------
    dudy : jnp.ndarray
        Vorticity magnitude |ω|.
    nuHat : jnp.ndarray
        SA working variable.
    y : jnp.ndarray
        Wall distance d.
        
    Returns
    -------
    val : jnp.ndarray
        r value (clamped to max=10).
    grad : jnp.ndarray
        d(r)/d(nuHat).
    """
    kappa = 0.41
    S_t, S_t_grad = _S_tilde(dudy, nuHat, y)
    denom_factor = kappa ** 2 * y ** 2
    denom = S_t * denom_factor
    r_raw = nuHat / denom
    grad_raw = (S_t - nuHat * S_t_grad) / (denom_factor * S_t ** 2)
    val = jnp.clip(r_raw, max=10.0)
    grad = jnp.where(r_raw > 10.0, jnp.zeros_like(grad_raw), grad_raw)
    return val, grad


@jax.jit
def g(dudy, nuHat, y):
    """
    Compute g = r + cw2·(r⁶ - r) and d(g)/d(nuHat).
    
    Parameters
    ----------
    dudy : jnp.ndarray
        Vorticity magnitude |ω|.
    nuHat : jnp.ndarray
        SA working variable.
    y : jnp.ndarray
        Wall distance d.
        
    Returns
    -------
    val : jnp.ndarray
        g value.
    grad : jnp.ndarray
        d(g)/d(nuHat).
    """
    cw2 = 0.3
    r_val, r_grad = r(dudy, nuHat, y)
    val = r_val + cw2 * (r_val ** 6 - r_val)
    dg_dr = 1.0 + cw2 * (6.0 * r_val ** 5 - 1.0)
    grad = dg_dr * r_grad
    return val, grad


@jax.jit
def fw(dudy, nuHat, y):
    """
    Compute fw and d(fw)/d(nuHat).
    
    Parameters
    ----------
    dudy : jnp.ndarray
        Vorticity magnitude |ω|.
    nuHat : jnp.ndarray
        SA working variable.
    y : jnp.ndarray
        Wall distance d.
        
    Returns
    -------
    val : jnp.ndarray
        fw value.
    grad : jnp.ndarray
        d(fw)/d(nuHat).
    """
    cw3 = 2.0
    g_val, g_grad = g(dudy, nuHat, y)
    g6 = g_val ** 6
    c6 = cw3 ** 6
    denom = g6 + c6
    top = 1.0 + c6
    ratio = top / denom
    radicand = ratio ** (1.0 / 6.0)
    val = g_val * radicand
    dfw_dg = radicand * (c6 / denom)
    grad = dfw_dg * g_grad
    return val, grad


@jax.jit
def spalart_allmaras_amplification(dudy, nuHat, y):
    """
    Compute SA production P = cb1·S̃·ν̃ and destruction D = cw1·fw·(ν̃/d)²
    with analytical gradients.
    
    Parameters
    ----------
    dudy : jnp.ndarray
        Vorticity magnitude |ω| (any shape).
    nuHat : jnp.ndarray
        SA working variable ν̃ (same shape as dudy).
    y : jnp.ndarray
        Wall distance d (same shape as dudy).
        
    Returns
    -------
    (prod_val, prod_grad) : tuple of jnp.ndarray
        Production term and d(Production)/d(nuHat).
    (dest_val, dest_grad) : tuple of jnp.ndarray
        Destruction term and d(Destruction)/d(nuHat).
    """
    cb1 = 0.1355
    cb2 = 0.622
    sigma = 2.0 / 3.0
    kappa = 0.41
    cw1 = cb1 / (kappa ** 2) + (1.0 + cb2) / sigma

    S_t, S_t_grad = _S_tilde(dudy, nuHat, y)
    prod_val = cb1 * S_t * nuHat
    prod_grad = cb1 * (S_t_grad * nuHat + S_t)

    fw_val, fw_grad = fw(dudy, nuHat, y)
    term_sq = (nuHat / y) ** 2
    dest_val = cw1 * fw_val * term_sq
    term_sq_grad = 2.0 * nuHat / (y ** 2)
    dest_grad = cw1 * (fw_grad * term_sq + fw_val * term_sq_grad)

    return (prod_val, prod_grad), (dest_val, dest_grad)


@jax.jit
def effective_viscosity_safe(nuHat, nu_laminar=1.0):
    """
    Compute effective viscosity with safety for negative nuHat.
    
    nu_eff = nu_laminar + max(0, nuHat * fv1(nuHat))
    
    Parameters
    ----------
    nuHat : jnp.ndarray
        SA working variable (may contain negative values).
    nu_laminar : float
        Laminar viscosity (default 1.0).
        
    Returns
    -------
    nu_eff : jnp.ndarray
        Effective viscosity, always >= nu_laminar.
    """
    nuHat_safe = jnp.clip(nuHat, min=0.0)
    fv1_val, _ = fv1(nuHat_safe)
    nu_turb = nuHat_safe * fv1_val
    return nu_laminar + nu_turb


@jax.jit
def sa_source_mask(nuHat):
    """
    Compute source term mask for non-negative safety.
    
    Returns ~1 for nuHat >= 0, ~0 for nuHat < 0.
    
    Parameters
    ----------
    nuHat : jnp.ndarray
        SA working variable.
        
    Returns
    -------
    mask : jnp.ndarray
        Source term multiplier.
    """
    return 1.0 / (1.0 + jnp.exp(-100.0 * nuHat))


@jax.jit
def compute_sa_production(omega, nuHat, d, nu_laminar=1e-6):
    """
    Compute SA production term with non-negative safety.
    
    P = cb1 * S_tilde * nuHat * mask(nuHat)
    
    Parameters
    ----------
    omega : jnp.ndarray
        Vorticity magnitude |ω|.
    nuHat : jnp.ndarray
        SA working variable (may contain negatives).
    d : jnp.ndarray
        Wall distance.
    nu_laminar : float
        Laminar kinematic viscosity (1/Re).
        
    Returns
    -------
    P : jnp.ndarray
        Production term.
    """
    cb1 = 0.1355
    kappa = 0.41
    cv1 = 7.1
    
    mask = sa_source_mask(nuHat)
    nuHat_safe = jnp.clip(nuHat, min=1e-10)
    
    # CRITICAL: chi = nuHat / nu_laminar, NOT chi = nuHat!
    chi = nuHat_safe / nu_laminar
    chi3 = chi ** 3
    fv1_val = chi3 / (chi3 + cv1 ** 3)
    fv2_val = 1.0 - chi / (1.0 + chi * fv1_val)
    
    inv_k2d2 = 1.0 / (kappa ** 2 * d ** 2 + 1e-20)
    S_tilde = omega + nuHat_safe * inv_k2d2 * fv2_val
    S_tilde = jnp.clip(S_tilde, min=1e-16)
    
    P = cb1 * S_tilde * nuHat_safe * mask
    return P


@jax.jit
def compute_sa_destruction(omega, nuHat, d, nu_laminar=1e-6):
    """
    Compute SA destruction term with non-negative safety.
    
    D = cw1 * fw * (nuHat / d)^2 * mask(nuHat)
    
    Parameters
    ----------
    omega : jnp.ndarray
        Vorticity magnitude |ω|.
    nuHat : jnp.ndarray
        SA working variable (may contain negatives).
    d : jnp.ndarray
        Wall distance.
    nu_laminar : float
        Laminar kinematic viscosity (1/Re).
        
    Returns
    -------
    D : jnp.ndarray
        Destruction term.
    """
    cb1 = 0.1355
    cb2 = 0.622
    sigma = 2.0 / 3.0
    kappa = 0.41
    cw1 = cb1 / (kappa ** 2) + (1.0 + cb2) / sigma
    cw2 = 0.3
    cw3 = 2.0
    cv1 = 7.1
    
    mask = sa_source_mask(nuHat)
    nuHat_safe = jnp.clip(nuHat, min=1e-10)
    
    # CRITICAL: chi = nuHat / nu_laminar, NOT chi = nuHat!
    chi = nuHat_safe / nu_laminar
    chi3 = chi ** 3
    fv1_val = chi3 / (chi3 + cv1 ** 3)
    fv2_val = 1.0 - chi / (1.0 + chi * fv1_val)
    
    inv_k2d2 = 1.0 / (kappa ** 2 * d ** 2 + 1e-20)
    S_tilde = omega + nuHat_safe * inv_k2d2 * fv2_val
    S_tilde = jnp.clip(S_tilde, min=1e-16)
    
    r_val = nuHat_safe / (S_tilde * kappa ** 2 * d ** 2 + 1e-20)
    r_val = jnp.clip(r_val, max=10.0)
    
    g_val = r_val + cw2 * (r_val ** 6 - r_val)
    
    c6 = cw3 ** 6
    fw_val = g_val * ((1.0 + c6) / (g_val ** 6 + c6)) ** (1.0 / 6.0)
    
    D = cw1 * fw_val * (nuHat_safe / d) ** 2 * mask
    return D


@jax.jit
def compute_diffusion_coefficient_safe(nuHat, sigma=2.0/3.0, nu_laminar=1.0):
    """
    Compute diffusion coefficient with safety.
    
    D = (nu_laminar + max(0, nuHat)) / sigma
    
    Parameters
    ----------
    nuHat : jnp.ndarray
        SA working variable (may contain negatives).
    sigma : float
        SA model constant (default 2/3).
    nu_laminar : float
        Laminar viscosity (default 1.0).
        
    Returns
    -------
    D : jnp.ndarray
        Diffusion coefficient, always > 0.
    """
    nuHat_safe = jnp.clip(nuHat, min=0.0)
    return (nu_laminar + nuHat_safe) / sigma
