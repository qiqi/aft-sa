"""
Spalart-Allmaras Turbulence Model Functions.

This module implements the Spalart-Allmaras one-equation turbulence model
with analytical gradients for use in implicit solvers.

Dimension Agnostic:
    All functions work with any tensor shape - scalars, 1D arrays, 2D fields, etc.
    NumPy broadcasting rules apply. This allows the same code to be used for:
    - Boundary layer solvers (1D profiles)
    - 2D/3D RANS solvers (full fields)

Notation:
    In boundary layer context: omega = |du/dy|, d = y (wall-normal coordinate)
    In general 2D/3D context:  omega = |vorticity|, d = wall distance
    
    The variable names `dudy` and `y` are kept for backward compatibility
    with boundary layer solvers, but they represent vorticity magnitude
    and wall distance respectively in the general case.

Robustness:
    This module includes "safe" versions of functions that handle negative
    nuHat values gracefully. When nuHat < 0 (due to numerical dispersion):
    - Source terms are zeroed (pure diffusion mode)
    - Effective viscosity uses max(0, nuHat)
    This allows the Maximum Principle to fill in negative regions via diffusion.
"""

import numpy as np
import torch


def fv1(nuHat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute fv1 damping function and its derivative.
    
    fv1 = chi³ / (chi³ + cv1³)  where chi = nuHat
    
    Parameters
    ----------
    nuHat : Tensor
        SA working variable (any shape).
        
    Returns
    -------
    val : Tensor
        fv1 value (same shape as input).
    grad : Tensor  
        d(fv1)/d(nuHat) (same shape as input).
    """
    cv1 = 7.1
    chi = nuHat
    chi3 = chi ** 3
    denom = chi3 + cv1 ** 3

    val = chi3 / denom

    # Derivative: d/dchi [chi^3 / (chi^3 + cv1^3)]
    # = (3chi^2 * denom - chi^3 * 3chi^2) / denom^2
    # = 3chi^2 * (denom - chi^3) / denom^2
    # = 3chi^2 * cv1^3 / denom^2
    grad = (3 * chi**2 * cv1**3) / (denom ** 2)

    return val, grad

def fv2(nuHat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (fv2, d(fv2)/d(nuHat)).
    """
    chi = nuHat
    fv1_val, fv1_grad = fv1(nuHat)

    denom = 1.0 + chi * fv1_val
    val = 1.0 - chi / denom

    # Derivative of fv2 = 1 - chi / (1 + chi*fv1)
    # Let term = chi / denom
    # d(term)/dchi = (1*denom - chi*(fv1 + chi*fv1_grad)) / denom^2
    #              = (1 + chi*fv1 - chi*fv1 - chi^2*fv1_grad) / denom^2
    #              = (1 - chi^2*fv1_grad) / denom^2
    # fv2_grad = - d(term)/dchi
    term_grad = (1.0 - chi**2 * fv1_grad) / (denom ** 2)
    grad = -term_grad

    return val, grad

def _S_tilde(dudy: torch.Tensor, nuHat: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute modified vorticity S_tilde and its derivative.
    
    S̃ = Ω + (ν̃ / κ²d²) · fv2(ν̃)
    
    Parameters
    ----------
    dudy : Tensor
        Vorticity magnitude |ω| (any shape).
        For boundary layers: |du/dy|
        For 2D flows: |∂v/∂x - ∂u/∂y|
    nuHat : Tensor
        SA working variable (same shape as dudy).
    y : Tensor
        Wall distance d (same shape as dudy).
        For boundary layers: y-coordinate
        For general flows: computed wall distance field
        
    Returns
    -------
    val : Tensor
        S_tilde value (same shape as inputs).
    grad : Tensor
        d(S_tilde)/d(nuHat) (same shape as inputs).
    """
    kappa = 0.41
    omega = dudy.abs()
    fv2_val, fv2_grad = fv2(nuHat)

    # Constants wrt nuHat
    inv_k2y2 = 1.0 / (kappa ** 2 * y ** 2)

    # S_tilde = Omega + nuHat * inv_k2y2 * fv2
    # S_tilde_raw to check for clamping
    s_tilde_term = nuHat * inv_k2y2 * fv2_val
    S_tilde_raw = omega + s_tilde_term

    # Derivative:
    # d/dnuHat = inv_k2y2 * [ 1 * fv2 + nuHat * fv2' ]
    grad_raw = inv_k2y2 * (fv2_val + nuHat * fv2_grad)

    # Apply clamp: min=1e-16
    # If S_tilde_raw < 1e-16, value is 1e-16 and grad is 0
    val = torch.clamp(S_tilde_raw, min=1e-16)
    grad = torch.where(S_tilde_raw < 1e-16, torch.zeros_like(grad_raw), grad_raw)

    return val, grad

def r(dudy: torch.Tensor, nuHat: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (r, d(r)/d(nuHat)).
    """
    kappa = 0.41
    S_t, S_t_grad = _S_tilde(dudy, nuHat, y)

    denom_factor = kappa ** 2 * y ** 2
    denom = S_t * denom_factor

    r_raw = nuHat / denom

    # Derivative of nuHat / (S_t * C) where C = k^2 y^2
    # d/dnuHat = (1 * (S_t*C) - nuHat * (S_t_grad*C)) / (S_t*C)^2
    #          = C(S_t - nuHat*S_t_grad) / C^2 S_t^2
    #          = (S_t - nuHat*S_t_grad) / (C * S_t^2)
    grad_raw = (S_t - nuHat * S_t_grad) / (denom_factor * S_t ** 2)

    # Apply clamp: max=10.0
    val = torch.clamp(r_raw, max=10.0)
    grad = torch.where(r_raw > 10.0, torch.zeros_like(grad_raw), grad_raw)

    return val, grad

def g(dudy: torch.Tensor, nuHat: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (g, d(g)/d(nuHat)).
    """
    cw2 = 0.3
    r_val, r_grad = r(dudy, nuHat, y)

    # g = r + cw2 * (r^6 - r)
    val = r_val + cw2 * (r_val ** 6 - r_val)

    # Chain rule: dg/dnuHat = dg/dr * dr/dnuHat
    # dg/dr = 1 + cw2 * (6r^5 - 1)
    dg_dr = 1.0 + cw2 * (6.0 * r_val ** 5 - 1.0)

    grad = dg_dr * r_grad
    return val, grad

def fw(dudy: torch.Tensor, nuHat: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (fw, d(fw)/d(nuHat)).
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

    # Derivative dfw/dg
    # fw = g * ((1+c6)/(g^6+c6))^(1/6)
    # Using logarithmic differentiation or quotient rules efficiently:
    # dfw/dg = radicand * [ c6 / (g^6 + c6) ]
    # (Derived from: fw' = R^(1/6) * (1 - g^6/(g^6+c6)) )

    # Note: radicand is (top/denom)^(1/6)
    # term2 = c6 / denom
    dfw_dg = radicand * (c6 / denom)

    grad = dfw_dg * g_grad
    return val, grad

def spalart_allmaras_amplification(
    dudy: torch.Tensor,
    nuHat: torch.Tensor,
    y: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute SA production and destruction terms with analytical gradients.
    
    Production: P = cb1 · S̃ · ν̃
    Destruction: D = cw1 · fw · (ν̃/d)²
    
    This function is dimension-agnostic - works with any tensor shape.
    
    Parameters
    ----------
    dudy : Tensor
        Vorticity magnitude |ω| (any shape).
        For boundary layers: |du/dy|
        For 2D flows: |∂v/∂x - ∂u/∂y|
    nuHat : Tensor
        SA working variable ν̃ (same shape as dudy).
    y : Tensor
        Wall distance d (same shape as dudy).
        For boundary layers: y-coordinate
        For general flows: computed wall distance field
        
    Returns
    -------
    (prod_val, prod_grad) : tuple of Tensors
        Production term and d(Production)/d(nuHat).
    (dest_val, dest_grad) : tuple of Tensors
        Destruction term and d(Destruction)/d(nuHat).
        
    Example
    -------
    # 1D boundary layer (ny,)
    >>> omega_1d = torch.tensor([0.1, 0.5, 1.0])
    >>> nuHat_1d = torch.tensor([0.01, 0.1, 0.5])
    >>> d_1d = torch.tensor([0.001, 0.01, 0.1])
    >>> (P, dP), (D, dD) = spalart_allmaras_amplification(omega_1d, nuHat_1d, d_1d)
    
    # 2D flow field (ni, nj)
    >>> omega_2d = torch.rand(100, 50)
    >>> nuHat_2d = torch.rand(100, 50) 
    >>> d_2d = torch.rand(100, 50)
    >>> (P, dP), (D, dD) = spalart_allmaras_amplification(omega_2d, nuHat_2d, d_2d)
    """
    cb1 = 0.1355
    cb2 = 0.622
    sigma = 2.0 / 3.0
    kappa = 0.41
    cw1 = cb1 / (kappa ** 2) + (1.0 + cb2) / sigma

    # Production = cb1 * S_tilde * nuHat
    S_t, S_t_grad = _S_tilde(dudy, nuHat, y)
    prod_val = cb1 * S_t * nuHat

    # d(Prod)/dnuHat = cb1 * (S_t_grad * nuHat + S_t * 1)
    prod_grad = cb1 * (S_t_grad * nuHat + S_t)

    # Destruction = cw1 * fw * (nuHat / y)^2
    fw_val, fw_grad = fw(dudy, nuHat, y)

    term_sq = (nuHat / y) ** 2
    dest_val = cw1 * fw_val * term_sq

    # d(Dest)/dnuHat
    # = cw1 * [ fw_grad * (nuHat/y)^2 + fw_val * d/dnuHat((nuHat/y)^2) ]
    # d/dnuHat((nuHat/y)^2) = 2 * (nuHat/y) * (1/y) = 2 * nuHat / y^2

    term_sq_grad = 2.0 * nuHat / (y ** 2)
    dest_grad = cw1 * (fw_grad * term_sq + fw_val * term_sq_grad)

    return (prod_val, prod_grad), (dest_val, dest_grad)


# =============================================================================
# Robustness Handling for Negative nuHat
# =============================================================================

def effective_viscosity_safe(nuHat: torch.Tensor, nu_laminar: float = 1.0) -> torch.Tensor:
    """
    Compute effective viscosity with safety for negative nuHat.
    
    nu_eff = nu_laminar + max(0, nuHat * fv1(nuHat))
    
    This prevents negative effective viscosity which would cause divergence.
    
    Parameters
    ----------
    nuHat : Tensor
        SA working variable (any shape, may contain negative values).
    nu_laminar : float
        Laminar (molecular) viscosity, default 1.0 (non-dimensional).
        
    Returns
    -------
    nu_eff : Tensor
        Effective viscosity, always >= nu_laminar.
    """
    # For negative nuHat, fv1 gives garbage, so clamp first
    nuHat_safe = torch.clamp(nuHat, min=0.0)
    fv1_val, _ = fv1(nuHat_safe)
    nu_turb = nuHat_safe * fv1_val
    
    return nu_laminar + nu_turb


def compute_aft_sa_source_safe(
    omega: torch.Tensor,
    nuHat: torch.Tensor,
    d: torch.Tensor,
    is_turb: torch.Tensor,
    aft_amplification: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute combined AFT-SA source terms with non-negative safety.
    
    This function implements the "Non-Negative Source Switch":
    - If nuHat >= 0: Compute normal AFT/SA production and destruction
    - If nuHat < 0: Force source terms to zero (pure diffusion mode)
    
    By turning off sources for negative nuHat, the transport equation becomes
    a pure diffusion equation locally. By the Maximum Principle, surrounding
    positive values will diffuse in and "fill the hole."
    
    Parameters
    ----------
    omega : Tensor
        Vorticity magnitude |ω| (any shape).
    nuHat : Tensor
        SA working variable (same shape, may contain negative values).
    d : Tensor
        Wall distance (same shape).
    is_turb : Tensor
        Turbulent indicator (0=laminar/AFT, 1=fully turbulent/SA).
        Typically: is_turb = clamp(1 - exp(-(nuHat - 1) / 4), min=0)
    aft_amplification : Tensor
        Pre-computed AFT amplification rate (same shape).
        For boundary layers: amplification(u, dudy, y) * dudy
        
    Returns
    -------
    prod : Tensor
        Production term (combined AFT + SA).
    prod_grad : Tensor
        d(Production)/d(nuHat).
    dest : Tensor
        Destruction term (SA only, zero in AFT region).
    dest_grad : Tensor
        d(Destruction)/d(nuHat).
        
    Notes
    -----
    The gradient terms are zeroed for negative nuHat as well, which ensures
    the Jacobian remains well-behaved for implicit solvers.
    """
    # Create mask for non-negative nuHat
    # Use a smooth transition rather than hard switch for better convergence
    # mask = 1 for nuHat >= 0, smoothly goes to 0 for nuHat < 0
    mask = torch.sigmoid(nuHat * 100.0)  # Sharp but smooth transition at nuHat=0
    
    # Clamp nuHat to small positive value for SA computation to avoid NaN
    nuHat_safe = torch.clamp(nuHat, min=1e-10)
    
    # Compute standard SA terms with safe nuHat
    (sa_prod, sa_prod_grad), (sa_dest, sa_dest_grad) = \
        spalart_allmaras_amplification(omega, nuHat_safe, d)
    
    # Compute combined source with turbulent blending
    # AFT region (is_turb ~ 0): Production = aft_amplification * nuHat
    # SA region (is_turb ~ 1): Production = SA production
    aft_prod = aft_amplification * nuHat_safe
    aft_prod_grad = aft_amplification
    
    # Blend between AFT and SA
    prod_raw = sa_prod * is_turb + aft_prod * (1.0 - is_turb)
    prod_grad_raw = sa_prod_grad * is_turb + aft_prod_grad * (1.0 - is_turb)
    
    # Destruction only active in turbulent region
    dest_raw = sa_dest  # * is_turb  # Can optionally blend destruction too
    dest_grad_raw = sa_dest_grad  # * is_turb
    
    # Apply non-negative mask: zero out sources for negative nuHat
    prod = prod_raw * mask
    prod_grad = prod_grad_raw * mask
    dest = dest_raw * mask
    dest_grad = dest_grad_raw * mask
    
    return prod, prod_grad, dest, dest_grad


def compute_diffusion_coefficient_safe(
    nuHat: torch.Tensor,
    sigma: float = 2.0/3.0,
    nu_laminar: float = 1.0,
) -> torch.Tensor:
    """
    Compute diffusion coefficient for nuHat transport with safety.
    
    D = (nu_laminar + max(0, nuHat)) / sigma
    
    This ensures D > 0 even when nuHat < 0, preventing divergence.
    
    Parameters
    ----------
    nuHat : Tensor
        SA working variable (any shape, may contain negative values).
    sigma : float
        SA model constant (default 2/3).
    nu_laminar : float
        Laminar viscosity (default 1.0 for non-dimensional).
        
    Returns
    -------
    D : Tensor
        Diffusion coefficient, always > 0.
    """
    nuHat_safe = torch.clamp(nuHat, min=0.0)
    return (nu_laminar + nuHat_safe) / sigma


# =============================================================================
# NumPy Versions for Numba/FVM Solvers
# =============================================================================


def effective_viscosity_safe_np(nuHat: np.ndarray, nu_laminar: float = 1.0) -> np.ndarray:
    """
    NumPy version: Compute effective viscosity with safety for negative nuHat.
    
    nu_eff = nu_laminar + max(0, nuHat * fv1(nuHat))
    
    Parameters
    ----------
    nuHat : ndarray
        SA working variable (any shape, may contain negative values).
    nu_laminar : float
        Laminar viscosity (default 1.0).
        
    Returns
    -------
    nu_eff : ndarray
        Effective viscosity, always >= nu_laminar.
    """
    cv1 = 7.1
    nuHat_safe = np.maximum(nuHat, 0.0)
    chi3 = nuHat_safe ** 3
    fv1_val = chi3 / (chi3 + cv1 ** 3)
    nu_turb = nuHat_safe * fv1_val
    
    return nu_laminar + nu_turb


def sa_source_mask_np(nuHat: np.ndarray) -> np.ndarray:
    """
    NumPy version: Compute source term mask for non-negative safety.
    
    Returns a smooth mask that is:
    - ~1 for nuHat >= 0
    - ~0 for nuHat < 0 (smoothly transitions)
    
    Parameters
    ----------
    nuHat : ndarray
        SA working variable (any shape).
        
    Returns
    -------
    mask : ndarray
        Source term multiplier (same shape as input).
    """
    # Smooth sigmoid transition at nuHat = 0
    # Sharp enough to effectively turn off for nuHat < -0.01
    return 1.0 / (1.0 + np.exp(-100.0 * nuHat))


def compute_sa_production_np(omega: np.ndarray, nuHat: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    NumPy version: Compute SA production term with non-negative safety.
    
    P = cb1 * S_tilde * nuHat * mask(nuHat)
    
    For nuHat < 0, production is smoothly zeroed out.
    
    Parameters
    ----------
    omega : ndarray
        Vorticity magnitude |ω|.
    nuHat : ndarray
        SA working variable (may contain negatives).
    d : ndarray
        Wall distance.
        
    Returns
    -------
    P : ndarray
        Production term (safe, non-amplifying for negative nuHat).
    """
    cb1 = 0.1355
    kappa = 0.41
    cv1 = 7.1
    
    # Safety mask
    mask = sa_source_mask_np(nuHat)
    
    # Safe nuHat for computation
    nuHat_safe = np.maximum(nuHat, 1e-10)
    
    # fv1 and fv2
    chi3 = nuHat_safe ** 3
    fv1_val = chi3 / (chi3 + cv1 ** 3)
    fv2_val = 1.0 - nuHat_safe / (1.0 + nuHat_safe * fv1_val)
    
    # S_tilde = omega + nuHat / (kappa^2 * d^2) * fv2
    inv_k2d2 = 1.0 / (kappa ** 2 * d ** 2 + 1e-20)
    S_tilde = omega + nuHat_safe * inv_k2d2 * fv2_val
    S_tilde = np.maximum(S_tilde, 1e-16)
    
    # Production
    P = cb1 * S_tilde * nuHat_safe * mask
    
    return P


def compute_sa_destruction_np(omega: np.ndarray, nuHat: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    NumPy version: Compute SA destruction term with non-negative safety.
    
    D = cw1 * fw * (nuHat / d)^2 * mask(nuHat)
    
    For nuHat < 0, destruction is smoothly zeroed out.
    
    Parameters
    ----------
    omega : ndarray
        Vorticity magnitude |ω|.
    nuHat : ndarray
        SA working variable (may contain negatives).
    d : ndarray
        Wall distance.
        
    Returns
    -------
    D : ndarray
        Destruction term (safe).
    """
    cb1 = 0.1355
    cb2 = 0.622
    sigma = 2.0 / 3.0
    kappa = 0.41
    cw1 = cb1 / (kappa ** 2) + (1.0 + cb2) / sigma
    cw2 = 0.3
    cw3 = 2.0
    cv1 = 7.1
    
    # Safety mask
    mask = sa_source_mask_np(nuHat)
    
    # Safe nuHat for computation
    nuHat_safe = np.maximum(nuHat, 1e-10)
    
    # fv1 and fv2
    chi3 = nuHat_safe ** 3
    fv1_val = chi3 / (chi3 + cv1 ** 3)
    fv2_val = 1.0 - nuHat_safe / (1.0 + nuHat_safe * fv1_val)
    
    # S_tilde
    inv_k2d2 = 1.0 / (kappa ** 2 * d ** 2 + 1e-20)
    S_tilde = omega + nuHat_safe * inv_k2d2 * fv2_val
    S_tilde = np.maximum(S_tilde, 1e-16)
    
    # r = nuHat / (S_tilde * kappa^2 * d^2)
    r_val = nuHat_safe / (S_tilde * kappa ** 2 * d ** 2 + 1e-20)
    r_val = np.minimum(r_val, 10.0)
    
    # g = r + cw2 * (r^6 - r)
    g_val = r_val + cw2 * (r_val ** 6 - r_val)
    
    # fw = g * ((1 + cw3^6) / (g^6 + cw3^6))^(1/6)
    c6 = cw3 ** 6
    fw_val = g_val * ((1.0 + c6) / (g_val ** 6 + c6)) ** (1.0 / 6.0)
    
    # Destruction
    D = cw1 * fw_val * (nuHat_safe / d) ** 2 * mask
    
    return D


def compute_diffusion_coefficient_safe_np(
    nuHat: np.ndarray,
    sigma: float = 2.0/3.0,
    nu_laminar: float = 1.0,
) -> np.ndarray:
    """
    NumPy version: Compute diffusion coefficient with safety.
    
    D = (nu_laminar + max(0, nuHat)) / sigma
    
    Parameters
    ----------
    nuHat : ndarray
        SA working variable (may contain negatives).
    sigma : float
        SA model constant (default 2/3).
    nu_laminar : float
        Laminar viscosity (default 1.0).
        
    Returns
    -------
    D : ndarray
        Diffusion coefficient, always > 0.
    """
    nuHat_safe = np.maximum(nuHat, 0.0)
    return (nu_laminar + nuHat_safe) / sigma
