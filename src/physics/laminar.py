"""
Laminar Flow Amplification Model for boundary layer transition.

Dimension Agnostic: Works with any array shape.

This module provides a simplified interface for boundary layer analysis scripts.
The core implementation is in src/numerics/aft_sources.py with tunable parameters.

Notes:
    The amplification model is primarily designed for boundary layer flows
    where the shape factor Gamma can be meaningfully defined. For general
    2D flows, consider using full turbulence models (SA, k-ω, etc.) instead.
"""

from typing import Union
from .jax_config import jax, jnp

# Import core AFT functions - single source of truth
from src.numerics.aft_sources import (
    compute_Re_Omega as _compute_Re_Omega,
    compute_gamma as _compute_gamma,
    compute_aft_amplification_rate as _compute_amplification_rate,
)

ArrayLike = Union[jnp.ndarray, float]


@jax.jit
def sigmoid(x: ArrayLike) -> jnp.ndarray:
    """Sigmoid function σ(x) = 1 / (1 + exp(-x))."""
    return 1 / (1 + jnp.exp(-x))


@jax.jit
def Re_Omega(dudy: ArrayLike, y: ArrayLike) -> jnp.ndarray:
    """
    Compute vorticity Reynolds number Re_Ω = y² |ω|.
    
    Boundary layer interface: uses dudy (velocity gradient) and y (wall distance).
    Equivalent to compute_Re_Omega(omega_mag=|dudy|, d=y, nu_laminar=1.0).
    """
    # In boundary layer: omega_mag ≈ |dudy|, d = y, nu = 1 (normalized)
    return _compute_Re_Omega(jnp.abs(dudy), y, nu_laminar=1.0)


@jax.jit
def compute_nondimensional_amplification_rate(Re_Omega_val: ArrayLike, Gamma: ArrayLike) -> jnp.ndarray:
    """
    Compute non-dimensional amplification rate from Re_Ω and shape factor.
    
    Uses default Drela-style correlation parameters.
    For tunable parameters, use src.numerics.aft_sources.compute_aft_amplification_rate.
    """
    return _compute_amplification_rate(Re_Omega_val, Gamma)


@jax.jit
def amplification(u: ArrayLike, dudy: ArrayLike, y: ArrayLike) -> jnp.ndarray:
    """
    Compute amplification rate for boundary layer transition.
    
    Gamma = 2(du/dy · y)² / (u² + (du/dy · y)²) characterizes the velocity profile.
    
    Parameters
    ----------
    u : array
        Streamwise velocity.
    dudy : array
        Velocity gradient ∂u/∂y (≈ vorticity in boundary layer).
    y : array
        Wall distance.
        
    Returns
    -------
    rate : array
        Non-dimensional amplification rate.
    """
    # Compute Gamma using the general formula
    # In BL: vel_mag ≈ u, omega_mag ≈ |dudy|, d = y
    Gamma = _compute_gamma(jnp.abs(dudy), u, y)
    rate = compute_nondimensional_amplification_rate(Re_Omega(dudy, y), Gamma)
    # Q4 band gate: Q4 = 1 - 2(w d)|u| / (cA A + (w d)^2 + u^2), A = (dw/dn d^2)^2,
    # cA = 4. Wall-normal gradient by nonuniform central differences along y.
    w = jnp.abs(dudy)
    dy_f = jnp.diff(y)
    dwdn_int = (w[2:] - w[:-2]) / (y[2:] - y[:-2])
    dwdn = jnp.concatenate([(w[1:2] - w[0:1]) / dy_f[0:1],
                            dwdn_int,
                            (w[-1:] - w[-2:-1]) / dy_f[-1:]])
    A4 = 4.0 * (dwdn * y * y) ** 2
    B = (w * y) ** 2
    q4 = 1.0 - 2.0 * w * y * jnp.abs(u) / (A4 + B + u * u + 1e-300)
    return rate * q4
