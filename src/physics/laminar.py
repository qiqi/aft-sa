"""
Laminar Flow Amplification Model for boundary layer transition.

Dimension Agnostic: Works with any array shape.

Notes:
    The amplification model is primarily designed for boundary layer flows
    where the shape factor Gamma can be meaningfully defined. For general
    2D flows, consider using full turbulence models (SA, k-ω, etc.) instead.
"""

from typing import Union
from .jax_config import jax, jnp

ArrayLike = Union[jnp.ndarray, float]


@jax.jit
def sigmoid(x: ArrayLike) -> jnp.ndarray:
    """Sigmoid function σ(x) = 1 / (1 + exp(-x))."""
    return 1 / (1 + jnp.exp(-x))


@jax.jit
def Re_Omega(dudy: ArrayLike, y: ArrayLike) -> jnp.ndarray:
    """Compute vorticity Reynolds number Re_Ω = y² |ω|."""
    return y**2 * jnp.abs(dudy)


@jax.jit
def compute_nondimensional_amplification_rate(Re_Omega_val: ArrayLike, Gamma: ArrayLike) -> jnp.ndarray:
    """Compute non-dimensional amplification rate from Re_Ω and shape factor."""
    a: jnp.ndarray = jnp.log10(jnp.abs(Re_Omega_val) / 1000) / 50 + Gamma
    return 0.2 / (1 + jnp.exp(-35 * (a - 1.04)))


@jax.jit
def amplification(u: ArrayLike, dudy: ArrayLike, y: ArrayLike) -> jnp.ndarray:
    """
    Compute amplification rate for boundary layer transition.
    
    Gamma = 2(du/dy · y)² / (u² + (du/dy · y)²) characterizes the velocity profile.
    """
    Gamma: jnp.ndarray = 2 * (dudy * y)**2 / (u**2 + (dudy * y)**2)
    return compute_nondimensional_amplification_rate(Re_Omega(dudy, y), Gamma)
