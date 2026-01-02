"""
Laminar Flow Amplification Model for boundary layer transition.

Dimension Agnostic: Works with any array shape.

Notes:
    The amplification model is primarily designed for boundary layer flows
    where the shape factor Gamma can be meaningfully defined. For general
    2D flows, consider using full turbulence models (SA, k-ω, etc.) instead.
"""

from .jax_config import jax, jnp


@jax.jit
def sigmoid(x):
    """
    Sigmoid function σ(x) = 1 / (1 + exp(-x)).
    
    Parameters
    ----------
    x : jnp.ndarray
        Input array (any shape).
        
    Returns
    -------
    result : jnp.ndarray
        Sigmoid of x (same shape as input).
    """
    return 1 / (1 + jnp.exp(-x))


@jax.jit
def Re_Omega(dudy, y):
    """
    Compute vorticity Reynolds number Re_Ω = y² |ω|.
    
    Parameters
    ----------
    dudy : jnp.ndarray
        Vorticity magnitude |ω| (any shape).
        For boundary layers: |du/dy|
    y : jnp.ndarray
        Wall distance (same shape as dudy).
        
    Returns
    -------
    Re_Omega : jnp.ndarray
        Vorticity Reynolds number (same shape as inputs).
    """
    return y**2 * jnp.abs(dudy)


@jax.jit
def compute_nondimensional_amplification_rate(Re_Omega_val, Gamma):
    """
    Compute non-dimensional amplification rate from Re_Ω and shape factor.
    
    Parameters
    ----------
    Re_Omega_val : jnp.ndarray
        Vorticity Reynolds number (any shape).
    Gamma : jnp.ndarray
        Shape factor parameter (same shape as Re_Omega_val).
        
    Returns
    -------
    amp_rate : jnp.ndarray
        Non-dimensional amplification rate (same shape as inputs).
    """
    a = jnp.log10(jnp.abs(Re_Omega_val) / 1000) / 50 + Gamma
    return 0.2 / (1 + jnp.exp(-35 * (a - 1.04)))


@jax.jit
def amplification(u, dudy, y):
    """
    Compute amplification rate for boundary layer transition.
    
    This function computes the local amplification rate based on the
    velocity profile shape. It is primarily intended for boundary layer
    flows where the shape factor Gamma is meaningful.
    
    Parameters
    ----------
    u : jnp.ndarray
        Streamwise velocity (any shape).
    dudy : jnp.ndarray
        Velocity gradient |du/dy| (same shape as u).
    y : jnp.ndarray
        Wall distance (same shape as u).
        
    Returns
    -------
    amp : jnp.ndarray
        Amplification rate (same shape as inputs).
        
    Notes
    -----
    Gamma = 2(du/dy · y)² / (u² + (du/dy · y)²) is a shape factor that
    characterizes the velocity profile. For Blasius boundary layer, 
    Gamma varies from 0 at the wall to ~0.5 in the freestream.
    """
    Gamma = 2 * (dudy * y)**2 / (u**2 + (dudy * y)**2)
    return compute_nondimensional_amplification_rate(Re_Omega(dudy, y), Gamma)
