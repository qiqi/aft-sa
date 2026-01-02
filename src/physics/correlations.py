"""
Drela-Giles (1987) transition correlations for boundary layer flows.

These correlations are used to predict laminar-turbulent transition
based on the shape factor H and Mach number.
"""

from .jax_config import jax, jnp


@jax.jit
def dN_dRe_theta(H, Mach=0.0):
    """
    Calculate dN/dRe_theta using the Drela-Giles (1987) correlation.
    
    Source: Eqn 29 and Eqn 9 from Drela & Giles (1987).
    
    Parameters
    ----------
    H : jnp.ndarray
        Shape factor (any shape).
    Mach : float or jnp.ndarray
        Mach number (default 0.0).
        
    Returns
    -------
    slope : jnp.ndarray
        dN/dRe_theta (same shape as H).
    """
    # Convert H to Kinematic Shape Factor Hk (Eqn 9)
    Hk = (H - 0.290 * Mach**2) / (1.0 + 0.113 * Mach**2)

    # Calculate the slope (Eqn 29)
    inner_term = 2.4 * Hk - 3.7 + 2.5 * jnp.tanh(1.5 * Hk - 4.65)
    slope = 0.01 * jnp.sqrt(inner_term**2 + 0.25)

    return slope


@jax.jit
def Re_theta0(H, Mach=0.0):
    """
    Calculate Critical Re_theta0 using the Drela-Giles (1987) correlation.
    
    Source: Eqn 30 and Eqn 9 from Drela & Giles (1987).
    
    Parameters
    ----------
    H : jnp.ndarray
        Shape factor (any shape).
    Mach : float or jnp.ndarray
        Mach number (default 0.0).
        
    Returns
    -------
    Re_theta0 : jnp.ndarray
        Critical Reynolds number (same shape as H).
    """
    # Convert H to Kinematic Shape Factor Hk (Eqn 9)
    Hk = (H - 0.290 * Mach**2) / (1.0 + 0.113 * Mach**2)

    # Safety Clip for Hk (singularity at Hk=1.0)
    Hk = jnp.clip(Hk, min=1.05)

    # Calculate log10(Re_theta0) (Eqn 30)
    term1 = (1.415 / (Hk - 1)) - 0.489
    term2 = jnp.tanh((20.0 / (Hk - 1)) - 12.9)
    term3 = (3.295 / (Hk - 1)) + 0.44

    log_Re_theta0 = term1 * term2 + term3

    return 10.0**log_Re_theta0


@jax.jit
def compute_nondimensional_spatial_rate(H, Mach=0.0):
    """
    Calculate the nondimensional spatial envelope growth rate: θ · dN/dx.
    
    Combines the Envelope Slope (Eq 29) with the Falkner-Skan
    spatial transformation (Eqs 33, 34) from Drela & Giles (1987).
    
    Parameters
    ----------
    H : jnp.ndarray
        Shape factor (any shape).
    Mach : float or jnp.ndarray
        Mach number (default 0.0).
        
    Returns
    -------
    theta_dN_dx : jnp.ndarray
        Growth rate of N per unit (x/theta).
        Values: ~0.002 (Blasius) to ~0.03 (Separation).
    """
    # Kinematic Shape Factor Hk (Eq 9)
    Hk = (H - 0.290 * Mach**2) / (1.0 + 0.113 * Mach**2)

    # Clip Hk for safety (singularity at Hk=1.0)
    Hk = jnp.clip(Hk, min=1.05)

    # Envelope Slope dN/dRe_theta (Eq 29)
    inner_term = 2.4 * Hk - 3.7 + 2.5 * jnp.tanh(1.5 * Hk - 4.65)
    dN_dRe = 0.01 * jnp.sqrt(inner_term**2 + 0.25)

    # Spatial Transformation Factors (Eqs 32-34)
    # l(Hk) from Eq 33
    l_Hk = (6.54 * Hk - 14.07) / (Hk**2)

    # product l(Hk) * m(Hk) derived from Eq 34
    term_m = 0.058 * ((Hk - 4.0)**2) / (Hk - 1.0)
    l_times_m = term_m - 0.068

    # The geometric conversion factor
    spatial_conversion = 0.5 * (l_times_m + l_Hk)

    # Clip to prevent negative growth rates
    spatial_conversion = jnp.clip(spatial_conversion, min=0.0)

    # Final Calculation
    theta_dN_dx = dN_dRe * spatial_conversion

    return theta_dN_dx
