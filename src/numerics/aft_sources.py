"""
AFT (Amplification Factor Transport) Source Terms for Laminar Transition.

This module implements the laminar instability amplification model that
drives transition in the AFT-SA combined model. The amplification rate
is based on Drela-style correlations relating growth rate to:
- Vorticity Reynolds number Re_Ω = d² |ω|
- Shape factor Γ characterizing the velocity profile

All constants are exposed as tunable parameters for ensemble studies
and future autodiff-based optimization.

References:
- Drela & Giles (1987), "Viscous-inviscid analysis..."
- src/physics/laminar.py for original boundary layer formulation
"""

from typing import Union
from src.physics.jax_config import jax, jnp

ArrayLike = Union[jnp.ndarray, float]

# =============================================================================
# DEFAULT CONSTANTS - All tunable for parameter studies
# =============================================================================

# Gamma calculation
AFT_GAMMA_COEFF = 2.0          # Coefficient in Γ numerator

# Amplification rate formula
AFT_RE_OMEGA_SCALE = 1000.0    # Re_Ω scaling in log term
AFT_LOG_DIVISOR = 50.0         # Divisor in log term
AFT_SIGMOID_CENTER = 1.04      # Center of sigmoid activation
AFT_SIGMOID_SLOPE = 35.0       # Steepness of sigmoid
AFT_RATE_SCALE = 0.2           # Overall rate magnitude


# =============================================================================
# CORE AFT FUNCTIONS
# =============================================================================

@jax.jit
def compute_gamma(omega_mag: ArrayLike, vel_mag: ArrayLike, d: ArrayLike,
                  gamma_coeff: float = 2.0) -> jnp.ndarray:
    """
    Compute shape factor Γ characterizing the velocity profile.
    
    FORMULA:
        Γ = 2 * (|ω|·d)² / (|V|² + (|ω|·d)²)
    
    PHYSICAL MEANING:
        Γ measures the "fullness" of the velocity profile:
        - Γ → 0: Velocity-dominated (flat profile, stable)
        - Γ → 2: Shear-dominated (inflectional profile, unstable)
    
    BOUNDARY LAYER CALIBRATION (gamma_coeff = 2.0):
        In a boundary layer near the wall, velocity varies linearly:
            |V| ≈ (∂u/∂y)|_wall × y
        And vorticity is approximately constant:
            |ω| ≈ (∂u/∂y)|_wall
        Therefore:
            |ω| × d ≈ (∂u/∂y)|_wall × y ≈ |V|
        
        This means (|ω|·d)² ≈ |V|², so:
            Γ = 2 × 0.5 = 1.0 at the wall
        
        The coefficient 2.0 is thus chosen so that Γ = 1 in the viscous
        sublayer of a canonical boundary layer profile.
    
    Parameters
    ----------
    omega_mag : array
        Vorticity magnitude |ω|.
    vel_mag : array
        Velocity magnitude |V| = sqrt(u² + v²).
    d : array
        Wall distance.
    gamma_coeff : float
        Coefficient (default 2.0, calibrated for Γ=1 at wall).
        
    Returns
    -------
    Gamma : array
        Shape factor Γ in [0, gamma_coeff].
    """
    omega_d = omega_mag * d
    omega_d_sq = omega_d ** 2
    vel_sq = vel_mag ** 2
    # Avoid 0/0: when both vel and omega_d are zero, Gamma is undefined but set to 0
    denom = vel_sq + omega_d_sq
    # Hardcoded coeff per user request
    return jnp.where(denom > 0, 2.0 * omega_d_sq / denom, 0.0)


@jax.jit
def compute_Re_Omega(omega_mag: ArrayLike, d: ArrayLike, 
                     nu_laminar: float = 1.0) -> jnp.ndarray:
    """
    Compute vorticity Reynolds number.
    
    FORMULA:
        Re_Ω = d² |ω| / ν
    
    PHYSICAL MEANING:
        Re_Ω characterizes the local instability potential:
        - Low Re_Ω: viscous damping dominates (stable)
        - High Re_Ω: instability growth possible
    
    NOTE: The division by nu_laminar makes this truly dimensionless.
    For boundary layer solvers where ν = 1 (normalized units), this
    reduces to the original formula Re_Ω = d² |ω|.
    
    Parameters
    ----------
    omega_mag : array
        Vorticity magnitude |ω|.
    d : array
        Wall distance.
    nu_laminar : float
        Laminar kinematic viscosity (default 1.0 for normalized BL solvers).
        
    Returns
    -------
    Re_Omega : array
        Vorticity Reynolds number (dimensionless).
    """
    return d ** 2 * omega_mag / nu_laminar


@jax.jit
def compute_aft_amplification_rate(Re_Omega: ArrayLike, Gamma: ArrayLike,
                                   re_scale: float = AFT_RE_OMEGA_SCALE,
                                   log_divisor: float = AFT_LOG_DIVISOR,
                                   sigmoid_center: float = AFT_SIGMOID_CENTER,
                                   sigmoid_slope: float = AFT_SIGMOID_SLOPE,
                                   rate_scale: float = AFT_RATE_SCALE) -> jnp.ndarray:
    """
    Compute non-dimensional amplification rate from Drela-style correlation.
    
    FORMULA:
        a = log10(Re_Ω / re_scale) / log_divisor + Γ
        rate = rate_scale / (1 + exp(-sigmoid_slope * (a - sigmoid_center)))
    
    PHYSICAL MEANING:
        The rate represents how fast TS waves amplify. The sigmoid ensures:
        - rate → 0 for small a (stable conditions)
        - rate → rate_scale for large a (unstable conditions)
        - Transition at a ≈ sigmoid_center
    
    TUNABLE PARAMETERS (for ensemble studies):
        - re_scale: Threshold Re_Ω for instability onset
        - log_divisor: Sensitivity to Re_Ω changes
        - sigmoid_center: Critical value of a for transition
        - sigmoid_slope: Sharpness of transition
        - rate_scale: Maximum growth rate
    
    Parameters
    ----------
    Re_Omega : array
        Vorticity Reynolds number.
    Gamma : array
        Shape factor.
    re_scale, log_divisor, sigmoid_center, sigmoid_slope, rate_scale : float
        Tunable correlation parameters.
        
    Returns
    -------
    rate : array
        Non-dimensional amplification rate (dimensionless).
    """
    # Activation variable
    # User modification: replace Gamma < 1 part with log(Gamma) + 1 (for continuity at 1)
    Gamma_mod = jnp.where(Gamma < 1.0, jnp.log(jnp.maximum(Gamma * 2 - 1, 1e-20)) / 2 + 1.0, Gamma)
    a = jnp.log10(jnp.abs(Re_Omega) / re_scale + 1e-20) / log_divisor + Gamma_mod
    
    # Sigmoid activation (using jax.nn.sigmoid for stability)
    # x = -slope * (a - center)
    # sigmoid(x) = 1 / (1 + exp(-x))
    # We want rate = rate_scale / (1 + exp(-slope*(a-center)))
    # This is equivalent to rate_scale * sigmoid(slope * (a - center))
    x_arg = sigmoid_slope * (a - sigmoid_center)
    return rate_scale * jax.nn.sigmoid(x_arg)


@jax.jit
def compute_aft_production(omega_mag: ArrayLike, vel_mag: ArrayLike,
                           d: ArrayLike, nuHat: ArrayLike,
                           nu_laminar: float = 1.0,
                           # Tunable parameters
                           gamma_coeff: float = AFT_GAMMA_COEFF,
                           re_scale: float = AFT_RE_OMEGA_SCALE,
                           log_divisor: float = AFT_LOG_DIVISOR,
                           sigmoid_center: float = AFT_SIGMOID_CENTER,
                           sigmoid_slope: float = AFT_SIGMOID_SLOPE,
                           rate_scale: float = AFT_RATE_SCALE) -> jnp.ndarray:
    """
    Compute AFT production term for laminar instability growth.
    
    FORMULA:
        P_aft = rate * |ω| * nuHat   (for nuHat > 0)
        P_aft = 0                     (for nuHat <= 0)
    
    DIMENSIONAL ANALYSIS:
        [rate] = dimensionless
        [|ω|] = 1/time
        [nuHat] = length²/time
        [P_aft] = length²/time² (same as SA production)
    
    NOTE: The factor of nuHat ensures:
        1. Same dimensions as SA production (P = cb1·S̃·ν̃)
        2. Zero production when nuHat <= 0 (stability)
        3. Growth rate proportional to current disturbance level
    
    Parameters
    ----------
    omega_mag : array
        Vorticity magnitude |ω|.
    vel_mag : array
        Velocity magnitude |V|.
    d : array
        Wall distance.
    nuHat : array
        SA working variable ν̃.
    nu_laminar : float
        Laminar kinematic viscosity (1/Re). Used to make Re_Omega dimensionless.
    gamma_coeff, re_scale, log_divisor, sigmoid_center, sigmoid_slope, rate_scale : float
        Tunable correlation parameters.
        
    Returns
    -------
    P_aft : array
        AFT production term with same dimensions as SA production.
    """
    # Compute shape factor and vorticity Reynolds number
    Gamma = compute_gamma(omega_mag, vel_mag, d, gamma_coeff)
    Re_Omega = compute_Re_Omega(omega_mag, d, nu_laminar)
    
    # Non-dimensional amplification rate
    rate = compute_aft_amplification_rate(
        Re_Omega, Gamma, re_scale, log_divisor, sigmoid_center, sigmoid_slope, rate_scale
    )
    
    # Production = rate * omega * nuHat (zero for nuHat <= 0)
    # This gives same dimensions as SA production: [1/s] * [L²/s] = [L²/s²]
    return jnp.where(nuHat > 0, rate * omega_mag * nuHat, 0.0)


@jax.jit
def compute_aft_source_jax(omega_mag: ArrayLike, vel_mag: ArrayLike,
                           d: ArrayLike, nuHat: ArrayLike,
                           nu_laminar: float = 1.0,
                           # Tunable parameters
                           gamma_coeff: float = AFT_GAMMA_COEFF,
                           re_scale: float = AFT_RE_OMEGA_SCALE,
                           log_divisor: float = AFT_LOG_DIVISOR,
                           sigmoid_center: float = AFT_SIGMOID_CENTER,
                           sigmoid_slope: float = AFT_SIGMOID_SLOPE,
                           rate_scale: float = AFT_RATE_SCALE) -> jnp.ndarray:
    """
    Compute AFT source term for 2D RANS solver.
    
    This is the main entry point for the AFT model in the RANS solver.
    Returns only the production term (no destruction in pure AFT).
    
    Parameters
    ----------
    omega_mag : array (NI, NJ)
        Vorticity magnitude from velocity gradients.
    vel_mag : array (NI, NJ)
        Velocity magnitude sqrt(u² + v²).
    d : array (NI, NJ)
        Wall distance.
    nuHat : array (NI, NJ)
        SA working variable.
    nu_laminar : float
        Laminar kinematic viscosity (1/Re).
    All other parameters are tunable for ensemble studies.
        
    Returns
    -------
    source : array (NI, NJ)
        AFT source term (production only).
    """
    return compute_aft_production(
        omega_mag, vel_mag, d, nuHat, nu_laminar,
        gamma_coeff, re_scale, log_divisor, sigmoid_center, sigmoid_slope, rate_scale
    )
