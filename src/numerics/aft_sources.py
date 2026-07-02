"""
AFT (Amplification Factor Transport) Source Terms for Laminar Transition.

This module implements the laminar instability amplification model that
drives transition in the SA-AF combined model. The amplification rate
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

# Amplification rate formula. Option A: kernel depends on Γ alone (no Re_Ω
# log term in the sigmoid argument). The barrier is now a Γ-dependent cliff
# (replacing the previous single-scalar Re_Ω cutoff):
#     Re_Ω_cliff(Γ) = max(reOmegaFloor, 10^(log10(reOmegaFloor) + (1-Γ)/tiltSlope))
# i.e. vertical wall at Re_Ω = reOmegaFloor for Γ >= 1, tilted segment for
# Γ < 1 sloping outward through (Re_Ω=100, Γ=1) and (1000, 0.85). Designed
# by tracing the outer envelope of Drela-onset Falkner-Skan trajectories.
# Must stay in sync with Flow360 ModelConstants.h.
AFT_SIGMOID_CENTER = 1.572
AFT_SIGMOID_SLOPE = 5.263
AFT_RATE_SCALE = 0.15          # a_max (no-tilt cliff_floor matches S-S ±12%, growth start at Drela 200)
AFT_RE_OMEGA_FLOOR = 100.0     # cliff floor (Γ-independent, no slanted cliff)
AFT_TILT_SLOPE = 1.0e6         # effectively disabled (favorable-PG suppression -> sigma_FPG(lambda_p))
AFT_BARRIER_POWER = 4.0        # p: sharp cliff barrier = 1 - (Re_Ω_floor/Re_Ω)^p
# Favorable-PG effect is carried ENTIRELY by an onset-delay cliff (no rate factor):
# Re_Omega_cliff(lambda_p) = floor * exp(K_lambda * max(0, lambda_p)).
# lambda_p = -d^2 * (u . grad_p) / (rho * nu * |u|^2)  -- local streamwise PG sensor.
# (The former sigma_FPG rate-suppression factor was removed: the Drela H(lambda_p)
#  analysis shows favorable-PG stabilization is dominated by onset delay, which the
#  cliff already provides; see paper Sec. calib.)
AFT_CLIFF_LAMBDA_SLOPE = 10.0  # K_λ — favorable-PG onset-delay slope (0 disables)


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
                                   lambda_p: ArrayLike = 0.0,
                                   sigmoid_center: float = AFT_SIGMOID_CENTER,
                                   sigmoid_slope: float = AFT_SIGMOID_SLOPE,
                                   rate_scale: float = AFT_RATE_SCALE,
                                   re_omega_floor: float = AFT_RE_OMEGA_FLOOR,
                                   tilt_slope: float = AFT_TILT_SLOPE,
                                   barrier_power: float = AFT_BARRIER_POWER,
                                   cliff_lambda_slope: float = AFT_CLIFF_LAMBDA_SLOPE) -> jnp.ndarray:
    """
    Non-dimensional amplification rate. Matches CUDA `SAAftTransition.h::__aftRate`.

    FORMULA (Option A kernel with Γ-dependent SHARP cliff barrier):
        Re_Ω_cliff(Γ) = max(re_omega_floor,
                            10^(log10(re_omega_floor) + (1-Γ)/tilt_slope))
        barrier       = ln(1 - (Re_Ω_cliff(Γ) / Re_Ω)^p)       (Re_Ω > cliff)
        x             = s * (Γ - g_c) + barrier
        rate          = a_max * sigmoid(x)                     (rate ≡ 0 for Re_Ω ≤ cliff)
    """
    # Cliff threshold with two modifiers:
    #   (1) Gamma-dependent tilt (disabled by default with tilt_slope=1e6).
    #   (2) FPG-dependent boost: cliff *= exp(cliff_lambda_slope * max(0, lambda_p))
    # so favorable-PG regions (lambda_p > 0) delay the onset.
    log_floor = jnp.log10(re_omega_floor)
    log_extra = jnp.where(Gamma < 1.0, (1.0 - Gamma) / tilt_slope, 0.0)
    lambda_pos = jnp.maximum(lambda_p, 0.0)
    fpg_boost = cliff_lambda_slope * lambda_pos / jnp.log(10.0)
    log_cliff = log_floor + log_extra + fpg_boost
    re_cliff = jnp.power(10.0, log_cliff)
    safe_Re = jnp.maximum(Re_Omega, re_cliff + 1e-12)
    # Sharp cliff: barrier_inside = 1 - (re_cliff/Re_Omega)^p, large p gives
    # near-Heaviside cutoff while leaving Re_Omega >> re_cliff unattenuated.
    ratio = re_cliff / safe_Re
    barrier_inside = jnp.maximum(1.0 - jnp.power(ratio, barrier_power), 1e-20)
    barrier = jnp.log(barrier_inside)
    x_arg = sigmoid_slope * (Gamma - sigmoid_center) + barrier
    rate = rate_scale * jax.nn.sigmoid(x_arg)
    # Favorable-PG suppression: sigma_FPG(lambda_p) = 1/(1+exp(slope*(lambda_p-star)))
    # lambda_p > 0 -> favorable PG -> sigma_FPG -> 0 (suppress);
    # lambda_p <= 0 -> Blasius/adverse PG -> sigma_FPG -> 1 (no suppression).
    # Favorable-PG effect enters only through re_cliff(lambda_p) (onset delay);
    # there is no sigma_FPG rate factor.
    return jnp.where(Re_Omega > re_cliff, rate, 0.0)


@jax.jit
def compute_aft_production(omega_mag: ArrayLike, vel_mag: ArrayLike,
                           d: ArrayLike, nuHat: ArrayLike,
                           nu_laminar: float = 1.0,
                           # Tunable parameters
                           gamma_coeff: float = AFT_GAMMA_COEFF,
                           sigmoid_center: float = AFT_SIGMOID_CENTER,
                           sigmoid_slope: float = AFT_SIGMOID_SLOPE,
                           rate_scale: float = AFT_RATE_SCALE,
                           re_omega_floor: float = AFT_RE_OMEGA_FLOOR,
                           tilt_slope: float = AFT_TILT_SLOPE) -> jnp.ndarray:
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
        Re_Omega, Gamma,
        sigmoid_center=sigmoid_center, sigmoid_slope=sigmoid_slope,
        rate_scale=rate_scale,
        re_omega_floor=re_omega_floor, tilt_slope=tilt_slope,
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
                           sigmoid_center: float = AFT_SIGMOID_CENTER,
                           sigmoid_slope: float = AFT_SIGMOID_SLOPE,
                           rate_scale: float = AFT_RATE_SCALE,
                           re_omega_floor: float = AFT_RE_OMEGA_FLOOR,
                           tilt_slope: float = AFT_TILT_SLOPE) -> jnp.ndarray:
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
        gamma_coeff=gamma_coeff,
        sigmoid_center=sigmoid_center, sigmoid_slope=sigmoid_slope,
        rate_scale=rate_scale,
        re_omega_floor=re_omega_floor, tilt_slope=tilt_slope,
    )
