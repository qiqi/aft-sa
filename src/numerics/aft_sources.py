"""
AFT (Amplification Factor Transport) Source Terms for Laminar Transition.

This module implements the laminar instability amplification model that
drives transition in the SA-AI combined model. The amplification rate
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
# Q4-gated model (explore-vortgrad-gate, Jul 2026): rate multiplied by the band
# gate Q4 = 1 - 2(w d)|u|/(cA A + (w d)^2 + u^2), A = (dw/dn d^2)^2, cA = 4
# (see compute_q4_gate). With the gate: a_max at the top of the Michalke band
# (free-shear limit of Q4 is exactly 1), floor at the envelope condition
# (cliff through the Blasius Drela-onset point, 2.19 x 242 = 529), sigmoid
# re-centered on the attached asymptote, K_lambda from the same worst-point
# self-consistency at the new kernel.
AFT_SIGMOID_CENTER = 0.9874
AFT_SIGMOID_SLOPE = 10.68
AFT_RATE_SCALE = 0.19          # a_max = Michalke free-shear ceiling (Q4 -> 1 there)
AFT_RE_OMEGA_FLOOR = 243.7     # THREE-ANCHOR kernel: (floor, g_c, s) solve three Drela-
                               # envelope conditions -- the Blasius envelope meets Drela's
                               # at N=1 and N=9, and the separation-limit profile's
                               # onset-to-transition mean rate equals Drela's there
                               # (gate band-weighting and laminar diffusion accounted
                               # for by the transport solve itself)
AFT_TILT_SLOPE = 1.0e6         # effectively disabled
AFT_BARRIER_POWER = 4.0        # p: sharp cliff barrier = 1 - (Re_Ω_floor/Re_Ω)^p
AFT_Q4_CA = 4.0                # weight on the vorticity-gradient invariant in Q4
# Favorable-PG effect enters twice, both via the local streamwise PG sensor
#   lambda_p = -d^2 * (u . grad_p) / (rho * nu * |u|^2):
# (1) onset-delay cliff  Re_Omega_cliff = floor * exp(K_lambda * max(0, lambda_p)),
# (2) rate factor        1 / (1 + (K_r * max(0, lambda_p))^2),
# both ==1 (inactive) for lambda_p <= 0, so Blasius and the adverse branch are
# untouched (paper Sec. calib).
AFT_CLIFF_LAMBDA_SLOPE = 6.20   # K_λ — worst-point fixed point at the three-anchor kernel
AFT_FPG_RATE_SLOPE = 5.80       # K_r — favorable-rate factor 1/(1+(K_r·max(0,λ_p))²); ==1 for
                               # λ_p≤0 (C¹ at the boundary); fit at the worst favorable point
                               # of the shape-factor family (β=0.35, H=2.34): mean rate = Drela


# --- FINAL composite two-pinch gate (paper v2, C++ AI_VG_GATE=7) -----------
AFT_LV_CV = 4.0     # c_V: Lambda_v weight in the smooth Q1 denominator
AFT_Q2_C2 = 8.0     # c_2: Q2 release softness outside the parabola loop


def compute_composite_gate(dudy, d2udn2, vel_mag, d,
                           cV: float = AFT_LV_CV, c2: float = AFT_Q2_C2):
    """FINAL band gate (paper v2): Q = Q1s * Q2.

    Q1s = 1 - sqrt(G(2-G)) / sqrt(1 + (cV Lv)^2) pins the WALL zero-advection
    locus (Gamma, Lambda_v) = (1, 0); Q2 = 1 - (G-1)_+^2/(1 + c2 max(P,0)),
    P = (Lv+G)^2 - G(2-G), pins the RECIRCULATION zero-advection locus: the
    pocket is the interior of the universal thin-bubble parabola loop (any
    quadratic profile with a wall zero crosses u=0 at exactly (2,-2)).
    Lambda_v = -u'u''d^3/((u'd)^2+u^2), with BOTH derivatives SIGNED (the
    sign convention makes Blasius all positive; in 3D the solver's
    lap(u).(n x omega) projection supplies the signed product directly)."""
    import numpy as _np
    den_core = (dudy*d)**2 + vel_mag**2 + 1e-300
    num = 2.0*_np.abs(dudy)*d*_np.abs(vel_mag)
    q1 = 1.0 - num/_np.sqrt(den_core**2 + (cV*dudy*d2udn2*d**3)**2)
    G = 2.0*(dudy*d)**2/den_core
    Lv = -dudy*d2udn2*d**3/den_core
    P = (Lv + G)**2 - G*(2.0 - G)
    q2 = 1.0 - _np.clip(G - 1.0, 0.0, None)**2/(1.0 + c2*_np.clip(P, 0.0, None))
    return q1*q2


def compute_q4_gate(dwdn, omega_mag, vel_mag, d, cA: float = AFT_Q4_CA):
    """Band gate Q4 = 1 - 2(w d)|u| / (cA A + (w d)^2 + u^2), A = (dw/dn d^2)^2.

    Equivalently, in the paper's dimensionless indicators (Eq. 4/6),
    Q4 = 1 - sqrt(Gamma(2-Gamma))/(1 + cA*Gamma_g) with
    Gamma_g = A/((w d)^2 + u^2). Pinched near zero on the attached wall
    (Gamma = 1, Gamma_g -> 0), exactly 1 in the free-shear limit.
    Multiplies the laminar amplification rate only."""
    import numpy as _np
    A = (dwdn*d*d)**2
    B = (omega_mag*d)**2
    return 1.0 - 2.0*_np.abs(omega_mag)*d*_np.abs(vel_mag)/(cA*A + B + vel_mag**2 + 1e-300)


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
                                   cliff_lambda_slope: float = AFT_CLIFF_LAMBDA_SLOPE,
                                   fpg_rate_slope: float = AFT_FPG_RATE_SLOPE) -> jnp.ndarray:
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
    # Favorable-rate factor (branch favorable-rate): the correlation's rate
    # DROP under acceleration, carried by lambda_p where it is a faithful
    # shape surrogate. ==1 for lambda_p <= 0; C^1 at the boundary.
    rate = rate / (1.0 + (fpg_rate_slope * lambda_pos)**2)
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
