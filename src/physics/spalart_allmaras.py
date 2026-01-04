"""
Spalart-Allmaras Turbulence Model with Embedded Physical Derivation Logic.

Based on: "A One-Equation Turbulence Model for Aerodynamic Flows"
          Spalart & Allmaras, AIAA Paper 92-0439, 1992.

================================================================================
PHYSICAL DERIVATION AND CALIBRATION
================================================================================

The SA model solves for ν̃ (nuHat), a working variable closely related to
turbulent kinematic viscosity νₜ. The model is CONSTRUCTED (not derived)
to satisfy specific physical constraints:

1. LOG-LAYER CALIBRATION (y⁺ > 30):
   ---------------------------------
   In the log layer, the balance is: PRODUCTION + DIFFUSION = DESTRUCTION
   
       P + Diff = D    (NOT P = D!)
   
   IMPORTANT: Unlike other turbulence models, DIFFUSION IS NOT NEGLIGIBLE
   in the SA model's log layer. Since ν̃ ∝ y (linear), ∇ν̃ is constant,
   making the diffusion terms non-zero and significant.
   
   This balance enforces the LOG LAW: u = (u_τ/κ) ln(y) + C
   
   CONSEQUENCE: The viscosity profile must be LINEAR:
       ν̃ = κ d u_τ   →   χ = ν̃/ν = κ y⁺
   
   THE P/D RATIO IS NOT 1.0:
       P/D = cb1/(cw1·κ²) ≈ 0.25
   
   DEBUGGING: If χ ≠ κ y⁺ in the log layer:
     - Check the FULL balance: P + Diff - D ≈ 0
     - Do NOT assert P = D (this is WRONG for SA)
   Possible causes:
     - Wrong wall distance d
     - Wrong vorticity Ω
     - Diffusion term has wrong sign
     - Numerical issues in the solver

2. VISCOUS SUBLAYER (y⁺ < 5):
   --------------------------
   The turbulent viscosity must vanish at the wall: νₜ → 0 as y → 0.
   
   DNS data shows: νₜ ∝ y³ (or y⁴) as y → 0
   
   MECHANISM: The damping function fv1(χ) = χ³/(χ³ + cv1³)
   - When χ → 0: fv1 → χ³/cv1³ → 0 (cubic damping)
   - This makes νₜ = ν̃ · fv1 vanish correctly even if ν̃ ≠ 0
   
   DEBUGGING: If ν̃ has correct profile but skin friction is wrong,
   the error is in fv1 or the νₜ calculation.

3. THE cw1 DERIVATION (Rigorous - determines the νhat slope):
   ----------------------------------------------------------
   See Appendix in docs/physics/SA_Theory_Deep_Dive.md for full derivation.
   
   KEY INSIGHT: We solve P + Diff = D for the UNKNOWN slope S_νhat,
   assuming νhat = S_νhat · y. Then cw1 is chosen to match S_νhat = κ·u_τ.
   
   Production:  P = cb1 · u_τ²  (slope S_νhat cancels out!)
   Destruction: D = cw1 · S_νhat²
   Diffusion:   Diff = (1+cb2)/σ · S_νhat²
   
   Solving P + Diff = D for S_νhat:
       S_νhat = u_τ · √(cb1 / (cw1 - (1+cb2)/σ))
   
   Setting S_νhat = κ·u_τ and solving for cw1:
       cw1 = cb1/κ² + (1+cb2)/σ ≈ 3.24
   
   CONCLUSION: cw1 is the TUNING KNOB that forces the SA model to produce
   the correct Von Kármán slope κ = 0.41.

4. THE r PARAMETER - EQUILIBRIUM INDICATOR:
   ----------------------------------------
   r = ν̃ / (S̃ · κ² · d²)
   
   In equilibrium (P + Diff = D), r ≈ 1 (which makes fw ≈ 1).
   - r > 1: ν̃ too high relative to S̃ → fw > 1 → enhanced destruction
   - r < 1: ν̃ too low relative to S̃ → fw < 1 → reduced destruction
   
   DEBUGGING: If r ≈ 1 everywhere but χ is wrong, the issue is in S̃ or d.

================================================================================
CAUSAL LOGIC HIERARCHY (CRITICAL FOR DEBUGGING)
================================================================================

The SA model does NOT cause the Log Law. The Log Law is a REQUIREMENT of the
Momentum Equation; the SA model is merely CALIBRATED to support it.

This distinction is vital for debugging: if the velocity profile isn't
logarithmic, debugging the SA production terms is a waste of time.

THE ORIGIN (Momentum Equation):
   In the log layer, Total Stress τ ≈ τ_w (constant).
   Since τ ≈ νₜ·∂u/∂y, and we observe u ∝ ln(y), physics DEMANDS that νₜ ∝ y.
   This requirement comes from the Navier-Stokes equations, NOT the SA model.

THE RESPONSE (SA Model):
   The SA terms (P, D, Diff) are ENGINEERED specifically so that the linear
   profile ν̃ = κ·u_τ·y is a stable solution to the transport equation.

DEBUGGING FLOWCHART:
   If χ ≠ κ·y⁺ (Linear Law violated):
   
   1. CHECK MOMENTUM FIRST: Is u⁺ logarithmic?
      → NO:  Error is in FLOW SOLVER (Fluxes/BCs). SA model is INNOCENT.
      → YES: Proceed to step 2.
   
   2. CHECK GEOMETRY: Is wall distance d correct?
      → NO:  Fix wall distance calculation.
      → YES: Proceed to step 3.
   
   3. CHECK GRID RESOLUTION: Is vorticity Ω correct in log layer?
      → Check Ω vs expected Ω = u_τ/(κy) at y⁺ = 50, 100, 200
      → If Ω << expected (ratio < 0.5): Grid is too coarse in outer BL!
      → Fix: Reduce stretching ratio from >2.0 to ~1.2
      → YES: Proceed to step 4.
   
   4. CHECK SA MODEL: P, D, Diff failing to balance.
      - Check signs of all terms
      - Check constants (cb1, cw1, sigma)
      - Check diffusion coefficient (ν + ν̃)/σ

GRID REQUIREMENTS (CRITICAL):
   SA model requires adequate resolution throughout the boundary layer!
   - Wall: y⁺ ≈ 1 for first cell
   - Log layer (y⁺ = 30-200): Need 10-15 cells
   - Stretching ratio: 1.15-1.25 (NOT 2.0+!)
   - If stretching too aggressive: Ω → 0 in outer BL → P → 0 → χ plateaus

================================================================================
MODEL CONSTANTS (Standard SA-neg)
================================================================================
"""

from typing import Tuple, Union, Dict, Any
from .jax_config import jax, jnp
import numpy as np

ArrayLike = Union[jnp.ndarray, float]

# =============================================================================
# PHYSICAL CONSTANTS - With Derivation Origins
# =============================================================================

# Von Karman constant - from log-law velocity profile: u⁺ = (1/κ) ln(y⁺) + B
KAPPA = 0.41

# SA Model Constants
CB1 = 0.1355   # Production coefficient - calibrated to match mixing length in log layer
CB2 = 0.622    # Diffusion gradient term coefficient
SIGMA = 2.0/3.0  # Diffusion Prandtl number
CV1 = 7.1      # Wall damping - controls fv1 transition (fv1 = 0.5 at χ = cv1)
CV2 = 0.7      # S̃ modification threshold (standard SA uses different form)
CW2 = 0.3      # Controls fw shape - determines r-to-fw mapping steepness  
CW3 = 2.0      # fw clipping parameter - limits destruction at large r

# DERIVED CONSTANT - THE "TUNING KNOB" FOR THE LOG LAW
# =============================================================================
# cw1 = cb1/κ² + (1 + cb2)/σ  ≈ 3.24
# 
# RIGOROUS DERIVATION (see docs/physics/SA_Theory_Deep_Dive.md Appendix):
# -----------------------------------------------------------------------
# 1. Physical Requirement (from momentum equation):
#    For τ = ρ·νₜ·∂u/∂y = ρ·u_τ² in log layer with u = (u_τ/κ)·ln(y),
#    we MUST have νₜ ∝ y with slope S_νhat = κ·u_τ.
#
# 2. SA Equation Balance (P + Diff = D):
#    Solving P + Diff = D for the slope S_νhat gives:
#    S_νhat = u_τ · √(cb1 / (cw1 - (1+cb2)/σ))
#
# 3. Calibration:
#    Setting S_νhat = κ·u_τ and solving for cw1:
#    cw1 = cb1/κ² + (1 + cb2)/σ
#
# COMPONENTS:
# - cb1/κ² ≈ 0.806: Production/Destruction ratio at equilibrium
# - (1+cb2)/σ ≈ 2.433: Diffusion contribution to destruction balance
#
# NOTE: P/D ≈ cb1/(cw1·κ²) ≈ 0.25, NOT 1.0! Diffusion provides ~75%!
# =============================================================================
CW1 = CB1 / KAPPA**2 + (1.0 + CB2) / SIGMA  # ≈ 3.24


# =============================================================================
# DAMPING FUNCTIONS - Control near-wall behavior
# =============================================================================

@jax.jit
def compute_fv1(chi: ArrayLike) -> jnp.ndarray:
    """
    Compute wall damping function fv1.
    
    PHYSICAL PURPOSE:
    ----------------
    Ensures νₜ → 0 as y → 0, even if ν̃ ≠ 0.
    This allows ν̃ to be well-behaved at the wall while νₜ vanishes correctly.
    
    FORMULA:
        fv1 = χ³ / (χ³ + cv1³)
    
    BEHAVIOR:
        - χ → 0: fv1 → χ³/cv1³ → 0 (cubic damping for νₜ = ν̃·fv1)
        - χ → ∞: fv1 → 1 (no damping far from wall)
        - χ = cv1: fv1 = 0.5 (transition point)
    
    DEBUGGING:
        - In the log layer (y⁺ > 30), χ ≈ κ·y⁺ ≈ 12-40, so fv1 ≈ 0.3-0.8
        - If fv1 is much lower, χ is too small → check ν̃ calculation
    
    Parameters
    ----------
    chi : array-like
        Ratio χ = ν̃/ν (dimensionless).
        
    Returns
    -------
    fv1 : array
        Damping function value, 0 ≤ fv1 ≤ 1.
    """
    chi3 = chi ** 3
    return chi3 / (chi3 + CV1 ** 3)


@jax.jit
def compute_fv2(chi: ArrayLike, fv1: ArrayLike) -> jnp.ndarray:
    """
    Compute fv2 function for S̃ modification.
    
    PHYSICAL PURPOSE:
    ----------------
    Modifies S̃ near the wall to prevent S̃ from becoming negative
    when the fv2 correction term dominates.
    
    FORMULA:
        fv2 = 1 - χ / (1 + χ·fv1)
    
    BEHAVIOR:
        - χ → 0: fv2 → 1 (full correction near wall)
        - χ → ∞: fv2 → 1 - 1/fv1 → 0 (no correction in log layer)
        - fv2 becomes negative when χ > 1/(1-fv1) ≈ 0.9-1.5
    
    DEBUGGING:
        - In log layer: fv2 ≈ 0, so S̃ ≈ Ω (vorticity dominates)
        - Near wall: fv2 ≈ 1, so fv2 correction is important
    
    Parameters
    ----------
    chi : array-like
        Ratio χ = ν̃/ν.
    fv1 : array-like
        Precomputed fv1 values.
        
    Returns
    -------
    fv2 : array
        S̃ modification function.
    """
    return 1.0 - chi / (1.0 + chi * fv1 + 1e-20)


@jax.jit
def compute_S_tilde(omega: ArrayLike, nuHat: ArrayLike, d: ArrayLike,
                    chi: ArrayLike, fv2: ArrayLike) -> jnp.ndarray:
    """
    Compute modified vorticity S̃.
    
    PHYSICAL PURPOSE:
    ----------------
    S̃ is the "effective strain rate" that drives production.
    The modification term prevents S̃ from going negative near the wall.
    
    FORMULA:
        S̃ = Ω + (ν̃ / κ²d²) · fv2
    
    LOG-LAYER BEHAVIOR (y⁺ > 30):
        - fv2 ≈ 0, so S̃ ≈ Ω
        - In log layer: Ω ≈ u_τ / (κd), so S̃ ≈ u_τ / (κd)
    
    NEAR-WALL BEHAVIOR:
        - fv2 ≈ 1, and the correction term can dominate
        - This prevents production from being too large near the wall
    
    DEBUGGING:
        If S̃ >> Ω in the log layer, the fv2 term is too large.
        This happens when χ is wrong (not following κy⁺).
    
    Parameters
    ----------
    omega : array-like
        Vorticity magnitude |ω| = |∂u/∂y - ∂v/∂x|.
    nuHat : array-like
        SA working variable ν̃.
    d : array-like
        Wall distance.
    chi : array-like
        Precomputed χ = ν̃/ν.
    fv2 : array-like
        Precomputed fv2.
        
    Returns
    -------
    S_tilde : array
        Modified vorticity (clipped to be positive).
    """
    inv_k2d2 = 1.0 / (KAPPA**2 * d**2 + 1e-20)
    S_tilde_raw = omega + nuHat * inv_k2d2 * fv2
    return jnp.maximum(S_tilde_raw, 1e-16)


@jax.jit
def compute_r(nuHat: ArrayLike, S_tilde: ArrayLike, d: ArrayLike) -> jnp.ndarray:
    """
    Compute equilibrium parameter r.
    
    PHYSICAL PURPOSE:
    ----------------
    r measures how far from equilibrium the turbulence is.
    r = 1 corresponds to Production = Destruction.
    
    FORMULA:
        r = ν̃ / (S̃ · κ² · d²)
    
    INTERPRETATION:
        - r = 1: Local equilibrium (P = D)
        - r > 1: ν̃ too high → Destruction > Production → ν̃ will decrease
        - r < 1: ν̃ too low → Production > Destruction → ν̃ will increase
    
    DEBUGGING:
        In log layer, r should be ≈ 1 for correct χ = κy⁺ profile.
        If r >> 1, either:
          - ν̃ is too high
          - S̃ is too low (wrong vorticity or fv2)
          - d is too small (wrong wall distance)
    
    Parameters
    ----------
    nuHat : array-like
        SA working variable ν̃.
    S_tilde : array-like
        Modified vorticity S̃.
    d : array-like
        Wall distance.
        
    Returns
    -------
    r : array
        Equilibrium parameter (clipped to max=10).
    """
    r_raw = nuHat / (S_tilde * KAPPA**2 * d**2 + 1e-20)
    return jnp.clip(r_raw, max=10.0)


@jax.jit
def compute_fw(r: ArrayLike) -> jnp.ndarray:
    """
    Compute destruction function fw.
    
    PHYSICAL PURPOSE:
    ----------------
    fw modulates the destruction term to:
    1. Equal 1 at equilibrium (r = 1) for correct P = D balance
    2. Increase for r > 1 to enhance destruction when ν̃ is too high
    3. Decrease for r < 1 to reduce destruction when ν̃ is too low
    
    FORMULA:
        g = r + cw2·(r⁶ - r)
        fw = g · [(1 + cw3⁶)/(g⁶ + cw3⁶)]^(1/6)
    
    BEHAVIOR:
        - r = 1: g ≈ 1, fw ≈ 1 (equilibrium)
        - r >> 1: g → large, fw → cw3^(1/6) ≈ 1.12 (clipped)
        - r << 1: g → small, fw → 0 (reduced destruction)
    
    DEBUGGING:
        If fw >> 1 everywhere, r is consistently > 1, meaning
        the destruction is trying to reduce ν̃ but can't.
        This suggests an imbalance in the source terms.
    
    Parameters
    ----------
    r : array-like
        Equilibrium parameter.
        
    Returns
    -------
    fw : array
        Destruction modifier function.
    """
    g = r + CW2 * (r**6 - r)
    c6 = CW3 ** 6
    return g * ((1.0 + c6) / (g**6 + c6 + 1e-20)) ** (1.0/6.0)


# =============================================================================
# PRODUCTION TERM - Turbulence generation by mean shear
# =============================================================================

@jax.jit
def compute_sa_production(omega: ArrayLike, nuHat: ArrayLike, d: ArrayLike,
                          nu_laminar: float = 1e-6) -> jnp.ndarray:
    """
    Compute SA production term.
    
    PHYSICAL MEANING:
    -----------------
    Production represents turbulence generation by mean shear.
    In physical terms: mean velocity gradients stretch and amplify
    turbulent eddies, converting mean flow energy to turbulence.
    
    FORMULA:
        P = cb1 · S̃ · ν̃
    
    DIMENSIONAL ANALYSIS:
        [P] = [1/time] × [length²/time] = [length²/time²]
        Same as D(ν̃)/Dt, which is correct.
    
    LOG-LAYER SCALING:
        With S̃ ≈ u_τ/(κd) and ν̃ = κd·u_τ:
        P = cb1 · (u_τ/κd) · (κd·u_τ) = cb1 · u_τ²
        
        Production is constant in the log layer (independent of y)!
        This is a key feature that enables the log-law solution.
    
    DEBUGGING:
        If P is NOT constant in log layer, either:
        - S̃ is wrong (vorticity or d)
        - ν̃ is not linear in d
        - Check cb1 value
    
    Parameters
    ----------
    omega : array
        Vorticity magnitude |ω|.
    nuHat : array
        SA working variable ν̃.
    d : array
        Wall distance.
    nu_laminar : float
        Laminar kinematic viscosity ν (= 1/Re).
        
    Returns
    -------
    P : array
        Production term P = cb1·S̃·ν̃ (per unit mass).
    """
    # Safety: ensure positive nuHat
    mask = 1.0 / (1.0 + jnp.exp(-100.0 * nuHat))  # Smooth mask for nuHat > 0
    nuHat_safe = jnp.maximum(nuHat, 1e-10)
    
    # Chi and damping functions
    chi = nuHat_safe / nu_laminar
    fv1_val = compute_fv1(chi)
    fv2_val = compute_fv2(chi, fv1_val)
    
    # Modified vorticity
    S_tilde = compute_S_tilde(omega, nuHat_safe, d, chi, fv2_val)
    
    # Production
    return CB1 * S_tilde * nuHat_safe * mask


# =============================================================================
# DESTRUCTION TERM - Turbulence dissipation near walls
# =============================================================================

@jax.jit
def compute_sa_destruction(omega: ArrayLike, nuHat: ArrayLike, d: ArrayLike,
                           nu_laminar: float = 1e-6) -> jnp.ndarray:
    """
    Compute SA destruction term.
    
    PHYSICAL MEANING:
    -----------------
    Destruction represents turbulence dissipation, primarily near walls.
    In physical terms: wall proximity constrains eddy sizes, and viscous
    effects dissipate turbulent energy at small scales.
    
    FORMULA:
        D = cw1 · fw · (ν̃/d)²
    
    DIMENSIONAL ANALYSIS:
        [D] = [length²/time / length]² = [length²/time²]
        Same as production - required for equilibrium!
    
    LOG-LAYER SCALING:
        With ν̃ = κd·u_τ and fw = 1 (equilibrium):
        D = cw1 · 1 · (κd·u_τ/d)² = cw1 · κ² · u_τ²
        
        For P = D: cb1·u_τ² = cw1·κ²·u_τ²
        This is satisfied by: cw1 = cb1/κ² + (1+cb2)/σ
        
        (The extra term accounts for the cb2 diffusion contribution)
    
    WHY 1/d² SCALING:
        - Production scales as S·ν̃ ~ (1/d)·d = constant
        - Destruction must also be constant for P = D
        - (ν̃/d)² = (κd·u_τ/d)² = κ²·u_τ² = constant ✓
    
    DEBUGGING:
        If D >> P in log layer, either:
        - fw > 1 (r > 1, meaning ν̃ is "too high" relative to S̃)
        - ν̃/d is larger than expected
        - Check cw1 calculation
    
    Parameters
    ----------
    omega : array
        Vorticity magnitude |ω|.
    nuHat : array
        SA working variable ν̃.
    d : array
        Wall distance.
    nu_laminar : float
        Laminar kinematic viscosity ν (= 1/Re).
        
    Returns
    -------
    D : array
        Destruction term D = cw1·fw·(ν̃/d)² (per unit mass).
    """
    # Safety
    mask = 1.0 / (1.0 + jnp.exp(-100.0 * nuHat))
    nuHat_safe = jnp.maximum(nuHat, 1e-10)
    
    # Chi and damping functions
    chi = nuHat_safe / nu_laminar
    fv1_val = compute_fv1(chi)
    fv2_val = compute_fv2(chi, fv1_val)
    
    # Modified vorticity
    S_tilde = compute_S_tilde(omega, nuHat_safe, d, chi, fv2_val)
    
    # Equilibrium parameter r (indicates P/D balance)
    r_val = compute_r(nuHat_safe, S_tilde, d)
    
    # Destruction modifier
    fw_val = compute_fw(r_val)
    
    # Destruction
    return CW1 * fw_val * (nuHat_safe / d) ** 2 * mask


# =============================================================================
# TURBULENT VISCOSITY - The final physical quantity
# =============================================================================

@jax.jit
def compute_turbulent_viscosity(nuHat: ArrayLike, nu_laminar: float) -> jnp.ndarray:
    """
    Compute turbulent kinematic viscosity νₜ from working variable ν̃.
    
    PHYSICAL MEANING:
    -----------------
    νₜ is the actual turbulent viscosity used in the momentum equations.
    The relationship νₜ = ν̃·fv1 serves two purposes:
    
    1. NEAR-WALL DAMPING:
       fv1 → 0 as χ → 0, so νₜ → 0 at the wall even if ν̃ ≠ 0.
       This gives νₜ ∝ y³ near the wall, matching DNS data.
    
    2. FAR FROM WALL:
       fv1 → 1 as χ → ∞, so νₜ ≈ ν̃ in the log layer.
    
    FORMULA:
        χ = ν̃/ν
        fv1 = χ³/(χ³ + cv1³)
        νₜ = ν̃ · fv1 = ν · χ · fv1
    
    ASYMPTOTIC BEHAVIOR:
        χ → 0: νₜ = ν·χ·(χ³/cv1³) = ν·χ⁴/cv1³ ∝ y⁴ (fourth-power damping)
        χ → ∞: νₜ = ν·χ·1 = ν̃ (no damping)
    
    DEBUGGING:
        If skin friction is wrong but ν̃ profile looks correct:
        - Check fv1 calculation
        - Verify χ = ν̃/ν is computed correctly (not χ = ν̃!)
    
    Parameters
    ----------
    nuHat : array
        SA working variable ν̃.
    nu_laminar : float
        Laminar kinematic viscosity ν.
        
    Returns
    -------
    nu_t : array
        Turbulent kinematic viscosity νₜ = ν̃·fv1.
    """
    nuHat_safe = jnp.maximum(nuHat, 0.0)
    chi = nuHat_safe / nu_laminar
    fv1_val = compute_fv1(chi)
    return nuHat_safe * fv1_val


@jax.jit
def compute_effective_viscosity(nuHat: ArrayLike, nu_laminar: float) -> jnp.ndarray:
    """
    Compute total effective kinematic viscosity ν_eff = ν + νₜ.
    
    Parameters
    ----------
    nuHat : array
        SA working variable ν̃.
    nu_laminar : float
        Laminar kinematic viscosity ν.
        
    Returns
    -------
    nu_eff : array
        Effective viscosity ν + νₜ.
    """
    nu_t = compute_turbulent_viscosity(nuHat, nu_laminar)
    return nu_laminar + nu_t


# =============================================================================
# DIFFUSION COEFFICIENT
# =============================================================================

@jax.jit
def compute_diffusion_coefficient(nuHat: ArrayLike, nu_laminar: float,
                                   sigma: float = SIGMA) -> jnp.ndarray:
    """
    Compute diffusion coefficient for the ν̃ equation.
    
    PHYSICAL MEANING:
    -----------------
    The diffusion term in the SA equation is:
        (1/σ) ∇·[(ν + ν̃)∇ν̃]
    
    The diffusion coefficient is (ν + ν̃)/σ.
    
    Note: This is NOT the same as the momentum diffusion coefficient!
    The SA variable ν̃ diffuses with its own effective viscosity.
    
    Parameters
    ----------
    nuHat : array
        SA working variable ν̃.
    nu_laminar : float
        Laminar kinematic viscosity ν.
    sigma : float
        SA Prandtl number (default 2/3).
        
    Returns
    -------
    D_coeff : array
        Diffusion coefficient (ν + ν̃)/σ.
    """
    nuHat_safe = jnp.maximum(nuHat, 0.0)
    return (nu_laminar + nuHat_safe) / sigma


# =============================================================================
# PHYSICS DIAGNOSTICS - Debug functions to check physical invariants
# =============================================================================

def check_log_layer_equilibrium(y_plus: np.ndarray, P: np.ndarray, D: np.ndarray,
                                 chi: np.ndarray, Diff: np.ndarray = None,
                                 verbose: bool = True) -> Dict[str, Any]:
    """
    Check log-layer equilibrium: Production + Diffusion = Destruction.
    
    PHYSICS ASSERTION:
    ------------------
    In the log layer (30 < y⁺ < 100), the SA equation balance is:
    
        P + Diff = D    (NOT P = D!)
    
    IMPORTANT: Unlike other turbulence models, diffusion is NOT negligible
    for SA in the log layer. Since ν̃ ∝ y, |∇ν̃| is constant, so diffusion
    contributes significantly to the balance.
    
    Expected P/D ratio: ~0.25 (NOT 1.0!)
    This is because Diff contributes ~75% of the destruction balance.
    
    Parameters
    ----------
    y_plus : array
        Wall units coordinate y⁺ = y·u_τ/ν.
    P : array
        Production term.
    D : array
        Destruction term.
    chi : array
        Viscosity ratio χ = ν̃/ν.
    Diff : array, optional
        Diffusion term. If provided, checks full balance P + Diff = D.
    verbose : bool
        Print diagnostic information.
        
    Returns
    -------
    diagnostics : dict
        - 'equilibrium_satisfied': bool
        - 'mean_P_over_D': float (expected ~0.25, NOT 1.0)
        - 'chi_slope': float (should be ≈ κ = 0.41)
        - 'message': str
    """
    # Select log layer region
    log_mask = (y_plus > 30) & (y_plus < 100)
    
    if not np.any(log_mask):
        return {
            'equilibrium_satisfied': None,
            'mean_P_over_D': None,
            'chi_slope': None,
            'message': 'No points in log layer (30 < y+ < 100)'
        }
    
    P_log = P[log_mask]
    D_log = D[log_mask]
    y_plus_log = y_plus[log_mask]
    chi_log = chi[log_mask]
    
    # Expected P/D ratio from SA theory:
    # P/D = cb1/(cw1·κ²) = 0.1355/(3.239·0.41²) ≈ 0.249
    expected_P_over_D = CB1 / (CW1 * KAPPA**2)  # ≈ 0.25
    
    # Check P/D ratio
    P_over_D = P_log / (D_log + 1e-20)
    mean_ratio = np.mean(P_over_D)
    
    # Check chi linearity: chi should be κ·y⁺
    if len(y_plus_log) > 2:
        slope, intercept = np.polyfit(y_plus_log, chi_log, 1)
    else:
        slope = chi_log[-1] / y_plus_log[-1] if len(y_plus_log) > 0 else 0
        intercept = 0
    
    # Expected slope is κ = 0.41
    expected_slope = KAPPA
    slope_error = abs(slope - expected_slope) / expected_slope
    
    # Check full balance if Diff is provided
    if Diff is not None:
        Diff_log = Diff[log_mask]
        full_balance = (P_log + Diff_log) / (D_log + 1e-20)
        mean_full_balance = np.mean(full_balance)
        balance_ok = (0.8 < mean_full_balance < 1.2)
    else:
        mean_full_balance = None
        # Without Diff, check if P/D is in expected range (~0.25)
        balance_ok = (0.15 < mean_ratio < 0.4)  # Allow range around 0.25
    
    chi_ok = (slope_error < 0.3)  # 30% tolerance
    
    message = []
    
    if Diff is not None:
        if not balance_ok:
            message.append(f"FULL BALANCE VIOLATED: (P+Diff)/D = {mean_full_balance:.2f} (should be ~1.0)")
        else:
            message.append(f"Full balance (P+Diff)/D = {mean_full_balance:.2f}: OK")
        message.append(f"P/D = {mean_ratio:.2f} (expected ~{expected_P_over_D:.2f}, NOT 1.0!)")
    else:
        if not balance_ok:
            message.append(f"P/D RATIO UNEXPECTED: {mean_ratio:.2f}")
            message.append(f"  Expected P/D ≈ {expected_P_over_D:.2f} (Diffusion provides the rest)")
            if mean_ratio > 0.4:
                message.append("  → P/D too high: either P too large or D too small")
            elif mean_ratio < 0.15:
                message.append("  → P/D too low: either P too small or D too large")
        else:
            message.append(f"P/D ratio = {mean_ratio:.2f} (expected ~{expected_P_over_D:.2f}): OK")
    
    if not chi_ok:
        message.append(f"CHI PROFILE INCORRECT: dχ/dy⁺ = {slope:.3f} (should be ~{expected_slope:.2f})")
        message.append(f"  → χ is only {slope/expected_slope*100:.0f}% of expected value")
        message.append("  → ν̃ is not growing fast enough with wall distance")
    else:
        message.append(f"χ slope = {slope:.3f} ({slope/expected_slope*100:.0f}% of κ): OK")
    
    result = {
        'equilibrium_satisfied': balance_ok and chi_ok,
        'mean_P_over_D': mean_ratio,
        'expected_P_over_D': expected_P_over_D,
        'mean_full_balance': mean_full_balance,
        'chi_slope': slope,
        'expected_chi_slope': expected_slope,
        'chi_slope_ratio': slope / expected_slope,
        'message': '\n'.join(message)
    }
    
    if verbose:
        print("\n" + "="*60)
        print("LOG LAYER BALANCE CHECK (P + Diff = D)")
        print("="*60)
        print(result['message'])
        print(f"\nDetailed metrics:")
        print(f"  P/D in log layer:      {mean_ratio:.3f} (expected: {expected_P_over_D:.2f})")
        if mean_full_balance is not None:
            print(f"  (P+Diff)/D:            {mean_full_balance:.3f} (expected: 1.0)")
        print(f"  χ slope (dχ/dy⁺):      {slope:.4f} (expected: {expected_slope:.2f})")
        print(f"  χ slope ratio:         {slope/expected_slope:.2%} of expected")
        print("="*60)
    
    return result


def verify_wall_scaling(y_plus: np.ndarray, chi: np.ndarray, 
                        nuHat: np.ndarray, nu_laminar: float,
                        verbose: bool = True) -> Dict[str, Any]:
    """
    Verify correct wall scaling of ν̃ and νₜ.
    
    PHYSICS ASSERTIONS:
    -------------------
    1. ν̃ should be linear in y near the wall: ν̃ ∝ y
       Actually, in the viscous sublayer, ν̃ = κ·y·u_τ·f(y⁺)
       where f(y⁺) → 1 as y⁺ → 0. So ν̃ ∝ y.
    
    2. νₜ should vanish as y³ or y⁴ at the wall.
       νₜ = ν̃·fv1 = ν̃·χ³/(χ³+cv1³)
       With χ = ν̃/ν ∝ y, we get νₜ ∝ y·y³ = y⁴ at leading order.
    
    3. fv1 should approach 0 with cubic behavior as χ → 0.
    
    Parameters
    ----------
    y_plus : array
        Wall units coordinate.
    chi : array
        Viscosity ratio χ = ν̃/ν.
    nuHat : array
        SA working variable.
    nu_laminar : float
        Laminar viscosity.
    verbose : bool
        Print diagnostics.
        
    Returns
    -------
    diagnostics : dict
        Wall scaling verification results.
    """
    # Focus on near-wall region y+ < 10
    wall_mask = y_plus < 10
    
    if not np.any(wall_mask):
        return {
            'wall_scaling_ok': None,
            'message': 'No points in near-wall region (y+ < 10)'
        }
    
    y_plus_wall = y_plus[wall_mask]
    chi_wall = chi[wall_mask]
    nuHat_wall = nuHat[wall_mask]
    
    # Compute fv1 and nu_t
    chi3 = chi_wall ** 3
    fv1_wall = chi3 / (chi3 + CV1**3)
    nu_t_wall = nuHat_wall * fv1_wall
    
    message = []
    
    # Check 1: chi behavior (should be roughly linear in y+)
    # chi = ν̃/ν, and in sublayer ν̃ ≈ κ·y·u_τ, so χ ≈ κ·y⁺
    if len(y_plus_wall) > 2:
        chi_slope, _ = np.polyfit(y_plus_wall, chi_wall, 1)
        expected_chi_slope = KAPPA
        chi_ratio = chi_slope / expected_chi_slope
        
        if chi_ratio < 0.5:
            message.append(f"NEAR-WALL χ GROWTH TOO SLOW: dχ/dy⁺ = {chi_slope:.4f}")
            message.append(f"  → Only {chi_ratio*100:.0f}% of expected slope (κ = {KAPPA})")
    
    # Check 2: fv1 behavior at wall
    # fv1 should be small at the wall (< 0.1 for y+ < 5)
    fv1_at_wall = fv1_wall[y_plus_wall < 5].mean() if np.any(y_plus_wall < 5) else None
    if fv1_at_wall is not None:
        if fv1_at_wall > 0.1:
            message.append(f"WARNING: fv1 too large near wall: {fv1_at_wall:.3f}")
            message.append("  → χ is too large near wall, damping may be inadequate")
        else:
            message.append(f"Wall damping (fv1 at y⁺<5): {fv1_at_wall:.4f} (OK, should be small)")
    
    # Check 3: chi value at y+ = 1 (should be ≈ κ ≈ 0.41)
    chi_at_1 = np.interp(1.0, y_plus, chi) if y_plus.min() < 1 < y_plus.max() else None
    if chi_at_1 is not None:
        expected_chi_at_1 = KAPPA * 1.0
        chi_1_ratio = chi_at_1 / expected_chi_at_1
        message.append(f"χ at y⁺=1: {chi_at_1:.3f} (expected: {expected_chi_at_1:.2f}, ratio: {chi_1_ratio:.1%})")
        if chi_1_ratio < 0.5:
            message.append("  → ν̃ is significantly too low in viscous sublayer!")
    
    # Overall assessment
    wall_ok = (chi_ratio > 0.5 if 'chi_ratio' in dir() else True) and \
              (fv1_at_wall < 0.2 if fv1_at_wall else True)
    
    if wall_ok:
        message.append("Near-wall scaling: Generally OK")
    
    result = {
        'wall_scaling_ok': wall_ok,
        'chi_slope_near_wall': chi_slope if 'chi_slope' in dir() else None,
        'fv1_at_wall': fv1_at_wall,
        'chi_at_yplus_1': chi_at_1,
        'message': '\n'.join(message)
    }
    
    if verbose:
        print("\n" + "="*60)
        print("NEAR-WALL SCALING VERIFICATION")
        print("="*60)
        print(result['message'])
        print("="*60)
    
    return result


def diagnose_sa_physics(y_plus: np.ndarray, nuHat: np.ndarray, 
                        omega: np.ndarray, d: np.ndarray,
                        nu_laminar: float, verbose: bool = True) -> Dict[str, Any]:
    """
    Comprehensive SA physics diagnostic.
    
    Runs all physics checks and provides actionable debugging guidance.
    
    Parameters
    ----------
    y_plus : array
        Wall units coordinate.
    nuHat : array
        SA working variable profile.
    omega : array
        Vorticity magnitude.
    d : array
        Wall distance.
    nu_laminar : float
        Laminar kinematic viscosity.
    verbose : bool
        Print detailed diagnostics.
        
    Returns
    -------
    diagnostics : dict
        Complete diagnostic results with physics interpretation.
    """
    # Compute derived quantities
    chi = nuHat / nu_laminar
    fv1_val = compute_fv1(chi)
    fv2_val = compute_fv2(chi, fv1_val)
    S_tilde = compute_S_tilde(omega, nuHat, d, chi, fv2_val)
    r_val = compute_r(nuHat, S_tilde, d)
    fw_val = compute_fw(r_val)
    
    # Convert JAX arrays to numpy for diagnostics
    chi = np.asarray(chi)
    fv1_val = np.asarray(fv1_val)
    fv2_val = np.asarray(fv2_val)
    S_tilde = np.asarray(S_tilde)
    r_val = np.asarray(r_val)
    fw_val = np.asarray(fw_val)
    
    # Compute P and D
    P = CB1 * S_tilde * nuHat
    D = CW1 * fw_val * (nuHat / d) ** 2
    
    # Run physics checks
    log_layer_check = check_log_layer_equilibrium(y_plus, P, D, chi, verbose=False)
    wall_check = verify_wall_scaling(y_plus, chi, nuHat, nu_laminar, verbose=False)
    
    if verbose:
        print("\n" + "="*70)
        print("SPALART-ALLMARAS PHYSICS DIAGNOSTIC")
        print("="*70)
        
        print("\n--- LOG LAYER EQUILIBRIUM ---")
        print(log_layer_check['message'])
        
        print("\n--- NEAR-WALL SCALING ---")
        print(wall_check['message'])
        
        # Summary statistics
        print("\n--- KEY METRICS ---")
        log_mask = (y_plus > 30) & (y_plus < 100)
        if np.any(log_mask):
            print(f"Log layer (30 < y⁺ < 100):")
            print(f"  Mean r:  {r_val[log_mask].mean():.3f} (equilibrium: r = 1)")
            print(f"  Mean fw: {fw_val[log_mask].mean():.3f} (equilibrium: fw ≈ 1)")
            print(f"  Mean χ:  {chi[log_mask].mean():.1f} (expected: κ·ȳ⁺ = {KAPPA * y_plus[log_mask].mean():.1f})")
        
        # Actionable guidance with CAUSAL LOGIC HIERARCHY
        print("\n--- DEBUGGING GUIDANCE (Follow This Order!) ---")
        if not log_layer_check['equilibrium_satisfied']:
            print("χ profile is INCORRECT (not following κ·y⁺).")
            print("")
            print("STEP 1: CHECK MOMENTUM FIRST - Is u⁺ logarithmic?")
            print("  → If NO: Error is in FLOW SOLVER (fluxes, BCs, pressure).")
            print("           The SA model is INNOCENT. Fix momentum first!")
            print("  → If YES: Proceed to step 2.")
            print("")
            print("STEP 2: CHECK GEOMETRY - Is wall distance d correct?")
            print("  → If NO: Fix wall distance calculation.")
            print("  → If YES: Proceed to step 3.")
            print("")
            print("STEP 3: CHECK SA MODEL (only if steps 1-2 pass)")
            if log_layer_check['chi_slope_ratio'] < 0.5:
                print("  PRIMARY ISSUE: χ is too low (ν̃ not growing correctly).")
                print("  Possible SA causes:")
                print("    • Diffusion has wrong sign (removing instead of adding ν̃)")
                print("    • Initial condition: ν̃ started too low")
                print("    • Numerical dissipation in ν̃ transport")
        else:
            print("Log-layer physics appears CORRECT.")
        
        print("")
        print("Remember: The SA model does NOT cause the log law.")
        print("The momentum equation produces u⁺ = (1/κ)ln(y⁺) + B.")
        print("The SA model is calibrated to SUPPORT this, not create it.")
        print("="*70)
    
    return {
        'log_layer': log_layer_check,
        'wall_scaling': wall_check,
        'chi': chi,
        'r': r_val,
        'fw': fw_val,
        'P': P,
        'D': D,
        'S_tilde': S_tilde
    }


# =============================================================================
# LEGACY COMPATIBILITY - Old function signatures
# =============================================================================

# Keep old names for backward compatibility
@jax.jit
def fv1(nuHat: ArrayLike) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Legacy fv1 with gradient. NOTE: Assumes chi = nuHat (normalized input)."""
    chi = nuHat
    chi3 = chi ** 3
    denom = chi3 + CV1 ** 3
    val = chi3 / denom
    grad = (3 * chi**2 * CV1**3) / (denom ** 2)
    return val, grad


@jax.jit
def fv1_value(nuHat: ArrayLike) -> jnp.ndarray:
    """Legacy fv1 value only. NOTE: Assumes chi = nuHat (normalized input)."""
    chi3 = nuHat ** 3
    return chi3 / (chi3 + CV1 ** 3)


@jax.jit
def fv2(nuHat: ArrayLike) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Legacy fv2 with gradient. NOTE: Assumes chi = nuHat (normalized input)."""
    chi = nuHat
    fv1_val, fv1_grad = fv1(nuHat)
    denom = 1.0 + chi * fv1_val
    val = 1.0 - chi / denom
    term_grad = (1.0 - chi**2 * fv1_grad) / (denom ** 2)
    grad = -term_grad
    return val, grad


@jax.jit
def effective_viscosity_safe(nuHat: ArrayLike, nu_laminar: float = 1.0) -> jnp.ndarray:
    """Compute effective viscosity with safety for negative nuHat."""
    nuHat_safe = jnp.clip(nuHat, min=0.0)
    fv1_val, _ = fv1(nuHat_safe)
    nu_turb = nuHat_safe * fv1_val
    return nu_laminar + nu_turb


@jax.jit
def sa_source_mask(nuHat: ArrayLike) -> jnp.ndarray:
    """Smooth mask for non-negative safety."""
    return 1.0 / (1.0 + jnp.exp(-100.0 * nuHat))


@jax.jit
def compute_diffusion_coefficient_safe(nuHat: ArrayLike, sigma: float = SIGMA,
                                       nu_laminar: float = 1.0) -> jnp.ndarray:
    """Compute diffusion coefficient with safety."""
    nuHat_safe = jnp.clip(nuHat, min=0.0)
    return (nu_laminar + nuHat_safe) / sigma
