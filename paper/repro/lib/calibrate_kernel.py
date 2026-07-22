"""Canonical Python home for the SA-AI kernel constants + the Tu->chi_inf map.

This module holds the canonical numeric constants for the SA-AI transition
kernel (A_MAX, SIGMOID_SLOPE, SIGMOID_CENTER, RE_OMEGA_FLOOR, K_LAMBDA,
BARRIER_POWER) and the Mack (1977) freestream-turbulence receptivity map
(A_TU, B_TU, C_V1 -> chi_inf). These MUST match src/numerics/aft_sources.py
and Flow360 ModelConstants.h; tests/test_constants_consistency.py enforces it.

Kernel:  a(Gamma) = a_max * Q4 * sigmoid(s * (Gamma - g_c)) , with the Gamma-
dependent onset cliff (paper Sec. calib). In the CURRENT (Q4-gated) model the
sigmoid (s, g_c) and floor are NOT set by a two-point (a_FP, a_PG) inversion:
they are fixed by the Blasius nuHat-transport envelope meeting the Drela-Giles
envelope at its N=1 and N=9 points (see repro/analytic/fig03_fs_transport_rows.py).

The sg_from_anchors / anchors_from_sg helpers below implement the LEGACY
two-anchor inversion. They are retained only as diagnostic utilities and are
NOT how the committed constants are determined.
"""
import math

# Canonical SA-AI kernel constants (paper Table; identical to src/numerics/aft_sources.py
# and Flow360 ModelConstants.h). Q4 band gate ON. See tests/test_constants_consistency.py.
A_MAX = 0.19            # a_max: Michalke free-shear ceiling (Q4 gate -> 1 in free shear)
RE_OMEGA_FLOOR = 243.7  # cliff floor; three-anchor solve (Blasius N=1, N=9; separation mean)
SIGMOID_SLOPE = 10.68    # s: separation-limit mean rate matches Drela (third anchor)
SIGMOID_CENTER = 0.9874  # g_c: attached-asymptote center (Blasius-slope anchor)
K_LAMBDA = 6.20          # favorable-PG onset-delay slope (worst-point self-consistency)
FPG_RATE_SLOPE = 5.80    # K_r: favorable-rate factor 1/(1+(K_r*max(0,lambda_p))^2), fit at beta=0.35
BARRIER_POWER = 4.0     # p: sharp cliff exponent

# Tu -> chi_inf mapping. Single source of truth for the SA-AI paper.
#
# N_crit = A_TU - B_TU * ln(Tu_fraction)     [Tu in fraction, NOT percent]
# chi_inf = c_v1 * exp(-N_crit)              [transition at chi = c_v1]
#
# We ADOPT Mack's (1977) e^N critical-N-factor correlation WHOLESALE:
#     N_crit = -8.43 - 2.4 ln(Tu_fraction)       (Mack 1977, JPL 77-15)
# This is the community-standard e^N translation of freestream turbulence to a
# transition threshold; it is BORROWED, not fit. The SA-AI seed/threshold pair
# (seed chi_inf, transition at chi=c_v1) is isomorphic to the e^N envelope/N_crit
# split, so N_crit = ln(c_v1/chi_inf) is exactly the e^N threshold and B_TU IS the
# Mack slope. K_lambda (favorable-gradient onset delay) is DERIVED, not fit: it is
# set by matching the cliff's slope to the rate Drela's critical Re_theta rises
# with pressure gradient, evaluated at the worst (most-triggering) point of the
# Falkner-Skan family -- K_lambda = (d ln Re_theta_c/dbeta)/(d lambda_p^worst/dbeta)
# ~ 12.3/2.0 ~ 6.1 (worst-point fixed point at the three-anchor kernel). See paper Sec. calib
# (repro/analytic/fig05_06_klambda.py). The model carries NO constant fit to a transition case.
#
# The flat plate (paper Sec. flat-plate) then VERIFIES that (i) the working
# variable reproduces the Blasius amplification envelope and (ii) onset falls
# within the e^N/AGS reference spread. Adopting Mack's slope (2.4) rather than the
# earlier AGS-fit slope (2.705) shifts predicted onset ahead of the
# Abu-Ghannam & Shaw correlation Re_theta_t = 163 + exp(6.91 - Tu%) -- consistent
# with the known scatter between e^N-based and momentum-thickness-based onset
# correlations. (History: an even earlier B=4.00 / A=-20.25 was fit to mislabeled
# "S-S" tabulated numbers; superseded.)
A_TU = -8.43    # offset (Mack 1977)
B_TU = 2.4      # ln(Tu) coefficient = Mack's e^N slope (positive => -B_TU*ln(Tu) term)
C_V1 = 7.1

import math as _math
def chi_inf_from_Tu_pct(Tu_pct):
    """SA-AI Tu (%) -> chi_inf seed for freestream BC (Anchor A)."""
    N_crit = A_TU - B_TU * _math.log(Tu_pct / 100.0)
    return C_V1 * _math.exp(-N_crit)

def Tu_pct_from_chi_inf(chi_inf):
    """Inverse: given chi_inf, what Tu (%) does the SA-AI model see?"""
    N_crit = _math.log(C_V1 / chi_inf)
    Tu_frac = _math.exp(-(N_crit - A_TU) / B_TU)
    return Tu_frac * 100.0

def sg_from_anchors(a_FP, a_PG, a_max=A_MAX):
    """Given (a_FP, a_PG, a_max), return (s, g_c) for the SA-AI kernel."""
    if not (0 < a_FP < a_max and 0 < a_PG < a_max):
        raise ValueError(f"need 0 < a_FP, a_PG < a_max={a_max}")
    z_FP = math.log(a_FP / (a_max - a_FP))
    z_PG = math.log(a_PG / (a_max - a_PG))
    s = z_PG - z_FP
    g_c = 1.0 - z_FP / s
    return s, g_c

def anchors_from_sg(s, g_c, a_max=A_MAX):
    """Inverse: given (s, g_c), return (a_FP, a_PG, a_at_g_c=a_max/2)."""
    a_FP = a_max / (1.0 + math.exp(-s * (1.0 - g_c)))
    a_PG = a_max / (1.0 + math.exp(-s * (2.0 - g_c)))
    return a_FP, a_PG

if __name__ == '__main__':
    print("=== Tu -> chi_inf mapping (Mack 1977 e^N, adopted wholesale) ===")
    print(f"  N_crit = {A_TU} - {B_TU} * ln(Tu_fraction)   [= Mack 1977]\n")
    print(f"{'Tu (%)':>7s}  {'chi_inf':>10s}  {'N_crit':>8s}")
    for Tu_pct in [0.04, 0.07, 0.08, 0.16, 0.30, 0.60]:
        chi = chi_inf_from_Tu_pct(Tu_pct)
        N = A_TU - B_TU * _math.log(Tu_pct/100.0)
        print(f"  {Tu_pct:>5.3f}  {chi:>10.3e}  {N:>8.3f}")
    print(f"\n  airfoil anchor N_crit=9 (chi_inf={C_V1*_math.exp(-9):.3e}) "
          f"<-> Tu={Tu_pct_from_chi_inf(C_V1*_math.exp(-9)):.4f}%")
    print()

    # Committed kernel (canonical, paper Table = ModelConstants.h ai_sigmoidSlope/
    # ai_sigmoidCenter = JAX AFT_SIGMOID_SLOPE/CENTER). Q4 gate ON; a_max=0.19,
    # reOmegaFloor=290, K_lambda=5.9, nuLamScale=1/12. (s, g_c) are set by the
    # Blasius nuHat-transport envelope meeting Drela-Giles at N=1 and N=9 -- NOT by
    # the two-anchor (a_FP,a_PG) inversion below (retained only as a legacy utility).
    S_COMMIT, GC_COMMIT = SIGMOID_SLOPE, SIGMOID_CENTER
    a1, a2 = anchors_from_sg(S_COMMIT, GC_COMMIT)
    print(f"=== Committed kernel: (s, g_c) = ({S_COMMIT}, {GC_COMMIT}), "
          f"a_max={A_MAX}, nuLamScale=1/12 ===")
    print(f"  implied anchors: a(Γ=1)=a_FP={a1:.5f}, a(Γ=2)=a_PG={a2:.5f}")
    print(f"  a(Γ=1.5) = {A_MAX/(1+math.exp(-S_COMMIT*(1.5-GC_COMMIT))):.5f}")
    print(f"  a(Γ=1.7) = {A_MAX/(1+math.exp(-S_COMMIT*(1.7-GC_COMMIT))):.5f}")
