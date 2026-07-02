"""Helper: map the two physically-anchored kernel constants (a_FP, a_PG)
to the internal (sigmoid slope s, center g_c) used by SAAftTransition.h.

Definition (single source of truth — applied identically in Flow360 ModelConstants.h
and the JAX implementation in src/numerics/aft_sources.py):

    a(Gamma) = a_max * sigmoid(s * (Gamma - g_c))

with the kernel calibrated by pinning two physical anchors:
    a_FP = a(Gamma=1, Re_Omega->inf)   # flat-plate (Blasius) amplification rate
    a_PG = a(Gamma=2, Re_Omega->inf)   # adverse-PG saturation rate

Inversion:
    z_FP = ln(a_FP / (a_max - a_FP))
    z_PG = ln(a_PG / (a_max - a_PG))
    s     = z_PG - z_FP
    g_c   = 1 - z_FP / s

Decoupling property: a(Gamma=1) is identically a_FP and a(Gamma=2) is identically
a_PG for ANY (a_FP, a_PG) pair. Tuning a_FP does not move a_PG, and vice versa.
The (s, g_c) re-solve internally to satisfy both anchors simultaneously.
"""
import math

A_MAX = 0.15   # no-tilt cliff_floor=100, p=4 matches S-S ±12% with growth start at Drela 200

# Tu -> chi_inf mapping. Single source of truth for the SA-AF paper.
#
# N_crit = A_TU - B_TU * ln(Tu_fraction)     [Tu in fraction, NOT percent]
# chi_inf = c_v1 * exp(-N_crit)              [transition at chi = c_v1]
#
# We ADOPT Mack's (1977) e^N critical-N-factor correlation WHOLESALE:
#     N_crit = -8.43 - 2.4 ln(Tu_fraction)       (Mack 1977, JPL 77-15)
# This is the community-standard e^N translation of freestream turbulence to a
# transition threshold; it is BORROWED, not fit. The SA-AF seed/threshold pair
# (seed chi_inf, transition at chi=c_v1) is isomorphic to the e^N envelope/N_crit
# split, so N_crit = ln(c_v1/chi_inf) is exactly the e^N threshold and B_TU IS the
# Mack slope. K_lambda (favorable-gradient onset delay) is then the ONLY fitted
# scalar in the model.
#
# The flat plate (paper Sec. flat-plate) then VERIFIES that (i) the working
# variable reproduces the Blasius amplification envelope and (ii) onset falls
# within the e^N/AGS reference spread. Adopting Mack's slope (2.4) rather than the
# earlier AGS-fit slope (2.705) shifts predicted onset ~8-15% ahead of the
# Abu-Ghannam & Shaw correlation Re_theta_t = 163 + exp(6.91 - Tu%) -- consistent
# with the known scatter between e^N-based and momentum-thickness-based onset
# correlations. (History: an even earlier B=4.00 / A=-20.25 was fit to mislabeled
# "S-S" tabulated numbers; superseded.)
A_TU = -8.43    # offset (Mack 1977)
B_TU = 2.4      # ln(Tu) coefficient = Mack's e^N slope (positive => -B_TU*ln(Tu) term)
C_V1 = 7.1

import math as _math
def chi_inf_from_Tu_pct(Tu_pct):
    """SA-AF Tu (%) -> chi_inf seed for freestream BC (Anchor A)."""
    N_crit = A_TU - B_TU * _math.log(Tu_pct / 100.0)
    return C_V1 * _math.exp(-N_crit)

def Tu_pct_from_chi_inf(chi_inf):
    """Inverse: given chi_inf, what Tu (%) does the SA-AF model see?"""
    N_crit = _math.log(C_V1 / chi_inf)
    Tu_frac = _math.exp(-(N_crit - A_TU) / B_TU)
    return Tu_frac * 100.0

def sg_from_anchors(a_FP, a_PG, a_max=A_MAX):
    """Given (a_FP, a_PG, a_max), return (s, g_c) for the SA-AFT kernel."""
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

    # Committed kernel: the (s, g_c) sigmoid is the single source of truth
    # (ModelConstants.h aft_sigmoidSlope/aft_sigmoidCenter, JAX
    # AFT_SIGMOID_SLOPE/CENTER, paper Table). a_max=0.15, nuLamScale=1/12.
    S_COMMIT, GC_COMMIT = 5.263, 1.572
    a1, a2 = anchors_from_sg(S_COMMIT, GC_COMMIT)
    print(f"=== Committed kernel: (s, g_c) = ({S_COMMIT}, {GC_COMMIT}), "
          f"a_max={A_MAX}, nuLamScale=1/12 ===")
    print(f"  implied anchors: a(Γ=1)=a_FP={a1:.5f}, a(Γ=2)=a_PG={a2:.5f}")
    print(f"  a(Γ=1.5) = {A_MAX/(1+math.exp(-S_COMMIT*(1.5-GC_COMMIT))):.5f}")
    print(f"  a(Γ=1.7) = {A_MAX/(1+math.exp(-S_COMMIT*(1.7-GC_COMMIT))):.5f}")
