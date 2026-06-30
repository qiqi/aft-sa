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

# Tu -> chi_inf mapping. Single source of truth for the AFT-SA paper.
#
# N_crit = A_TU - B_TU * ln(Tu_fraction)     [Tu in fraction, NOT percent]
# chi_inf = c_v1 * exp(-N_crit)              [Anchor A]
#
# Mack 1977 e^N envelope (for reference, NOT what we use):
#     N_crit_Mack = -8.43 - 2.4 ln(Tu_fraction)
# Calibrated to the Abu-Ghannam & Shaw (1980) natural-transition onset
# correlation  Re_theta_t = 163 + exp(6.91 - Tu%)  over Tu = 0.04-0.60%, the
# community-standard low-Tu reference (a smooth fit through Schubauer-Skramstad-
# type data). The model's transition onset (chi=1) is matched to AGS Re_theta_t
# via the model's amplification envelope, giving B=2.705 (between Mack's 2.4 and
# the previous, mis-sourced B=4.00). NOTE: the earlier B=4.00 / A=-20.25 were fit
# to tabulated "S-S" Re_theta values that match neither Schubauer-Skramstad's
# actual Fig. 7 band (NACA Rep. 909) nor AGS; they were Mack-N_crit-via-envelope
# numbers mislabeled as experiment. See paper Sec. flat-plate.
A_TU = -9.088   # offset
B_TU = 2.705    # ln(Tu) coefficient (positive => -B_TU*ln(Tu_fraction) term)
C_V1 = 7.1

import math as _math
def chi_inf_from_Tu_pct(Tu_pct):
    """AFT-SA Tu (%) -> chi_inf seed for freestream BC (Anchor A)."""
    N_crit = A_TU - B_TU * _math.log(Tu_pct / 100.0)
    return C_V1 * _math.exp(-N_crit)

def Tu_pct_from_chi_inf(chi_inf):
    """Inverse: given chi_inf, what Tu (%) does the AFT-SA model see?"""
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
    print("=== Tu -> chi_inf mapping (AFT-SA-tuned) ===")
    print(f"  N_crit = {A_TU} - {B_TU} * ln(Tu_fraction)")
    print(f"  vs Mack:  N_crit = -8.43 - 2.4 * ln(Tu_fraction)\n")
    print(f"{'Tu (%)':>7s}  {'chi_inf':>10s}  {'N_crit_AFT':>11s}  {'N_crit_Mack':>11s}")
    for Tu_pct in [0.026, 0.06, 0.18, 0.30, 0.85]:
        chi = chi_inf_from_Tu_pct(Tu_pct)
        N_AFT = A_TU - B_TU * _math.log(Tu_pct/100.0)
        N_Mack = -8.43 - 2.4 * _math.log(Tu_pct/100.0)
        print(f"  {Tu_pct:>5.3f}  {chi:>10.3e}  {N_AFT:>11.3f}  {N_Mack:>11.3f}")
    print()

    print("=== Current committed model: (a_FP, a_PG) = (0.0094, 0.181), nuLamScale=0.25 ===")
    s, gc = sg_from_anchors(0.0094, 0.181)
    print(f"  -> (s, g_c) = ({s:.4f}, {gc:.4f})")
    a1, a2 = anchors_from_sg(s, gc)
    print(f"  Round-trip: a(Γ=1)={a1:.5f}, a(Γ=2)={a2:.5f}")
    print(f"  a(Γ=1.5) = {A_MAX/(1+math.exp(-s*(1.5-gc))):.5f}")
    print(f"  a(Γ=1.7) = {A_MAX/(1+math.exp(-s*(1.7-gc))):.5f}")
