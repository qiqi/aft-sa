"""eq:tumap -> printed table. Freestream turbulence Tu -> chi_inf seed -> N_crit.

Imports the canonical map from scripts.calibrate_kernel (Mack 1977 e^N, adopted
wholesale): N_crit = A_TU - B_TU ln(Tu_frac), chi_inf = c_v1 exp(-N_crit)."""
import _saai
from _saai import A_TU, B_TU, C_V1, chi_inf_from_Tu_pct
import math


def main():
    print(f"N_crit = {A_TU} - {B_TU} * ln(Tu_fraction)   [Mack 1977 e^N, adopted]")
    print(f"chi_inf = c_v1 * exp(-N_crit),  c_v1 = {C_V1}\n")
    print(f"{'Tu (%)':>7s}  {'chi_inf':>11s}  {'N_crit':>8s}")
    for Tu_pct in [0.04, 0.07, 0.08, 0.16, 0.30, 0.60]:
        chi = chi_inf_from_Tu_pct(Tu_pct)
        N = A_TU - B_TU*math.log(Tu_pct/100.0)
        print(f"  {Tu_pct:>5.3f}  {chi:>11.3e}  {N:>8.3f}")
    chi9 = C_V1*math.exp(-9.0)
    print(f"\nairfoil anchor N_crit=9: chi_inf={chi9:.3e}")
    print(f"Mack slope B_TU={B_TU}, offset A_TU={A_TU}")


if __name__ == '__main__':
    main()
