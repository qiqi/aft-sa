"""Ramp-width sensitivity quoted in Sec. II.D (the w=0.35 justification).

Marches the calibration instrument at w = {0.175, 0.35, 0.7} for three
wedges and prints the N=1 stations: the text quotes "<1% when w is halved,
<2.5% when doubled". Along a growing layer Re_Omega ~ Re_theta^2 sweeps
through the tanh ramp quickly, so the gate width matters weakly and the
anchor absorbs the common shift.

Run from paper/: python3 repro/analytic/explore_ramp_width.py  (~15 min)
"""
import numpy as np

import _saai  # noqa: F401
import fig04_shapefactor as f4
from fig04_shapefactor import measures_for_beta

BETAS = (0.0, -0.10, 0.10)
W_CANON = f4.RAMP_W


def main():
    base = {}
    for w in (W_CANON, 0.5*W_CANON, 2.0*W_CANON):
        f4.RAMP_W = w
        for beta in BETAS:
            H, s_m, s_l, Rt1 = measures_for_beta(beta, verbose=False)
            tag = ''
            if w == W_CANON:
                base[beta] = Rt1
            elif beta in base:
                tag = f"  ({(Rt1/base[beta]-1)*100:+.1f}% vs w={W_CANON})"
            print(f"w={w:5.3f} beta={beta:+.2f}: N=1 at Re_theta={Rt1:6.0f}{tag}",
                  flush=True)
    f4.RAMP_W = W_CANON


if __name__ == '__main__':
    main()
