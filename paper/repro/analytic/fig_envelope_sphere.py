"""Section IV figure -> paper/figs/envelope_sphere.pdf.

The indicator sphere of Fig. 1, front view, with the flows of Section IV that
are exact solutions: the Blasius profile for reference, the asymptotic suction
profile u = 1 - exp(-eta), the spanwise profile of the swept attachment line
(swept Hiemenz), and the single point (1, 2, 0)/sqrt(5) of a fluid in
solid-body rotation with its wall. Contours of Omega_hat*I_hat as in Fig. 1.

Run from paper/: python3 repro/analytic/fig_envelope_sphere.py
"""
import _saai  # noqa: F401
import numpy as np

import fig01_indicator_sphere as F
from fs_frozen_profile import build_profile
from _family_colors import beta_color
from envelope_bounds import swept_hiemenz

COL_SUCTION = '#7b3294'      # purple
COL_ATTACH = '#1b7837'       # green


def main():
    blasius = build_profile(0.0, None)
    s = np.linspace(0.0, 30.0, 6001)
    suction = dict(eta=s, u=1.0 - np.exp(-s), up=np.exp(-s), upp=-np.exp(-s))
    eta, g, gp, gpp = swept_hiemenz()
    attach = dict(eta=eta, u=g, up=gp, upp=gpp)
    profiles = [(blasius, beta_color(0.0), None),
                (suction, COL_SUCTION, None),
                (attach, COL_ATTACH, None)]
    F.render('envelope_sphere.pdf', 0.0, 0.0, 1.39, F.LABEL_AT_FRONT, profiles,
             points=[((1.0, 2.0, 0.0), 'k')])


if __name__ == '__main__':
    main()
