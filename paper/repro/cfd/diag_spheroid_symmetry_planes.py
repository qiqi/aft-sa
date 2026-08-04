"""Why the measured (and Stock's) transition is near-symmetric between the
windward and leeward symmetry planes, when their pressure-gradient histories
are nothing alike.

Crossflow is identically zero on both planes (they are symmetry planes), so
crossflow growth cannot explain any difference OR similarity between them.
Yet the measurement puts the two fronts within 0.018 x/L at alpha = 2.5 and
0.002 at alpha = 5, while the two meridians start with opposite pressure
gradients -- windward from the stagnation point and accelerating, leeward
already accelerated around the nose and decelerating.

What this script establishes, from committed data only:

  1. The gradient histories do NOT converge downstream.  The windward
     meridian stays favorable to x/L ~ 0.75 (alpha=5); the leeward turns
     adverse at x/L ~ 0.15.  The gap widens, it does not close.

  2. A textbook Drela-Giles envelope e^N, marched (Mangler-weighted Thwaites)
     on each meridian's OWN edge velocity, does NOT reproduce the symmetry
     either: it puts the leeward front 0.067 (alpha=2.5) and 0.146 (alpha=5)
     AHEAD of the windward one at N=8.  So the near-symmetry is not a
     property of the edge-velocity histories read through a 2-D envelope.

  3. What distinguishes Stock is his THRESHOLD, not his growth.  His limit is
     a coupled N_TS-N_CF curve (his Fig. 11c): N_TS = 8.0 while N_CF < 1, and
     N_TS decays linearly as N_CF rises.  N_CF = 0 exactly on both symmetry
     planes, so his TS threshold is at its MAXIMUM there and relaxes
     everywhere in between.  A threshold that is highest at both planes and
     lower between them produces a front late at both planes and early in the
     middle -- the observed U -- with no appeal to the gradient asymmetry.
     The script shows the front sensitivity to threshold: 2 units of N is
     worth ~0.12 x/L, the right size for the observed dip.

CAVEAT, stated because it matters: Stock's N_TS(N_CF) decay was FITTED to
this same DFVLR dataset, so his reproduction of the U is not independent
confirmation of the mechanism.  The U itself is in the measurement and is
real; the coupled-threshold explanation is his model of it.

Run from paper/:  python3 repro/cfd/diag_spheroid_symmetry_planes.py
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
CACHE = os.path.join(HERE, 'cache_spheroid_surface')

from spheroid_a0_physics import thwaites_eN          # noqa: E402

CONDS = [('a2p5', 2.5, 7.20e6), ('a5hi', 5.0, 6.49e6)]
# measured windward / leeward fronts at the two conditions (digitized)
MEAS = {2.5: (0.450, 0.432), 5.0: (0.567, 0.565)}
# Stock's computed front interpolated at the same azimuths
STOCK = {2.5: (0.460, 0.430), 5.0: (0.493, 0.550)}


def meridian(tag, phi):
    d = np.load(glob.glob(os.path.join(CACHE, f'*{tag}_*'))[0])
    xl, phd, cp = d['xl'], d['phi_deg'], d['cp']
    j = int(np.argmin(abs(phd - phi)))
    ue = np.sqrt(np.clip(1.0 - cp[j], 1e-9, None))
    return xl, ue


def main():
    for tag, al, Re in CONDS:
        nu = 1.0 / Re
        print(f'=== alpha = {al}, Re_L = {Re:.3g} ===')
        res = {}
        for nm, phi in (('windward', 0.0), ('leeward', 180.0)):
            xl, ue = meridian(tag, phi)
            m = (xl > 0.02) & (xl < 0.97)
            r = thwaites_eN(xl[m], ue[m], nu, (6, 8, 9))
            res[nm] = r
            g = np.gradient(ue[m], xl[m])
            flip = xl[m][1:][np.diff(np.sign(g)) != 0]
            print(f'  {nm:9s} onset {r["onset_x"]:.3f}  '
                  f'N6 {r["crossings"]["N6"]:.3f}  N8 {r["crossings"]["N8"]:.3f}'
                  f'  N9 {r["crossings"]["N9"]:.3f}   '
                  f'du_e/dx first flips at x/L '
                  f'{flip[0]:.3f}' if len(flip) else '')
        w, l = res['windward']['crossings'], res['leeward']['crossings']
        print(f'  plain Drela-Giles e^N, windward - leeward: '
              f'N6 {w["N6"]-l["N6"]:+.3f}  N8 {w["N8"]-l["N8"]:+.3f}  '
              f'N9 {w["N9"]-l["N9"]:+.3f}')
        mw, ml = MEAS[al]
        sw, sl = STOCK[al]
        print(f'  measured  windward - leeward: {mw-ml:+.3f}')
        print(f'  Stock     windward - leeward: {sw-sl:+.3f}')
        print(f'  threshold sensitivity: N=8 -> N=6 moves the windward front '
              f'{w["N6"]-w["N8"]:+.3f}, the leeward {l["N6"]-l["N8"]:+.3f}')
        print()
    print('Conclusion: the edge-velocity histories, read through a 2-D')
    print('envelope, give leeward-early by 0.07-0.15.  The measurement gives')
    print('~0.  So the symmetry is not in the growth; in Stock it is in the')
    print('coupled threshold, which peaks exactly where N_CF vanishes -- on')
    print('both symmetry planes.')


if __name__ == '__main__':
    main()
