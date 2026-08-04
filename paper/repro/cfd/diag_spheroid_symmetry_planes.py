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

  3. RETRACTED.  An earlier version of this file argued that Stock's
     coupled N_TS-N_CF threshold explains it: N_TS = 8.0 while N_CF < 1,
     decaying as N_CF rises, so the threshold peaks where crossflow
     vanishes.  That argument is WRONG for the question asked.  N_CF = 0 at
     BOTH symmetry planes, by symmetry, so Stock's threshold is identically
     8.0 at both and cannot produce either a difference or a similarity
     between them.  The coupled threshold does explain why his front is late
     at both planes RELATIVE TO THE MIDDLE; it says nothing about why the two
     planes agree with each other.  Only the growth can do that.

  4. And the growth, as computed here, does not.  Worse, the one 3-D effect
     the axisymmetric march omits pushes the wrong way: the lateral strain
     (1/r0) d(u_phi)/d(phi) is strongly POSITIVE at the windward plane
     (streamtube diverging, layer thinned, transition later) and strongly
     NEGATIVE at the leeward plane (converging, thickened, earlier).  Both
     signs add to the leeward-early bias rather than cancelling it.

  So this is an OPEN question, not an answered one.  The measured windward and
  leeward fronts agree to 0.002 x/L at alpha=5 while every estimate available
  here says the leeward should transition 0.15 or more ahead.

  A KNOWN FLAW in the estimate above, which must be removed before drawing any
  conclusion: the edge velocity is taken from OUR OWN computed c_p, and the
  cached fields are the measured-seed cases whose leeward front sits at
  x/L = 0.273.  The march accumulates N from ~5 to 8 downstream of that, i.e.
  through a region our own solution has already made turbulent, so the input
  is contaminated exactly where it matters.  Two clean next steps: march on
  exact potential-theory u_e for the inclined spheroid (Stock reports
  potential theory matches the measured c_p well wherever the flow is attached
  and laminar), or probe the calibrated-seed fields, whose laminar run extends
  past x/L = 0.45 on the leeward meridian.

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
    print('OPEN: a 2-D envelope on these edge velocities gives leeward-early')
    print('by 0.07-0.15; the measurement gives ~0.  The coupled N_TS-N_CF')
    print('threshold CANNOT explain it -- N_CF = 0 at both planes, so that')
    print('threshold is identical there.  Only the growth can, and the growth')
    print('computed here does the opposite.  See the docstring for the known')
    print('contamination in this estimate and the two ways to remove it.')


if __name__ == '__main__':
    main()
