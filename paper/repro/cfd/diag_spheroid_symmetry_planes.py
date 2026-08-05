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

  4. ALSO RETRACTED.  An earlier version claimed the one 3-D effect this march
     omits pushes the wrong way, quoting a "lateral strain" of +5858 at the
     windward plane and -4363 at the leeward one.  Those numbers are garbage.
     They were computed from the cache's `up` field, which is the azimuthal
     WALL SHEAR, not an azimuthal velocity -- see the |us| ~ 760 at x/L = 0.05
     rising to ~1520 across transition, which is a c_f signature, not a speed.
     So the magnitudes are meaningless AND the field is the wrong one: the term
     the momentum integral needs is the divergence of the INVISCID EDGE
     streamlines, (1/r0) d(u_e,phi)/d(phi), which these caches do not carry.
     Nothing is established about the 3-D term, in either direction.

  5. The contamination worry raised earlier was real but secondary.  Our own
     c_p does asymmetrise the two meridians GENUINELY, not just downstream of
     our own front: over x/L = 0.05-0.27, where both planes are still laminar
     in our own solution, the mean Falkner-Skan beta is +0.092 windward against
     +0.006 leeward, and u_e at x/L = 0.05 is 0.931 windward against 1.039
     leeward.  The leeward meridian is a faster, flat-plate-like run (beta ~ 0)
     and the windward one is a slower, still-accelerating run.  Both of those
     differences say leeward-early under any 2-D envelope.  So the +0.146 is
     not an artefact of marching through our own turbulent region.

  WHAT THE LITERATURE SETTLES, and what it does not.  See
  ../spheroid-literature-after-stock.md.  The 1st AIAA CFD Transition Modeling
  and Prediction Workshop (2021) ran this exact condition as its Case 3
  (M = 0.13, alpha = 5/10/15, Re_L = 6.5e6, Tu = 0.15%) and compared alpha = 5
  meridian by meridian at phi = 0, 60 and 180.  Its two independent LST + e^N
  submittals put the front at 0.50-0.56 at BOTH symmetry planes and match the
  measurement at both.  So the near-symmetry is a reproducible property of
  e^N, not an artefact of Stock's calibration, and point 2 above is a real
  deficiency of the axisymmetric envelope rather than evidence against Stock.

  Meanwhile every workshop transport model that transitions in the interior
  transitions EARLIER at phi = 180 than at phi = 0, by 0.24 to 0.73 x/L -- with
  SA + AFT2017b, the amplification-factor transport model closest in spirit to
  SA-AI, the worst of the set.  Our leeward-early bias is the family signature,
  not an SA-AI quirk.

  STILL OPEN: what the LST marches see that a local model cannot.  The leading
  candidate is streamwise history along the true 3-D surface streamline,
  including the symmetry-plane divergence term retracted in point 4 -- which
  needs the exact potential-theory edge field for the inclined spheroid, since
  neither our caches nor our own solution can supply it uncontaminated.

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
# Measured windward / leeward fronts, read from the digitized hot-film points.
# At alpha=2.5 (fig14b) there ARE stations on both symmetry planes, phi = 0.0
# and 180.0 exactly.  At alpha=5 (fig14c) there are NOT: the nearest circles
# sit at phi = 10.3 and 159.9, so that pair is near-plane, not on-plane.
MEAS = {2.5: (0.450, 0.432), 5.0: (0.567, 0.565)}
# Stock's computed front at (or nearest to) the two symmetry planes, read from
# the same digitized polylines -- NOT assumed.  Earlier revisions of this file
# carried alpha=2.5 as (0.460, 0.430), which is WINDWARD AND LEEWARD SWAPPED:
# fig14b's computed_ts_front runs 0.430 at phi=1.8 to 0.460 at phi=178.4.  So
# Stock is leeward-LATE at both incidences, the opposite sign to us.
STOCK = {2.5: (0.430, 0.460),      # fig14b, phi 1.8 and 178.4
         5.0: (0.488, 0.507)}      # fig14c, phi 3.4 and 175.9 (piece2)
# The alpha=5 windward value was 0.432 until a nine-point horizontal TAIL SPUR
# was removed from the fig14c trace -- points spanning 0.052 x/L at constant
# phi ~ 2.4-2.7, the tracer running sideways along near-horizontal ink at the
# panel bottom.  See repro/cfd/retrace_stock_fig14c_windward.py.  The corrected
# 0.488 agrees with a 2200-dpi render of the corner (~0.49) and with Stock's own
# Fig. 18 (~0.50 on both planes).  So Stock at alpha=5 is NEAR-SYMMETRIC
# (-0.019), not asymmetric by -0.075 as this file previously reported.


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
    print('A 2-D envelope on these edge velocities gives leeward-early by')
    print('0.07-0.15; the measurement gives ~0.  The coupled N_TS-N_CF')
    print('threshold cannot explain that -- N_CF = 0 at both planes, so the')
    print('threshold is identical there.  AIAA TMPW-1 (2021) Case 3 settles')
    print('the empirical half: its two independent LST+e^N submittals put the')
    print('alpha=5 front at 0.50-0.56 at BOTH planes and match measurement at')
    print('both, so the near-symmetry is a property of e^N, not of Stock.')
    print('Every workshop TRANSPORT model instead transitions 0.24-0.73 x/L')
    print('earlier at phi=180 than at phi=0 -- SA+AFT2017b worst.  Our bias is')
    print('the family signature.  What e^N sees that a local model cannot is')
    print('still open; see the docstring and spheroid-literature-after-stock.md.')


if __name__ == '__main__':
    main()
