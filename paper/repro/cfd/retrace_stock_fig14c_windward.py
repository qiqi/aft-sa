"""Remove the spurious windward tail spur from Stock's Fig. 14c computed front
(alpha = 5, Re_L = 6.49e6) and report the corrected symmetry-plane values.

WHAT WAS WRONG.  `data/stock2006_fig14c_digitized.json` ->
`stock_computed_lines.front_re6p49e6` ends its windward TS segment at
x/L = 0.4321, phi = 2.37, and we were reading that as Stock's windward
symmetry-plane front.  It is not.  The trace's last nine points are

    phi 2.37 2.38 2.38 2.40 2.63 2.64 2.64 2.71 2.73
    x/L .4321 .4344 .4415 .4486 .4556 .4626 .4697 .4768 .4837

-- nine points spanning 0.052 in x/L at essentially CONSTANT phi.  That is not
a transition front, it is the tracer running sideways along a near-horizontal
piece of ink at the bottom of the panel (the phi ~ 2.5 deg streamline, or the
axis frame) before terminating.  A front cannot be horizontal in this plane:
every other part of the traced curve advances in phi monotonically.

The genuine front is the vertical branch immediately above it, which is clean
and self-consistent:

    phi   3.43  5.55  7.64  9.67 11.67 13.69 15.69 17.68 19.68
    x/L  .4888 .4888 .4900 .4921 .4943 .4964 .4989 .5013 .5038

Extrapolating that branch to the symmetry plane gives x/L ~ 0.488, and a
2200-dpi render of the Fig. 14c windward corner puts the front meeting the
phi = 0 axis at X/a ~ -0.02, i.e. x/L ~ 0.49.  Stock's own Fig. 18, which
overlays the computed front for all incidences at comparable Re, puts alpha = 5
near X/a ~ 0.0 (x/L ~ 0.50) on both symmetry planes.  Three independent reads
agree on ~0.49; the 0.4321 was the far end of the spur.

SCOPE.  This is a nine-point tail artefact, NOT a bad windward half.  An
earlier note in this repo claimed the whole windward half needed re-checking to
0.07 x/L and that Fig. 14c and Fig. 18 disagreed at phi = 60 as well; both
claims were wrong.  Everything at phi >= 3.43 agrees with the render to about
0.005 x/L, which is the digitizer's own stated accuracy.

CONSEQUENCE, and it matters.  With the spur dropped, Stock's alpha = 5 front is
    windward  x/L ~ 0.489  (phi 3.43, the first sound point)
    leeward   x/L ~ 0.507  (phi 175.92, front_re6p49e6_piece2)
i.e. windward - leeward ~ -0.018, near-symmetric.  We had been carrying -0.075.
That agrees with Stock's own statement that the windward/leeward split opens
only above alpha = 5, and with the measured +0.002.

Run from paper/:  python3 repro/cfd/retrace_stock_fig14c_windward.py [--write]
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, '..', '..'))
JSON = os.path.join(PAPER, 'data', 'stock2006_fig14c_digitized.json')

KEY = 'front_re6p49e6'
# a run of >= MIN_N consecutive points whose phi span is under DPHI while their
# x/L span exceeds DXL is a sideways excursion, not a front
DPHI, DXL, MIN_N = 1.0, 0.020, 4


def spur(a):
    """-> boolean mask of points belonging to a horizontal tail excursion."""
    bad = np.zeros(len(a), bool)
    for end in (0, len(a) - 1):                 # only the two trace ends
        step = 1 if end == 0 else -1
        j = end
        run = [j]
        while 0 <= j + step < len(a):
            k = j + step
            if abs(a[k, 0] - a[end, 0]) > DPHI:
                break
            run.append(k)
            j = k
        if len(run) >= MIN_N and np.ptp(a[run, 1]) > DXL:
            bad[run] = True
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()

    d = json.load(open(JSON))
    blk = d['stock_computed_lines'][KEY]
    a = np.array(sorted((p['phi_deg'], p['xL']) for p in blk['points']))
    bad = spur(a)
    print(f'{KEY}: {len(a)} points, {bad.sum()} flagged as tail spur')
    for p, x in a[bad]:
        print(f'   DROP  phi {p:6.2f}  x/L {x:.4f}')
    keep = a[~bad]
    print(f'\nkept {len(keep)}: phi {keep[0,0]:.2f}..{keep[-1,0]:.2f}, '
          f'x/L {keep[:,1].min():.4f}..{keep[:,1].max():.4f}')

    p2 = d['stock_computed_lines'].get('front_re6p49e6_piece2')
    lee = None
    if p2:
        b = np.array(sorted((q['phi_deg'], q['xL']) for q in p2['points']))
        lee = float(b[-1, 1])
        print(f'leeward branch (piece2) ends phi {b[-1,0]:.2f} '
              f'-> x/L {lee:.4f}')
    wind = float(keep[0, 1])
    print(f'\ncorrected symmetry planes: windward {wind:.3f} (phi '
          f'{keep[0,0]:.2f}), leeward {lee:.3f}  -> windward - leeward '
          f'{wind-lee:+.3f}')
    print('was (0.432, 0.507) -> -0.075;  measured +0.002;  Fig. 18 ~0 at both')

    if args.write:
        blk['points'] = [{'phi_deg': round(float(p), 2),
                          'xL': round(float(x), 4)} for p, x in keep]
        blk['n_points'] = len(keep)
        segs = blk.get('mechanism_segments') or []
        for s in segs:                       # the windward TS segment
            if abs(s.get('phi_to', 1e9) - 2.37) < 0.2:
                s['phi_to'] = round(float(keep[0, 0]), 2)
                s['xL_to'] = round(float(keep[0, 1]), 4)
        blk['spur_removed'] = {
            'n_dropped': int(bad.sum()),
            'why': 'nine points spanning 0.052 x/L at constant phi ~ 2.4-2.7 '
                   '-- the tracer running sideways along near-horizontal ink '
                   'at the panel bottom, not a transition front',
            'checked_against': 'a 2200-dpi render of the windward corner '
                               '(front meets phi=0 at X/a ~ -0.02, x/L ~ 0.49) '
                               'and Stock Fig. 18 (alpha=5 near x/L ~ 0.50 on '
                               'both symmetry planes)',
            'by': 'repro/cfd/retrace_stock_fig14c_windward.py',
        }
        json.dump(d, open(JSON, 'w'), indent=1)
        print(f'\npatched {JSON}')


if __name__ == '__main__':
    main()
