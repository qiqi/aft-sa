"""Verify the digitized Stock replots against the ERCOFTAC numerical originals.

The ERCOFTAC Classic Collection Case 074 files (data/ercoftac_case074/, see its
README) are the original DFVLR measurement files.  One of the three cases,
`nwg30` (alpha = 29.7 deg, Re_L = 6.54e6, NATURAL transition), is the same
condition as the circles of Stock's Fig. 16c, which we digitized from the
printed panel.  That overlap is a direct check on the digitization.

Two independent checks:

  1. GEOMETRY -- the 12 true station positions (X0/2A from the file headers)
     against the digitized station ladder.
  2. FRONT -- the paper's k = 1.5 c_f-rise criterion applied to the numerical
     c_f(x, phi) against the digitized measured-transition symbols.  Also
     evaluates the c_f-minimum as an alternative detection convention, to
     establish which one Kreplin's hot films correspond to.

Run from paper/:  python3 repro/cfd/verify_ercoftac_vs_digitized.py
Exits nonzero if the geometry check regresses past the digitizer's stated
accuracy (+/-0.0025 x/L).
"""
import glob
import json
import os
import re
import sys

import numpy as np

_H = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(_H, '..', '..'))
ERC = os.path.join(PAPER, 'data', 'ercoftac_case074', 'nwg30')
DIG = os.path.join(PAPER, 'data', 'stock2006_fig16c_digitized.json')

STATION_TOL = 0.0025   # digitizer's quoted accuracy
K_RISE = 1.5           # the paper's C_f-rise criterion


def read_cf(fn):
    """-> (X0/2A, array of (phi_deg, cf)) sorted by phi."""
    txt = open(fn, errors='replace').read()
    x = float(re.search(r'X0/2A=\s*([\d.]+)', txt).group(1))
    rows = []
    for line in txt.split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        p = line.split()
        if len(p) >= 3:
            try:
                rows.append((float(p[0]), float(p[1])))
            except ValueError:
                pass
    a = np.array(rows)
    return x, a[np.argsort(a[:, 0])]


def main():
    files = sorted(glob.glob(os.path.join(ERC, 'nwg30_cf_*.dat')))
    if not files:
        sys.exit(f'no ERCOFTAC cf files under {ERC}')
    S = sorted((read_cf(f) for f in files), key=lambda s: s[0])
    xs = np.array([s[0] for s in S])
    dig = json.load(open(DIG))

    # ---- 1. station geometry ------------------------------------------------
    ladder = sorted({round(v, 4) for v in dig['hotfilm_stations_xL']})
    dev = [min(abs(x - l) for l in ladder) for x in xs]
    print('ERCOFTAC nwg30 stations :', ' '.join(f'{x:.3f}' for x in xs))
    print(f'max |true - digitized| station x/L = {max(dev):.4f} '
          f'(tolerance {STATION_TOL})')

    # ---- 2. front ----------------------------------------------------------
    circles = sorted(dig['measured_re6p54e6_circles'], key=lambda p: p['phi_deg'])
    print()
    print(f'{"phi":>6} {"digitized":>10} {"cf-min":>8} {"k=1.5 rise":>11}')
    d_min, d_rise = [], []
    for p in circles:
        cf = np.array([np.interp(p['phi_deg'], s[1][:, 0], s[1][:, 1])
                       for s in S])
        x_min = xs[int(np.argmin(cf))]
        run = np.minimum.accumulate(cf)
        ir = next((i for i in range(len(cf))
                   if run[i] > 0 and cf[i] > K_RISE * run[i]), None)
        x_rise = xs[ir] if ir is not None else np.nan
        d_min.append(x_min - p['xL'])
        d_rise.append(x_rise - p['xL'])
        print(f"{p['phi_deg']:6.1f} {p['xL']:10.3f} {x_min:8.3f} {x_rise:11.3f}")

    d_min, d_rise = np.array(d_min), np.array(d_rise)
    ok = ~np.isnan(d_rise)
    print()
    print('cf-minimum   vs digitized: mean %+.3f rms %.3f max|d| %.3f'
          % (d_min.mean(), np.sqrt((d_min ** 2).mean()), abs(d_min).max()))
    print('k=1.5 rise   vs digitized: mean %+.3f rms %.3f max|d| %.3f (n=%d)'
          % (d_rise[ok].mean(), np.sqrt((d_rise[ok] ** 2).mean()),
             abs(d_rise[ok]).max(), ok.sum()))
    print('station spacing ~%.3f -- the rise criterion cannot resolve finer,'
          % np.diff(xs).mean())
    print('and cannot represent the multivalued front "hook" near phi ~ 52 deg')
    print('where the printed line doubles back (the large residuals).')

    if max(dev) > STATION_TOL:
        sys.exit(f'GEOMETRY REGRESSION: {max(dev):.4f} > {STATION_TOL}')
    print('\nGEOMETRY CHECK PASS')


if __name__ == '__main__':
    main()
