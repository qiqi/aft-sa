"""Quantitative comparison of the computed spheroid transition fronts
(Re_L=1.5e6, alpha=10) against Stock 2006 Fig. 15a's measured points
(DFVLR hot films, Re=1.52e6; digitized in
paper/data/stock2006_fig15a_digitized.json).

Front definitions from the surface-map .npz dumps (surface_map.py):
  chi front    : first x/L where the wall-normal max chi crosses c_v1
  cf-rise front: first x/L (past the nose) where cf exceeds
                 1.5x its running minimum + 2e-4 (the hot-film criterion
                 analogue: the shear rise out of the laminar decay)

Outputs: overlay figure (L2 cf map + measured points + computed fronts)
-> paper/figs/spheroid_front_compare.pdf/png, and a comparison table
printed at the measured phis for L0/L1/L2 (grid convergence).

Run from paper/: python3 ../spheroid/compare_stock_fronts.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER = '/home/qiqi/flexcompute/sa-ai/paper'
CV1 = 7.1


def fronts(npz):
    d = np.load(npz)
    xl, ph = d['xl'], d['phi_deg']
    chi, cf = d['chimax'], d['cf']
    n = len(ph)
    f_chi = np.full(n, np.nan)
    f_cf = np.full(n, np.nan)
    for i in range(n):
        c = chi[i]
        hits = np.where(np.isfinite(c) & (c > CV1))[0]
        if len(hits) and hits[0] > 0:
            j = hits[0]
            f = (CV1 - c[j-1]) / (c[j] - c[j-1])
            f_chi[i] = xl[j-1] + f * (xl[j] - xl[j-1])
        v = cf[i]
        runmin = np.inf
        for j in range(len(xl)):
            if not np.isfinite(v[j]) or xl[j] < 0.05:
                continue
            runmin = min(runmin, v[j])
            if xl[j] > 0.2 and v[j] > 1.5 * runmin + 2e-4:
                f_cf[i] = xl[j]
                break
    return xl, ph, f_chi, f_cf, cf


D = json.load(open(f'{PAPER}/data/stock2006_fig15a_digitized.json'))
SQ = D['re_1p52e6_alpha10_squares']

res = {}
for lev in ('L0', 'L1', 'L2'):
    npz = f'{PAPER}/figs/spheroid_maps_{lev}.npz'
    res[lev] = fronts(npz)

# ---- comparison table at the measured phis ---------------------------------
print(f"{'phi':>6} {'meas x/L':>9} |"
      + ''.join(f" {lev+' chi':>8} {lev+' cf':>8}" for lev in ('L0','L1','L2'))
      + " | d(L2 cf) ")
for s in SQ:
    row = [f"{s['phi_deg']:6.1f} {s['xL']:9.3f} |"]
    for lev in ('L0', 'L1', 'L2'):
        xl, ph, f_chi, f_cf, _ = res[lev]
        fc = np.interp(s['phi_deg'], ph, f_chi)
        ff = np.interp(s['phi_deg'], ph, f_cf)
        row.append(f" {fc:8.3f} {ff:8.3f}")
    xl, ph, f_chi, f_cf, _ = res['L2']
    ff = np.interp(s['phi_deg'], ph, f_cf)
    row.append(f" | {ff - s['xL']:+7.3f}")
    print(''.join(row))

# ---- overlay figure ---------------------------------------------------------
xl, ph, f_chi, f_cf, cf = res['L2']
fig, ax = plt.subplots(figsize=(9.6, 4.4))
m = ax.contourf(xl, ph, cf * 1e3, levels=np.linspace(0, 6, 25),
                cmap='viridis', extend='max')
fig.colorbar(m, ax=ax, label=r'$c_f \times 10^3$')
for lev, c, ls in (('L0', 'w', ':'), ('L1', 'w', '--'), ('L2', 'w', '-')):
    _, phL, fchiL, fcfL, _ = res[lev]
    ax.plot(fcfL, phL, ls, color=c, lw=1.4,
            label=f'{lev} $C_f$-rise front')
ax.plot(f_chi, ph, '-', color='cyan', lw=1.2, label=r'L2 $\chi=c_{v1}$ front')
ax.plot([s['xL'] for s in SQ], [s['phi_deg'] for s in SQ], 's',
        color='red', mfc='none', ms=9, mew=2,
        label='measured (Stock Fig. 15a, DFVLR)')
ax.set_xlabel('$x/L$')
ax.set_ylabel(r'$\phi$ [deg]  (0 = windward)')
ax.set_ylim(0, 180)
ax.set_xlim(0, 1)
ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax.set_title(r'6:1 spheroid, $Re_L=1.5\times10^6$, $\alpha=10^\circ$: '
             'computed fronts vs measured transition')
fig.tight_layout()
fig.savefig(f'{PAPER}/figs/spheroid_front_compare.pdf')
fig.savefig(f'{PAPER}/figs/spheroid_front_compare.png', dpi=140)
print('wrote spheroid_front_compare.pdf/png')
