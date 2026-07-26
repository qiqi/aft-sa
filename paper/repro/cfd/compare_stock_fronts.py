"""Quantitative comparison of the computed spheroid transition fronts
(Re_L=1.5e6, alpha=10) against Stock 2006 Fig. 15a's measured points
(DFVLR hot films, Re=1.52e6; digitized in
paper/data/stock2006_fig15a_digitized.json).

Front definitions from the surface-map .npz dumps (surface_map.py):
  chi fronts   : first x/L where the wall-normal max chi crosses 1
                 (blend onset) and c_v1 (half-saturation); the two
                 nearly coincide (<=0.05 L over the azimuth grid) --
                 the fast-handover diagnostic
                 (the model-native front; at this Re it sits at
                 0.92-0.97 x/L for every phi and does NOT track the
                 measured points -- the finding, not a bug: the
                 measured points are the laminar-separation-line
                 shear rise, Stock Sec. III.C)
  cf-rise front: first x/L (past the nose, sub-cell interpolated) where
                 cf exceeds k x its running minimum, k = 1.5 (the
                 hot-film analogue: the resultant-shear rise out of the
                 laminar decay; Kreplin's detection quantity). The
                 criterion sensitivity is quantified with k = 1.25 and
                 2.0 and reported as a band.

Outputs: overlay figure (L2 cf map + measured points + computed fronts)
-> paper/figs/spheroid_front_compare.pdf/png, and a comparison table
printed at the measured phis for L0/L1/L2 (grid convergence).

Run from paper/: python3 repro/cfd/compare_stock_fronts.py
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
    f_chi1 = np.full(n, np.nan)
    f_chi = np.full(n, np.nan)
    for i in range(n):
        c = chi[i]
        for lev, arr in ((1.0, f_chi1), (CV1, f_chi)):
            hits = np.where(np.isfinite(c) & (c > lev))[0]
            if len(hits) and hits[0] > 0:
                j = hits[0]
                f = (lev - c[j-1]) / (c[j] - c[j-1])
                arr[i] = xl[j-1] + f * (xl[j] - xl[j-1])
    return xl, ph, f_chi1, f_chi, cf


def cf_front(xl, cf_row, k):
    """First sub-cell-interpolated x/L (x>0.2) where cf exceeds k x its
    running minimum. Pure relative criterion (no absolute offset)."""
    runmin = np.inf
    prev_ex = None
    for j in range(len(xl)):
        v = cf_row[j]
        if not np.isfinite(v) or xl[j] < 0.05:
            continue
        runmin = min(runmin, v)
        ex = v - k * runmin
        if xl[j] > 0.2 and ex > 0 and prev_ex is not None and prev_ex[1] <= 0:
            xj_1, e0 = prev_ex
            f = -e0 / (ex - e0)
            return xj_1 + f * (xl[j] - xj_1)
        prev_ex = (xl[j], ex)
    return np.nan


D = json.load(open(f'{PAPER}/data/stock2006_fig15a_digitized.json'))
SQ = D['re_1p52e6_alpha10_squares']

res = {}
for lev in ('L0', 'L1', 'L2'):
    npz = f'{PAPER}/figs/spheroid_maps_{lev}.npz'
    xl, ph, f_chi1, f_chi, cf = fronts(npz)
    fcf = {k: np.array([cf_front(xl, cf[i], k) for i in range(len(ph))])
           for k in (1.25, 1.5, 2.0)}
    res[lev] = (xl, ph, f_chi1, f_chi, fcf, cf)

# ---- comparison table at the measured phis ---------------------------------
print(f"{'phi':>6} {'meas x/L':>9} |"
      + ''.join(f" {lev+' cf':>8}" for lev in ('L0', 'L1', 'L2'))
      + " |  L2 band(k=1.25..2) | d(L2,k=1.5)")
for s in SQ:
    row = [f"{s['phi_deg']:6.1f} {s['xL']:9.3f} |"]
    for lev in ('L0', 'L1', 'L2'):
        _, ph, _, _, fcf, _ = res[lev]
        ff = np.interp(s['phi_deg'], ph, fcf[1.5])
        row.append(f" {ff:8.3f}")
    _, ph, f_chi1, f_chi, fcf, _ = res['L2']
    lo = np.interp(s['phi_deg'], ph, fcf[1.25])
    hi = np.interp(s['phi_deg'], ph, fcf[2.0])
    ff = np.interp(s['phi_deg'], ph, fcf[1.5])
    c1 = np.interp(s['phi_deg'], ph, f_chi1)
    cv = np.interp(s['phi_deg'], ph, f_chi)
    row.append(f" |  [{lo:5.3f},{hi:5.3f}] | {ff - s['xL']:+7.3f}"
               f" | chi1 {c1:5.3f} cv1 {cv:5.3f}")
    print(''.join(row))

# ---- overlay figure ---------------------------------------------------------
xl, ph, f_chi1, f_chi, fcf, cf = res['L2']
fig, ax = plt.subplots(figsize=(9.6, 4.4))
m = ax.contourf(xl, ph, cf * 1e3, levels=np.linspace(0, 6, 25),
                cmap='viridis', extend='max')
fig.colorbar(m, ax=ax, label=r'$c_f \times 10^3$')
ax.fill_betweenx(ph, fcf[1.25], fcf[2.0], color='w', alpha=0.25, lw=0,
                 label=r'L2 criterion band ($k=1.25$--$2$)')
for lev, c, ls in (('L0', 'w', ':'), ('L1', 'w', '--'), ('L2', 'w', '-')):
    _, phL, _, _, fcfL, _ = res[lev]
    ax.plot(fcfL[1.5], phL, ls, color=c, lw=1.4,
            label=f'{lev} $C_f$-rise front ($k=1.5$)')
ax.plot(f_chi1, ph, ':', color='cyan', lw=1.2, label=r'L2 $\chi=1$ front')
ax.plot(f_chi, ph, '-', color='cyan', lw=1.2, label=r'L2 $\chi=c_{v1}$ front')
sep = D.get('stock_computed_separation_line', {}).get('points', [])
if sep:
    ax.plot([q['xL'] for q in sep], [q['phi_deg'] for q in sep], '-.',
            color='0.35', lw=1.3,
            label='free-vortex separation line (Stock, computed)')
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
summary = []
for sq in SQ:
    row = dict(phi=sq['phi_deg'], meas=sq['xL'])
    for lev in ('L0', 'L1', 'L2'):
        _, phL, f1L, fvL, fcfL, _ = res[lev]
        row[f'cf_{lev}'] = round(float(np.interp(sq['phi_deg'], phL, fcfL[1.5])), 4)
    _, phL, f1L, fvL, fcfL, _ = res['L2']
    row['band_lo'] = round(float(np.interp(sq['phi_deg'], phL, fcfL[1.25])), 4)
    row['band_hi'] = round(float(np.interp(sq['phi_deg'], phL, fcfL[2.0])), 4)
    row['chi1_L2'] = round(float(np.interp(sq['phi_deg'], phL, f1L)), 4)
    row['cv1_L2'] = round(float(np.interp(sq['phi_deg'], phL, fvL)), 4)
    summary.append(row)
json.dump(dict(condition='Re_L=1.5e6 (measured 1.52e6), alpha=10',
               criterion='cf-rise k=1.5 x running min, band k=1.25-2',
               rows=summary),
          open(f'{PAPER}/data/spheroid_front_summary.json', 'w'), indent=1)
print('wrote spheroid_front_compare.pdf/png + front summary json')
