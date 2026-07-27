"""Figure for the attachment-anchored-branch appendix: standard SA on the
frozen Hiemenz field with EXACTLY zero freestream seed.

Left: the surviving turbulent branch -- steady max chi vs sqrt(Re_r)=L,
with the collapsed cases on the floor and the bisected critical band.
Right: line contours of log10 chi for the sustained L=3000 wedge
(flat-plate figure conventions: dashed = laminar levels chi<1, solid =
1, c_v1, 30).

Inputs: data/stagnation_bistability.json, data/stagnation_field_L3000.npz
(both written by stagnation_bistability.py).
-> figs/stagnation_bistability.pdf
Run from paper/: python3 repro/analytic/regen_stagnation_figure.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_H = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(_H, '..', '..'))

d = json.load(open(f'{PAPER}/data/stagnation_bistability.json'))
fld = np.load(f'{PAPER}/data/stagnation_field_L3000.npz')

fig, (ax, axf) = plt.subplots(
    1, 2, figsize=(10.2, 3.5), gridspec_kw=dict(width_ratios=[1, 1.5]))

sus = sorted([r for r in d['results'] if r['sustained']], key=lambda r: r['L'])
col = sorted([r for r in d['results'] if not r['sustained']],
             key=lambda r: r['L'])
FLOOR = 0.05
ax.plot([r['L'] for r in sus], [r['maxchi'] for r in sus], 'o-', color='k',
        mfc='none', ms=5, label='sustained (turbulent init)')
ax.plot([r['L'] for r in col], [FLOOR]*len(col), 'x', color='0.45', ms=6,
        label='collapses to $\\chi=0$')
c = d['critical']
ax.axvspan(c['L_lo'], c['L_hi'], color='0.85', zorder=0)
ax.axhline(1.0, color='0.7', lw=0.7, ls=':')
ax.axhline(7.1, color='0.7', lw=0.7, ls='--')
ax.text(33, 1.25, '$\\chi=1$', fontsize=8, color='0.4')
ax.text(33, 8.6, '$\\chi=c_{v1}$', fontsize=8, color='0.4')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(FLOOR*0.6, 200)
ax.set_xlabel('$L=x_{\\max}/\\delta=\\sqrt{Re_r}$')
ax.set_ylabel('steady $\\max\\chi$')
ax.legend(fontsize=8, loc='upper left', frameon=False)
# no in-figure titles (removed paper-wide by user order); the caption
# carries the panel assignment
ax.grid(alpha=0.25, which='both')

x, y, chi = fld['x'], fld['y'], fld['chi']
X, Y = np.meshgrid(x, y, indexing='ij')
lg = np.log10(np.maximum(chi, 1e-12))
axf.contour(X, Y, lg, levels=[-8, -6, -4, -2, -1], colors='0.6',
            linewidths=0.5, linestyles='dashed')
axf.contour(X, Y, lg, levels=[0.0], colors='k', linewidths=1.2)
axf.contour(X, Y, lg, levels=[np.log10(7.1), np.log10(30.0)], colors='k',
            linewidths=0.7)
axf.set_xlabel('$x/\\delta$')
axf.set_ylabel('$y/\\delta$')
axf.set_ylim(0, 40)
fig.tight_layout()
out = f'{PAPER}/figs/stagnation_bistability.pdf'
fig.savefig(out)
print('wrote', out)
