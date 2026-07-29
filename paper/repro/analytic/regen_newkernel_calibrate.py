"""Whitepaper Fig 25 -> figs/newkernel_calibrate.png.

Low-H Falkner-Skan family (H 2.216..2.59): the MAIN-PAPER canon single-branch
sphere kernel (form='add', eps=0 -- exactly fig:calibrate) vs the vg two-branch
kernel, both against Drela-Giles, computed with one machinery (fpg
recalibration S.row = measures_for_beta with the patched rate) so they are
directly comparable. The canon low-H onset explodes (rate collapses to ~0):
Rt1 = 2.5e6 at H=2.216 (369x the DG N=1 station), 4.7e6 at H=2.285 -- it runs
off the panel top/bottom on Drela-range axes. vg tracks DG to ~0.6-1.0x.

Caches rows to figs_explore/newkernel_calibrate_rows.json (delete to recompute).
Run from paper/: python3 repro/analytic/regen_newkernel_calibrate.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fpg_recalibration_study as S
from lib.correlations import dN_dRe_theta, Re_theta0

BETAS = [1.0, 0.55, 0.35, 0.2, 0.1, 0.05, 0.0]
EPS_R = 0.1455
BC = 129.74
CACHE = 'repro/analytic/figs_explore/newkernel_calibrate_rows.json'


def compute():
    S.FORM[0] = 'add'; S.EPS[0] = 0.0; S.AC[0] = None; S.BC[0] = 0.0
    canon = [S.row(b) for b in BETAS]
    S.FORM[0] = 'vg'; S.EPS[0] = EPS_R; S.AC[0] = None; S.BC[0] = BC
    vg = [S.row(b) for b in BETAS]
    return canon, vg


if os.path.exists(CACHE):
    d = json.load(open(CACHE)); canon, vg = d['canon'], d['vg']
else:
    canon, vg = compute()
    json.dump({'canon': canon, 'vg': vg}, open(CACHE, 'w'), indent=1)


def cols(rows):
    H = np.array([r['H'] for r in rows]); sl = np.array([r['s_late'] for r in rows])
    R = np.array([r['Rt1'] for r in rows]); o = np.argsort(H)
    return H[o], sl[o], R[o]


fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.2, 4.3))
fig.patch.set_facecolor('white')
Hg = np.geomspace(2.2, 2.62, 200)
dg = np.asarray(dN_dRe_theta(Hg)); Rtc = np.asarray(Re_theta0(Hg)); N1 = Rtc + 1.0/dg
for rows, col, lab, mk in [(canon, '0.55', 'canon (main paper, single-branch)', 'o'),
                           (vg, 'C0', 'vg (two-branch gate)', '^')]:
    H, sl, R = cols(rows)
    axa.semilogy(H, sl, '-'+mk, color=col, lw=1.4, ms=5, label=lab)
    axb.semilogy(H, R, '-'+mk, color=col, lw=1.4, ms=5, label=lab)
axa.semilogy(Hg, dg, 'k--', lw=1.8, label='Drela--Giles')
axb.semilogy(Hg, Rtc, '--', color='0.5', lw=1.2, label=r'Drela crit $Re_{\theta0}$')
axb.semilogy(Hg, N1, 'k--', lw=1.8, label=r'DG $N=1$ station')
axa.set_xlabel('$H$'); axa.set_ylabel(r'$dN/dRe_\theta$ (late secant)')
axa.grid(alpha=0.3, which='both'); axa.legend(fontsize=8); axa.set_title('(a) rate')
axa.set_ylim(1e-3, 2e-2)      # Drela range; canon low-H rate runs off the bottom
axb.set_xlabel('$H$'); axb.set_ylabel(r'onset $Re_\theta$')
axb.grid(alpha=0.3, which='both'); axb.legend(fontsize=8); axb.set_title('(b) onset')
axb.set_ylim(2e2, 2e4)        # Drela range; canon low-H onset runs off the top
plt.tight_layout()
fig.savefig('figs/newkernel_calibrate.png', dpi=150, facecolor='white')
print('wrote figs/newkernel_calibrate.png')
