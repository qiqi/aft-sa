"""Low-H (vg two-branch) in-solver result plot for the whitepaper appendix,
in the canon drag-crisis style (cf. fig:dragcrisiscd / fig:dragcrisisangles):
vg vs canon C_d and transition angle theta_tr across Re_D = 2e6..1e10.

Data source: the budgeted overnight validation
agent-paper-review/2026-07-29-0320-vg-kernel-validation.md (Phase B tables;
vg from /local_data .../vg_summary.jsonl + vg_theta_tr.json, canon from
matrix_summary.jsonl, up-ladder, Tu=0.2%). highre mesh 2e6..2e7; ultra mesh
2e7..1e10 (the 2e7 seam appears on both meshes, as in fig:dragcrisiscd).
theta_tr = radial-ray chi=1 front.

-> figs/newkernel_vg_results.png  (grayscale, e-ink)
Run from paper/: python3 repro/analytic/regen_vg_lowH_results.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_H = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(_H, '..', '..'))

# (Re_D, canon Cd, vg Cd, canon theta_tr, vg theta_tr)
HIGHRE = [
    (2.0e6, 0.2214, 0.2088, 98.0, 99.0),
    (4.0e6, 0.2035, 0.2159, 96.5, 94.0),
    (7.0e6, 0.1970, 0.2090, 91.0, 90.0),
    (1.0e7, 0.1974, 0.2055, 88.5, 88.0),
    (2.0e7, 0.1992, 0.2048, 83.0, 82.5),
]
ULTRA = [
    (2.0e7, 0.1917, 0.1916, 89.0, 89.0),
    (5.0e7, 0.1867, 0.1888, 81.0, 78.0),
    (1.0e8, 0.1971, 0.2018, 63.0, 61.0),
    (3.0e8, 0.2057, 0.2107, 39.5, 33.5),
    (1.0e9, 0.2056, 0.2091, 16.5, 15.0),
    (1.0e10, 0.1889, 0.1989, 3.0, 2.5),
]


def cols(rows, i):
    return [r[0] for r in rows], [r[i] for r in rows]


fig, (axCd, axTh) = plt.subplots(1, 2, figsize=(7.4, 3.1))

for rows, mk in ((HIGHRE, 'highre'), (ULTRA, 'ultra')):
    re, cCd = cols(rows, 1)
    _, vCd = cols(rows, 2)
    _, cTh = cols(rows, 3)
    _, vTh = cols(rows, 4)
    lab_c = 'canon' if rows is HIGHRE else None
    lab_v = 'vg (low-$H$)' if rows is HIGHRE else None
    axCd.semilogx(re, cCd, 'o-', color='k', mfc='none', ms=5, lw=1.2, label=lab_c)
    axCd.semilogx(re, vCd, 's--', color='0.45', mfc='none', ms=5, lw=1.2, label=lab_v)
    axTh.semilogx(re, cTh, 'o-', color='k', mfc='none', ms=5, lw=1.2)
    axTh.semilogx(re, vTh, 's--', color='0.45', mfc='none', ms=5, lw=1.2)

axCd.set_xlabel('$Re_D$')
axCd.set_ylabel('$C_d$')
axCd.set_ylim(0.15, 0.25)
axCd.grid(alpha=0.25, which='both')
axCd.legend(fontsize=8, frameon=False, loc='upper left')
axCd.text(0.03, 0.05, '(a)', transform=axCd.transAxes, fontsize=11, va='bottom')

axTh.set_xlabel('$Re_D$')
axTh.set_ylabel(r'$\theta_{tr}$ ($\chi=1$)  [deg]')
axTh.set_ylim(0, 105)
axTh.grid(alpha=0.25, which='both')
axTh.text(0.03, 0.05, '(b)', transform=axTh.transAxes, fontsize=11, va='bottom')

fig.tight_layout()
out = f'{PAPER}/figs/newkernel_vg_results.png'
fig.savefig(out, dpi=150, facecolor='white')
fig.savefig('/tmp/newkernel_vg_results.png', dpi=130, facecolor='white')
print('wrote', out)
