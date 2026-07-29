"""Low-H (vg two-branch) in-solver result plot for the whitepaper appendix,
in the canon drag-crisis style (cf. fig:dragcrisiscd / fig:dragcrisisangles):
vg vs canon C_d and transition angle theta_tr across Re_D = 2e6..1e10.

Data source: the budgeted overnight validation
agent-paper-review/2026-07-29-0320-vg-kernel-validation.md (Phase B tables;
vg from /local_data .../vg_summary.jsonl + vg_theta_tr.json, canon from
matrix_summary.jsonl, up-ladder, Tu=0.2%). highre mesh 2e6..2e7; ultra mesh
2e7..1e10 (the 2e7 seam appears on both meshes, as in fig:dragcrisiscd).
theta_tr = radial-ray chi=1 front.

Panel (c): radial-ray max-chi(theta) profiles, canon (solid) vs vg (dashed),
for Re_D = 2e6, 1e8, 1e9, 1e10 -- the SAME radial-ray extractor the paper
uses (dragcrisis_transition_angle.ray_profiles). Canon profiles come from the
already-cached dragcrisis_theta_tr.json (2e6) and the systematic campaign
summary systematic_Tu0.2_summary.jsonl (ultra ladder); vg profiles from
vg_maxchi_profiles.json (repro/cfd/vg_maxchi_profile_extract.py).

-> figs/newkernel_vg_results.png  (grayscale, e-ink)
Run from paper/: python3 repro/analytic/regen_vg_lowH_results.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_H = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(_H, '..', '..'))

# max-chi(theta) profile sources (theta grid: 0..180 step 0.5, 361 pts)
THETA = np.arange(0.0, 180.0 + 1e-9, 0.5)
DATA = os.path.join(_H, '..', 'cfd', 'figs_explore', 'data')
CANON_THETA_JSON = os.path.join(DATA, 'dragcrisis_theta_tr.json')
VG_PROFILE_JSON = os.path.join(DATA, 'vg_maxchi_profiles.json')
SYS_SUMMARY = ('/local_data/qiqi/sa-ai/dragcrisis_matrix/'
               'systematic_Tu0.2_summary.jsonl')

# (Re_D, canon case key, vg case key, label)
PROFILE_RE = [
    (2.0e6,  'cyl_Re2000000_Tu0.2_up',            'cyl_Re2000000_Tu0.2_up_highre_vg'),
    (1.0e8,  'cyl_Re100000000_Tu0.2_up_sys_ultra', 'cyl_Re100000000_Tu0.2_up_ultra_vg'),
    (1.0e9,  'cyl_Re1000000000_Tu0.2_up_sys_ultra', 'cyl_Re1000000000_Tu0.2_up_ultra_vg'),
    (1.0e10, 'cyl_Re10000000000_Tu0.2_up_sys_ultra', 'cyl_Re10000000000_Tu0.2_up_ultra_vg'),
]


def _re_label(re):
    e = int(round(np.log10(re)))
    m = re / 10.0 ** e
    return (rf'$Re_D=10^{{{e}}}$' if abs(m - 1.0) < 1e-6
            else rf'$Re_D={m:.0f}\times10^{{{e}}}$')


def load_maxchi_profiles():
    """Return {re: (canon_maxchi[361], vg_maxchi[361])} (linear chi)."""
    cj = json.load(open(CANON_THETA_JSON))['cases']
    vj = json.load(open(VG_PROFILE_JSON))['cases']
    sys_rows = {}
    with open(SYS_SUMMARY) as f:
        for ln in f:
            d = json.loads(ln)
            sys_rows[d['case']] = d
    out = {}
    for re, canon_key, vg_key in PROFILE_RE:
        if canon_key in cj:
            can = np.asarray(cj[canon_key]['profiles']['log10_maxchi'])
        else:
            can = np.asarray(sys_rows[canon_key]['log10_maxchi'])
        vg = np.asarray(vj[vg_key]['log10_maxchi'])
        out[re] = (10.0 ** can, 10.0 ** vg)
    return out

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


fig = plt.figure(figsize=(7.4, 6.0))
gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.55,
                      left=0.09, right=0.98, top=0.95, bottom=0.09)
axCd = fig.add_subplot(gs[0, 0:2])
axTh = fig.add_subplot(gs[0, 2:4])

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

# --- panel (c): radial-ray max-chi(theta), canon (solid) vs vg (dashed) ---
prof = load_maxchi_profiles()
for j, (re, canon_key, vg_key) in enumerate(PROFILE_RE):
    ax = fig.add_subplot(gs[1, j])
    can, vg = prof[re]
    ax.semilogy(THETA, np.clip(can, 1e-3, None), '-', color='k', lw=1.2,
                label='canon')
    ax.semilogy(THETA, np.clip(vg, 1e-3, None), '--', color='0.45', lw=1.2,
                label='vg')
    ax.axhline(1.0, color='0.6', lw=0.8, ls=':')       # chi = 1 front
    ax.set_xlim(0, 180)
    ax.set_ylim(1e-2, 5e8)
    ax.set_xticks([0, 90, 180])
    ax.grid(alpha=0.2, which='major')
    ax.set_title(_re_label(re), fontsize=8.5)
    ax.set_xlabel(r'$\theta$ [deg]', fontsize=9)
    if j == 0:
        ax.set_ylabel(r'$\max_s\,\chi$', fontsize=10)
        ax.legend(fontsize=7.5, frameon=False, loc='lower right')
        ax.text(0.06, 0.90, '(c)', transform=ax.transAxes, fontsize=11,
                va='top')
    else:
        ax.tick_params(labelleft=False)

out = f'{PAPER}/figs/newkernel_vg_results.png'
fig.savefig(out, dpi=150, facecolor='white')
fig.savefig('/tmp/newkernel_vg_results.png', dpi=130, facecolor='white')
print('wrote', out)
