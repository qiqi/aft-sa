#!/usr/bin/env python3
"""Surface distributions and wall-normal sheets at Re = 1e5 and 3e5.

The Re = 2e5 counterparts are the C_f page series and the per-incidence chi
sheets. These are the same two figures for the other two Reynolds numbers of
the incidence sweeps.

Two things differ from the Re = 2e5 versions, both forced by the data:

  * two rows, not six. The 1e5 and 3e5 incidence sweeps were run on the L2
    pair only, so there is no L0/L1 to draw.
  * chi and Re_Omega are normalized by each case's OWN nu (= muRef = M/Re),
    not by the Re = 2e5 constant. nu is 1e-6 at Re = 1e5 and 3.33e-7 at 3e5
    against 5e-7 at 2e5, so the Re = 2e5 constant would scale both by the
    Reynolds ratio. regen_eppler_v2.nu_of() now reads it per case.

Usage: python3 regen_epp_re_sheets.py [cf|sheets|all]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H)
import regen_eppler_v2 as R
import regen_chi_sheets as CS
from tm4062_re_data import STATIONS, NATURAL_TRANSITION

B = os.environ.get('SAAI_CFD_ROOT', '/home/qiqi/flexcompute/sa-ai/flow360_fv1')
FIGS = os.path.abspath(os.path.join(_H, '..', '..', 'figs'))

TAG = {-2: 'am2', 0: 'a0', 1: 'a1', 2: 'a2', 3: 'a3', 4: 'a4',
       6: 'a6', 8: 'a8', 10: 'a10'}

SWEEPS = [
    ('Re100k', 1.0e5, r'10^5',            (-2, 0, 2, 4, 6, 8, 10)),
    ('Re300k', 3.0e5, r'3\times10^5',     (-2, 0, 1, 2, 3, 4, 6, 8, 10)),
]


def case(retag, mesh, alpha):
    return f'{B}/sweep_{mesh}L2_{retag}_{TAG[alpha]}'


def pages(alphas, per=2):
    return [alphas[i:i + per] for i in range(0, len(alphas), per)]


def do_cf(retag, Re, relabel, alphas):
    """C_f page series, two incidences per page, two grid rows.

    Every reference overlay in make_cf_figure is R = 200,000 data: the
    oil-flow LSB band, the natural-transition line, the tabulated
    experimental C_p and the mfoil e^9 envelope. Drawing those on a 1e5 or
    3e5 page would put one Reynolds number's measurement under another's
    computation. The bands are swapped for this Reynolds number's own Table
    III entries; C_p and mfoil have no counterpart here and are switched off.
    """
    keep = (R.EXP_LSB, R.EXP_NT, R.EXP_CP_TAB, R.mn, R.xn)
    R.CASE_DIR_FN = lambda mesh, level, a, _rt=retag: case(_rt, mesh, a)
    R.LEVELS = ['L2']
    R.EXP_LSB = dict(STATIONS[int(Re)])
    R.EXP_NT = dict(NATURAL_TRANSITION[int(Re)])
    R.EXP_CP_TAB = {}
    R.mn, R.xn = {}, {}
    try:
        for k, group in enumerate(pages(alphas), start=1):
            out = f'eppler_cf_{retag}_p{k}'
            R.make_cf_figure(list(group), out,
                             f'Eppler 387, $Re={relabel}$, L2 pair')
            print(f'  wrote {out}.pdf  alphas={group}', flush=True)
    finally:
        R.CASE_DIR_FN = None
        R.LEVELS = ['L0', 'L1', 'L2']
        R.EXP_LSB, R.EXP_NT, R.EXP_CP_TAB, R.mn, R.xn = keep


def do_sheets(retag, Re, relabel, alphas):
    """Wall-normal sheets, one per incidence, cavity L2 over O-grid L2."""
    L_up = CS.AF_SETUP['eppler387']['L_up'] * np.sqrt(2e5 / Re)
    for a in alphas:
        rows = [('cav', 'unstructured L2'), ('str', 'O-grid L2')]
        have = [(f, lab, case(retag, f, a)) for f, lab in rows
                if os.path.exists(f'{case(retag, f, a)}/slice_centerSpan.pvtu')]
        if not have:
            print(f'  SKIP {retag} a={a}: no slices', flush=True)
            continue
        fig, axes = plt.subplots(len(have), 2, figsize=(11.5, 2.1 * len(have) + 0.6),
                                 sharex=True, sharey=True)
        if len(have) == 1:
            axes = np.asarray([axes])
        for r, (fam, lab, cd) in enumerate(have):
            x, d, chi, ux = CS.scan(R, 'eppler387', cd, 'upper', L_up)
            # scan() divides nuHat by the benchmark Re/Mach; rescale to this one
            chi = chi * (Re / 2e5)
            axU, axC = axes[r]
            cs = axU.contour(x, d * Re, ux, levels=CS.UX_LEV, colors='k', linewidths=0.6)
            axU.clabel(cs, levels=CS.UX_LABEL, fmt='%g', fontsize=6.5, inline_spacing=2)
            lg = np.log10(np.clip(chi, 1e-8, None))
            axC.contour(x, d * Re, lg, levels=CS.CHI_MINOR, colors='k', linewidths=0.4)
            cs = axC.contour(x, d * Re, lg, levels=CS.CHI_MAJOR, colors='k', linewidths=0.8)
            axC.clabel(cs, fmt=CS.CHI_FMT, fontsize=6.5, inline_spacing=2)
            axU.set_xlim(0, 1); axU.set_ylim(0, L_up * Re)
            axU.set_ylabel(f'{lab}\n' + r'$d\,U_\infty/\nu$', fontsize=9)
        axes[0, 0].set_title(r'$u_x/U_\infty$', fontsize=10)
        axes[0, 1].set_title(r'$\log_{10}\chi$', fontsize=10)
        for c in range(2):
            axes[-1, c].set_xlabel('wall-anchor x/c')
        out = os.path.join(FIGS, f'chi_sheet_eppler387_{retag}_{TAG[a]}_upper.pdf')
        fig.tight_layout(); fig.savefig(out); plt.close(fig)
        print(f'  wrote {os.path.basename(out)}', flush=True)


if __name__ == '__main__':
    what = (sys.argv[1] if len(sys.argv) > 1 else 'all').lower()
    for retag, Re, relabel, alphas in SWEEPS:
        if what in ('cf', 'all'):
            print(f'=== {retag} cf pages', flush=True)
            do_cf(retag, Re, relabel, alphas)
        if what in ('sheets', 'all'):
            print(f'=== {retag} sheets', flush=True)
            do_sheets(retag, Re, relabel, alphas)
    print('ALL DONE', flush=True)
