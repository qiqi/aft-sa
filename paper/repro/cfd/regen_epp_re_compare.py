#!/usr/bin/env python3
"""Eppler 387 across three Reynolds numbers, against the measurements only.

Two figures:
  eppler_re_polar.pdf     drag polar, R = 1e5 / 2e5 / 3e5
  eppler_re_bubble.pdf    upper-surface separation and reattachment stations

Both carry the L2 pair only. That is not a presentation choice: the R = 1e5
and 3e5 incidence sweeps were run on the L2 pair alone, so putting L0/L1 on
the R = 2e5 curve would compare a different grid set at one Reynolds number
than at the other two.

Only the experiment is shown alongside. The other transition models in
Figs. eppolar/eppbubble were published at R = 2e5 and have no counterpart at
the other two Reynolds numbers.

Run on a host holding flow360_fv1 (014). Writes into paper/figs.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regen_eppler_v2 as R
from tm4062_re_data import POLARS, STATIONS, NATURAL_TRANSITION, check
from bubble_stations import stations as bl_stations, chi_front

B = os.environ.get('SAAI_CFD_ROOT', '/home/qiqi/flexcompute/sa-ai/flow360_fv1')
FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figs')

# alpha -> case-name token, for the two sweep Reynolds numbers and the main one
TAG_MAIN = {-4: 'am4', -2: 'am2', -1: 'am1', 0: 'a0', 1: 'a1', 2: 'a2', 3: 'a3',
            4: 'a4x', 5: 'a5', 6: 'a6', 7: 'a7', 8: 'a8', 8.5: 'a8p5', 9: 'a9'}
TAG_SWEEP = {-2: 'am2', 0: 'a0', 1: 'a1', 2: 'a2', 3: 'a3', 4: 'a4',
             5: 'a5', 6: 'a6', 8: 'a8', 10: 'a10'}

# (Re, label, colour, alphas, case-name builder)
CASES = [
    (1.0e5, r'$10^5$', 'C0', (-2, 0, 2, 4, 6, 8, 10),
     lambda fam, a: f'sweep_{fam}L2_Re100k_{TAG_SWEEP[a]}'),
    (2.0e5, r'$2\times10^5$', 'C1', (-4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8.5, 9),
     lambda fam, a: f'{fam}L2prop_eppler387_Re200k_{TAG_MAIN[a]}'),
    (3.0e5, r'$3\times10^5$', 'C2', (-2, 0, 1, 2, 3, 4, 6, 8, 10),
     lambda fam, a: f'sweep_{fam}L2_Re300k_{TAG_SWEEP[a]}'),
]


def collect():
    """Computed forces and stations for every (Re, family, alpha) present."""
    out = {}
    for Re, _, _, alphas, build in CASES:
        for fam in ('str', 'cav'):
            rows = []
            for a in alphas:
                d = os.path.join(B, build(fam, a))
                f = R.converged_clcd(d)
                if f is None:
                    print(f'  MISSING forces {os.path.basename(d)}', flush=True)
                    continue
                ls = tr = None
                state = 'unknown'
                if os.path.exists(f'{d}/slice_centerSpan.pvtu'):
                    try:
                        (xu, cfu, _), _ = R.airfoil_walk_contour(d)
                        state, ls, tr = bl_stations(xu, cfu)
                        if state == 'attached':
                            # No bubble: the transition front IS the station,
                            # so separation and reattachment collapse onto it.
                            ls = tr = chi_front(R, d)
                    except Exception as e:
                        print(f'  station read failed {os.path.basename(d)}: {e}',
                              flush=True)
                rows.append(dict(alpha=a, cl=f[0], cd=f[1], ls=ls, tr=tr,
                                 state=state))
            out[(Re, fam)] = rows
            print(f'  {Re:.0e} {fam}: {len(rows)} incidences', flush=True)
    return out


def fig_polar(data):
    # Colour carries the Reynolds number for the measurement as well as the
    # model. Drawing all three experimental curves in black made them
    # mutually indistinguishable, which is the one thing this figure exists
    # to separate. Marker and line style carry the source instead.
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    for Re, lab, col, _, _ in CASES:
        e = POLARS[int(Re)]
        # repeat readings at one incidence: kept, they show the scatter
        ax.plot([r[2] for r in e], [r[1] for r in e], '-o', color=col,
                ms=3.2, lw=1.0, alpha=0.9, zorder=3)
        for fam, ls, mk in (('str', '-', 's'), ('cav', '--', '^')):
            rows = data[(Re, fam)]
            ax.plot([r['cd'] for r in rows], [r['cl'] for r in rows], ls,
                    color=col, marker=mk, ms=5.5, mfc='none', mew=1.3,
                    lw=1.6, zorder=4)
    ax.set_xlabel('$C_d$'); ax.set_ylabel('$C_l$')
    ax.set_xlim(0.0, 0.06); ax.set_ylim(-0.1, 1.45)
    ax.grid(alpha=0.3)
    h = [Line2D([], [], color='0.35', ls='-', marker='o', ms=3.2, lw=1.0,
                label='Experiment (LTPT, TM-4062)'),
         Line2D([], [], color='0.35', ls='-', marker='s', mfc='none', ms=5.5,
                label='SA-AI, structured O-grid (L2)'),
         Line2D([], [], color='0.35', ls='--', marker='^', mfc='none', ms=5.5,
                label='SA-AI, unstructured (L2)')]
    h += [Line2D([], [], color=c, lw=2.4, label=f'$Re=${l}')
          for _, l, c, _, _ in CASES]
    ax.legend(handles=h, fontsize=8.5, loc='lower right', framealpha=0.95)
    out = os.path.join(FIGS, 'eppler_re_polar.pdf')
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print('wrote', out, flush=True)


def fig_bubble(data):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    ax_ls, ax_tr = axs
    for Re, lab, col, _, _ in CASES:
        # Where the oil flow records natural transition and no bubble, the
        # separation and reattachment stations coincide at the transition
        # station, so the measurement enters the curve as an ordinary point
        # at the same x in both panels rather than as a separate symbol.
        st = dict(STATIONS[int(Re)])
        for a, v in NATURAL_TRANSITION[int(Re)].items():
            st[a] = (v, v)
        aa = sorted(st)
        ax_ls.plot([st[a][0] for a in aa], aa, '-o', color=col,
                   ms=3.6, lw=1.0, zorder=3)
        ax_tr.plot([st[a][1] for a in aa], aa, '-o', color=col,
                   ms=3.6, lw=1.0, zorder=3)
        for fam, mk in (('str', 's'), ('cav', '^')):
            rows = data[(Re, fam)]
            for ax, key in ((ax_ls, 'ls'), (ax_tr, 'tr')):
                pts = [(r[key], r['alpha']) for r in rows if r[key] is not None]
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], mk,
                            color=col, mfc='none', ms=5.5, mew=1.2, zorder=4)
    ax_ls.set_xlabel('laminar separation $x_{LS}/c$')
    ax_tr.set_xlabel('turbulent reattachment $x_R/c$')
    ax_ls.set_ylabel(r'$\alpha$ (deg)')
    for ax in axs:
        ax.set_xlim(0, 1.0); ax.grid(alpha=0.3)
    ax_ls.set_ylim(-5.0, 10.5)   # -4 deg runs at Re = 2e5
    h = [Line2D([], [], color='0.35', marker='o', ms=3.6, lw=1.0,
                label='Experiment, oil flow (TM-4062 Table III)'),
         Line2D([], [], color='0.35', marker='s', ls='none', mfc='none', ms=5.5,
                label='SA-AI, structured O-grid (L2)'),
         Line2D([], [], color='0.35', marker='^', ls='none', mfc='none', ms=5.5,
                label='SA-AI, unstructured (L2)')]
    h += [Line2D([], [], color=c, lw=2.4, label=f'$Re=${l}')
          for _, l, c, _, _ in CASES]
    ax_tr.legend(handles=h, fontsize=7.5, loc='lower left', framealpha=0.95)
    out = os.path.join(FIGS, 'eppler_re_bubble.pdf')
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    check(verbose=False)
    d = collect()
    fig_polar(d)
    fig_bubble(d)
