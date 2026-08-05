"""Drag polar for the two-element ladder.

Forces are parsed out of the run logs rather than re-derived, and the step count
is carried through with them: a case that stopped at its ceiling instead of at
the 1e-9 momentum tolerance is NOT converged, and is drawn hollow so an
unconverged point cannot be mistaken for a converged one.

CL against CD on a LINEAR drag axis clipped to [0, 0.025]: the post-stall
points (CD ~ 0.08 at +7, ~0.14 at -9) are an order of magnitude outside the
bucket and would flatten it, so they are allowed off-plot and their incidence is
annotated at the clip edge instead.

Run:  python3 drag_polar.py out.pdf log [log ...]
"""
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import plot_paper_style as PS

# Per-level step ceiling with STEP_SCALE = 4, from run_ladder_v2.LEVELS.
CEIL = {'L0': 32000, 'L1': 48000, 'L2': 64000}
COL = {'L0': '#c44e52', 'L1': '#1f4e9c', 'L2': '#2ca02c'}


def parse_logs(paths):
    """-> {level: [(alpha, CL, CD, step, converged), ...]}"""
    out = {}
    for p in paths:
        try:
            txt = open(p).read()
        except OSError:
            print('missing %s' % p)
            continue
        cur = None
        for line in txt.splitlines():
            m = re.match(r'#+\s*(L\d)\s+alpha=([+-]?[\d.]+)', line.strip())
            if m:
                cur = (m.group(1), float(m.group(2)))
                continue
            if cur and 'forces:' in line:
                def g(pat):
                    r = re.search(pat, line)
                    return float(r.group(1)) if r else np.nan
                step = g(r"'step': ([0-9]+)")
                cl, cd = g(r"'CL': ([-0-9.e+]+)"), g(r"'CD': ([-0-9.e+]+)")
                lvl, al = cur
                # step is the LAST written step; converged means it stopped
                # short of the ceiling
                conv = np.isfinite(step) and step < CEIL.get(lvl, 1e9) - 5
                out.setdefault(lvl, []).append((al, cl, cd, step, conv))
                cur = None
    for lvl in out:
        out[lvl] = sorted(out[lvl])
    return out


def main():
    out = sys.argv[1]
    data = parse_logs(sys.argv[2:])
    print('%-4s %-7s %9s %10s %8s %s' % ('lvl', 'alpha', 'CL', 'CD', 'step',
                                         'converged'))
    for lvl in sorted(data):
        for al, cl, cd, st, conv in data[lvl]:
            print('%-4s %+7.1f %9.4f %10.5f %8.0f %s'
                  % (lvl, al, cl, cd, st, 'yes' if conv else 'NO'))

    CD_MAX = 0.025
    fig, axp = plt.subplots(1, 1, figsize=(6.4, 5.2))
    for lvl in sorted(data):
        r = np.array([(a, cl, cd, s, c) for a, cl, cd, s, c in data[lvl]],
                     dtype=float)
        al, cl, cd, conv = r[:, 0], r[:, 1], r[:, 2], r[:, 4].astype(bool)
        c = COL.get(lvl, 'k')
        lw = PS.LEVEL_LW.get(lvl, 1.6)
        axp.plot(cd, cl, '-', color=c, lw=lw, label=lvl, zorder=2)
        axp.plot(cd[conv], cl[conv], 'o', color=c, ms=5, zorder=3)
        axp.plot(cd[~conv], cl[~conv], 'o', mfc='none', mec=c, ms=7, mew=1.4,
                 zorder=3)
        inside = cd <= CD_MAX
        for a_, x_, y_ in zip(al[inside], cd[inside], cl[inside]):
            axp.annotate('%+.0f' % a_, (x_, y_), fontsize=7,
                         textcoords='offset points', xytext=(4, 3), color=c)
        # off-plot points: mark the incidence at the clip edge so a missing
        # point is not read as a missing solution
        for a_, y_ in zip(al[~inside], cl[~inside]):
            axp.annotate(r'%+.0f$\rightarrow$' % a_, (CD_MAX, y_), fontsize=7,
                         ha='right', va='center', color=c,
                         textcoords='offset points', xytext=(-2, 0))
    axp.set_xlim(0.0, CD_MAX)
    axp.set_xlabel('$C_D$')
    axp.set_ylabel('$C_L$')
    axp.axhline(0, color='0.6', lw=.8)
    axp.grid(alpha=.3)
    h = [Line2D([], [], color=COL[l], lw=PS.LEVEL_LW[l], marker='o', ms=4,
                label=l) for l in sorted(data)]
    h += [Line2D([], [], color='0.3', marker='o', mfc='none', ls='none', ms=7,
                 label='not converged')]
    axp.legend(handles=h, fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print('\nwrote', out)


if __name__ == '__main__':
    main()
