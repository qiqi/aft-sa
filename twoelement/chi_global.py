"""Global chi = nuHat/nu field as LINE contours, one row per angle of attack,
one PDF per refinement level.

Line contours rather than filled: with a filled map the eye reads the bright
region as "turbulent" and the dark as "laminar", which hides where the c_v1
crossing actually sits. Lines put the level values on the picture, so the
amplified fluid can be traced back to the surface it grew on.

Levels are decades from 1e-3 to 1e2, plus c_v1 = 7.1 drawn heavy in black --
c_v1 is where the SA-AI variable stops being a passive seed and starts behaving
like eddy viscosity, so that contour is the meaningful boundary -- and a faint
chi_inf e^2 line marking where the seed has begun to grow at all.

Run:  python3 chi_global.py out.pdf case_dir [case_dir ...]
      Rows are ordered by the alpha parsed from each case directory name.
"""
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

import plot_solution_mesh as PM
import plot_paper_style as PS
from measure_l1_spacing import read_contours

C_V1 = 7.1
CHI_INF = C_V1*np.exp(-9.0)
LEVELS = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
NORM = LogNorm(1e-4, 1e2)
CMAP = plt.get_cmap('magma')
XLIM, ZLIM = (-0.13, 1.33), (-0.09, 0.20)
# Fraction of the figure width the axes actually occupy once the colorbar
# and labels are taken out. The row height has to be derived from this and
# the data aspect, or set_aspect('equal') pads inside each axes and the
# rows appear gapped however small hspace is.
AX_FRAC = 0.80


def alpha_of(case):
    m = re.search(r'_a([+-][\d.]+)', case)
    return float(m.group(1)) if m else -1.0


def main():
    out = sys.argv[1]
    cases = sorted(sys.argv[2:], key=alpha_of)
    n = len(cases)
    # No gaps and no frames: the rows are one continuous field, and internal
    # borders just eat vertical space and imply the panels are separate plots.
    W = 11.2
    row_h = AX_FRAC*W*(ZLIM[1] - ZLIM[0])/(XLIM[1] - XLIM[0])
    fig, axs = plt.subplots(n, 1, figsize=(W, row_h*n + 0.85), squeeze=False,
                            sharex=True, sharey=True,
                            gridspec_kw=dict(hspace=0.0))
    for r, case in enumerate(cases):
        ax = axs[r, 0]
        hdr, cpts, curves = read_contours('%s/contours_L1.txt' % case)
        for w in [nn for ww, nn in curves if ww]:
            q = cpts[w]
            ax.fill(q[:, 0], q[:, 1], color='0.88', zorder=2)
            ax.plot(q[:, 0], q[:, 1], '-', color='0.25', lw=1.0, zorder=3)
        ax.set_xlim(*XLIM)
        ax.set_ylim(*ZLIM)
        ax.set_aspect('equal')
        ax.set_ylabel(r'$\alpha=%+.0f^\circ$' % alpha_of(case), fontsize=10)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        if r != n - 1:
            ax.spines['bottom'].set_visible(False)
        ax.set_yticks([])
        ax.tick_params(axis='x', which='both', length=0 if r != n - 1 else 3)

        g, pts, arr = PM.read_vtu('%s/volume_proc0.vtu' % case)
        idx, tris, P2 = PM.midplane_tris(g, pts)
        if 'nuHat' not in arr:
            ax.text(0.5, 0.90, '%s: no nuHat -- rerun with the extended '
                    'VOLUME_FIELDS' % case, transform=ax.transAxes,
                    ha='center', va='top', fontsize=8, color='crimson')
            continue
        nu = PS.nu_of(case)
        # nuHat is exactly 0 at the walls, and tricontour with a LogNorm
        # refuses non-positive data outright, so floor it below the
        # lowest contour level rather than let it fail.
        chi = np.clip(arr['nuHat'][idx]/nu, 1e-6, None)
        T = mtri.Triangulation(P2[:, 0], P2[:, 1], tris)
        cs = ax.tricontour(T, chi, levels=LEVELS, norm=NORM, cmap=CMAP,
                           linewidths=0.9)
        ax.clabel(cs, fmt='%g', fontsize=6, inline=True)
        ax.tricontour(T, chi, levels=[C_V1], colors='k', linewidths=2.0)
        ax.tricontour(T, chi, levels=[CHI_INF*np.e**2], colors='0.55',
                      linewidths=0.5, linestyles=':')
    axs[-1, 0].set_xlabel('$x$')
    sm = ScalarMappable(norm=NORM, cmap=CMAP)
    cb = fig.colorbar(sm, ax=axs[:, 0], fraction=0.014, pad=0.01)
    cb.set_label(r'$\chi=\tilde\nu/\nu$  (contour level)')
    fig.legend(handles=[Line2D([], [], color='k', lw=2.0,
                               label=r'$\chi=c_{v1}=7.1$'),
                        Line2D([], [], color='0.55', lw=0.5, ls=':',
                               label=r'$\chi=\chi_\infty e^2$')],
               loc='upper right', fontsize=8, ncol=2)
    fig.suptitle('SA-AI amplification variable $\\chi$, mid-plane', fontsize=10)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('wrote %s  (%d rows)' % (out, n))


if __name__ == '__main__':
    main()
