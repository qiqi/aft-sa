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
XLIM, ZLIM = (-0.13, 1.33), (-0.07, 0.15)
# Fraction of the figure width the axes actually occupy once the colorbar
# and labels are taken out. The row height has to be derived from this and
# the data aspect, or set_aspect('equal') pads inside each axes and the
# rows appear gapped however small hspace is.
AX_FRAC = 0.80
# Target aspect so the figure fills the PAGE WIDTH rather than the page
# height. A LaTeX page is about 6.5 x 9 in of text, ratio 0.72; if the
# figure is taller than W/0.72 the height constraint binds and the width
# is left unused. Keeping W/H above ~0.85 leaves margin.
MIN_W_OVER_H = 0.85
# Resampling grid for contouring.
NGRID_X, NGRID_Z = 1600, 260


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
    H = row_h*n + 0.85
    if W/H < MIN_W_OVER_H:
        print('note: %d rows at z-span %.3f gives W/H = %.2f; the page height '
              'would bind' % (n, ZLIM[1] - ZLIM[0], W/H))
    fig, axs = plt.subplots(n, 1, figsize=(W, H), squeeze=False,
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
        # Incidence label and legend go INSIDE the axes: as a y-label and a
        # figure legend they sat in the margins, which costs page width the
        # contours could be using.
        ax.text(0.008, 0.90, r'$\alpha=%+.0f^\circ$' % alpha_of(case),
                transform=ax.transAxes, fontsize=10, va='top', ha='left')
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
        # Contour on a REGULAR grid, not on the 300k-triangle mesh. tricontour
        # follows mesh edges, so it emits a path segment per crossed triangle:
        # megabytes of vector, and visibly jagged. Rasterizing that is not the
        # fix either -- at 300 dpi the embedded raster came out LARGER than the
        # paths. Resampling first gives both a smaller file and smoother lines.
        f = mtri.LinearTriInterpolator(T, chi)
        gx = np.linspace(XLIM[0], XLIM[1], NGRID_X)
        gz = np.linspace(ZLIM[0], ZLIM[1], NGRID_Z)
        GX, GZ = np.meshgrid(gx, gz)
        G = np.ma.filled(f(GX, GZ), np.nan)
        cs = ax.contour(GX, GZ, G, levels=LEVELS, norm=NORM, cmap=CMAP,
                        linewidths=0.9)
        ax.clabel(cs, fmt='%g', fontsize=6, inline=True)
        ax.contour(GX, GZ, G, levels=[C_V1], colors='k', linewidths=2.0)
        ax.contour(GX, GZ, G, levels=[CHI_INF*np.e**2], colors='0.55',
                   linewidths=0.5, linestyles=':')
    axs[-1, 0].set_xlabel('$x$')
    sm = ScalarMappable(norm=NORM, cmap=CMAP)
    cb = fig.colorbar(sm, ax=axs[:, 0], fraction=0.014, pad=0.01)
    cb.set_label(r'$\chi=\tilde\nu/\nu$  (contour level)')
    axs[0, 0].legend(handles=[Line2D([], [], color='k', lw=2.0,
                                     label=r'$\chi=c_{v1}=7.1$'),
                              Line2D([], [], color='0.55', lw=0.5, ls=':',
                                     label=r'$\chi=\chi_\infty e^2$')],
                     loc='upper right', fontsize=8, ncol=1, framealpha=0.85,
                     borderpad=0.3, handlelength=1.6)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print('wrote %s  (%d rows)' % (out, n))


if __name__ == '__main__':
    main()
