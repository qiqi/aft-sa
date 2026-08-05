"""Picture of the L1 tangential-spacing defect.

Row 1: the contour's own edge length vs local index, against the value the
       metric actually read (global-array lookup with a local edge index).
Row 2: the same two curves mapped onto the geometry, coloured by ratio.
Row 3: realised near-wall triangle size along the surface, upper vs lower.

Usage: python3 plot_spacing_diagnosis.py [case_dir] [out.pdf]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from measure_l1_spacing import read_contours, read_vtk_tris, surface_frame, project

FARF, FORE0, FLAP0 = 360, 360, 1562


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else 'case_L1_fixed'
    out = sys.argv[2] if len(sys.argv) > 2 else 'l1_spacing_diagnosis.pdf'
    hdr, pts, curves = read_contours('%s/contours_L1.txt' % case)
    hwall = hdr['HWALL']
    walls = [n for w, n in curves if w]
    names = ['fore', 'flap']

    def load_mesh(cdir):
        P, T = read_vtk_tris('%s/mesh2d.vtk' % cdir)
        tp = P[T]
        cen = tp.mean(axis=1)
        e0, e1 = tp[:, 1] - tp[:, 0], tp[:, 2] - tp[:, 0]
        area = 0.5*np.abs(e0[:, 0]*e1[:, 1] - e0[:, 1]*e1[:, 0])
        return cen, np.sqrt(4.0*area/np.sqrt(3.0)), len(T)

    frames = {}
    for k, nodes in enumerate(walls):
        p, seg, s, ile, u2 = surface_frame(pts, nodes)
        frames[names[k]] = (p, seg, s, ile, nodes, u2)

    # the reference case, plus an optional comparison case overlaid in row 3
    cmp_dir = sys.argv[3] if len(sys.argv) > 3 else None
    meshes = {}
    for tag, cdir in [('bug', case)] + ([('fix', cmp_dir)] if cmp_dir else []):
        cen, hcell, ncell = load_mesh(cdir)
        dists, segidx = [], []
        for nm in names:
            d, i = project(cen, frames[nm][0])
            dists.append(d); segidx.append(i)
        dists, segidx = np.array(dists), np.array(segidx)
        which = np.argmin(dists, axis=0)
        ar = np.arange(len(cen))
        meshes[tag] = (hcell, which, dists[which, ar], segidx[which, ar],
                       ncell, cdir)
    hcell, which, dmin, imin, ncell, _ = meshes['bug']

    fig, axes = plt.subplots(3, 2, figsize=(13.5, 11.5))
    for k, nm in enumerate(names):
        p, seg, s, ile, nodes, u2 = frames[nm]
        n = len(seg)
        gi = np.arange(n)
        got = np.linalg.norm(pts[gi+1] - pts[gi], axis=1)
        eff_got = np.minimum(got, hwall)
        eff_want = np.minimum(seg, hwall)
        # NOTE contour runs lower TE -> LE -> upper TE, so idx > ile is UPPER
        LOW, UPP = slice(0, ile), slice(ile, n)

        # ---- row 1: spacing vs index
        ax = axes[0, k]
        ax.semilogy(gi, eff_want, lw=1.0, color='#1f77b4',
                    label=r'intended  $\min(h_{\rm local},h_{\rm wall})$')
        ax.semilogy(gi, eff_got, lw=1.0, color='#d62728',
                    label=r'as evaluated (global lookup)')
        ax.axvline(ile, color='k', ls=':', lw=1.0)
        ax.axvline(FARF - 1, color='gray', ls='--', lw=1.0)
        ax.text(ile, ax.get_ylim()[1], ' LE', va='top', fontsize=8)
        ax.text(FARF - 1, ax.get_ylim()[0], ' farfield lookup ends',
                va='bottom', fontsize=8, color='gray')
        ax.text(0.02, 0.04, 'LOWER surface', transform=ax.transAxes,
                fontsize=8, color='#555')
        ax.text(0.98, 0.04, 'UPPER surface', transform=ax.transAxes,
                fontsize=8, color='#555', ha='right')
        ax.set_title('%s: tangential spacing source (local edge index)' % nm)
        ax.set_xlabel('local edge index  (lower TE $\\to$ LE $\\to$ upper TE)')
        ax.set_ylabel('spacing')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=0.3, which='both')

        # ---- row 2: ratio on the geometry
        ax = axes[1, k]
        r = eff_got/eff_want
        segs = np.stack([p[:-1], p[1:]], axis=1)
        lc = LineCollection(segs, cmap='coolwarm',
                            norm=matplotlib.colors.LogNorm(vmin=0.05, vmax=20.0),
                            linewidths=3.0)
        lc.set_array(r)
        ax.add_collection(lc)
        for kk, nn in enumerate(walls):
            q = pts[nn]
            ax.plot(q[:, 0], q[:, 1], color='0.75', lw=0.5, zorder=0)
        ax.set_aspect('equal')
        ax.autoscale_view()
        ax.set_title('%s: evaluated / intended spacing' % nm)
        plt.colorbar(lc, ax=ax, fraction=0.03, label='ratio (<1 = too fine)')

        # ---- row 3: realised near-wall cell size along the surface
        ax = axes[2, k]
        edges = np.linspace(0, s[-1], 61)
        mid = 0.5*(edges[:-1] + edges[1:])
        axt = ax.twinx()
        style = {'bug': ('#d62728', '-', 'buggy index'),
                 'fix': ('#2ca02c', '-', 'index fixed')}
        for tag, (hc, wh, dm, im, nc, cd) in meshes.items():
            col, ls, lbl = style[tag]
            sel = (wh == k) & (dm < 5e-3)
            sc = s[im][sel]
            hs = hc[sel]
            med = np.full(60, np.nan); cnt = np.zeros(60)
            for j in range(60):
                m = (sc >= edges[j]) & (sc < edges[j+1])
                cnt[j] = m.sum()
                if m.sum():
                    med[j] = np.median(hs[m])
            ax.semilogy(mid, med, color=col, ls=ls, lw=1.4,
                        label='%s (%d cells)' % (lbl, nc))
            axt.plot(mid, cnt, color=col, lw=0.9, alpha=0.35)
        ax.semilogy(s[:-1], eff_want, color='#1f77b4', lw=1.0, alpha=0.8,
                    label='intended')
        ax.axvline(s[ile], color='k', ls=':', lw=1.0)
        ax.set_xlabel('arc length from lower TE  (LE dotted; UPPER surface to the right)')
        ax.set_ylabel('near-wall cell size')
        ax.set_title('%s: realised near-wall size ($d<0.005$)' % nm)
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(alpha=0.3, which='both')
        axt.set_ylabel('cells per bin (faint)', fontsize=8)

    fig.suptitle('L1 mesh: upper-surface over-refinement traced to a local '
                 'edge index used as a global point index', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out)
    print('wrote', out)


if __name__ == '__main__':
    main()
