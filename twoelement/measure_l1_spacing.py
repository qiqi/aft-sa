"""Measure the L1 mesh's near-wall spacing, upper surface vs lower surface, and
compare against what the metric in mesh2d.cpp actually evaluates.

Two things are computed independently:

  1. GEOMETRY / METRIC side. From contours.txt (the exact file the mesher read),
     reproduce `hLocal` the way SpaldingMetric does it --

         a = w->points[info.minEdgeIdx];  b = w->points[info.minEdgeIdx + 1];

     -- where `points` is the GLOBAL point array and `minEdgeIdx` is the index
     the KD-tree returns.  Compare it against the local edge length at the same
     place, i.e. what the code intended.

  2. MESH side. From mesh2d.vtk, the realised triangle size as a function of
     arc position along each element, split into upper and lower surface, so the
     asymmetry can be quantified rather than eyeballed off a picture.

Usage: python3 measure_l1_spacing.py [case_dir]
"""
import sys

import numpy as np
from scipy.spatial import cKDTree


def read_contours(path):
    tok = open(path).read().split()
    i = 0
    hdr = {}
    pts = None
    curves = []
    while i < len(tok):
        t = tok[i]
        if t in ('H0', 'GROWTH', 'HWALL', 'HMAX'):
            hdr[t] = float(tok[i+1]); i += 2
        elif t == 'NPOINTS':
            n = int(tok[i+1]); i += 2
            pts = np.array(tok[i:i+2*n], dtype=float).reshape(n, 2)
            i += 2*n
        elif t == 'NCURVES':
            m = int(tok[i+1]); i += 2
            for _ in range(m):
                isw, cnt = int(tok[i]), int(tok[i+1]); i += 2
                nodes = np.array(tok[i:i+cnt], dtype=np.int64); i += cnt
                curves.append((isw, nodes))
        elif t == 'NBANDS':
            nb = int(tok[i+1]); i += 2
            for _ in range(nb):
                hdr.setdefault('bands', []).append(
                    (int(tok[i]), float(tok[i+1]), float(tok[i+2]), int(tok[i+3])))
                i += 4
        else:
            i += 1
    return hdr, pts, curves


def read_vtk_tris(path):
    with open(path) as fh:
        txt = fh.read()
    tok = txt.split()
    i = tok.index('POINTS')
    npt = int(tok[i+1])
    P = np.array(tok[i+3:i+3+3*npt], dtype=float).reshape(npt, 3)[:, :2]
    j = tok.index('CELLS')
    ncell = int(tok[j+1])
    blk = np.array(tok[j+3:j+3+4*ncell], dtype=np.int64).reshape(ncell, 4)
    return P, blk[:, 1:]


def surface_frame(pts, nodes):
    """Per-element arc coordinate, edge lengths, and the LE index.

    Orientation is detected, not assumed: contours.txt comes out of the pipeline
    ordered lower TE -> LE -> upper TE (the reverse of what contour_te.blunt_contour
    returns), so `idx > ile` is the UPPER surface here.
    """
    p = pts[nodes]                       # nodes closes the loop (last == first)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    ile = int(np.argmin(p[:, 0]))        # global-frame LE (min x)
    upper_is_second = p[ile:, 1].mean() > p[:ile, 1].mean()
    return p, seg, s, ile, upper_is_second


def project(centroids, p):
    """Nearest point on the polyline `p` for each centroid: distance, segment
    index, and the along-segment parameter."""
    a, b = p[:-1], p[1:]
    mid = 0.5*(a + b)
    tree = cKDTree(mid)
    K = 24
    _, cand = tree.query(centroids, k=min(K, len(mid)))
    e = b - a
    L2 = np.maximum((e*e).sum(1), 1e-300)
    best_d = np.full(len(centroids), np.inf)
    best_i = np.zeros(len(centroids), dtype=np.int64)
    for c in range(cand.shape[1]):
        idx = cand[:, c]
        w = centroids - a[idx]
        t = np.clip((w*e[idx]).sum(1)/L2[idx], 0.0, 1.0)
        proj = a[idx] + t[:, None]*e[idx]
        d = np.linalg.norm(centroids - proj, axis=1)
        m = d < best_d
        best_d[m], best_i[m] = d[m], idx[m]
    return best_d, best_i


def band_stats(name, mask, h, tag):
    if mask.sum() == 0:
        print('  %-28s   (none)' % name)
        return
    v = h[mask]
    print('  %-28s n=%7d   h: med %8.2e  p10 %8.2e  p90 %8.2e  min %8.2e'
          % (name, mask.sum(), np.median(v), np.percentile(v, 10),
             np.percentile(v, 90), v.min()))


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else 'case_L1_fixed'
    hdr, pts, curves = read_contours('%s/contours_L1.txt' % case)
    hwall, growth, hmax = hdr['HWALL'], hdr['GROWTH'], hdr['HMAX']
    walls = [(i, n) for i, (w, n) in enumerate(curves) if w]
    names = ['fore', 'flap']

    print('contours: %d points, %d curves (%d walls);  hwall=%.4g growth=%.3g '
          'hmax=%.3g' % (len(pts), len(curves), len(walls), hwall, growth, hmax))
    for ci, (w, n) in enumerate(curves):
        print('  curve %d  isWall=%d  nodes %d..%d (%d)'
              % (ci, w, n[0], n[-2], len(n)))

    # ---------------------------------------------------------------- part 1
    print('\n=== hLocal as the metric evaluates it, vs as intended ===')
    frames = {}
    for k, (ci, nodes) in enumerate(walls):
        p, seg, s, ile, u2 = surface_frame(pts, nodes)
        frames[names[k]] = (p, seg, s, ile, nodes, u2)
        nseg = len(seg)
        # what the code does: global-array lookup with a LOCAL edge index
        gi = np.arange(nseg)
        got = np.linalg.norm(pts[gi+1] - pts[gi], axis=1)
        eff_got = np.minimum(got, hwall)
        eff_want = np.minimum(seg, hwall)
        print('\n%s (wall %d, %d segments, LE at local index %d, arc %.4f/%.4f)'
              % (names[k], k, nseg, ile, s[ile], s[-1]))
        # which source region each lookup lands in
        # absolute global-array boundaries, not this curve's own base
        f0, a0 = walls[0][1][0], walls[1][1][0]
        src = np.where(gi + 1 < f0, 'farfield',
                       np.where(gi + 1 < a0, 'fore', 'flap'))
        for tag in ('farfield', 'fore', 'flap'):
            m = src == tag
            if m.sum():
                print('   lookup lands in %-9s for %5d of %d segments '
                      '(local idx %d..%d)'
                      % (tag, m.sum(), nseg, gi[m].min(), gi[m].max()))
        a, b = ('lower', 'upper') if u2 else ('upper', 'lower')
        for lbl, lo, hi in (('%s (TE->LE)' % a, 0, ile),
                            ('%s (LE->TE)' % b, ile, nseg)):
            sl = slice(lo, hi)
            print('   %-15s intended h: med %8.2e min %8.2e max %8.2e '
                  '| effective ht(wall): med %8.2e min %8.2e'
                  % (lbl, np.median(seg[sl]), seg[sl].min(), seg[sl].max(),
                     np.median(eff_got[sl]), eff_got[sl].min()))
            print('   %-15s ratio effective/intended: med %6.2f  '
                  'frac finer than intended %5.1f%%  frac at hwall %5.1f%%'
                  % ('', np.median(eff_got[sl]/eff_want[sl]),
                     100*np.mean(eff_got[sl] < 0.5*eff_want[sl]),
                     100*np.mean(eff_got[sl] >= hwall - 1e-15)))

    # ---------------------------------------------------------------- part 2
    print('\n=== realised mesh ===')
    P, T = read_vtk_tris('%s/mesh2d.vtk' % case)
    tp = P[T]
    cen = tp.mean(axis=1)
    e0, e1 = tp[:, 1] - tp[:, 0], tp[:, 2] - tp[:, 0]
    area = 0.5*np.abs(e0[:, 0]*e1[:, 1] - e0[:, 1]*e1[:, 0])
    h = np.sqrt(4.0*area/np.sqrt(3.0))    # equilateral-equivalent edge
    print('%d triangles, %d points; h: med %.3e  min %.3e  max %.3e'
          % (len(T), len(P), np.median(h), h.min(), h.max()))

    dists, segidx, which = [], [], []
    for nm in names:
        p = frames[nm][0]
        d, i = project(cen, p)
        dists.append(d); segidx.append(i)
    dists = np.array(dists); segidx = np.array(segidx)
    which = np.argmin(dists, axis=0)
    dmin = dists[which, np.arange(len(cen))]
    imin = segidx[which, np.arange(len(cen))]

    for k, nm in enumerate(names):
        p, seg, s, ile, nodes, u2 = frames[nm]
        sel = which == k
        print('\n%s: %d cells nearest' % (nm, sel.sum()))
        first = sel & (imin < ile)
        second = sel & (imin >= ile)
        arc1, arc2 = s[ile], s[-1] - s[ile]
        sides = ([('lower', first, arc1), ('upper', second, arc2)] if u2 else
                 [('upper', first, arc1), ('lower', second, arc2)])
        dens = {}
        for lbl, side, arc in sides:
            for dlo, dhi in ((0, 2e-3), (2e-3, 1e-2), (1e-2, 6e-2), (6e-2, 0.5)):
                m = side & (dmin >= dlo) & (dmin < dhi)
                band_stats('%s  d in [%.0e,%.0e)' % (lbl, dlo, dhi), m, h, nm)
            dens[lbl] = (side.sum(), arc, side.sum()/arc)
            print('   %s: %d cells over arc %.4f  ->  %.0f cells per unit arc'
                  % (lbl, side.sum(), arc, side.sum()/arc))
        print('   UPPER/LOWER cells = %.2f   density ratio = %.2f'
              % (dens['upper'][0]/max(dens['lower'][0], 1),
                 dens['upper'][2]/dens['lower'][2]))

    # near-wall profile vs arc, so the fine patches are locatable
    print('\n=== near-wall cell size vs arc position (d < 0.01) ===')
    for k, nm in enumerate(names):
        p, seg, s, ile, nodes, u2 = frames[nm]
        sel = (which == k) & (dmin < 1e-2)
        print('\n%s   (LE at arc %.4f, %s surface first)'
              % (nm, s[ile], 'lower' if u2 else 'upper'))
        print('   %-14s %8s %10s %10s %10s' % ('arc band', 'cells', 'h med',
                                               'h min', 'intended h'))
        edges = np.linspace(0.0, s[-1], 21)
        sc = s[imin]
        for j in range(20):
            m = sel & (sc >= edges[j]) & (sc < edges[j+1])
            ms = (s[:-1] >= edges[j]) & (s[:-1] < edges[j+1])
            below = edges[j] < s[ile]
            side = ('L' if below else 'U') if u2 else ('U' if below else 'L')
            print('   %s %5.3f-%5.3f %8d %10.2e %10.2e %10.2e'
                  % (side, edges[j], edges[j+1], m.sum(),
                     np.median(h[m]) if m.sum() else np.nan,
                     h[m].min() if m.sum() else np.nan,
                     np.median(seg[ms]) if ms.sum() else np.nan))


if __name__ == '__main__':
    main()
