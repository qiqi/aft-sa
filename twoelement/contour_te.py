"""Blunt-TE contour generation, following the recipe used for the paper's
airfoil meshes (flow360/eppler_contour_te.py, nlf_contour_te.py).

Two things my earlier cosine contour was missing:

  1. EXPLICIT BLUNT-FACE POINTS. Opening the trailing edge to a finite base is
     not enough -- the base needs its own edges, or the two prism stacks meet
     across a single segment. The recipe uses
         n_per_side = round(half_base / h_te)
     so the face refines with the level exactly as the surface does.

  2. TE-ANCHORED EXPONENTIAL GRADING. Arc-length spacing starts at h_te at the
     trailing-edge corner and grows at r_te, then blends by a QUARTER SINE into
     leading-edge clustering. The quarter-sine is what avoids the "double
     growth pattern" a piecewise full cosine produces (see the note in
     eppler_contour_te._build_half).

Ladder values mirror the campaign: h_te halves per level, (r_te - 1) halves per
level.
"""
import numpy as np
from scipy.interpolate import CubicSpline

import panel2e as M


def _build_half(n_half, h_te, r_te, dxds_0):
    """n_half arc-length values in [0, 0.5]: geometric from the TE, then a
    quarter-sine blend into LE clustering."""
    s = [0.0]
    step = h_te/abs(dxds_0)
    while len(s) < n_half:
        nxt = s[-1] + step
        if nxt >= 0.5:
            break
        if step > (0.5 - s[-1])/max(n_half - len(s), 1):
            break
        s.append(nxt)
        step *= r_te
    anchor = s[-1]
    k = n_half - len(s)
    if k > 0:
        i = np.arange(1, k + 1)
        s.extend((anchor + (0.5 - anchor)*np.sin(0.5*np.pi*i/k)).tolist())
    return np.array(s[:n_half])


def blunt_contour(n_surf, h_te, r_te, te_thick, chord, inc, x_le, z_le,
                  m, p, t, modified=False, xm=0.30, ik=6.0, km=1.1,
                  le_blend=0.15, dense=6000):
    """Closed contour: upper TE corner -> LE -> lower TE corner -> blunt face.

    n_surf   points on the airfoil surface (excludes the blunt-face points)
    h_te     arc-length step at the TE corner, in chord units of THIS element
    te_thick full base thickness, in chord units of THIS element
    """
    nd = M.airfoil_nodes(dense, m, p, t, modified=modified, xm=xm, ik=ik,
                         km=km, le_blend=le_blend, te_thick=te_thick)
    # nd runs lower TE -> LE -> upper TE; reverse to upper TE -> LE -> lower TE
    nd = nd[::-1]
    d = np.hypot(np.diff(nd[:, 0]), np.diff(nd[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(d)])
    s /= s[-1]
    spx, spz = CubicSpline(s, nd[:, 0]), CubicSpline(s, nd[:, 1])
    eps = 1e-5
    dxds0 = (spx(eps) - spx(0.0))/eps

    n_half = n_surf//2 + 1
    su = _build_half(n_half, h_te, r_te, dxds0)
    sl = 1.0 - su[::-1]
    sa = np.concatenate([su, sl[1:]])
    if len(sa) > n_surf:
        sa = sa[:n_surf]
    xa, za = spx(sa), spz(sa)

    half = 0.5*te_thick
    n_side = max(1, int(round(half/h_te)))
    zl = np.linspace(za[-1], 0.0, n_side + 1)[1:-1]
    zu = np.linspace(0.0, za[0], n_side + 1)[1:-1]
    zb = np.concatenate([zl, [0.0], zu])
    xb = np.full_like(zb, 1.0)

    xx = np.concatenate([xa, xb])
    zz = np.concatenate([za, zb])
    pts = np.column_stack([xx, zz])
    return M.place(pts, chord, inc, x_le, z_le), 2*n_side


if __name__ == '__main__':
    import json
    c = json.load(open('twoelement_adopted.json'))
    F, FL = c['fore'], c['flap']
    TE = 0.003
    print('%5s %6s %8s %6s | %7s %7s %9s %8s'
          % ('lvl', 'N', 'h_te', 'r_te', 'pts', 'face', 'ang_LE', 'ang_max'))
    for tag, N, hte_f, rte in (('L0', 600, 2.0e-3, 2.00),
                               ('L1', 1200, 1.0e-3, 1.50),
                               ('L2', 2400, 5.0e-4, 1.25)):
        nd, nface = blunt_contour(N, hte_f, rte, TE/c['chord'], c['chord'],
                                  F['inc'], 0.0, 0.0, F['m'], F['p'], F['t'],
                                  modified=True, xm=F['xm'], km=F['km'])
        d = np.diff(nd, axis=0)
        n = np.linalg.norm(d, axis=1)
        u = d/np.maximum(n, 1e-30)[:, None]
        ang = np.degrees(np.arccos(np.clip((u[:-1]*u[1:]).sum(1), -1, 1)))
        i = int(np.argmin(nd[:, 0]))
        w = np.arange(max(i-6, 0), min(i+7, len(ang)))
        print('%5s %6d %8.1e %6.2f | %7d %7d %9.2f %8.2f'
              % (tag, N, hte_f, rte, len(nd), nface, ang[w].max(), ang.max()))
