"""Surface contour generation for the adopted geometry, replacing
contour_te.blunt_contour.

Two reasons it had to be rewritten:

  1. The fore element is no longer a NACA section. It is a CST camber line plus
     a CST thickness distribution, so the old generator -- which calls
     panel2e.airfoil_nodes with NACA digits -- cannot describe it.

  2. contour_te._build_half has a defect that the metric fix now exposes
     directly to the mesher. It distributes arc length symmetrically about
     normalised s = 0.5 and blends into the leading edge with a quarter sine
     whose derivative vanishes there, so

       - the last step collapses as 1/k^2: the measured minimum segment was
         2.44e-6, 537x below the median, a degenerate point rather than
         resolution, and it is what produced the 3.7e-6 cells in the L1 mesh;
       - the cluster sits at arc fraction 0.5, but the geometric leading edge
         does not. On the flap (NACA 9416, heavily cambered) the true leading
         edge is at 0.479 -- the finest points landed 109 nodes away from it,
         on the lower surface, while the leading edge itself got 2.34e-4.

Both are avoided here by prescribing a SPACING FUNCTION and integrating it,
rather than composing analytic segments:

    h(s) = min( h_te * g^(d_te/h_te),  h_le * g^(d_le/h_le),  h_mid )

with d_te and d_le the arc distances to the nearest trailing-edge corner and to
the TRUE leading edge (located on the dense curve, not assumed). Nodes are then
placed at equal increments of N(s) = int ds/h(s). Because h is bounded below by
min(h_te, h_le) everywhere, a degenerate segment cannot occur, and because the
leading-edge anchor is the real one, the clustering lands where the curvature is.

The blunt trailing-edge face keeps the recipe that worked: explicit face points,
n_per_side = round(half_base / h_te), so the face refines with the level.
"""
import numpy as np
from scipy.interpolate import CubicSpline


def _dense_arc(pts, n=8000):
    """Resample a closed contour densely and return (points, arc, total)."""
    d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    spx, spz = CubicSpline(s, pts[:, 0]), CubicSpline(s, pts[:, 1])
    sd = np.linspace(0.0, total, n)
    return np.column_stack([spx(sd), spz(sd)]), sd, total, spx, spz


def redistribute(pts, n_surf, h_te, r_te, h_le=None, grade=0.25,
                 theta_max=np.radians(2.5), dense=8000):
    """Redistribute a closed contour (TE corner -> LE -> TE corner) onto n_surf
    points using the spacing function described above.

    pts    dense contour, first and last points are the two TE corners
    h_te   arc spacing at the trailing-edge corners, same units as the contour
    r_te   retained for interface compatibility; the gradation is set by `grade`
    h_le   arc spacing at the leading edge (defaults to h_te). Choose it from
           the leading-edge radius and the turning angle you will tolerate:
           theta ~ h_le / R_LE, so 3 deg on R_LE = 0.005 wants h_le = 2.6e-4.
    grade  d(spacing)/d(arc) away from an anchor
    theta_max  maximum turning angle per segment, enforced as h <= theta_max*R
    """
    P, sd, total, spx, spz = _dense_arc(pts, dense)
    h_le = h_te if h_le is None else h_le

    # TRUE leading edge: minimum x on the dense curve, refined by a parabola
    i = int(np.argmin(P[:, 0]))
    i = min(max(i, 1), len(P) - 2)
    x0, x1, x2 = P[i-1, 0], P[i, 0], P[i+1, 0]
    den = (x0 - 2.0*x1 + x2)
    frac = 0.5*(x0 - x2)/den if abs(den) > 1e-30 else 0.0
    s_le = sd[i] + np.clip(frac, -1.0, 1.0)*(sd[1] - sd[0])

    # Spacing grows LINEARLY with distance from an anchor, h = h0 + G*d, which
    # is what a geometric progression of ratio r actually gives: after k cells
    # the distance is h0(r^k - 1)/(r - 1), so h = h0 + (r-1)*d. Writing it as
    # h0*r^(d/h0) instead -- the obvious-looking form -- grows by r every SINGLE
    # h0 of arc, so it blows past any mid-chord cap within one cell and the
    # anchors stop binding at all: the result was a near-uniform distribution
    # with 27 deg of turning at the leading edge.
    d_te = np.minimum(sd, total - sd)
    d_le = np.abs(sd - s_le)
    grow = np.minimum(h_te + grade*d_te, h_le + grade*d_le)

    # CURVATURE CAP. Anchoring on distance from the leading edge alone gives a
    # cluster whose width is arbitrary: with grade 0.25 it spans only ~4 cells
    # and the turning angle a few nodes away was still 8.6 deg. What actually
    # has to be bounded is the turning per segment, theta ~ h/R, so cap the
    # spacing at theta_max * R(s) directly. This clusters by exactly the amount
    # the local curvature demands, at the leading edge and anywhere else, and
    # needs no per-element tuning.
    x1d, z1d = spx(sd, 1), spz(sd, 1)
    x2d, z2d = spx(sd, 2), spz(sd, 2)
    speed = np.maximum(np.hypot(x1d, z1d), 1e-30)
    kappa = np.abs(x1d*z2d - z1d*x2d)/speed**3
    # smooth over a few samples: spline second derivatives are noisy
    w = np.ones(9)/9.0
    kappa = np.convolve(np.pad(kappa, 4, mode='edge'), w, mode='valid')
    R = 1.0/np.maximum(kappa, 1e-12)
    grow = np.minimum(grow, np.maximum(theta_max*R, min(h_te, h_le)))

    def count(h_mid):
        inv = 1.0/np.minimum(grow, h_mid)
        return np.trapezoid(inv, sd)

    # h_mid is NOT free: it is whatever makes the point budget come out right.
    # Capping the growth at the total arc length instead (the obvious thing)
    # leaves the mid-chord spacing effectively unbounded, so the nodes pile up
    # at the trailing and leading edges and a handful of enormous segments carry
    # the rest -- measured median 3.8e-5 against a max of 1.6e-2.
    lo, hi = min(h_te, h_le)*1e-3, total
    if count(hi) > n_surf - 1:
        h_mid = hi                                # even uncapped it is too fine
    else:
        for _ in range(200):                      # bisection: count decreases in h_mid
            mid = np.sqrt(lo*hi)
            if count(mid) > n_surf - 1:
                lo = mid
            else:
                hi = mid
        h_mid = np.sqrt(lo*hi)

    inv = 1.0/np.minimum(grow, h_mid)
    N = np.concatenate([[0.0], np.cumsum(0.5*(inv[1:] + inv[:-1])*np.diff(sd))])
    N *= (n_surf - 1)/N[-1]
    s_nodes = np.interp(np.arange(n_surf), N, sd)
    return np.column_stack([spx(s_nodes), spz(s_nodes)]), s_le, total


def blunt_contour_from_surface(surf, n_surf, h_te, r_te, te_base,
                               h_le=None, grade=0.25,
                               theta_max=np.radians(2.5)):
    """Closed contour with an explicit blunt trailing-edge face.

    surf     dense closed loop, lower TE corner -> LE -> upper TE corner, with
             the trailing edge ALREADY opened to te_base
    returns  (nodes, n_face_edges)
    """
    nd, s_le, total = redistribute(surf, n_surf, h_te, r_te, h_le=h_le,
                                   grade=grade, theta_max=theta_max)
    lo_te, up_te = nd[0], nd[-1]
    n_side = max(1, int(round(0.5*te_base/h_te)))
    t = np.linspace(0.0, 1.0, 2*n_side + 1)[1:-1]
    face = lo_te[None, :] + t[:, None]*(up_te - lo_te)[None, :]
    # loop: lower TE -> LE -> upper TE -> down the face -> back to lower TE
    return np.vstack([nd, face[::-1]]), 2*n_side


def report(nd, tag):
    """Segment statistics and leading-edge turning, measured not assumed."""
    d = np.diff(nd, axis=0, append=nd[:1])
    L = np.linalg.norm(d, axis=1)
    u = d/np.maximum(L, 1e-30)[:, None]
    ang = np.degrees(np.arccos(np.clip((u[:-1]*u[1:]).sum(1), -1.0, 1.0)))
    i = int(np.argmin(nd[:, 0]))
    w = np.arange(max(i - 8, 0), min(i + 9, len(ang)))
    print('  %-6s n=%4d  seg: min %.3e med %.3e max %.3e  (min/med %.4f)'
          % (tag, len(nd), L.min(), np.median(L), L.max(),
             L.min()/np.median(L)))
    print('  %-6s LE at index %d, turning: max %.2f deg near LE, %.2f deg overall'
          % ('', i, ang[w].max(), ang.max()))
    return L.min()/np.median(L), ang[w].max()
