"""RANS counterpart of the inviscid figure: geometry + streamlines on top,
surface Cp below, x aligned.

Streamlines are integrated in the mid-span plane of the quasi-2D solution with
a linear triangulation interpolant; Cp comes from the solver's own surface
output (surface_fluid_<elem>_proc0.vtu), not from the volume.

Run:  python3 plot_rans_cp.py case_L1 [out.pdf]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import plot_solution_mesh as PM


def surface_cp(path):
    """Mid-plane wall points ordered around the loop, split at LE and TE."""
    g, pts, arr = PM.read_vtu(path)
    nun = [len(np.unique(np.round(pts[:, i], 9))) for i in range(3)]
    span = int(np.argmin(nun))
    keep = np.isclose(pts[:, span], np.unique(pts[:, span])[0], atol=1e-9)
    P = np.delete(pts[keep], span, axis=1)
    cp = arr['Cp'][keep]
    cf = arr['Cf'][keep] if arr['Cf'].ndim == 1 else \
        np.linalg.norm(arr['Cf'][keep], axis=1)
    c = P.mean(axis=0)
    order = np.argsort(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))
    P, cp, cf = P[order], cp[order], cf[order]
    i_le, i_te = int(np.argmin(P[:, 0])), int(np.argmax(P[:, 0]))
    n = len(P)
    a = np.arange(i_le, i_le + n) % n            # LE -> ... -> LE
    k = int(np.where(a == i_te)[0][0])
    br1, br2 = a[:k + 1], a[k:]
    z1 = P[br1][:, 1].mean(); z2 = P[br2][:, 1].mean()
    up, lo = (br1, br2) if z1 > z2 else (br2, br1)
    return P, cp, cf, up, lo


def streamlines(case, T, U, V, seeds, x0, x1):
    fu = mtri.LinearTriInterpolator(T, U)
    fv = mtri.LinearTriInterpolator(T, V)
    out = []
    for z0 in seeds:
        xs, zs, x, z = [x0], [z0], x0, z0
        for _ in range(2600):
            u, v = fu(x, z), fv(x, z)
            if np.ma.is_masked(u) or np.ma.is_masked(v):
                break
            s = float(np.hypot(u, v))
            if s < 1e-9:
                break
            h = 0.0016
            x += h*float(u)/s
            z += h*float(v)/s
            if x > x1 or abs(z) > 0.6:
                break
            xs.append(x); zs.append(z)
        out.append((np.array(xs), np.array(zs)))
    return out


if __name__ == '__main__':
    case = sys.argv[1] if len(sys.argv) > 1 else 'case_L1'
    out = sys.argv[2] if len(sys.argv) > 2 else '%s_rans_cp.pdf' % case

    g, pts, arr = PM.read_vtu('%s/volume_proc0.vtu' % case)
    idx, tris, P2 = PM.midplane_tris(g, pts)
    vel = arr['velocity'][idx]
    T = mtri.Triangulation(P2[:, 0], P2[:, 1], tris)
    U, V = vel[:, 0], vel[:, 2]                  # plane is x-z
    print('mid-plane %d nodes; |u| max %.4f' % (len(P2), np.hypot(U, V).max()))

    surf = {}
    for nm in ('fore', 'flap'):
        surf[nm] = surface_cp('%s/surface_fluid_%s_proc0.vtu' % (case, nm))
        P, cp, cf, up, lo = surf[nm]
        print('%s: %d wall pts, Cp %.3f..%.3f' % (nm, len(P), cp.min(), cp.max()))

    allP = np.vstack([surf[n][0] for n in surf])
    x0, x1 = allP[:, 0].min() - 0.30, allP[:, 0].max() + 0.42
    z0, z1 = allP[:, 1].min() - 0.15, allP[:, 1].max() + 0.16
    X, Z = x1 - x0, z1 - z0
    W = 8.6
    L, R, Bm, T_, G = 0.085, 0.02, 0.075, 0.05, 0.05
    axw = 1.0 - L - R
    geo_h = (Z/X)*(axw*W); cp_h = 3.7
    H = geo_h + cp_h + (Bm + T_ + G)*6.0
    fig = plt.figure(figsize=(W, H))
    gh, ch = geo_h/H, cp_h/H
    axc = fig.add_axes([L, Bm, axw, ch])
    axg = fig.add_axes([L, Bm + ch + G, axw, gh])

    sl = streamlines(case, T, U, V, np.linspace(z0 + 0.012, z1 - 0.012, 34),
                     x0 + 0.005, x1)
    for xs, zs in sl:
        axg.plot(xs, zs, '-', color='0.6', lw=0.6, zorder=1)
    for nm, c in (('fore', '#1f4e9c'), ('flap', '#1a8a5a')):
        P = surf[nm][0]
        axg.fill(P[:, 0], P[:, 1], color=c, alpha=.22, zorder=3)
        axg.plot(np.append(P[:, 0], P[0, 0]), np.append(P[:, 1], P[0, 1]),
                 '-', color=c, lw=1.5, zorder=4)
    axg.set_xlim(x0, x1); axg.set_ylim(z0, z1)
    axg.set_xticklabels([]); axg.set_yticks([])
    for sp in ('left', 'right', 'top'):
        axg.spines[sp].set_visible(False)
    axg.set_title('%s : SA-AI RANS, Re=1e6, M=0.1, '
                  r'$\alpha=-1^\circ$ -- streamlines' % case, fontsize=10)

    for nm, c in (('fore', '#1f4e9c'), ('flap', '#1a8a5a')):
        P, cp, cf, up, lo = surf[nm]
        axc.plot(P[up][:, 0], cp[up], '-', color=c, lw=1.5, label=nm + ' upper')
        axc.plot(P[lo][:, 0], cp[lo], '--', color=c, lw=1.2, alpha=.85,
                 label=nm + ' lower')
    axc.axhline(0, color='0.55', lw=.8, zorder=0)
    axc.set_xlim(x0, x1); axc.invert_yaxis()
    axc.set_xlabel('$x/c$'); axc.set_ylabel('$C_p$')
    axc.grid(alpha=.25, lw=.6); axc.legend(fontsize=7.5, loc='lower right')
    fig.savefig(out)
    print('wrote', out)
