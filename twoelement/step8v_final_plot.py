"""Final inviscid picture of the adopted two-element geometry: geometry and
streamlines on top, surface Cp below, x aligned.

What the solver sees is the fore element plus the flap's DISPLACEMENT body
(surface + delta* frozen in from mfoil at Re_flap = 3e5), which is what carries
the correct flap circulation -- cl 1.431 against 1.459 viscous, where the plain
inviscid section gives 1.985. The true flap section is drawn underneath as a
faint dashed line so the displacement thickness is visible rather than implied.

Streamlines are integrated by ARC LENGTH, not in x: dz/dx blows up near the
stagnation points, and marching in x cannot round a leading edge at all.
Integration stops on leaving the window, entering either body, or the speed
collapsing at a stagnation point.

Run:  python3 step8v_final_plot.py [out.pdf] [alpha]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path

import panel2e as M
import step1v_flap_viscous as V
import flap_displacement as FD

ALPHA = -1.0


def streamline(P, x0, z0, x1, paths, h=0.0035, nmax=4000):
    """Trace one streamline forward by arc length through the panel field."""
    xs, zs = [x0], [z0]
    x, z = x0, z0
    for _ in range(nmax):
        u, w = M.field_velocity(np.array([x]), np.array([z]), P)
        u, w = float(u[0]), float(w[0])
        q = np.hypot(u, w)
        if not np.isfinite(q) or q < 1e-3:
            break
        # RK2 in arc length
        xm, zm = x + 0.5*h*u/q, z + 0.5*h*w/q
        u2, w2 = M.field_velocity(np.array([xm]), np.array([zm]), P)
        u2, w2 = float(u2[0]), float(w2[0])
        q2 = np.hypot(u2, w2)
        if not np.isfinite(q2) or q2 < 1e-3:
            break
        x, z = x + h*u2/q2, z + h*w2/q2
        if x > x1 or x < x0 - 0.05 or abs(z) > 0.55:
            break
        if any(pp.contains_point((x, z)) for pp in paths):
            break
        xs.append(x); zs.append(z)
    return np.array(xs), np.array(zs)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'step8v_final.pdf'
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else ALPHA

    d = np.load('step7v_final.npz')
    fore, flap = d['fore'], d['flap']
    x_end = float(d['x_end'])
    P, res = M.solve_elements([fore, flap], alpha)
    print('adopted geometry, alpha = %+.1f: CL = %.4f  CM = %.4f'
          % (alpha, res['Cl'], res.get('Cm', float('nan'))))
    print('  fore: x_end %.4f, camber TE slope %.2f deg, gap %.4f'
          % (x_end, np.degrees(np.arctan(float(d['s_te']))), float(d['gap'])))

    # the true flap section, for reference under the displacement body
    flap_true = M.place(M.airfoil_nodes(400, V.FLAP['m'], V.FLAP['p'],
                                        V.FLAP['t'], modified=False,
                                        te_thick=V.TE_THICK),
                        FD.FLAP_CHORD, FD.FLAP_INC, *FD.FLAP_LE)

    allp = np.vstack([fore, flap])
    x0, x1 = allp[:, 0].min() - 0.32, allp[:, 0].max() + 0.36
    z0, z1 = allp[:, 1].min() - 0.17, allp[:, 1].max() + 0.19
    paths = [Path(np.vstack([fore, fore[:1]])), Path(np.vstack([flap, flap[:1]]))]

    # ------------------------------------------------------------- layout --
    X, Z = x1 - x0, z1 - z0
    W = 9.0
    L, R, Bm, T_, G = 0.085, 0.02, 0.075, 0.075, 0.045
    axw = 1.0 - L - R
    geo_h = (Z/X)*(axw*W)
    cp_h = 3.9
    H = geo_h + cp_h + (Bm + T_ + G)*6.5
    fig = plt.figure(figsize=(W, H))
    gh, ch = geo_h/H, cp_h/H
    axc = fig.add_axes([L, Bm, axw, ch])
    axg = fig.add_axes([L, Bm + ch + G, axw, gh])

    seeds = np.linspace(z0 + 0.010, z1 - 0.010, 42)
    for zs0 in seeds:
        sx, sz = streamline(P, x0 + 0.004, zs0, x1, paths)
        if len(sx) > 3:
            axg.plot(sx, sz, '-', color='0.62', lw=0.55, zorder=1)

    axg.plot(np.append(flap_true[:, 0], flap_true[0, 0]),
             np.append(flap_true[:, 1], flap_true[0, 1]), '--',
             color='#1a8a5a', lw=0.9, alpha=.75, zorder=3,
             label='true flap section')
    for nd, c, lbl in ((fore, '#1f4e9c', 'fore element'),
                       (flap, '#1a8a5a', r'flap + $\delta^*$ (what is solved)')):
        axg.fill(nd[:, 0], nd[:, 1], color=c, alpha=.20, zorder=4)
        axg.plot(np.append(nd[:, 0], nd[0, 0]), np.append(nd[:, 1], nd[0, 1]),
                 '-', color=c, lw=1.5, zorder=5, label=lbl)
    axg.set_xlim(x0, x1); axg.set_ylim(z0, z1)
    axg.set_xticklabels([]); axg.set_yticks([])
    for sp in ('left', 'right', 'top'):
        axg.spines[sp].set_visible(False)
    axg.legend(fontsize=7.5, loc='upper left', framealpha=.9)
    axg.set_title('Adopted two-element geometry, inviscid: '
                  r'$\alpha=%+.1f^\circ$, $C_L=%.3f$' % (alpha, res['Cl']),
                  fontsize=10)

    for k, (nm, c) in enumerate((('fore', '#1f4e9c'), ('flap', '#1a8a5a'))):
        lo, up = M.surfaces(P, k)
        axc.plot(P.xc[up], P.Cp[up], '-', color=c, lw=1.6,
                 label='%s upper' % nm)
        axc.plot(P.xc[lo], P.Cp[lo], '--', color=c, lw=1.2, alpha=.85,
                 label='%s lower' % nm)
    axc.axhline(0, color='0.55', lw=.8, zorder=0)
    axc.set_xlim(x0, x1); axc.invert_yaxis()
    axc.set_xlabel('$x$'); axc.set_ylabel('$C_p$')
    axc.grid(alpha=.25, lw=.6)
    axc.legend(fontsize=8, loc='lower right')
    fig.savefig(out)
    print('wrote', out)
