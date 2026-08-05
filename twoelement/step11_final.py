"""STEP 11 -- the alpha = -1 configuration: geometry + streamlines + Cp.

Settled through steps 1-10:
    alpha        -1 deg  (monotone streamline Cp; alpha=-2's mid-chord adverse
                 pocket would need x_m ~ 0.70, outside the classic NACA families,
                 and would double the trailing-edge closure slope)
    fore chord   0.700   (the 4-digit camber fit gives out here, short of the
                 0.7795 pressure-minimum hard limit)
    fore mean    NACA 4-digit, m = -4.1% at p = 0.74, mounted +7.85 deg
    fore thick   NACA 4-digit MODIFIED, x_m = 0.50, I = 6, K_m = 0.7
    flap         NACA 9416, chord 0.30, LE (0.70, 0.04), inc -8 deg

Thickness is re-picked here by the COUPLED solve rather than the thin-airfoil
shape match.

Run:  python3 step11_final.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import panel2e as M
from step9_extent import flap_at, streamline_and_cp, fit4

ALPHA = -1.0
CHORD = 0.700
XM, IK, KM = 0.50, 6.0, 0.7
NPAN = 300


def build(t, m_cam, p_cam, inc):
    n1 = M.place(M.airfoil_nodes(NPAN, m_cam, p_cam, t, modified=True,
                                 xm=XM, ik=IK, km=KM, le_blend=0.15),
                 CHORD, inc, 0.0, 0.0)
    n2 = M.place(M.airfoil_nodes(NPAN, 0.09, 0.40, 0.16), 0.30, -8.0,
                 0.70, 0.04)
    return [n1, n2]


def surf_metrics(P, k=0, lo=0.02, hi=0.98):
    out = {}
    l_, u_ = M.surfaces(P, k)
    for tag, idx in (('up', u_), ('lo', l_)):
        s = P.xc[idx]
        s = (s - s.min())/(s.max() - s.min() + 1e-30)
        m = (s >= lo) & (s <= hi)
        cp = P.Cp[idx][m]
        g = np.gradient(cp, s[m])
        out[tag] = (float(cp.max() - cp.min()),
                    float(np.clip(g, 0, None).mean()),
                    float(np.clip(g, 0, None).max()))
    return out


def trace(P, x0, z0, x_end, n=800):
    xs = np.linspace(x0, x_end, n); h = xs[1] - xs[0]
    zs = np.empty(n); zs[0] = z0; z = z0
    for i in range(n - 1):
        def sl(xx, zz):
            Vx, Vz = M.field_velocity(xx, zz, P)
            return float(Vz[0]/(Vx[0] + 1e-30))
        k1 = sl(xs[i], z); k2 = sl(xs[i]+.5*h, z+.5*h*k1)
        k3 = sl(xs[i]+.5*h, z+.5*h*k2); k4 = sl(xs[i]+h, z+h*k3)
        z += (h/6.)*(k1+2*k2+2*k3+k4); zs[i+1] = z
        if abs(z) > 1.2:
            return xs[:i+2], zs[:i+2]
    return xs, zs


if __name__ == '__main__':
    nd_f, Pf, resf = flap_at(ALPHA)
    xs, zs, cps = streamline_and_cp(Pf)
    _, _, _, par, rms = fit4(xs, zs, CHORD)
    m_cam, p_cam, inc = par
    print('mean line: NACA m=%+.4f p=%.3f mounted %+.2f deg (fit rms %.5f c)'
          % (m_cam, p_cam, inc, rms))

    print('\n%6s | %-26s | %-26s | %8s' %
          ('t/c', 'UPPER rng advMean advMax', 'LOWER rng advMean advMax', 'Cl'))
    best = None
    for t in (0.05, 0.06, 0.07, 0.08, 0.09, 0.10):
        P, res = M.solve_elements(build(t, m_cam, p_cam, inc), ALPHA)
        mt = surf_metrics(P)
        score = mt['up'][0] + mt['lo'][0] + 2*(mt['up'][1] + mt['lo'][1])
        print('%6.3f | %7.3f %8.3f %8.3f | %7.3f %8.3f %8.3f | %8.3f'
              % (t, *mt['up'], *mt['lo'], res['Cl']))
        if best is None or score < best[0]:
            best = (score, t, P, res, mt)
    _, t_b, P, res, mt = best
    print('\nchosen t/c = %.3f ; Cl = %.4f (fore %.4f, flap %.4f)'
          % (t_b, res['Cl'], res['Cl1'], res['Cl2']))
    els = build(t_b, m_cam, p_cam, inc)
    gap = float(np.min(np.hypot(els[0][:, 0][:, None] - els[1][:, 0][None, :],
                                els[0][:, 1][:, None] - els[1][:, 1][None, :])))
    print('slot gap %.4f c' % gap)

    # ------------------------------------------------------------- figure --
    allx = np.concatenate([e[:, 0] for e in els])
    allz = np.concatenate([e[:, 1] for e in els])
    x0, x1 = allx.min() - 0.35, allx.max() + 0.45
    z0, z1 = allz.min() - 0.16, allz.max() + 0.16
    X, Z = x1 - x0, z1 - z0
    W = 8.6
    L, R, B, T, G = 0.085, 0.02, 0.075, 0.05, 0.05
    axw = 1.0 - L - R
    geo_h = (Z/X)*(axw*W)
    cp_h = 3.6
    H = geo_h + cp_h + (B + T + G)*6.0
    fig = plt.figure(figsize=(W, H))
    gh, ch = geo_h/H, cp_h/H
    axc = fig.add_axes([L, B, axw, ch])
    axg = fig.add_axes([L, B + ch + G, axw, gh])

    for z_s in np.linspace(z0 + 0.02, z1 - 0.02, 34):
        xx, zz = trace(P, x0, z_s, x1)
        axg.plot(xx, zz, '-', color='0.62', lw=0.6, zorder=1)
    for nd, c in zip(els, ('#1f4e9c', '#1a8a5a')):
        axg.fill(nd[:, 0], nd[:, 1], color=c, alpha=.22, zorder=3)
        axg.plot(nd[:, 0], nd[:, 1], '-', color=c, lw=1.7, zorder=4)
    axg.set_xlim(x0, x1); axg.set_ylim(z0, z1)
    axg.set_xticklabels([]); axg.set_yticks([])
    for sp in ('left', 'right', 'top'):
        axg.spines[sp].set_visible(False)
    axg.set_title('two-element wake-interaction article: inviscid streamlines, '
                  r'$\alpha=%.0f^\circ$' % ALPHA, fontsize=10)

    for k, c in enumerate(('#1f4e9c', '#1a8a5a')):
        l_, u_ = M.surfaces(P, k)
        axc.plot(P.xc[u_], P.Cp[u_], '-', color=c, lw=1.6,
                 label=['fore', 'flap'][k] + ' upper')
        axc.plot(P.xc[l_], P.Cp[l_], '--', color=c, lw=1.2, alpha=.85,
                 label=['fore', 'flap'][k] + ' lower')
    axc.axhline(0, color='0.55', lw=.8, zorder=0)
    axc.set_xlim(x0, x1); axc.invert_yaxis()
    axc.set_xlabel('$x/c$'); axc.set_ylabel('$C_p$')
    axc.grid(alpha=.25, lw=.6); axc.legend(fontsize=7.5, loc='lower right')
    txt = ('t/c = %.3f   gap = %.4f\n$C_l$ = %.4f  (fore %.3f, flap %.3f)\n'
           'fore upper: rng %.3f  adv %.3f\nfore lower: rng %.3f  adv %.3f'
           % (t_b, gap, res['Cl'], res['Cl1'], res['Cl2'],
              mt['up'][0], mt['up'][1], mt['lo'][0], mt['lo'][1]))
    axc.text(0.012, 0.04, txt, transform=axc.transAxes, fontsize=7.5,
             va='bottom', family='monospace',
             bbox=dict(fc='white', ec='0.8', lw=.6, pad=4, alpha=.93))
    fig.savefig('step11_final.pdf')
    print('wrote step11_final.pdf')
