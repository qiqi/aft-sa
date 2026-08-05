"""STEP 14 -- the ADOPTED configuration (blend lambda = 0.20).

Geometry, all from public NACA definitions:

    alpha        -1 deg
    fore chord   0.700 (LE at the origin)
    fore mean    NACA 4-digit, m = -3.67% at p = 0.714, mounted +7.73 deg
    fore thick   NACA 4-digit MODIFIED, t/c = 0.077, x_m = 0.470, I = 6,
                 K_m = 0.666
    flap         NACA 9416, chord 0.30, LE (0.70, 0.04), inc -8 deg

lambda = 0.20 on the line between the streamline-fitted start (lambda = 0) and
the slot-shaped fine tune (lambda = 1); user's choice over the 0.30 that
minimised the lower-surface gradient.

Run:  python3 step14_adopted.py
"""
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import panel2e as M
import step12_slot_shaping as S

LAM = 0.20
A = dict(m=-0.0409, p=0.743, inc=7.85, t=0.070, xm=0.50, km=0.70)
B = dict(m=-0.0200, p=0.600, inc=7.24, t=0.105, xm=0.35, km=0.53)
PAR = {k: A[k] + LAM*(B[k] - A[k]) for k in A}


def trace(P, x0, z0, x_end, n=760):
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
    print('ADOPTED (lambda = %.2f)' % LAM)
    for k in ('m', 'p', 'inc', 't', 'xm', 'km'):
        print('   %-4s %9.4f' % (k, PAR[k]))
    a = S.analyse(PAR)
    P, els = a['P'], a['els']
    print('  upper: rng %.3f  adv %.3f  spike %.3f'
          % (a['up']['rng'], a['up']['adv'], a['up']['spike']))
    print('  lower: rng %.3f  adv %.3f  spike %.3f'
          % (a['lo']['rng'], a['lo']['adv'], a['lo']['spike']))
    print('  gap %.4f   Cl %.4f' % (a['gap'], a['Cl']))
    json.dump(dict(alpha=S.ALPHA, chord=S.CHORD, lam=LAM, fore=PAR,
                   flap=S.FLAP), open('twoelement_adopted.json', 'w'), indent=2)

    for k, nd in enumerate(els):
        np.savetxt('adopted_elem%d.dat' % (k+1), nd, fmt='%12.8f',
                   header='two-element wake article, element %d' % (k+1))

    allx = np.concatenate([e[:, 0] for e in els])
    allz = np.concatenate([e[:, 1] for e in els])
    x0, x1 = allx.min() - 0.32, allx.max() + 0.42
    z0, z1 = allz.min() - 0.15, allz.max() + 0.15
    X, Z = x1 - x0, z1 - z0
    W = 8.6
    L, R, Bm, T, G = 0.085, 0.02, 0.075, 0.05, 0.05
    axw = 1.0 - L - R
    geo_h = (Z/X)*(axw*W); cp_h = 3.7
    H = geo_h + cp_h + (Bm + T + G)*6.0
    fig = plt.figure(figsize=(W, H))
    gh, ch = geo_h/H, cp_h/H
    axc = fig.add_axes([L, Bm, axw, ch])
    axg = fig.add_axes([L, Bm + ch + G, axw, gh])

    for zs0 in np.linspace(z0 + 0.015, z1 - 0.015, 36):
        xx, zz = trace(P, x0, zs0, x1)
        axg.plot(xx, zz, '-', color='0.63', lw=0.6, zorder=1)
    for nd, c in zip(els, ('#1f4e9c', '#1a8a5a')):
        axg.fill(nd[:, 0], nd[:, 1], color=c, alpha=.22, zorder=3)
        axg.plot(nd[:, 0], nd[:, 1], '-', color=c, lw=1.7, zorder=4)
    axg.set_xlim(x0, x1); axg.set_ylim(z0, z1)
    axg.set_xticklabels([]); axg.set_yticks([])
    for sp in ('left', 'right', 'top'):
        axg.spines[sp].set_visible(False)
    axg.set_title('two-element wake-interaction article, adopted geometry: '
                  r'inviscid streamlines, $\alpha=-1^\circ$', fontsize=10)

    for k, c in enumerate(('#1f4e9c', '#1a8a5a')):
        li, ui = M.surfaces(P, k)
        axc.plot(P.xc[ui], P.Cp[ui], '-', color=c, lw=1.6,
                 label=['fore', 'flap'][k] + ' upper')
        axc.plot(P.xc[li], P.Cp[li], '--', color=c, lw=1.2, alpha=.85,
                 label=['fore', 'flap'][k] + ' lower')
    axc.axhline(0, color='0.55', lw=.8, zorder=0)
    axc.set_xlim(x0, x1); axc.invert_yaxis()
    axc.set_xlabel('$x/c$'); axc.set_ylabel('$C_p$')
    axc.grid(alpha=.25, lw=.6); axc.legend(fontsize=7.5, loc='lower right')
    txt = ('NACA mean line m=%.3f p=%.3f @ %+.2f$^\\circ$\n'
           'NACA mod thickness t/c=%.3f x$_m$=%.3f K$_m$=%.3f\n'
           'flap NACA 9416, gap %.4f\n'
           '$C_l$=%.3f (fore %.3f, flap %.3f)'
           % (PAR['m'], PAR['p'], PAR['inc'], PAR['t'], PAR['xm'], PAR['km'],
              a['gap'], a['Cl'], a['Cl'] - 0, 0))
    axc.text(0.012, 0.04, txt, transform=axc.transAxes, fontsize=7.2,
             va='bottom', family='monospace',
             bbox=dict(fc='white', ec='0.8', lw=.6, pad=4, alpha=.93))
    fig.savefig('step14_adopted.pdf')
    print('wrote step14_adopted.pdf, twoelement_adopted.json, adopted_elem{1,2}.dat')
