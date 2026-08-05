"""STEP 9 -- pick the anchoring alpha and the fore-element EXTENT.

User's constraints (2026-08-04):
  * the mean line was traced at alpha = 0, but that need not be the most
    laminar anchoring condition; at alpha = -1 the streamline climbs less and
    the gap to the flap closes;
  * the fore element's chord is free -- go as LONG as possible, but the camber
    fit must stay good, and
  * HARD LIMIT: the fore element may not extend beyond the MINIMUM-PRESSURE
    station along its own streamline (above the flap). Past that point the flow
    decelerates, and a closing trailing edge cannot cancel an adverse gradient.

So: for each alpha, trace the flap-alone streamline through the fore LE, find
where Cp along it is minimum -> x_max, then sweep the chord up to x_max and
report where the classic 4-digit camber fit stops being good.

Run:  python3 step9_extent.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import panel2e as M
from step1_flap_and_camber import FLAP, FLAP_CHORD, FLAP_LE, FLAP_INC, NPAN

VN_LIMIT = 0.012          # mean |v_n|/Vinf we are willing to accept
ALPHAS = (-3.0, -2.0, -1.0, 0.0)


def flap_at(alpha):
    nd = M.place(M.airfoil_nodes(NPAN, FLAP['m'], FLAP['p'], FLAP['t']),
                 FLAP_CHORD, FLAP_INC, *FLAP_LE)
    P, res = M.solve_elements([nd], alpha)
    return nd, P, res


def streamline_and_cp(P, x_end=0.98, n=700):
    xs, zs = M.streamline_camber(P, 0.0, 0.0, x_end, n=n)
    Vx, Vz = M.field_velocity(xs, zs, P)
    return xs, zs, 1.0 - (Vx**2 + Vz**2)


def fit4(xs, zs, chord):
    """Fit NACA 4-digit (m, p) + incidence to the streamline over [0, chord]."""
    m_ = xs <= chord
    x, z = xs[m_], zs[m_]
    xn = x/chord

    def model(par):
        mm, pp, th = par
        yc, _ = M.naca_camber(np.clip(xn, 1e-9, 1.0), mm, np.clip(pp, .05, .95))
        t = np.radians(th)
        return (xn*np.sin(t) + yc*np.cos(t))*chord
    r = least_squares(lambda p: model(p) - z, [-0.03, 0.7, 5.0],
                      bounds=([-0.20, 0.05, -20.0], [0.25, 0.95, 30.0]))
    zf = model(r.x)
    return x, zf, np.gradient(zf, x), r.x, float(np.sqrt(np.mean((zf - z)**2)))


def vn_of(P, x, zf, dzf):
    Vx, Vz = M.field_velocity(x, zf, P)
    nrm = np.hypot(1.0, dzf)
    return np.abs((-Vx*dzf + Vz)/nrm)


if __name__ == '__main__':
    print('%7s %10s %10s %10s %10s %10s' %
          ('alpha', 'Cl_flap', 'x(Cp_min)', 'z@x_max', 'Cp_min', 'gap@x_max'))
    data = {}
    for a in ALPHAS:
        nd, P, res = flap_at(a)
        xs, zs, cp = streamline_and_cp(P)
        i = int(np.argmin(cp))
        xmax, zmax = xs[i], zs[i]
        gap = float(np.min(np.hypot(nd[:, 0] - xmax, nd[:, 1] - zmax)))
        data[a] = (nd, P, xs, zs, cp, xmax)
        print('%7.1f %10.4f %10.4f %10.4f %10.4f %10.4f'
              % (a, res['Cl'], xmax, zmax, cp[i], gap))

    print('\nchord sweep -- how long can the fore element be before the fit'
          ' degrades?  (|v_n| limit %.3f)' % VN_LIMIT)
    print('%7s %8s %10s %10s %9s %8s %8s' %
          ('alpha', 'chord', 'rms/c', 'vn mean', 'vn max', 'm', 'p'))
    best = {}
    for a in ALPHAS:
        nd, P, xs, zs, cp, xmax = data[a]
        keep = None
        for chord in np.arange(0.45, xmax + 1e-9, 0.025):
            x, zf, dzf, par, rms = fit4(xs, zs, chord)
            vn = vn_of(P, x, zf, dzf)
            ok = vn.mean() <= VN_LIMIT
            if ok:
                keep = (chord, rms, vn.mean(), vn.max(), par)
        if keep:
            chord, rms, vm, vx, par = keep
            gap = float(np.min(np.hypot(nd[:, 0] - chord,
                                        nd[:, 1] - np.interp(chord, xs, zs))))
            best[a] = keep + (gap,)
            print('%7.1f %8.3f %10.5f %10.4f %9.4f %8.3f %8.2f   gap %.4f'
                  % (a, chord, rms, vm, vx, par[0], par[1], gap))
        else:
            print('%7.1f   no chord meets the limit' % a)

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(2, 1, figsize=(9.4, 7.6))
    for a, c in zip(ALPHAS, ('#1f4e9c', '#0891b2', '#d97706', '#b03060')):
        nd, P, xs, zs, cp, xmax = data[a]
        ax[0].plot(xs, zs, '-', color=c, lw=1.7, label='$\\alpha$=%.0f$^\\circ$' % a)
        i = int(np.argmin(cp))
        ax[0].plot(xs[i], zs[i], 'o', color=c, ms=6)
        if a in best:
            ch = best[a][0]
            ax[0].plot(ch, np.interp(ch, xs, zs), 's', color=c, ms=7, mfc='none')
        ax[1].plot(xs, cp, '-', color=c, lw=1.6)
        ax[1].plot(xs[i], cp[i], 'o', color=c, ms=6)
    nd0 = data[ALPHAS[0]][0]
    ax[0].fill(nd0[:, 0], nd0[:, 1], color='#1a8a5a', alpha=.18)
    ax[0].plot(nd0[:, 0], nd0[:, 1], '-', color='#1a8a5a', lw=1.5)
    ax[0].set_aspect('equal'); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[0].set_title('streamline through the fore LE; circle = $C_p$ minimum '
                    '(hard limit), square = longest good fit', fontsize=10)
    ax[0].set_ylabel('z')
    ax[1].invert_yaxis(); ax[1].grid(alpha=.3)
    ax[1].set_xlabel('x'); ax[1].set_ylabel('$C_p$ along the streamline')
    fig.tight_layout()
    fig.savefig('step9_extent.pdf'); fig.savefig('step9_extent.png', dpi=125)
    print('\nwrote step9_extent.pdf / .png')
