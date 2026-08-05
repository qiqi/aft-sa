"""STEP 1 -- choose the flap, then ask what mean line the fore element wants.

Two questions, in order:

  (a) FLAP GEOMETRY. NACA 9416 (user's suggestion, 2026-08-04): 9% camber at
      40% chord, 16% thick. High camber to carry the loading, and a well
      rounded nose so the fore element's trailing edge -- which has to tuck in
      just ahead of it -- does not need a knife edge.

  (b) MEAN LINE. Solve the FLAP ALONE. Trace the streamline that would pass
      through the fore element's leading edge. Then ask whether a CLASSIC
      camber line can be fitted to that streamline, and quantify what is lost:

        residual    velocity normal to the FITTED line, as a fraction of
                    V_inf. Zero on the streamline itself by definition, so
                    this is exactly the turning the fore element would have to
                    impose -- i.e. the loading, and the suction peaks, that a
                    classic mean line cannot avoid.
        pressure    Cp along the line: the variation the THICKNESS
                    distribution then has to cancel.

Run:  python3 step1_flap_and_camber.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import panel2e as M

# ------------------------------------------------------------------ setup --
ALPHA = 0.0            # freestream incidence
FLAP = dict(m=0.09, p=0.40, t=0.16)        # NACA 9416
FLAP_CHORD = 0.30
FLAP_LE = (0.70, 0.04)
FLAP_INC = -8.0
FORE_CHORD = 0.68      # fore LE at the origin, TE at x = FORE_CHORD
NPAN = 200


def flap_nodes():
    nd = M.airfoil_nodes(NPAN, FLAP['m'], FLAP['p'], FLAP['t'], modified=False)
    return M.place(nd, FLAP_CHORD, FLAP_INC, *FLAP_LE)


def fit_camber(xs, zs, kind):
    """Least-squares fit of a classic mean line (plus incidence) to the
    streamline. Returns (z_fit, dz_fit, params, rms)."""
    c = FORE_CHORD
    xn = xs/c

    def model(par):
        if kind == '4digit':
            m, p, th = par
            yc, _ = M.naca_camber(np.clip(xn, 1e-9, 1), m, np.clip(p, .05, .95))
        else:
            cli, a, th = par
            yc, _ = M.naca_a_camber(xn, cli, np.clip(a, 0.05, 0.999))
        # rotate the (chord, camber) pair by the incidence th
        t = np.radians(th)
        return (xn*np.sin(t) + yc*np.cos(t))*c

    if kind == '4digit':
        p0 = [0.04, 0.5, 4.0]
        bnd = ([-0.20, 0.05, -20.0], [0.25, 0.95, 30.0])
    else:
        p0 = [0.4, 0.8, 4.0]
        bnd = ([-1.5, 0.05, -20.0], [1.5, 0.999, 30.0])
    r = least_squares(lambda p: model(p) - zs, p0, bounds=bnd)
    zf = model(r.x)
    return zf, np.gradient(zf, xs), r.x, float(np.sqrt(np.mean((zf - zs)**2)))


def normal_residual(P, xs, zf, dzf):
    """Velocity normal to a given line, per unit V_inf."""
    Vx, Vz = M.field_velocity(xs, zf, P)
    tx, tz = 1.0/np.hypot(1, dzf), dzf/np.hypot(1, dzf)
    return -Vx*tz + Vz*tx, np.hypot(Vx, Vz)


if __name__ == '__main__':
    n2 = flap_nodes()
    Pf, res = M.solve_elements([n2], ALPHA)
    print('FLAP: NACA %d%d%02d  chord %.2f  LE (%.2f, %.2f)  inc %+.1f deg'
          % (FLAP['m']*100, FLAP['p']*10, FLAP['t']*100, FLAP_CHORD,
             *FLAP_LE, FLAP_INC))
    print('  flap alone at alpha=%.1f:  Cl=%.4f  Cm=%.4f' %
          (ALPHA, res['Cl'], res['Cm']))

    xs, zs = M.streamline_camber(Pf, 0.0, 0.0, FORE_CHORD, n=400)
    dzs = np.gradient(zs, xs)
    vn_s, V_s = normal_residual(Pf, xs, zs, dzs)
    cp_s = 1.0 - V_s**2
    print('\nSTREAMLINE through the fore LE: rises to z=%.4f, slope %+.3f -> %+.3f'
          % (zs[-1], dzs[0], dzs[-1]))
    print('  |v_n|/Vinf on the streamline itself: max %.2e  (zero by definition)'
          % np.abs(vn_s).max())
    print('  Cp along it: %.3f -> %.3f   (range %.3f)'
          % (cp_s[0], cp_s[-1], cp_s.max() - cp_s.min()))

    fits = {}
    for kind in ('4digit', 'aseries'):
        zf, dzf, par, rms = fit_camber(xs, zs, kind)
        vn, V = normal_residual(Pf, xs, zf, dzf)
        cp = 1.0 - V**2
        fits[kind] = (zf, dzf, par, rms, vn, cp)
        lbl = ('m=%.3f p=%.2f inc=%+.1f' % tuple(par) if kind == '4digit'
               else 'cli=%.3f a=%.2f inc=%+.1f' % tuple(par))
        print('\nFIT %-8s %s' % (kind, lbl))
        print('  geometric rms  %.5f  (%.2f%% of fore chord)'
              % (rms, 100*rms/FORE_CHORD))
        print('  |v_n|/Vinf     mean %.4f   max %.4f   <-- residual turning'
              % (np.abs(vn).mean(), np.abs(vn).max()))
        print('  Cp along fit   %.3f -> %.3f   (range %.3f)'
              % (cp[0], cp[-1], cp.max() - cp.min()))

    # ------------------------------------------------------------- figure --
    fig, ax = plt.subplots(3, 1, figsize=(8.4, 9.2),
                           gridspec_kw=dict(height_ratios=[1.0, 1.0, 1.0]))
    ax[0].plot(n2[:, 0], n2[:, 1], '-', color='#1a8a5a', lw=1.6)
    ax[0].fill(n2[:, 0], n2[:, 1], color='#1a8a5a', alpha=.12)
    ax[0].plot(xs, zs, '-', color='#b03060', lw=2.2, label='streamline (exact)')
    ax[0].plot(xs, fits['4digit'][0], '--', color='#1f4e9c', lw=1.5,
               label='NACA 4-digit fit')
    ax[0].plot(xs, fits['aseries'][0], ':', color='#d97706', lw=1.9,
               label='NACA a-series fit')
    for k in range(1, 5):
        z0 = 0.0 + 0.012*k
        xx, zz = M.streamline_camber(Pf, 0.0, z0, FORE_CHORD, n=200)
        ax[0].plot(xx, zz, '-', color='0.75', lw=0.7, zorder=0)
    ax[0].set_aspect('equal'); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[0].set_title('NACA 9416 flap alone, $\\alpha$=%.0f$^\\circ$: streamlines '
                    'and classic mean-line fits' % ALPHA, fontsize=10)
    ax[0].set_ylabel('z')

    for kind, c, ls in (('4digit', '#1f4e9c', '--'), ('aseries', '#d97706', ':')):
        ax[1].plot(xs, fits[kind][4], ls, color=c, lw=1.7, label=kind)
    ax[1].plot(xs, vn_s, '-', color='#b03060', lw=2.0, label='streamline (=0)')
    ax[1].axhline(0, color='0.6', lw=.8)
    ax[1].set_ylabel('$v_n / V_\\infty$')
    ax[1].set_title('residual normal velocity along the fitted line '
                    '(= turning the fore element must impose)', fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    ax[2].plot(xs, cp_s, '-', color='#b03060', lw=2.0, label='streamline')
    for kind, c, ls in (('4digit', '#1f4e9c', '--'), ('aseries', '#d97706', ':')):
        ax[2].plot(xs, fits[kind][5], ls, color=c, lw=1.7, label=kind)
    ax[2].invert_yaxis()
    ax[2].set_xlabel('x'); ax[2].set_ylabel('$C_p$')
    ax[2].set_title('pressure along the line (flap-alone field) -- what the '
                    'THICKNESS must cancel', fontsize=10)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig('step1_flap_and_camber.pdf')
    fig.savefig('step1_flap_and_camber.png', dpi=125)
    print('\nwrote step1_flap_and_camber.pdf / .png')
