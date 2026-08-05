"""STEP 1v -- the fore element's mean line, re-traced through the VISCOUS
flap-alone field.

Replaces step1_flap_and_camber.py, which traced the streamline through the
INVISCID flap field. The flap carries a thick trailing-edge layer and a
separation bubble at Re_flap = 3e5, so its real circulation is far below the
inviscid Kutta value:

    inviscid, sharp TE (what step 1 used)   cl = 2.080
    inviscid, real 0.01c TE base            cl = 1.805
    mfoil viscous, Re = 3e5, ncrit 9        cl = 1.459
    SA-AI RANS, flap alone                  cl = 1.356   (CL_total 0.4069 / 0.3)

Tracing the mean line through a field with ~50% too much circulation gives a
line that is too curved and at too much incidence. The fore element then has to
impose the difference, which is what shows up as the leading-edge suction spike
measured on its LOWER surface (Cp ~ -0.85 instead of the intended flat rooftop).

Both streamlines are traced here -- inviscid via panel2e, viscous via the RANS
mid-plane field -- and classic mean lines are fitted to each, so the change is
quantified rather than asserted.

Run:  python3 step1v_camber.py [out.pdf]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.optimize import least_squares

import panel2e as M
import plot_solution_mesh as PM
from contour_te import blunt_contour

ALPHA = -1.0                             # freestream, matching the adopted case
FLAP = dict(m=0.09, p=0.40, t=0.16)
FLAP_CHORD, FLAP_INC, FLAP_LE = 0.30, -8.0, (0.70, 0.04)
TE_BASE = 0.003
# the fore element runs from the origin to its trailing edge; the adopted
# geometry has chord 0.7 at +7.728 deg, so the TE sits at x = 0.6936
FORE_X_END = 0.6936
MACH = 0.10


class RansField:
    """Mid-plane velocity interpolant from the flap-alone solution."""

    def __init__(self, case='case_F1'):
        g, pts, arr = PM.read_vtu('%s/volume_proc0.vtu' % case)
        idx, tris, P2 = PM.midplane_tris(g, pts)
        vel = arr['velocity'][idx]
        T = mtri.Triangulation(P2[:, 0], P2[:, 1], tris)
        self.fu = mtri.LinearTriInterpolator(T, vel[:, 0])
        self.fv = mtri.LinearTriInterpolator(T, vel[:, 2])
        # freestream speed, for Cp and for normalising
        self.vinf = MACH
        print('   RANS mid-plane: %d nodes' % len(P2))

    def __call__(self, x, z):
        u = np.ma.filled(self.fu(np.atleast_1d(x), np.atleast_1d(z)), np.nan)
        v = np.ma.filled(self.fv(np.atleast_1d(x), np.atleast_1d(z)), np.nan)
        return np.asarray(u, float), np.asarray(v, float)


def trace(vel, x0, z0, x_end, n=400):
    """Streamline through (x0, z0), RK4 in x, using a velocity callable."""
    xs = np.linspace(x0, x_end, n)
    h = xs[1] - xs[0]
    zs = np.empty(n)
    zs[0] = z = z0

    def slope(xx, zz):
        u, v = vel(xx, zz)
        if not np.isfinite(u[0]) or abs(u[0]) < 1e-12:
            return 0.0
        return float(v[0]/u[0])

    for i in range(n - 1):
        k1 = slope(xs[i], z)
        k2 = slope(xs[i] + 0.5*h, z + 0.5*h*k1)
        k3 = slope(xs[i] + 0.5*h, z + 0.5*h*k2)
        k4 = slope(xs[i] + h, z + h*k3)
        z += (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        zs[i + 1] = z
    return xs, zs


def fit_camber(xs, zs, kind, chord):
    """Least-squares fit of a classic mean line plus incidence."""
    xn = xs/chord

    def model(par):
        if kind == '4digit':
            m, p, th = par
            yc, _ = M.naca_camber(np.clip(xn, 1e-9, 1), m, np.clip(p, .05, .95))
        else:
            cli, a, th = par
            yc, _ = M.naca_a_camber(xn, cli, np.clip(a, 0.05, 0.999))
        t = np.radians(th)
        return (xn*np.sin(t) + yc*np.cos(t))*chord

    if kind == '4digit':
        p0, bnd = [0.04, 0.5, 4.0], ([-0.20, 0.05, -20.0], [0.25, 0.95, 30.0])
    else:
        p0, bnd = [0.4, 0.8, 4.0], ([-1.5, 0.05, -20.0], [1.5, 0.999, 30.0])
    r = least_squares(lambda p: model(p) - zs, p0, bounds=bnd)
    zf = model(r.x)
    return zf, np.gradient(zf, xs), r.x, float(np.sqrt(np.mean((zf - zs)**2)))


def residual(vel, xs, zf, dzf, vinf):
    """Velocity normal to a given line, per unit V_inf, and Cp along it."""
    u, v = vel(xs, zf)
    den = np.hypot(1.0, dzf)
    tx, tz = 1.0/den, dzf/den
    vn = (-u*tz + v*tx)/vinf
    q = np.hypot(u, v)/vinf
    return vn, 1.0 - q**2


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'step1v_camber.pdf'
    chord = FORE_X_END

    # ---------------------------------------------------------- inviscid --
    # Use the same sharp-TE section step 1 used. Feeding solve_elements the
    # blunt_contour output instead gives Cl = 0.011 -- the explicit
    # trailing-edge face breaks its Kutta condition, so the field ends up with
    # no circulation at all and the "inviscid" streamline is pure thickness
    # blockage running into the flap nose. That would not be a fair baseline.
    nd = M.place(M.airfoil_nodes(400, FLAP['m'], FLAP['p'], FLAP['t'],
                                 modified=False),
                 FLAP_CHORD, FLAP_INC, *FLAP_LE)
    Pf, res = M.solve_elements([nd], ALPHA)
    print('INVISCID flap alone at alpha=%+.1f: Cl=%.4f (flap chords)'
          % (ALPHA, res['Cl']))

    def vel_inv(x, z):
        u, v = M.field_velocity(np.atleast_1d(x), np.atleast_1d(z), Pf)
        return np.asarray(u, float), np.asarray(v, float)

    xi, zi = trace(vel_inv, 0.0, 0.0, chord)

    # ----------------------------------------------------------- viscous --
    rf = RansField()
    xv, zv = trace(rf, 0.0, 0.0, chord)

    print('\nSTREAMLINE through the fore LE (0,0) -> x=%.4f' % chord)
    for tag, xs, zs in (('inviscid', xi, zi), ('viscous ', xv, zv)):
        dz = np.gradient(zs, xs)
        print('  %s  z(TE) = %+.4f   slope %+.3f -> %+.3f   rise %.4f c'
              % (tag, zs[-1], dz[0], dz[-1], (zs[-1] - zs[0])/chord))
    print('  DIFFERENCE at the fore TE: %+.4f  (%.1f%% of the inviscid rise)'
          % (zv[-1] - zi[-1], 100*(zv[-1] - zi[-1])/max(abs(zi[-1]), 1e-9)))

    fits = {}
    for tag, xs, zs, vel, vinf in (('inviscid', xi, zi, vel_inv, 1.0),
                                   ('viscous', xv, zv, rf, MACH)):
        print('\n--- %s field ---' % tag)
        vn_s, cp_s = residual(vel, xs, zs, np.gradient(zs, xs), vinf)
        print('  on the streamline itself: |v_n| max %.2e (zero by definition)'
              '   Cp %.3f -> %.3f' % (np.nanmax(np.abs(vn_s)), cp_s[0],
                                      cp_s[-1]))
        for kind in ('4digit', 'aseries'):
            zf, dzf, par, rms = fit_camber(xs, zs, kind, chord)
            vn, cp = residual(vel, xs, zf, dzf, vinf)
            fits[(tag, kind)] = (zf, par, rms, vn, cp)
            lbl = ('m=%+.4f p=%.2f inc=%+.2f' % tuple(par) if kind == '4digit'
                   else 'cli=%+.4f a=%.2f inc=%+.2f' % tuple(par))
            print('  FIT %-8s %s' % (kind, lbl))
            print('      rms %.5f (%.2f%% of fore chord)   |v_n|/Vinf mean '
                  '%.4f max %.4f' % (rms, 100*rms/chord,
                                     np.nanmean(np.abs(vn)),
                                     np.nanmax(np.abs(vn))))

    # ------------------------------------------------------------ figure --
    fig, ax = plt.subplots(3, 1, figsize=(9.0, 10.4),
                           gridspec_kw=dict(height_ratios=[1.25, 1.0, 1.0]))
    ax[0].plot(nd[:, 0], nd[:, 1], '-', color='#1a8a5a', lw=1.5)
    ax[0].fill(nd[:, 0], nd[:, 1], color='#1a8a5a', alpha=.12)
    ax[0].plot(xi, zi, '--', color='#c44e52', lw=2.0,
               label='streamline, INVISCID flap (what step 1 used)')
    ax[0].plot(xv, zv, '-', color='#1f4e9c', lw=2.4,
               label='streamline, VISCOUS flap (SA-AI, Re$_{flap}$=3e5)')
    ax[0].plot(xv, fits[('viscous', '4digit')][0], ':', color='#000', lw=1.6,
               label='NACA 4-digit fit to viscous')
    ax[0].plot(xv, fits[('viscous', 'aseries')][0], '-.', color='#d97706',
               lw=1.5, label='NACA a-series fit to viscous')
    ax[0].set_aspect('equal')
    ax[0].set_xlim(-0.03, 1.05)
    ax[0].legend(fontsize=7.5, loc='upper left')
    ax[0].set_ylabel('$z$')
    ax[0].set_title('Fore-element mean line: streamline through (0,0) of the '
                    'flap-alone field', fontsize=10)

    ax[1].plot(xi, zi, '--', color='#c44e52', lw=1.8, label='inviscid')
    ax[1].plot(xv, zv, '-', color='#1f4e9c', lw=2.0, label='viscous')
    ax[1].plot(xv, zv - zi, '-', color='#555', lw=1.2,
               label='viscous $-$ inviscid')
    ax[1].axhline(0, color='0.7', lw=0.8)
    ax[1].set_ylabel('$z$')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.25, lw=.6)

    for tag, c, ls in (('inviscid', '#c44e52', '--'), ('viscous', '#1f4e9c', '-')):
        xs = xi if tag == 'inviscid' else xv
        for kind, lw in (('4digit', 1.6), ('aseries', 1.0)):
            vn = fits[(tag, kind)][3]
            ax[2].plot(xs, vn, ls, color=c, lw=lw, alpha=.9 if kind == '4digit'
                       else .55, label='%s, %s' % (tag, kind))
    ax[2].axhline(0, color='0.7', lw=0.8)
    ax[2].set_xlabel('$x$')
    ax[2].set_ylabel('$v_n/V_\\infty$ on the fitted line')
    ax[2].set_title('residual turning a classic mean line cannot avoid',
                    fontsize=9)
    ax[2].legend(fontsize=7.5, ncol=2)
    ax[2].grid(alpha=.25, lw=.6)

    fig.tight_layout()
    fig.savefig(out)
    print('\nwrote', out)
    np.savez('step1v_camber.npz', xi=xi, zi=zi, xv=xv, zv=zv,
             fit4=fits[('viscous', '4digit')][0],
             fita=fits[('viscous', 'aseries')][0],
             par4=fits[('viscous', '4digit')][1],
             para=fits[('viscous', 'aseries')][1])
