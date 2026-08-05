"""STEP 2 -- three questions before committing the geometry.

Q1  How PRECISE must the 4-digit camber line be? What does rounding the fitted
    (m, p) to catalogue integers cost in residual normal velocity?

Q2  Cl = 2.19 -- on flap chord or overall chord? Since the main element is
    meant to carry little lift, the configuration number wants overall-chord
    scaling.

Q3  Given the Cp actually reached at the fore trailing edge, how THICK must the
    main element be, by thin-airfoil theory, for the pressure to stay constant
    all the way to the TE?

Run:  python3 step2_questions.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import panel2e as M
from step1_flap_and_camber import (ALPHA, FLAP, FLAP_CHORD, FLAP_LE, FLAP_INC,
                                   FORE_CHORD, NPAN, flap_nodes, fit_camber,
                                   normal_residual)

OVERALL = 1.0        # the configuration reference chord (fore LE -> flap TE)


def line_from(m, p, inc, xs):
    yc, _ = M.naca_camber(np.clip(xs/FORE_CHORD, 1e-9, 1.0), m, p)
    t = np.radians(inc)
    return (xs/FORE_CHORD*np.sin(t) + yc*np.cos(t))*FORE_CHORD


def refit_inc(m, p, xs, zs):
    """Best incidence for a FIXED (m, p) -- the catalogue section is given, we
    are only free to mount it at some angle."""
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda th: np.sum((line_from(m, p, th, xs) - zs)**2),
                        bounds=(-20, 30), method='bounded')
    return float(r.x)


if __name__ == '__main__':
    n2 = flap_nodes()
    Pf, res = M.solve_elements([n2], ALPHA)
    xs, zs = M.streamline_camber(Pf, 0.0, 0.0, FORE_CHORD, n=400)

    # ------------------------------------------------------------------ Q2 --
    cref_flap = n2[:, 0].max() - n2[:, 0].min()
    print('=' * 72)
    print('Q2  Cl reference chord')
    print('  solver cref for the flap-alone solve : %.4f  (the flap extent)'
          % cref_flap)
    print('  Cl on FLAP chord                     : %.3f' % res['Cl'])
    print('  Cl on OVERALL chord (%.2f)            : %.3f'
          % (OVERALL, res['Cl']*cref_flap/OVERALL))
    print('  -> the configuration number is %.3f; the 2.19 was flap-chord.'
          % (res['Cl']*cref_flap/OVERALL))

    # ------------------------------------------------------------------ Q1 --
    print('=' * 72)
    print('Q1  how precise must the camber line be?')
    zf, dzf, par, rms0 = fit_camber(xs, zs, '4digit')
    vn0, _ = normal_residual(Pf, xs, zf, dzf)
    print('  exact fit      m=%+.4f p=%.3f inc=%+.2f   rms %.5f  '
          '|vn| mean %.4f max %.4f' % (*par, rms0, np.abs(vn0).mean(),
                                       np.abs(vn0).max()))
    print('  %-22s %8s %8s %8s %8s' % ('rounded (m, p)', 'inc', 'rms',
                                       'vn mean', 'vn max'))
    cands = [(-0.04, 0.70), (-0.03, 0.70), (-0.04, 0.60), (-0.04, 0.80),
             (-0.05, 0.70), (-0.02, 0.70), (0.0, 0.70)]
    rows = []
    for m, p in cands:
        inc = refit_inc(m, p, xs, zs)
        zl = line_from(m, p, inc, xs)
        vn, _ = normal_residual(Pf, xs, zl, np.gradient(zl, xs))
        rms = float(np.sqrt(np.mean((zl - zs)**2)))
        rows.append((m, p, inc, rms, np.abs(vn).mean(), np.abs(vn).max(), zl))
        tag = 'NACA %d%d%s' % (abs(m)*100, p*10, 'xx')
        print('  %-22s %+8.2f %8.5f %8.4f %8.4f  (%s%s)'
              % ('m=%+.2f p=%.2f' % (m, p), inc, rms, np.abs(vn).mean(),
                 np.abs(vn).max(), '-' if m < 0 else '', tag))
    print('  (|vn|/Vinf = local flow-angle error in radians; x57.3 for degrees)')

    # ------------------------------------------------------------------ Q3 --
    print('=' * 72)
    print('Q3  thickness needed to hold Cp constant to the TE')
    Vx, Vz = M.field_velocity(xs, zf, Pf)
    Vb = np.hypot(Vx, Vz)
    print('  base flow along the mean line: V/Vinf %.4f (LE) -> %.4f (TE)'
          % (Vb[0], Vb[-1]))
    print('  Cp %.3f -> %.3f ;  required thickness DECREMENT = %.4f V_inf'
          % (1 - Vb[0]**2, 1 - Vb[-1]**2, Vb[-1] - Vb[0]))

    # thickness-alone perturbation: symmetric section, uniform stream
    def u_thick(t, xm, ik=6.0, km=1.1, npan=240):
        nd = M.airfoil_nodes(npan, 0.0, 0.0, t, modified=True, xm=xm, ik=ik,
                             km=km, le_blend=0.0)
        nd = M.place(nd, FORE_CHORD, 0.0, 0.0, 0.0)
        P, _ = M.solve_elements([nd], 0.0)
        _, up = M.surfaces(P, 0)
        return P.xc[up], np.abs(P.Vt[up]) - 1.0

    print('  %-8s %-8s %10s %10s %10s' % ('t/c', 'x_m', 'peak u_t',
                                          'u_t at TE', 'drop'))
    best = None
    for t in (0.10, 0.15, 0.20, 0.25, 0.30):
        for xm in (0.40, 0.50, 0.60):
            xu, ut = u_thick(t, xm)
            m = (xu > 0.02*FORE_CHORD) & (xu < 0.97*FORE_CHORD)
            drop = float(ut[m].max() - ut[m][-1])
            print('  %-8.2f %-8.2f %10.4f %10.4f %10.4f'
                  % (t, xm, ut[m].max(), ut[m][-1], drop))
            need = Vb[-1] - Vb[0]
            if best is None or abs(drop - need) < abs(best[0] - need):
                best = (drop, t, xm)
    print('  required drop %.4f  ->  closest: t/c=%.2f at x_m=%.2f (drop %.4f)'
          % (Vb[-1] - Vb[0], best[1], best[2], best[0]))
    print('  NOTE first-order superposition (thickness + base flow perturbations'
          ' added); the coupled solve will differ.')

    # ------------------------------------------------------------- figure --
    fig, ax = plt.subplots(2, 1, figsize=(8.2, 6.4))
    ax[0].plot(xs, vn0, '-', color='#b03060', lw=2.0,
               label='exact fit (m=%+.3f p=%.2f)' % (par[0], par[1]))
    for (m, p, inc, rms, vm, vx, zl) in rows[:4]:
        vn, _ = normal_residual(Pf, xs, zl, np.gradient(zl, xs))
        ax[0].plot(xs, vn, '--', lw=1.3, label='m=%+.2f p=%.2f' % (m, p))
    ax[0].axhline(0, color='0.6', lw=.8)
    ax[0].set_ylabel('$v_n/V_\\infty$'); ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=7.5, ncol=2)
    ax[0].set_title('Q1: cost of rounding the camber line to catalogue values',
                    fontsize=10)

    for t in (0.10, 0.20, 0.30):
        xu, ut = u_thick(t, 0.50)
        ax[1].plot(xu, 1.0 + ut, '-', lw=1.4, label='t/c=%.2f (x$_m$=0.5)' % t)
    ax[1].plot(xs, Vb, '-', color='#b03060', lw=2.0, label='base flow (flap)')
    ax[1].axhline(1.0, color='0.6', lw=.8)
    ax[1].set_xlabel('x'); ax[1].set_ylabel('$V/V_\\infty$')
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    ax[1].set_title('Q3: base acceleration vs thickness-alone velocity',
                    fontsize=10)
    fig.tight_layout()
    fig.savefig('step2_questions.pdf'); fig.savefig('step2_questions.png', dpi=125)
    print('\\nwrote step2_questions.pdf / .png')
