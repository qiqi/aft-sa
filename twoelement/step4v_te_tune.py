"""STEP 4v -- fine-tune the fore element's trailing-edge region, where the
first-order instruments are the CAMBER SLOPE AT THE TE and the TE INCLUDED
ANGLE, not the thickness distribution at large.

Starting point (anchor 0, step2v):
    camber TE slope        +23.5 deg   -- points far too steeply upward
    TE included angle       43.0 deg   -- far too blunt
    upper surface leaves at  +3.7 deg, lower at +43.4 deg
    lower-surface Cp over x/c 0.85-0.99 reaches -0.825

Both are single coefficients in the CST form, which is why this is tractable:

  MEAN LINE   z = x_end*xh(1-xh)*sum A_i B_i^n(xh) + xh*dz_te
              dz/dx at the TE = dz_te/x_end - A_n
              so the last camber coefficient sets the TE slope outright.

  THICKNESS   t = x_end*[xh^0.5 (1-xh)*sum a_i B_i^m(xh) + xh*te_half]
              dt/dx at the TE = te_half - a_m
              so the last thickness coefficient sets the half-angle outright,
              and the included angle is 2*atan|te_half - a_m|.

The mean line is therefore refit to the traced streamline under an EQUALITY
CONSTRAINT on its TE slope -- it follows the streamline as closely as it still
can while leaving at the angle we ask for. Releasing the TE slope means the mean
line no longer follows the streamline near the trailing edge; that deviation is
real and is reported, and it is the price of not turning the flow at the slot.

Design variables: the TE camber slope, plus 8 CST thickness coefficients.
Constraints: TE included angle <= 10 deg, max t/c ~ 0.09, R_LE >= 0.005,
positive interior thickness, and the slot gap held at what step3v bought.

Run:  python3 step4v_te_tune.py [out.pdf]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import panel2e as M
import step1v_camber as S
import step2v_build as B
from step1v_kulfan import cst_design, cst_line
from flap_displacement import flap_body

ANCHOR = 0.0
OPER = -1.0
ALPHAS = (-1.5, -1.0, -0.5, 0.0)
CP_TARGET = -0.75
N_CAM = B.N_CAM
N_THK = 7                      # 8 coefficients
TE_ANGLE_MAX = 10.0            # degrees, included
GAP_MIN = 0.022


def fit_cst_te(xs, zs, x_end, n, s_te):
    """Least-squares CST fit to (xs, zs) with dz/dx at the trailing edge pinned
    to s_te. Solved as an equality-constrained linear problem (KKT), since the
    model is linear in the coefficients and the constraint is linear too."""
    D = cst_design(xs, x_end, n)
    c = np.zeros(n + 2)
    c[n] = -1.0
    c[n+1] = 1.0/x_end
    A = np.zeros((n + 3, n + 3))
    A[:n+2, :n+2] = D.T @ D
    A[:n+2, n+2] = c
    A[n+2, :n+2] = c
    b = np.concatenate([D.T @ zs, [s_te]])
    sol = np.linalg.solve(A, b)
    par = sol[:n+2]
    return par, float(np.sqrt(np.mean((D @ par - zs)**2)))


def te_angle(a, te_half):
    return 2.0*np.degrees(np.arctan(abs(te_half - a[-1])))


def build(p, xs_s, zs_s, x_end):
    s_te, a = p[0], p[1:]
    cam, rms = fit_cst_te(xs_s, zs_s, x_end, N_CAM, s_te)
    nd, xf, zc, th = B.fore_nodes(a, cam, x_end)
    return nd, cam, rms, th, xf, zc


def cost(p, xs_s, zs_s, x_end, flap_nd, t_target=0.09, rle_min=0.005,
         verbose=False):
    s_te, a = p[0], p[1:]
    te_half = 0.5*B.TE_BASE/x_end
    nd, cam, rms, th, xf, zc = build(p, xs_s, zs_s, x_end)
    pen = 0.0
    ti = th[1:-1]
    if ti.min() < 1e-6*x_end:
        return 1e3
    tmax = 2.0*th.max()/x_end
    pen += 12.0*(tmax - t_target)**2/t_target**2
    rle = 0.5*a[0]**2
    pen += 4.0*max(0.0, rle_min - rle)**2/rle_min**2
    ang = te_angle(a, te_half)
    pen += 6.0*max(0.0, ang - TE_ANGLE_MAX)**2/TE_ANGLE_MAX**2
    gap = float(np.min(np.hypot(flap_nd[:, 0] - nd[0, 0],
                                flap_nd[:, 1] - nd[0, 1])))
    pen += 20.0*max(0.0, GAP_MIN - gap)**2/GAP_MIN**2
    # staying near the traced streamline is still worth something away from the
    # trailing edge; the constraint already forces a deviation near it
    pen += 30.0*(rms/x_end)**2*1e2
    tot = 0.0
    for al in ALPHAS:
        try:
            P, res = M.solve_elements([nd, flap_nd], al)
        except Exception:
            return 1e3
        c, d = M.fore_flatness(P)
        if not np.isfinite(c):
            return 1e3
        # weight the slot-side aft region explicitly: that is the target
        lo, up = M.surfaces(P, 0)
        xn = P.xc[lo]/x_end
        m = xn >= 0.80
        spike = float(-P.Cp[lo][m].min()) if m.any() else 0.0
        tot += c + 0.8*max(0.0, spike - 0.35)
    return tot/len(ALPHAS) + pen


def report(p, xs_s, zs_s, x_end, flap_nd, tag):
    s_te, a = p[0], p[1:]
    te_half = 0.5*B.TE_BASE/x_end
    nd, cam, rms, th, xf, zc = build(p, xs_s, zs_s, x_end)
    ang = te_angle(a, te_half)
    gap = float(np.min(np.hypot(flap_nd[:, 0] - nd[0, 0],
                                flap_nd[:, 1] - nd[0, 1])))
    up_v = nd[-1] - nd[-2]
    lo_v = nd[1] - nd[0]
    print('\n=== %s ===' % tag)
    print('  camber TE slope   %+.4f  (%+.2f deg)' % (s_te, np.degrees(np.arctan(s_te))))
    print('  TE included angle %.2f deg   (upper leaves %+.2f, lower %+.2f)'
          % (ang, np.degrees(np.arctan2(up_v[1], up_v[0])),
             np.degrees(np.arctan2(-lo_v[1], -lo_v[0]))))
    print('  max t/c %.4f at x/c %.3f   R_LE %.5f   gap %.4f'
          % (2*th.max()/x_end, xf[int(np.argmax(th))]/x_end, 0.5*a[0]**2, gap))
    print('  mean line rms from the streamline %.5f (%.4f%% of x_end)'
          % (rms, 100*rms/x_end))
    print('  %-6s %8s %8s %8s %8s %8s %8s' % ('alpha', 'flat', 'rms_up',
                                              'rms_lo', 'Cp_lo_aft', 'CL',
                                              'flapPk'))
    outP = {}
    for al in ALPHAS:
        P, res = M.solve_elements([nd, flap_nd], al)
        c, d = M.fore_flatness(P)
        lo, up = M.surfaces(P, 0)
        xn = P.xc[lo]/x_end
        m = xn >= 0.80
        sp, lvl = M.flap_suction(P)
        print('  %-6.1f %8.4f %8.4f %8.4f %8.3f %8.4f %8.3f'
              % (al, c, d['rms_up'], d['rms_lo'], P.Cp[lo][m].min(), res['Cl'],
                 lvl))
        outP[al] = P
    return nd, cam, th, xf, zc, outP, gap, ang


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'step4v_te_tune.pdf'
    d0 = np.load('step2v_build.npz')
    x_end = float(d0['x_end'])
    thk0 = d0['thk']

    rf = S.RansField('case_A%+05.1f' % ANCHOR)
    xs_s, zs_s = S.trace(rf, 0.0, 0.0, x_end, n=900)
    flap_nd, finfo = flap_body(with_wake=False)

    # baseline: step2v, expressed in the new parameterisation
    te_half = 0.5*B.TE_BASE/x_end
    s_te0 = -d0['cam_par'][N_CAM] + d0['cam_par'][N_CAM+1]/x_end
    p0 = np.concatenate([[s_te0], np.append(thk0, thk0[-1])])
    report(p0, xs_s, zs_s, x_end, flap_nd, 'BASELINE (step2v, anchor 0)')

    # start the search from a sane trailing edge rather than the 43 deg one
    pstart = p0.copy()
    pstart[0] = 0.10                                  # ~5.7 deg camber TE slope
    pstart[-1] = np.tan(np.radians(0.5*TE_ANGLE_MAX)) + te_half
    print('\ntuning [s_te, a_0..a_%d] over alpha = %s ...' % (N_THK, ALPHAS))
    r = minimize(cost, pstart, args=(xs_s, zs_s, x_end, flap_nd),
                 method='Nelder-Mead',
                 options=dict(maxiter=9000, maxfev=12000, xatol=1e-6,
                              fatol=1e-8, adaptive=True))
    print('  cost %.5f -> %.5f in %d evaluations'
          % (cost(pstart, xs_s, zs_s, x_end, flap_nd), r.fun, r.nfev))
    p = r.x
    print('  s_te = %+.5f' % p[0])
    print('  A_thk = %s' % np.array2string(p[1:], precision=5,
                                           floatmode='fixed'))
    nd, cam, th, xf, zc, Ps, gap, ang = report(p, xs_s, zs_s, x_end, flap_nd,
                                               'TUNED')

    np.savez('step4v_te_tune.npz', cam_par=cam, thk=p[1:], s_te=p[0],
             x_end=x_end, anchor=ANCHOR, fore=nd, flap=flap_nd,
             te_angle=ang, gap=gap)
    np.savetxt('step4v_fore.dat', nd, fmt='%12.8f')

    # ------------------------------------------------------------ figure ---
    nd0, cam0, _, th0, xf0, zc0 = build(p0, xs_s, zs_s, x_end)
    fig = plt.figure(figsize=(9.4, 10.4))
    axg = fig.add_axes([0.09, 0.72, 0.88, 0.24])
    axt = fig.add_axes([0.09, 0.44, 0.88, 0.22])
    axc = fig.add_axes([0.09, 0.06, 0.88, 0.32])
    for a_, c_, lbl in ((nd0, '#c44e52', 'baseline (TE 43.0 deg)'),
                        (nd, '#1f4e9c', 'tuned (TE %.1f deg)' % ang)):
        axg.plot(np.append(a_[:, 0], a_[0, 0]), np.append(a_[:, 1], a_[0, 1]),
                 '-', color=c_, lw=1.4, label=lbl)
    axg.plot(np.append(flap_nd[:, 0], flap_nd[0, 0]),
             np.append(flap_nd[:, 1], flap_nd[0, 1]), '-', color='#1a8a5a',
             lw=1.2)
    axg.fill(flap_nd[:, 0], flap_nd[:, 1], color='#1a8a5a', alpha=.10)
    axg.set_aspect('equal'); axg.set_xlim(-0.04, 1.06)
    axg.legend(fontsize=7.5, loc='upper left'); axg.set_xticklabels([])
    axg.set_title('Fore element: trailing-edge fine-tune', fontsize=10)

    for a_, c_, lbl in ((nd0, '#c44e52', 'baseline'), (nd, '#1f4e9c', 'tuned')):
        axt.plot(np.append(a_[:, 0], a_[0, 0]), np.append(a_[:, 1], a_[0, 1]),
                 '-', color=c_, lw=1.6, label=lbl)
    axt.plot(flap_nd[:, 0], flap_nd[:, 1], '-', color='#1a8a5a', lw=1.4)
    axt.set_aspect('equal')
    axt.set_xlim(x_end - 0.13, x_end + 0.10)
    axt.set_ylim(zc[-1] - 0.055, zc[-1] + 0.045)
    axt.legend(fontsize=8); axt.grid(alpha=.25, lw=.6)
    axt.set_title('slot region, zoomed', fontsize=9)

    for al, c_ in zip(ALPHAS, plt.cm.viridis(np.linspace(.05, .85,
                                                         len(ALPHAS)))):
        P = Ps[al]
        lo, up = M.surfaces(P, 0)
        axc.plot(P.xc[up], P.Cp[up], '-', color=c_, lw=1.4,
                 label=r'$\alpha=%+.1f^\circ$' % al)
        axc.plot(P.xc[lo], P.Cp[lo], '--', color=c_, lw=1.1)
    P0, _ = M.solve_elements([nd0, flap_nd], OPER)
    lo0, up0 = M.surfaces(P0, 0)
    axc.plot(P0.xc[lo0], P0.Cp[lo0], ':', color='#c44e52', lw=1.6,
             label=r'baseline lower, $\alpha=-1^\circ$')
    axc.axhline(0, color='0.6', lw=.8)
    axc.invert_yaxis(); axc.set_xlim(-0.02, x_end + 0.02)
    axc.set_xlabel('$x$'); axc.set_ylabel('$C_p$ on the fore element')
    axc.legend(fontsize=7.5, loc='lower left'); axc.grid(alpha=.25, lw=.6)
    axc.set_title('solid upper, dashed lower', fontsize=9)
    fig.savefig(out)
    print('\nwrote %s, step4v_fore.dat, step4v_te_tune.npz' % out)
