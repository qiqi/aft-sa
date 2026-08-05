"""STEP 5v -- reshape the fore element's thickness taper to a prescription
rather than to an optimiser's taste:

    keep the ORIGINAL (step2v) distribution out to x = 0.4
    then thin gradually
    with the largest shrinkage rate as far aft as it will go (x = 0.68)
    ending at the 10 deg included trailing-edge wedge

All stations are GLOBAL x, the coordinate every plot in this thread uses; the
fore element runs x = 0 to x_end = 0.7039, so x = 0.4 is 57% of it and
x = 0.68 is 97%.

Holding the original distribution PAST x = 0.4 was tried and does not pay: it
forces the same total thinning into a shorter run, and flatness goes 1.23 (hold
to 0.40) -> 1.32 (0.45) -> 1.52 (0.50) -> 1.72 (0.55). Pushing the peak aft is
what buys the improvement; holding longer works against it.

Rather than fit a target and hope, the SLOPE profile is built directly, which is
the only way to place the maximum shrinkage rate exactly:

    t'(x) is interpolated through three anchors --
        x = 0.4      t' = t0'(0.4)                (slope continuity)
        x = x_peak   t' = m_min                   (the maximum shrinkage rate)
        x = x_end    t' = -tan(5 deg)             (the 10 deg wedge)
    with a Beta-shaped bump peaking at x_peak, and m_min set by the requirement
    that the integral lands on the trailing-edge half-thickness.

The resulting distribution is then fitted back into CST form so the geometry
stays in one parameterisation, and the fit error is reported rather than assumed.

Run:  python3 step5v_thickness_shape.py [out.pdf]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import comb

import panel2e as M
import step1v_camber as S
import step2v_build as B
import step4v_te_tune as T
from step1v_kulfan import cst_line
from flap_displacement import flap_body

X_HOLD = 0.40          # keep the original distribution out to here (global x)
X_PEAK = 0.680         # maximum shrinkage rate (global x). Swept: everything
                       # improves monotonically as this moves aft -- flatness
                       # 1.89 (0.625) -> 1.54 (0.650) -> 1.23 (0.680) -- and
                       # then collapses beyond it, because the remaining
                       # thinning has to fit into a shorter run before the 10 deg
                       # wedge and the rate runs away: d(2t)/dx = -0.62 at 0.680,
                       # -0.91 at 0.690, -1.30 at 0.695 (flatness 3.27).
Q_BUMP = 1.0           # bump width. Narrow bumps (q=3) concentrate the same
                       # total thinning into a shorter run and cost ~0.5 in
                       # flatness; q=1 spreads it and recovers most of that.
TE_ANGLE = 10.0        # included, degrees
N_FIT = 9              # CST order for refitting the prescribed thickness
ALPHAS = T.ALPHAS
OPER = -1.0


def t0_profile(thk, cam, x_end, npts=1201):
    """The original (step2v) half-thickness on a dense uniform x grid."""
    xh = np.linspace(0.0, 1.0, npts)
    te_half = 0.5*B.TE_BASE/x_end
    return xh*x_end, B.half_thickness(thk, xh, te_half)*x_end


def beta_bump(s, s_p, q=Q_BUMP):
    """Bump on [0,1], zero at both ends, peak 1 at s_p."""
    p = q*s_p/(1.0 - s_p)
    out = np.zeros_like(s)
    m = (s > 0) & (s < 1)
    out[m] = (s[m]/s_p)**p*((1.0 - s[m])/(1.0 - s_p))**q
    return out


def prescribed(thk0, cam, x_end, x_hold=X_HOLD, x_peak=X_PEAK,
               te_angle=TE_ANGLE):
    """Half-thickness following the prescription. Returns (x, t, t')."""
    x0, t0 = t0_profile(thk0, cam, x_end)
    dt0 = np.gradient(t0, x0)
    ia = int(np.argmin(np.abs(x0 - x_hold)))
    xa, ta, ma = x0[ia], t0[ia], dt0[ia]
    t_te = 0.5*B.TE_BASE
    m_te = -np.tan(np.radians(0.5*te_angle))

    xs = x0[ia:]
    s = (xs - xa)/(x_end - xa)
    s_p = (x_peak - xa)/(x_end - xa)
    Bmp = beta_bump(s, s_p, q=Q_BUMP)

    def t_end(m_min):
        # t' = linear from ma to m_te, plus a bump of depth (m_min - blend)
        lin = ma + (m_te - ma)*s
        base = lin[np.argmin(np.abs(s - s_p))]
        dt = lin + (m_min - base)*Bmp
        return np.trapezoid(dt, xs) + ta, dt

    lo, hi = -8.0, ma
    f = lambda m: t_end(m)[0] - t_te
    if f(lo)*f(hi) > 0:
        raise RuntimeError('no bracketing shrinkage rate: f(%.2f)=%.4g '
                           'f(%.2f)=%.4g' % (lo, f(lo), hi, f(hi)))
    m_min = brentq(f, lo, hi, xtol=1e-12)
    _, dt = t_end(m_min)
    t_aft = ta + np.concatenate([[0.0], np.cumsum(0.5*(dt[1:] + dt[:-1])
                                                 * np.diff(xs))])
    x = np.concatenate([x0[:ia], xs])
    t = np.concatenate([t0[:ia], t_aft])
    return x, t, np.gradient(t, x), m_min, (xa, ta, ma)


def fit_thickness_cst(x, t, x_end, n=N_FIT):
    """Least squares CST coefficients for a prescribed half-thickness, with the
    trailing-edge slope pinned so the 10 deg wedge survives the refit."""
    xh = np.clip(x/x_end, 0.0, 1.0)
    te_half = 0.5*B.TE_BASE/x_end
    i = np.arange(n + 1)[:, None]
    Bm = comb(n, i)*xh[None, :]**i*(1.0 - xh[None, :])**(n - i)
    D = (np.sqrt(np.maximum(xh, 0.0))*(1.0 - xh))[:, None]*Bm.T*x_end
    rhs = t - xh*te_half*x_end
    # constraint: dt/dx at the TE = te_half - a_n  ->  a_n fixed
    m_te = -np.tan(np.radians(0.5*TE_ANGLE))
    a_n = te_half - m_te
    rhs = rhs - D[:, n]*a_n
    a_rest, *_ = np.linalg.lstsq(D[:, :n], rhs, rcond=1e-12)
    a = np.concatenate([a_rest, [a_n]])
    fit = D @ a + xh*te_half*x_end
    return a, float(np.sqrt(np.mean((fit - t)**2)))


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'step5v_thickness_shape.pdf'
    d2 = np.load('step2v_build.npz')
    d4 = np.load('step4v_te_tune.npz')
    x_end = float(d2['x_end'])
    flap_nd, _ = flap_body(with_wake=False)
    rf = S.RansField('case_A%+05.1f' % T.ANCHOR)
    xs_s, zs_s = S.trace(rf, 0.0, 0.0, x_end, n=900)

    # camber from the step4v tune (the trailing edge you said looks right)
    s_te = float(d4['s_te'])
    cam, cam_rms = T.fit_cst_te(xs_s, zs_s, x_end, B.N_CAM, s_te)
    print('camber: TE slope %+.2f deg (from step4v), rms from streamline %.5f'
          % (np.degrees(np.arctan(s_te)), cam_rms))

    x, t, dt, m_min, anch = prescribed(d2['thk'], cam, x_end)
    xa, ta, ma = anch
    print('\nPRESCRIBED half-thickness (global x, x_end = %.4f)' % x_end)
    print('  held to x = %.3f  (t = %.5f, slope %+.4f there)' % (xa, ta, ma))
    print('  maximum shrinkage rate  dt/dx = %+.4f  at x = %.3f'
          % (dt[int(np.argmin(dt))], x[int(np.argmin(dt))]))
    print('  trailing edge: t = %.5f, dt/dx = %+.4f -> included angle %.2f deg'
          % (t[-1], dt[-1], 2*np.degrees(np.arctan(abs(dt[-1])))))
    print('  max t/c %.4f at x = %.3f' % (2*t.max()/x_end, x[int(np.argmax(t))]))

    a, frms = fit_thickness_cst(x, t, x_end)
    print('\nCST refit (n=%d): rms %.3e (%.4f%% of x_end)'
          % (N_FIT, frms, 100*frms/x_end))
    print('  A = %s' % np.array2string(a, precision=5, floatmode='fixed'))

    nd, xf, zc, th = B.fore_nodes(a, cam, x_end)
    gap = float(np.min(np.hypot(flap_nd[:, 0] - nd[0, 0],
                                flap_nd[:, 1] - nd[0, 1])))
    te_half = 0.5*B.TE_BASE/x_end
    ang = 2*np.degrees(np.arctan(abs(te_half - a[-1])))
    print('  realised: max t/c %.4f at x %.3f, TE angle %.2f deg, gap %.4f'
          % (2*th.max()/x_end, xf[int(np.argmax(th))], ang, gap))

    print('\n%-7s %8s %8s %8s %10s %8s %8s'
          % ('alpha', 'flat', 'rms_up', 'rms_lo', 'Cp_lo_aft', 'CL', 'flapPk'))
    Ps = {}
    for al in ALPHAS:
        P, res = M.solve_elements([nd, flap_nd], al)
        c, d = M.fore_flatness(P)
        lo, up = M.surfaces(P, 0)
        xn = P.xc[lo]/x_end
        m = xn >= 0.80
        sp, lvl = M.flap_suction(P)
        Ps[al] = P
        print('%-7.1f %8.4f %8.4f %8.4f %10.3f %8.4f %8.3f'
              % (al, c, d['rms_up'], d['rms_lo'], P.Cp[lo][m].min(), res['Cl'],
                 lvl))

    np.savez('step5v_thickness_shape.npz', thk=a, cam_par=cam, s_te=s_te,
             x_end=x_end, fore=nd, flap=flap_nd, te_angle=ang, gap=gap,
             x_hold=X_HOLD, x_peak=X_PEAK)
    np.savetxt('step5v_fore.dat', nd, fmt='%12.8f')

    # ------------------------------------------------------------ figure ---
    x0, t0 = t0_profile(d2['thk'], cam, x_end)
    nd4 = d4['fore']
    # fore_nodes returns its grid COSINE-spaced; pairing it with a linspace
    # misplaces every point, so take the x it actually returns.
    _, x4, _, th4 = B.fore_nodes(d4['thk'], d4['cam_par'], x_end)

    fig, ax = plt.subplots(4, 1, figsize=(9.2, 12.0),
                           gridspec_kw=dict(height_ratios=[1.0, 1.0, 1.0, 1.2]))
    ax[0].plot(x0, 2*t0, '-', color='#c44e52', lw=1.6, label='original (step2v)')
    ax[0].plot(x4, 2*th4, '-.', color='#7f7f7f', lw=1.3,
               label='step4v optimiser')
    ax[0].plot(x, 2*t, '-', color='#1f4e9c', lw=2.0, label='prescribed')
    ax[0].axvline(X_HOLD, color='#2ca02c', ls=':', lw=1.2)
    ax[0].axvline(X_PEAK, color='#d97706', ls='--', lw=1.3)
    ax[0].set_ylabel('thickness $2t$'); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.25, lw=.6)
    ax[0].set_title('hold to $x=%.2f$ (green dotted), peak shrinkage rate at '
                    '$x=%.3f$ (amber dashed)' % (X_HOLD, X_PEAK), fontsize=10)

    ax[1].plot(x0, np.gradient(2*t0, x0), '-', color='#c44e52', lw=1.4,
               label='original')
    ax[1].plot(x4, np.gradient(2*th4, x4), '-.', color='#7f7f7f', lw=1.2,
               label='step4v optimiser')
    ax[1].plot(x, 2*dt, '-', color='#1f4e9c', lw=1.8, label='prescribed')
    ax[1].axvline(X_HOLD, color='#2ca02c', ls=':', lw=1.2)
    ax[1].axvline(X_PEAK, color='#d97706', ls='--', lw=1.3)
    ax[1].axvline(x[int(np.argmin(dt))], color='#1f4e9c', ls='--', lw=1.0)
    ax[1].set_ylim(-0.6, 0.6)
    ax[1].set_ylabel(r'shrinkage rate $d(2t)/dx$'); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.25, lw=.6)

    ax[2].plot(np.append(nd[:, 0], nd[0, 0]), np.append(nd[:, 1], nd[0, 1]),
               '-', color='#1f4e9c', lw=1.5, label='prescribed')
    ax[2].plot(np.append(nd4[:, 0], nd4[0, 0]),
               np.append(nd4[:, 1], nd4[0, 1]), '-.', color='#7f7f7f', lw=1.2,
               label='step4v')
    ax[2].plot(flap_nd[:, 0], flap_nd[:, 1], '-', color='#1a8a5a', lw=1.2)
    ax[2].set_aspect('equal'); ax[2].set_xlim(-0.04, 1.06)
    ax[2].legend(fontsize=8); ax[2].set_ylabel('$z$')

    for al, c_ in zip(ALPHAS, plt.cm.viridis(np.linspace(.05, .85,
                                                         len(ALPHAS)))):
        P = Ps[al]
        lo, up = M.surfaces(P, 0)
        ax[3].plot(P.xc[up], P.Cp[up], '-', color=c_, lw=1.4,
                   label=r'$\alpha=%+.1f^\circ$' % al)
        ax[3].plot(P.xc[lo], P.Cp[lo], '--', color=c_, lw=1.1)
    ax[3].axhline(0, color='0.6', lw=.8); ax[3].invert_yaxis()
    ax[3].set_xlim(-0.02, x_end + 0.02)
    ax[3].set_xlabel('$x$'); ax[3].set_ylabel('$C_p$ on the fore element')
    ax[3].legend(fontsize=8, loc='lower left'); ax[3].grid(alpha=.25, lw=.6)
    ax[3].set_title('solid upper, dashed lower', fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    print('\nwrote %s, step5v_fore.dat, step5v_thickness_shape.npz' % out)
