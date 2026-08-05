"""STEP 10 -- can an ANALYTIC thickness form cancel the alpha=-2 adverse pocket?

Physics (user, 2026-08-04): the thickness-induced velocity increment u_t rises
toward the maximum-thickness station. So an adverse pocket at x_a is cancelled
by putting max thickness JUST DOWNSTREAM of x_a -- the rising u_t there
supplies the missing acceleration.

alpha = -2 has an adverse pocket at x = 0.50-0.60 (of 0.700 chord), i.e. at
x/c_fore = 0.71-0.86. That would want x_m ~ 0.85, far aft of anything the
classic NACA families offer (modified 4-digit and 6-series both stop at 0.60).

This script asks the question numerically rather than by assertion:
  * required u_t shape for each alpha,
  * best-matching x_m, swept PAST the classic limit,
  * what pushing x_m aft costs at the trailing edge (the aft polynomial's
    TE slope d1 grows fast, which is exactly the recovery we already fight).

Run:  python3 step10_xm_limit.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import panel2e as M
from step9_extent import flap_at, streamline_and_cp, fit4

CHORD = 0.700
XM_CLASSIC = 0.60            # NACA modified 4-digit / 6-series upper limit
WIN = (0.05, 0.95)


def u_thick(t, xm, ik=6.0, km=0.7, npan=300):
    nd = M.airfoil_nodes(npan, 0.0, 0.0, t, modified=True, xm=xm, ik=ik,
                         km=km, le_blend=0.0)
    nd = M.place(nd, CHORD, 0.0, 0.0, 0.0)
    P, _ = M.solve_elements([nd], 0.0)
    _, up = M.surfaces(P, 0)
    return P.xc[up]/CHORD, np.abs(P.Vt[up]) - 1.0


def te_slope(t, xm, km=0.7):
    """d1 * t = the trailing-edge thickness slope of the aft polynomial."""
    wm = 1.0 - xm
    Km = min(km/(xm*xm), 2.5/(wm*wm))
    d3 = (0.5 - 0.5*Km*wm*wm)/wm**3
    d1 = Km*wm + 3.0*d3*wm*wm
    return d1*t


def match(xn_req, req, xn, ut):
    m = (xn >= WIN[0]) & (xn <= WIN[1])
    r = np.interp(xn[m], xn_req, req)
    u = ut[m]
    r0, u0 = r - r.mean(), u - u.mean()
    s = float(np.dot(r0, u0)/np.dot(u0, u0))
    return s, float(np.sqrt(np.mean((r0 - s*u0)**2))), \
        float(np.corrcoef(r0, u0)[0, 1])


if __name__ == '__main__':
    reqs = {}
    for a in (-2.0, -1.0):
        nd, P, res = flap_at(a)
        xs, zs, cp = streamline_and_cp(P)
        x, zf, dzf, par, rms = fit4(xs, zs, CHORD)
        Vx, Vz = M.field_velocity(x, zf, P)
        Vb = np.hypot(Vx, Vz)
        reqs[a] = (x/CHORD, -Vb, Vb)
        g = np.gradient(1 - Vb**2, x/CHORD)
        print('alpha=%+.0f : V_base %.3f -> %.3f ; max adverse dCp/dx(fore) %.3f'
              % (a, Vb[0], Vb[-1], np.clip(g, 0, None).max()))

    print('\n%6s | %-38s | %-38s' % ('x_m', 'alpha = -2   (scale  rms   corr)',
                                     'alpha = -1   (scale  rms   corr)'))
    print('-'*88)
    best = {a: None for a in reqs}
    for xm in (0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85):
        xn, ut = u_thick(0.08, xm)
        line = '%6.2f |' % xm
        for a in (-2.0, -1.0):
            s, rms, corr = match(reqs[a][0], reqs[a][1], xn, ut)
            line += ' %8.3f %8.4f %8.4f    |' % (0.08*s, rms, corr)
            if best[a] is None or rms < best[a][1]:
                best[a] = (xm, rms, corr, 0.08*s)
        flag = '' if xm <= XM_CLASSIC else '   <-- beyond the classic families'
        print(line + flag)

    print()
    for a in (-2.0, -1.0):
        xm, rms, corr, tneed = best[a]
        print('alpha=%+.0f best x_m = %.2f  (rms %.4f, corr %.3f, t/c %.3f) %s'
              % (a, xm, rms, corr, tneed,
                 'IN family' if xm <= XM_CLASSIC else 'OUT of family'))

    print('\ncost of pushing x_m aft -- trailing-edge thickness slope at t/c=0.08:')
    print('%8s %14s %14s' % ('x_m', 'TE slope', 'vs x_m=0.50'))
    ref = te_slope(0.08, 0.50)
    for xm in (0.40, 0.50, 0.60, 0.70, 0.80, 0.85):
        s = te_slope(0.08, xm)
        print('%8.2f %14.4f %13.1fx' % (xm, s, s/ref))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, c in ((-2.0, '#0891b2'), (-1.0, '#d97706')):
        xn_r, req, _ = reqs[a]
        m = (xn_r >= WIN[0]) & (xn_r <= WIN[1])
        ax[0].plot(xn_r[m], req[m] - req[m].mean(), '-', color=c, lw=2.2,
                   label='required, $\\alpha$=%.0f$^\\circ$' % a)
    for xm, ls in ((0.50, '--'), (0.60, '-.'), (0.85, ':')):
        xn, ut = u_thick(0.08, xm)
        m = (xn >= WIN[0]) & (xn <= WIN[1])
        s, _, _ = match(reqs[-2.0][0], reqs[-2.0][1], xn, ut)
        ax[0].plot(xn[m], s*(ut[m] - ut[m].mean()), ls, color='0.35', lw=1.3,
                   label='$x_m$=%.2f' % xm)
    ax[0].set_xlabel('$x/c_{fore}$'); ax[0].set_ylabel('$u_t$ (mean removed)')
    ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3)
    ax[0].set_title('required vs available thickness perturbation', fontsize=10)

    xms = np.linspace(0.30, 0.88, 25)
    ax[1].plot(xms, [te_slope(0.08, x) for x in xms], '-', color='#b91c1c', lw=2)
    ax[1].axvline(XM_CLASSIC, color='0.4', ls='--', lw=1.4,
                  label='classic NACA limit $x_m$=0.60')
    ax[1].set_xlabel('$x_m$'); ax[1].set_ylabel('TE thickness slope')
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    ax[1].set_title('what aft $x_m$ costs at the trailing edge', fontsize=10)
    fig.tight_layout()
    fig.savefig('step10_xm_limit.pdf'); fig.savefig('step10_xm_limit.png', dpi=125)
    print('\nwrote step10_xm_limit.pdf / .png')
