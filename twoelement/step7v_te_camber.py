"""STEP 7v -- cancel the residual loading near the fore element's trailing edge
by adjusting the camber there, with the adjustment backed out of thin-airfoil
theory rather than searched for.

The measured loading (alpha = -1) is essentially all in the aft 30%:

    x/c    0.30   0.50   0.70   0.80   0.90   0.95
    dCp   -0.00  +0.029 +0.076 +0.175 +0.253 +0.304

A pressure difference across the surface IS a bound vortex sheet,

    dCp(x) = 2 gamma(x) / V_inf,

so cancelling it means laying down Delta_gamma = -gamma there. The camber slope
that supports a given sheet follows from flow tangency,

    dz/dx = alpha - (1/2pi) PV int gamma(xi)/V / (x - xi) dxi,

so the slope correction required to add Delta_gamma is

    d(dz/dx)(x) = -(1/4pi) PV int d(dCp)(xi) / (x - xi) dxi

where d(dCp) is the CHANGE in loading. To cancel the measured loading,
d(dCp) = -dCp_measured, and the two minus signs cancel:

    d(dz/dx)(x) = +(1/4pi) PV int dCp_measured(xi) / (x - xi) dxi

(Getting this sign wrong is not subtle in its consequences -- the iteration
becomes positive feedback and the loading quadruples in four passes.)

which is a Cauchy principal value, evaluated here on a STAGGERED grid -- the
loading lives at cell centres and the induced slope is evaluated at cell edges,
so the singular point never coincides with a sample and no ad hoc excision is
needed. Integrating the slope gives the camber increment, pinned at the leading
edge.

The induced velocity is nonlocal, so cancelling the trailing-edge loading also
perturbs the camber upstream; that is physics, not error, and the resulting
change is reported. Two or three iterations converge because the relation is
close to linear.

Run:  python3 step7v_te_camber.py [out.pdf]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import panel2e as M
import step2v_build as B
from step1v_kulfan import cst_design

OPER = -1.0
ALPHAS = (-1.5, -1.0, -0.5, 0.0)
X_RAMP0, X_RAMP1 = 0.75, 0.85     # target only the last 15%
X_KUTTA = 0.985                   # taper the last stretch so the sheet ends at 0
N_GRID = 400
RELAX = 0.7
N_ITER = 5


def loading(nd, flap_nd, x_end, alpha=OPER, n=N_GRID):
    """dCp = Cp_lower - Cp_upper on a uniform x/c grid."""
    P, res = M.solve_elements([nd, flap_nd], alpha)
    lo, up = M.surfaces(P, 0)
    g = np.linspace(0.01, 0.999, n)
    cu = np.interp(g, P.xc[up]/x_end, P.Cp[up])
    cl = np.interp(g, P.xc[lo]/x_end, P.Cp[lo])
    return g, cl - cu, P, res


def window(g):
    """Smooth ramp in over [X_RAMP0, X_RAMP1], with a Kutta taper at the end."""
    w = np.clip((g - X_RAMP0)/(X_RAMP1 - X_RAMP0), 0.0, 1.0)
    w = w*w*(3.0 - 2.0*w)                       # smoothstep
    tail = np.clip((1.0 - g)/(1.0 - X_KUTTA), 0.0, 1.0)
    return w*tail


def induced_slope(g, dcp_res, x_end):
    """d(dz/dx)(x) = (1/4pi) PV int dCp_res(xi)/(x - xi) dxi, staggered."""
    xi = 0.5*(g[1:] + g[:-1])                   # cell centres
    f = 0.5*(dcp_res[1:] + dcp_res[:-1])
    dxi = np.diff(g)
    # evaluate at the ORIGINAL nodes, which lie between the centres
    X = g[:, None]
    K = 1.0/(X - xi[None, :])
    return (K*(f*dxi)[None, :]).sum(axis=1)/(4.0*np.pi)


def apply_correction(cam, g, dslope, x_end, n_cam=B.N_CAM):
    """Integrate the slope increment and refit the CST camber to z + dz.

    The increment is pinned at BOTH ends -- dz(0) = dz(x_end) = 0 -- by removing
    the linear part. Without that pin the correction bodily lowers the trailing
    edge instead of reshaping the camber near it: the slot gap collapsed from
    0.0238 to 0.0098 and the trailing-edge slope ran to 36 deg, which is a
    change of geometry, not the local tune that was asked for. Pinning makes it
    a pure shape change that preserves both the extent and the gap.
    """
    x = g*x_end
    dz = np.concatenate([[0.0],
                         np.cumsum(0.5*(dslope[1:] + dslope[:-1])*np.diff(x))])
    dz = dz - dz[-1]*(x - x[0])/(x[-1] - x[0])
    D = cst_design(x, x_end, n_cam)
    z_old = D @ cam
    z_new = z_old + RELAX*dz
    par, *_ = np.linalg.lstsq(D, z_new, rcond=1e-12)
    return par, dz


def report(nd, flap_nd, x_end, tag):
    print('\n=== %s ===' % tag)
    print('  %-7s %8s %9s %9s %8s %8s'
          % ('alpha', 'flat', 'dCp>0.7', 'dCpmaxTE', 'CL', 'flapPk'))
    out = {}
    for al in ALPHAS:
        g, dcp, P, res = loading(nd, flap_nd, x_end, al)
        c, d = M.fore_flatness(P)
        m = g > 0.70
        _, lvl = M.flap_suction(P)
        out[al] = (g, dcp, P)
        print('  %-7.1f %8.4f %+9.4f %+9.4f %8.4f %8.3f'
              % (al, c, dcp[m].mean(), dcp[m][int(np.argmax(np.abs(dcp[m])))],
                 res['Cl'], lvl))
    return out


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'step7v_te_camber.pdf'
    d = np.load('step6v_final.npz')
    x_end = float(d['x_end'])
    thk, cam0, flap_nd = d['thk'], d['cam_par'], d['flap']

    nd0, _, zc0, _ = B.fore_nodes(thk, cam0, x_end)
    report(nd0, flap_nd, x_end, 'BEFORE (step6v)')

    cam = cam0.copy()
    hist = []
    prev_mean = np.inf
    for it in range(N_ITER):
        nd, xf, zc, th = B.fore_nodes(thk, cam, x_end)
        g, dcp, P, res = loading(nd, flap_nd, x_end)
        res_load = dcp*window(g)
        dslope = induced_slope(g, res_load, x_end)
        cam_new, dz = apply_correction(cam, g, dslope, x_end)
        m = g > 0.85
        s_te = -cam[B.N_CAM] + cam[B.N_CAM+1]/x_end
        gp = float(np.min(np.hypot(flap_nd[:, 0] - nd[0, 0],
                                   flap_nd[:, 1] - nd[0, 1])))
        print('iter %d: mean dCp(x/c>0.85) %+.4f  max %+.4f  |  TE slope '
              '%.2f deg, gap %.4f, max|dz| %.5f'
              % (it, dcp[m].mean(), dcp[m][int(np.argmax(np.abs(dcp[m])))],
                 np.degrees(np.arctan(s_te)), gp, np.abs(dz).max()))
        hist.append((g.copy(), dcp.copy(), dz.copy()))
        if it > 0 and abs(dcp[m].mean()) > abs(prev_mean)*1.05:
            print('  residual growing -- stopping before it runs away')
            break
        prev_mean = dcp[m].mean()
        cam = cam_new

    nd, xf, zc, th = B.fore_nodes(thk, cam, x_end)
    s_te = -cam[B.N_CAM] + cam[B.N_CAM+1]/x_end
    gap = float(np.min(np.hypot(flap_nd[:, 0] - nd[0, 0],
                                flap_nd[:, 1] - nd[0, 1])))
    after = report(nd, flap_nd, x_end, 'AFTER')
    print('\n  camber TE slope %.2f deg (was %.2f), gap %.4f, max t/c %.4f'
          % (np.degrees(np.arctan(s_te)),
             np.degrees(np.arctan(-cam0[B.N_CAM] + cam0[B.N_CAM+1]/x_end)),
             gap, 2*th.max()/x_end))

    np.savez('step7v_final.npz', cam_par=cam, thk=thk, x_end=x_end, fore=nd,
             flap=flap_nd, s_te=s_te, gap=gap)
    np.savetxt('step7v_fore.dat', nd, fmt='%12.8f')

    # ------------------------------------------------------------- figure --
    fig, ax = plt.subplots(3, 1, figsize=(9.2, 10.0))
    g0, dcp0, _ = hist[0][0], hist[0][1], None
    ax[0].plot(g0, dcp0, '-', color='#c44e52', lw=1.8, label='before')
    for k, (gg, dd, _) in enumerate(hist[1:], start=1):
        ax[0].plot(gg, dd, '-', lw=1.0, alpha=.5,
                   color=plt.cm.Blues(0.3 + 0.2*k), label='iter %d' % k)
    gA, dcpA, _ = after[OPER]
    ax[0].plot(gA, dcpA, '-', color='#1f4e9c', lw=2.0, label='after')
    ax[0].axhline(0, color='0.6', lw=.8)
    ax[0].axvspan(X_RAMP0, 1.0, color='#d97706', alpha=.10)
    ax[0].set_ylabel(r'$\Delta C_p = C_{p,l}-C_{p,u}$')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.25, lw=.6)
    ax[0].set_title(r'residual loading on the fore element, $\alpha=-1^\circ$',
                    fontsize=10)

    ax[1].plot(g0*x_end, hist[0][2], '-', color='#1f4e9c', lw=1.6,
               label='camber increment, iteration 0')
    tot = sum(h[2] for h in hist)
    ax[1].plot(g0*x_end, tot, '-', color='#000', lw=1.8, label='total')
    ax[1].axhline(0, color='0.6', lw=.8)
    ax[1].set_ylabel(r'$\delta z$'); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.25, lw=.6)

    for al, c_ in zip(ALPHAS, plt.cm.viridis(np.linspace(.05, .85,
                                                         len(ALPHAS)))):
        P = after[al][2]
        lo, up = M.surfaces(P, 0)
        ax[2].plot(P.xc[up], P.Cp[up], '-', color=c_, lw=1.4,
                   label=r'$\alpha=%+.1f^\circ$' % al)
        ax[2].plot(P.xc[lo], P.Cp[lo], '--', color=c_, lw=1.1)
    ax[2].axhline(0, color='0.6', lw=.8); ax[2].invert_yaxis()
    ax[2].set_xlabel('$x$'); ax[2].set_ylabel('$C_p$')
    ax[2].legend(fontsize=8, loc='lower left'); ax[2].grid(alpha=.25, lw=.6)
    ax[2].set_title('solid upper, dashed lower', fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    print('\nwrote %s, step7v_fore.dat, step7v_final.npz' % out)
