"""STEP 1v -- the flap alone, VISCOUS, at its own local Reynolds number.

Why this replaces the inviscid step 1. The fore element's mean line was traced
along a streamline of the INVISCID flap-alone field. At Re_flap = 3e5 the flap
carries a thick layer and a separation bubble near its trailing edge, so its
real circulation is far below the inviscid Kutta value. A mean line traced
through the inviscid streamline therefore over-curves, and the fore element has
to impose the difference -- which shows up as the leading-edge suction spike
measured on its lower surface (Cp ~ -0.85 instead of a flat rooftop).

Reynolds number. The pipeline sets Re = 1e6 per unit length with the TOTAL chord
at ~1, and the flap chord is 0.3, so the flap's own chord Reynolds number is
3e5 -- not 1e6. That factor of 3.3 matters a great deal for the bubble.

Angle of attack. The flap chord line sits at -8 deg in the global frame
(place() rotates by +incidence, and inc = -8). The freestream is at
alpha = -1 deg. The flap therefore sees +7 deg.

Solver: mfoil (Fidkowski), the XFOIL-faithful viscous coupled panel method
vendored at flexfoil/mfoil. ncrit = 9 to match the SA-AI seed
chi_inf = c_v1 exp(-9) used in the RANS ladder.

Run:  python3 step1v_flap_viscous.py
"""
import sys

import numpy as np

sys.path.insert(0, '/home/qiqi/flexcompute/flexfoil/mfoil')
import mfoil as MF                                            # noqa: E402
import panel2e as M                                           # noqa: E402

FLAP = dict(m=0.09, p=0.40, t=0.16)      # NACA 9416
FLAP_CHORD = 0.30
FLAP_INC = -8.0
ALPHA_GLOBAL = -1.0
AOA = ALPHA_GLOBAL - FLAP_INC            # +7 deg on the flap's own chord line
RE_UNIT = 1.0e6
RE_FLAP = RE_UNIT*FLAP_CHORD             # 3e5
MACH = 0.10
NCRIT = 9.0


def _set_coords(Mo, X):
    """mfoil.set_coords is unusable as shipped: it calls X.shape(1) instead of
    X.shape[1], and its orientation test flips the points the wrong way -- it
    turns the ordering mfoil actually wants into the one build_wake rejects.

    What mfoil needs is what its own naca_points() produces: lower TE -> LE ->
    upper TE, trailing-edge point repeated if sharp. build_wake forms
    t = (n[1], -n[0]) from n = x[-1] - x[0] and asserts t[0] > 0, which is
    exactly the statement that the LAST point is the upper trailing edge.
    panel2e.airfoil_nodes already returns that order, so store as given and
    only flip if the test fails.

    The orientation test must NOT compare the two trailing-edge endpoints: with
    a closed trailing edge both sit at z = 0 (+0.0 and -0.0), so the comparison
    is meaningless. Compare mid-surface points instead."""
    X = np.asarray(X, float)
    if X.shape[0] > X.shape[1]:
        X = X.T
    n = X.shape[1]
    if X[1, 3*n//4] < X[1, n//4]:           # second half must be the upper side
        X = np.fliplr(X)
    Mo.geom.npoint = X.shape[1]
    Mo.geom.xpoint = X
    Mo.geom.chord = X[0, :].max() - X[0, :].min()


MF.set_coords = _set_coords


TE_BASE = 0.003                          # total-chord units, as the mesher uses
TE_THICK = TE_BASE/FLAP_CHORD            # 0.01 in the flap's own chord units


def flap_own_frame(n_panels=260):
    """Flap section in its own chord frame: chord 1, LE at origin.

    The trailing edge carries the same finite base the CFD geometry has
    (0.003 total chord = 0.01 flap chord). mfoil needs that anyway: build_wake
    forms the wake direction from n = x[-1] - x[0] and asserts n[1] > 0, which
    a perfectly sharp trailing edge cannot satisfy.
    """
    return M.airfoil_nodes(n_panels, FLAP['m'], FLAP['p'], FLAP['t'],
                           modified=False, te_thick=TE_THICK)


def run(aoa, re, visc, ncrit=NCRIT, npanel=259):
    m = MF.mfoil(coords=flap_own_frame(), npanel=npanel)
    m.param.doplot = False
    m.param.verb = 0
    m.param.ncrit = ncrit
    m.setoper(alpha=aoa, Ma=MACH, visc=visc, **({'Re': re} if visc else {}))
    m.oper.viscous = visc
    m.solve()
    return m


def summarise(tag, m, visc):
    p = m.post
    print('%-26s cl = %8.4f   cm = %8.4f' % (tag, p.cl, p.cm), end='')
    if visc:
        print('   cd = %.5f (cdf %.5f, cdp %.5f)' % (p.cd, p.cdf, p.cdp))
        Xt = np.asarray(m.vsol.Xt)
        print('%-26s transition x/c: lower %.4f   upper %.4f'
              % ('', Xt[0, 1], Xt[1, 1]))
        ds, th = np.asarray(p.ds), np.asarray(p.th)
        Hk = np.asarray(p.Hk)
        Is = m.vsol.Is
        for k, side in ((0, 'lower'), (1, 'upper')):
            idx = np.asarray(Is[k])
            print('%-26s %-6s  delta*_TE = %.5f  theta_TE = %.5f  '
                  'Hk max = %.2f' % ('', side, ds[idx][-1], th[idx][-1],
                                     Hk[idx].max()))
    else:
        print()
    return p.cl


if __name__ == '__main__':
    print('FLAP alone: NACA %d%d%02d, chord %.2f of total, inc %+.1f deg'
          % (FLAP['m']*100, FLAP['p']*10, FLAP['t']*100, FLAP_CHORD, FLAP_INC))
    print('freestream alpha %+.1f deg  ->  flap sees %+.1f deg'
          % (ALPHA_GLOBAL, AOA))
    print('Re per unit length %.1e, flap chord %.2f  ->  Re_flap = %.2e'
          % (RE_UNIT, FLAP_CHORD, RE_FLAP))
    print('ncrit = %.1f, M = %.2f\n' % (NCRIT, MACH))

    cl_inv = summarise('INVISCID', run(AOA, None, False), False)
    mv = run(AOA, RE_FLAP, True)
    cl_vis = summarise('VISCOUS Re=3e5', mv, True)

    print('\nCIRCULATION DEFICIT')
    print('  cl inviscid %.4f  ->  viscous %.4f   = %.1f%% of inviscid'
          % (cl_inv, cl_vis, 100*cl_vis/cl_inv))
    print('  the streamline through the fore LE is set by this circulation,')
    print('  so tracing it from the inviscid field over-turns the flow by ~%.0f%%'
          % (100*(cl_inv/cl_vis - 1)))

    # what the WRONG Reynolds number would have told us, for the record
    cl_1e6 = run(AOA, RE_UNIT, True).post.cl
    print('\n  for reference, viscous at Re=1e6 (the total-chord value, wrong')
    print('  for this element): cl = %.4f  -- %.1f%% of inviscid'
          % (cl_1e6, 100*cl_1e6/cl_inv))

    np.savez('step1v_flap_viscous.npz',
             cl_inv=cl_inv, cl_vis=cl_vis, aoa=AOA, re_flap=RE_FLAP,
             x=np.asarray(mv.post.ue)*0 + np.asarray(mv.post.ue),
             ds=np.asarray(mv.post.ds), th=np.asarray(mv.post.th),
             Hk=np.asarray(mv.post.Hk), cp=np.asarray(mv.post.cp),
             cpi=np.asarray(mv.post.cpi))
    print('\nwrote step1v_flap_viscous.npz')
