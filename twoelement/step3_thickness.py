"""STEP 3 -- is there a CLASSIC thickness distribution whose thin-airfoil
velocity perturbation matches what the pressure distribution demands?

Settled so far (user, 2026-08-04):
    flap        NACA 9416, chord 0.30, LE (0.70, 0.04), inc -8 deg
    mean line   NACA 4-digit, m = -3.5% at p = 70%  (the exact fit was -3.58%;
                -4% would nearly DOUBLE the residual cross-camber velocity,
                -3.5% does not)
    thickness   target t/c = 0.10, leaving ~5% of the 15% "constant-Cp"
                requirement as margin for local adverse-gradient cancellation

Question: along the mean line the base (flap-alone) flow accelerates from
V/Vinf ~ 1.00 to 1.305. For the surface pressure to stay flat the thickness
must supply a velocity perturbation u_t(x) = C - V_base(x) -- flat over the
front, dropping over the aft third. Which classic thickness form has that
SHAPE?

Run:  python3 step3_thickness.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import panel2e as M
from step1_flap_and_camber import (ALPHA, FORE_CHORD, flap_nodes)
from step2_questions import line_from, refit_inc

M_CAM, P_CAM, T_TARGET = -0.035, 0.70, 0.10
WIN = (0.05, 0.95)          # fraction of fore chord used for shape matching


def u_thick(t, xm=None, ik=6.0, km=1.1, npan=260):
    """Thickness-alone surface velocity perturbation, symmetric, uniform flow.
    xm=None -> standard (unmodified) NACA 4-digit thickness."""
    mod = xm is not None
    nd = M.airfoil_nodes(npan, 0.0, 0.0, t, modified=mod,
                         xm=(xm or 0.30), ik=ik, km=km, le_blend=0.0)
    nd = M.place(nd, FORE_CHORD, 0.0, 0.0, 0.0)
    P, _ = M.solve_elements([nd], 0.0)
    _, up = M.surfaces(P, 0)
    return P.xc[up]/FORE_CHORD, np.abs(P.Vt[up]) - 1.0


def match(xn_req, req, xn, ut):
    """Best scale s so that s*ut ~ req (both mean-removed) on the window.
    Returns (scale, rms, corr). scale ~ how much of the candidate's t/c is
    needed; rms is the shape mismatch that thickness CANNOT remove."""
    m = (xn >= WIN[0]) & (xn <= WIN[1])
    r = np.interp(xn[m], xn_req, req)
    u = ut[m]
    r0, u0 = r - r.mean(), u - u.mean()
    s = float(np.dot(r0, u0)/np.dot(u0, u0))
    res = r0 - s*u0
    return s, float(np.sqrt(np.mean(res**2))), float(
        np.corrcoef(r0, u0)[0, 1])


if __name__ == '__main__':
    n2 = flap_nodes()
    Pf, _ = M.solve_elements([n2], ALPHA)
    xs, zs = M.streamline_camber(Pf, 0.0, 0.0, FORE_CHORD, n=400)
    inc = refit_inc(M_CAM, P_CAM, xs, zs)
    zc = line_from(M_CAM, P_CAM, inc, xs)
    Vx, Vz = M.field_velocity(xs, zc, Pf)
    Vb = np.hypot(Vx, Vz)
    xn_req = xs/FORE_CHORD
    req = -Vb                       # u_t must be proportional to -V_base
    print('mean line NACA m=%+.3f p=%.2f at inc %+.2f deg' % (M_CAM, P_CAM, inc))
    print('base flow V/Vinf %.4f -> %.4f  (needs u_t to fall %.4f)'
          % (Vb[0], Vb[-1], Vb[-1] - Vb[0]))

    print('\n%-34s %8s %8s %8s' % ('thickness form', 't/c needed', 'shape rms',
                                   'corr'))
    cands = [('4-digit (standard)', dict()),
             ('4-digit mod  x_m=0.30', dict(xm=0.30)),
             ('4-digit mod  x_m=0.40', dict(xm=0.40)),
             ('4-digit mod  x_m=0.45', dict(xm=0.45)),
             ('4-digit mod  x_m=0.50', dict(xm=0.50)),
             ('4-digit mod  x_m=0.55', dict(xm=0.55)),
             ('4-digit mod  x_m=0.60', dict(xm=0.60)),
             ('4-digit mod  x_m=0.50 I=3', dict(xm=0.50, ik=3.0)),
             ('4-digit mod  x_m=0.50 I=9', dict(xm=0.50, ik=9.0)),
             ('4-digit mod  x_m=0.50 Km=0.7', dict(xm=0.50, km=0.7)),
             ('4-digit mod  x_m=0.50 Km=1.6', dict(xm=0.50, km=1.6))]
    out = []
    for name, kw in cands:
        xn, ut = u_thick(0.10, **kw)
        s, rms, corr = match(xn_req, req, xn, ut)
        out.append((name, kw, s, rms, corr, xn, ut))
        print('%-34s %8.3f %8.4f %8.4f' % (name, 0.10*s, rms, corr))

    best = min(out, key=lambda r: r[3])
    print('\nbest shape match: %s   (t/c needed %.3f, shape rms %.4f)'
          % (best[0], 0.10*best[2], best[3]))

    # ------- what does t/c = 0.10 actually leave on the table? --------------
    print('\nat the TARGET t/c = %.2f with that form:' % T_TARGET)
    xn, ut = u_thick(T_TARGET, **best[1])
    m = (xn >= WIN[0]) & (xn <= WIN[1])
    Vtot = np.interp(xn[m], xn_req, Vb) + ut[m]
    cp = 1.0 - Vtot**2
    print('  V/Vinf %.3f -> %.3f ;  Cp %.3f -> %.3f  (range %.3f)'
          % (Vtot[0], Vtot[-1], cp[0], cp[-1], cp.max() - cp.min()))
    Vb_w = np.interp(xn[m], xn_req, Vb)
    cpb = 1.0 - Vb_w**2
    print('  uncorrected Cp range would be %.3f -> thickness removes %.0f%%'
          % (cpb.max() - cpb.min(),
             100*(1 - (cp.max() - cp.min())/(cpb.max() - cpb.min()))))

    # --------------------------------------------------------------- figure -
    fig, ax = plt.subplots(2, 1, figsize=(8.2, 7.0))
    r0 = req - req[(xn_req >= WIN[0]) & (xn_req <= WIN[1])].mean()
    ax[0].plot(xn_req, r0, '-', color='#b03060', lw=2.4,
               label='required $u_t$ shape ($-V_{base}$)')
    for name, kw, s, rms, corr, xn_, ut_ in out:
        if kw.get('xm') in (0.30, 0.45, 0.60) or not kw:
            mm = (xn_ >= WIN[0]) & (xn_ <= WIN[1])
            ax[0].plot(xn_[mm], s*(ut_[mm] - ut_[mm].mean()), '--', lw=1.3,
                       label='%s (t/c %.2f)' % (name, 0.10*s))
    ax[0].set_ylabel('$u_t/V_\\infty$ (mean removed)')
    ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3)
    ax[0].set_title('Q: which classic thickness form has the required shape?',
                    fontsize=10)

    ax[1].plot(xn[m], cpb, '-', color='#b03060', lw=2.0,
               label='mean line alone (no thickness)')
    ax[1].plot(xn[m], cp, '-', color='#1f4e9c', lw=2.0,
               label='+ %s at t/c=%.2f' % (best[0], T_TARGET))
    ax[1].invert_yaxis(); ax[1].axhline(0, color='0.6', lw=.8)
    ax[1].set_xlabel('$x/c_{fore}$'); ax[1].set_ylabel('$C_p$')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[1].set_title('thin-airfoil estimate of the resulting $C_p$', fontsize=10)
    fig.tight_layout()
    fig.savefig('step3_thickness.pdf'); fig.savefig('step3_thickness.png', dpi=125)
    print('\nwrote step3_thickness.pdf / .png')
