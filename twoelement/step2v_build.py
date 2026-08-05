"""STEP 2v -- build the fore element on the corrected (viscous) mean line, with a
Kulfan/CST thickness distribution tuned against the flap's DISPLACEMENT body.

Pieces, all established earlier:

  mean line   CST n=10 fitted to the streamline through (0,0) of the flap-alone
              SA-AI field, out to x_end where that streamline reaches Cp = -0.75
              (x_end = 0.7054). Follows the streamline to rms 0.01% of chord.
  flap        equivalent inviscid body: surface + delta* frozen in from mfoil at
              Re_flap = 3e5. Reproduces cl 1.431 against 1.459 viscous, where
              the plain inviscid section gives 1.985.
  thickness   CST half-thickness, class function xh^0.5 (1-xh), tuned here.

The thickness is tuned over THREE incidences (-2, -1, 0 deg), not one. At a
single incidence the optimum drives the leading-edge radius toward zero, which
would give exactly one angle at which both surfaces are laminar; the design needs
a small range.

Run:  python3 step2v_build.py [out.pdf]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import comb

import panel2e as M
import step1v_camber as S
from step1v_kulfan import fit_cst, cst_line
from flap_displacement import flap_body

ALPHA = -1.0                        # OPERATING incidence
ALPHAS = (-2.0, -1.0, 0.0)          # the range both surfaces must stay clean over
# ANCHORING incidence for the mean-line trace. Separate from the operating
# incidence: it only sets the fore element's shape, and it is the slot-gap knob.
# The gain is almost pure freestream tilt -- the fore trailing edge rises by
# x_end*tan(d_alpha) ~ 0.0123 per degree, while x_end itself moves < 0.003 and
# the flap's own circulation contributes almost nothing. Anchor 0 doubles the
# shortest fore-to-flap distance, 0.0118 -> 0.0240.
def _anchor_from_argv(default=0.0):
    """Read the anchor from argv only when it parses as a number. This module is
    imported by other drivers that have their own arguments (run_ladder_v2 takes
    level names), and parsing argv unconditionally at import time made
    `run_ladder_v2.py L0 L1` die on float('L1')."""
    if len(sys.argv) > 2:
        try:
            return float(sys.argv[2])
        except ValueError:
            pass
    return default


ANCHOR = _anchor_from_argv()
CP_TARGET = -0.75                   # streamline Cp that sets the aft limit
N_CAM = 10                          # CST order of the mean line
N_THK = 5                           # CST order of the half-thickness
TE_BASE = 0.003                     # total-chord units, as the mesher uses
NPTS = 121                          # points per surface


def bern(n, xh):
    i = np.arange(n + 1)[:, None]
    return comb(n, i)*xh[None, :]**i*(1.0 - xh[None, :])**(n - i)


def half_thickness(a, xh, te_half):
    """CST half-thickness: xh^0.5 (1-xh) * sum A_i B_i + xh * te_half."""
    n = len(a) - 1
    return (np.sqrt(np.maximum(xh, 0.0))*(1.0 - xh)*(a[:, None]*bern(n, xh)).sum(0)
            + xh*te_half)


def fore_nodes(a, cam_par, x_end, npts=NPTS):
    """Closed loop, lower TE -> LE -> upper TE, offset normal to the mean line."""
    beta = np.linspace(0.0, np.pi, npts)
    xh = 0.5*(1.0 - np.cos(beta))                  # 0 -> 1, cosine spaced
    xs = xh*x_end
    zc = cst_line(cam_par, xs, x_end, N_CAM)
    dz = np.gradient(zc, xs)
    den = np.hypot(1.0, dz)
    nx, nz = -dz/den, 1.0/den                      # unit normal to the mean line
    th = half_thickness(a, xh, 0.5*TE_BASE/x_end)*x_end
    up = np.column_stack([xs + th*nx, zc + th*nz])
    lo = np.column_stack([xs - th*nx, zc - th*nz])
    return np.vstack([lo[::-1][:-1], up]), xs, zc, th


def evaluate(a, cam_par, flap_nd, x_end, alphas=ALPHAS):
    """Mean fore-flatness cost over the incidence range, plus diagnostics."""
    nd, xs, zc, th = fore_nodes(a, cam_par, x_end)
    if th.min() < -1e-9 or np.any(np.diff(nd[:, 0]) == 0):
        return 1e3, {}
    tot, det = 0.0, {}
    for al in alphas:
        try:
            P, res = M.solve_elements([nd, flap_nd], al)
        except Exception:
            return 1e3, {}
        c, d = M.fore_flatness(P)
        if not np.isfinite(c):
            return 1e3, {}
        tot += c
        det[al] = (c, d, res, P)
    return tot/len(alphas), det


def cost(p, cam_par, flap_nd, x_end, t_target=0.09, rle_min=0.005):
    a = p
    nd, xs, zc, th = fore_nodes(a, cam_par, x_end)
    tmax = 2.0*th.max()/x_end
    pen = 0.0
    # Positivity applies to the INTERIOR only. With the class function
    # xh^0.5 (1-xh) the half-thickness is exactly zero at the leading edge by
    # construction, so testing th.min() over the whole surface fires always and
    # swamps everything else.
    ti = th[1:-1]
    if ti.min() < 1e-6*x_end:
        pen += 50.0*(1e-6*x_end - ti.min())/(1e-6*x_end)
    pen += 12.0*(tmax - t_target)**2/t_target**2          # hold max thickness
    # For a CST half-thickness with N1 = 0.5 the leading-edge radius is
    # R_LE = A_0^2/2 in chord units. Keep a real nose so more than one incidence
    # can stay laminar on both surfaces, but do not force it blunt.
    rle = 0.5*a[0]**2
    pen += 4.0*max(0.0, rle_min - rle)**2/rle_min**2
    c, _ = evaluate(a, cam_par, flap_nd, x_end)
    return c + pen


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'step2v_build.pdf'

    # ------------------------------------------------- mean line (viscous) --
    case = 'case_F1' if ANCHOR == -1.0 else 'case_A%+05.1f' % ANCHOR
    rf = S.RansField(case)
    # aft limit: where the streamline reaches CP_TARGET, upstream of its minimum
    xf, zf = S.trace(rf, 0.0, 0.0, 0.80, n=1600)
    uu, vv = rf(xf, zf)
    cpf = 1.0 - (np.hypot(uu, vv)/S.MACH)**2
    kmin = int(np.nanargmin(np.where(np.isfinite(cpf), cpf, np.inf)))
    X_END = float(xf[int(np.argmin(np.abs(cpf[:kmin+1] - CP_TARGET)))])
    xs_s, zs_s = S.trace(rf, 0.0, 0.0, X_END, n=900)
    _, _, cam_par, cam_rms = fit_cst(xs_s, zs_s, X_END, n=N_CAM)
    print('anchor alpha %+.1f (%s), operating alpha %+.1f' % (ANCHOR, case, ALPHA))
    print('mean line: CST n=%d, rms %.5f (%.4f%% of x_end=%.4f), Cp target %.2f'
          % (N_CAM, cam_rms, 100*cam_rms/X_END, X_END, CP_TARGET))

    # -------------------------------------------------- flap displacement --
    flap_nd, finfo = flap_body(with_wake=False)
    _, rflap = M.solve_elements([flap_nd], ALPHA)
    print('flap displacement body alone: cl = %.4f (mfoil %.4f)'
          % (rflap['Cl'], finfo['cl_mfoil']))

    # ------------------------------------------------------ tune thickness --
    a0 = np.array([0.30, 0.28, 0.26, 0.24, 0.22, 0.20])
    print('\ntuning %d CST thickness coefficients over alpha = %s ...'
          % (len(a0), ALPHAS))
    r = minimize(cost, a0, args=(cam_par, flap_nd, X_END), method='Nelder-Mead',
                 options=dict(maxiter=4000, maxfev=6000, xatol=1e-5,
                              fatol=1e-7, adaptive=True))
    a = r.x
    print('  cost %.5f -> %.5f in %d evaluations'
          % (cost(a0, cam_par, flap_nd, X_END), r.fun, r.nfev))
    print('  A = %s' % np.array2string(a, precision=5, floatmode='fixed'))

    nd, xs, zc, th = fore_nodes(a, cam_par, X_END)
    tmax = 2.0*th.max()/X_END
    print('  max thickness %.4f of x_end at x/c = %.3f'
          % (tmax, xs[int(np.argmax(th))]/X_END))
    ov, gap = None, float(np.min(np.hypot(flap_nd[:, 0] - nd[0, 0],
                                         flap_nd[:, 1] - nd[0, 1])))
    print('  fore TE (%.4f, %.4f); shortest distance to the flap body %.4f'
          % (nd[0, 0], nd[0, 1], gap))

    cbest, det = evaluate(a, cam_par, flap_nd, X_END)
    print('\n%-7s %8s %8s %8s %8s %8s %8s %8s'
          % ('alpha', 'cost', 'rms_up', 'rms_lo', 'adv_up', 'adv_lo',
             'CL_tot', 'spike'))
    for al in ALPHAS:
        c, d, res, P = det[al]
        sp, lvl = M.flap_suction(P)
        print('%-7.1f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.3f'
              % (al, c, d['rms_up'], d['rms_lo'], d['adv_up'], d['adv_lo'],
                 res['Cl'], lvl))

    np.savez('step2v_build.npz', cam_par=cam_par, thk=a, x_end=X_END, anchor=ANCHOR,
             n_cam=N_CAM, te_base=TE_BASE, fore=nd, flap=flap_nd)
    np.savetxt('step2v_fore.dat', nd, fmt='%12.8f')
    np.savetxt('step2v_flap_displacement.dat', flap_nd, fmt='%12.8f')

    # ------------------------------------------------------------ figure ---
    fig = plt.figure(figsize=(9.2, 8.6))
    axg = fig.add_axes([0.09, 0.66, 0.88, 0.28])
    axc = fig.add_axes([0.09, 0.08, 0.88, 0.52])
    axg.plot(np.append(nd[:, 0], nd[0, 0]), np.append(nd[:, 1], nd[0, 1]),
             '-', color='#1f4e9c', lw=1.5)
    axg.fill(nd[:, 0], nd[:, 1], color='#1f4e9c', alpha=.15)
    axg.plot(np.append(flap_nd[:, 0], flap_nd[0, 0]),
             np.append(flap_nd[:, 1], flap_nd[0, 1]), '-', color='#1a8a5a',
             lw=1.3)
    axg.fill(flap_nd[:, 0], flap_nd[:, 1], color='#1a8a5a', alpha=.12)
    axg.plot(xs, zc, ':', color='#000', lw=1.0, label='CST mean line')
    axg.plot(xs_s, zs_s, '--', color='#c44e52', lw=1.0,
             label='viscous streamline')
    axg.set_aspect('equal'); axg.set_xlim(-0.04, 1.06)
    axg.set_xticklabels([]); axg.legend(fontsize=7.5, loc='upper left')
    axg.set_title('Fore element on the viscous mean line, flap as its '
                  'displacement body', fontsize=10)

    for al, c in zip(ALPHAS, ('#9ecae1', '#1f4e9c', '#08306b')):
        P = det[al][3]
        for k, (nm, ls) in enumerate((('fore', '-'), ('flap', '--'))):
            lo, up = M.surfaces(P, k)
            for idx, lw in ((up, 1.5), (lo, 1.0)):
                axc.plot(P.xc[idx], P.Cp[idx], ls, color=c, lw=lw,
                         alpha=1.0 if k == 0 else 0.45)
        axc.plot([], [], '-', color=c, lw=1.5, label=r'$\alpha=%+.0f^\circ$' % al)
    axc.axhline(0, color='0.6', lw=.8)
    axc.invert_yaxis(); axc.set_xlim(-0.04, 1.06)
    axc.set_xlabel('$x$'); axc.set_ylabel('$C_p$')
    axc.grid(alpha=.25, lw=.6); axc.legend(fontsize=8, loc='lower right')
    axc.set_title('solid fore, faint dashed flap; both fore surfaces held flat',
                  fontsize=9)
    fig.savefig(out)
    print('\nwrote %s, step2v_fore.dat, step2v_build.npz' % out)
