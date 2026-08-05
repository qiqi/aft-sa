"""STEP 1v (b) -- Kulfan/CST mean line fitted to the VISCOUS streamline, and the
fore element pushed downstream to just short of the streamline's pressure
minimum.

Why CST rather than a NACA mean line. With the flap's real (reduced)
circulation the streamline stays low for most of the fore element's length and
then turns up steeply as it approaches the flap nose. A NACA 4-digit mean line
cannot follow that terminal steepening: the residual turning |v_n|/Vinf jumps
from 0.048 at x_end = 0.64 to 0.229 at 0.694. A Kulfan class/shape function with
enough Bernstein terms follows it, which lets the fore element run further aft
and keeps the slot small.

Representation, on xh = x/x_end:

    z(xh) = xh*(1 - xh) * sum_i A_i * B_i^n(xh)  +  xh * dz_te

The class function xh*(1-xh) pins the line at the leading edge and lets the
Bernstein sum shape the interior; the xh*dz_te term carries the net rise, i.e.
what a classic line would call incidence. n+2 free parameters.

The aft limit is the pressure minimum ALONG THE STREAMLINE: past it the fore
element would be sitting in the flap's own suction recovery, which is what the
extent constraint was always about.

Run:  python3 step1v_kulfan.py [out.pdf]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import comb

import panel2e as M
import step1v_camber as S

# order of the Bernstein sum, and the adopted fore trailing-edge station
N_CST = 16
X_END_PICK = 0.6936


def bernstein(n, xh):
    """(n+1, len(xh)) matrix of Bernstein basis polynomials of order n."""
    i = np.arange(n + 1)[:, None]
    return comb(n, i)*xh[None, :]**i*(1.0 - xh[None, :])**(n - i)


def cst_design(xs, x_end, n):
    """Design matrix: z = D @ par, par = [A_0..A_n, dz_te]. The model is LINEAR
    in the coefficients, so it must be solved as a linear least-squares problem.
    Driving it through least_squares() instead leaves the high-order Bernstein
    conditioning unresolved and gives residuals that are non-monotone in n --
    which reads as a geometric limit when it is only a solver artefact."""
    xh = xs/x_end
    B = bernstein(n, xh)                       # (n+1, N)
    D = np.empty((len(xs), n + 2))
    D[:, :n+1] = (x_end*xh*(1.0 - xh))[:, None]*B.T
    D[:, n+1] = xh
    return D


def cst_line(par, xs, x_end, n):
    """z(x) for CST coefficients par = [A_0..A_n, dz_te]."""
    return cst_design(xs, x_end, n) @ np.asarray(par, float)


def fit_cst(xs, zs, x_end, n=5, rcond=1e-12):
    D = cst_design(xs, x_end, n)
    par, *_ = np.linalg.lstsq(D, zs, rcond=rcond)
    zf = D @ par
    return zf, np.gradient(zf, xs), par, float(np.sqrt(np.mean((zf - zs)**2)))


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'step1v_kulfan.pdf'
    rf = S.RansField()

    # -------------------------------------------- where is the Cp minimum? --
    xs_far, zs_far = S.trace(rf, 0.0, 0.0, 0.90, n=900)
    u, v = rf(xs_far, zs_far)
    q = np.hypot(u, v)/S.MACH
    cp_far = 1.0 - q**2
    ok = np.isfinite(cp_far)
    kmin = int(np.nanargmin(np.where(ok, cp_far, np.inf)))
    x_cpmin, cp_min = xs_far[kmin], cp_far[kmin]
    print('\nSTREAMLINE from (0,0), traced to x = 0.90')
    print('  pressure minimum at x = %.4f  (z = %.4f), Cp = %.3f'
          % (x_cpmin, zs_far[kmin], cp_min))
    print('  flap leading edge is at x = %.2f' % S.FLAP_LE[0])
    for xq in (0.60, 0.6936, 0.72, 0.75, x_cpmin):
        j = int(np.argmin(np.abs(xs_far - xq)))
        print('    x=%.4f  z=%.4f  Cp=%+.3f' % (xs_far[j], zs_far[j],
                                                cp_far[j]))

    # ------------------------------------------------- extent sweep, CST ---
    print('\nCST mean line (n=%d) vs NACA 4-digit, viscous field' % N_CST)
    print('%-8s | %-24s | %-24s' % ('', 'CST', 'NACA 4-digit'))
    print('%-8s | %9s %7s %6s | %9s %7s %6s'
          % ('x_end', 'rms/c %', 'vn_max', 'dCp', 'rms/c %', 'vn_max', 'dCp'))
    # Candidates span from comfortably ahead of the flap LE to just short of
    # the streamline Cp minimum. NOTE the binding constraint is NOT the Cp-min
    # location and NOT the mean-line fit: with enough CST terms the fit is
    # essentially exact everywhere. It is dCp along the line -- what the fore
    # element's THICKNESS has to cancel -- which explodes once the line enters
    # the flap's nose acceleration past x = 0.70.
    cand = [0.6800, 0.6936, 0.7000, 0.7200, 0.7400, round(x_cpmin - 0.01, 4)]
    store = {}
    for xe in cand:
        xs, zs = S.trace(rf, 0.0, 0.0, xe, n=500)
        row = []
        zc, dzc, par, rmsc = fit_cst(xs, zs, xe, n=N_CST)
        vnc, cpc = S.residual(rf, xs, zc, dzc, S.MACH)
        row.append((100*rmsc/xe, np.nanmax(np.abs(vnc)),
                    np.nanmax(cpc) - np.nanmin(cpc)))
        zn, dzn, parn, rmsn = S.fit_camber(xs, zs, '4digit', xe)
        vnn, cpn = S.residual(rf, xs, zn, dzn, S.MACH)
        row.append((100*rmsn/xe, np.nanmax(np.abs(vnn)),
                    np.nanmax(cpn) - np.nanmin(cpn)))
        store[xe] = (xs, zs, zc, par, zn, parn, vnc, vnn, cpc)
        print('%-8.4f | %9.4f %7.3f %6.3f | %9.4f %7.3f %6.3f'
              % (xe, *row[0], *row[1]))

    x_pick = X_END_PICK
    xs, zs, zc, par, zn, parn, vnc, vnn, cpc = store[x_pick]
    print('\nADOPTED extent x_end = %.4f (Cp min at %.4f, stopping %.4f short)'
          % (x_pick, x_cpmin, x_cpmin - x_pick))
    print('  CST coefficients A_0..A_%d = %s' % (N_CST, np.array2string(
        par[:N_CST+1], precision=5, floatmode='fixed')))
    print('  dz_te = %+.5f   (net rise %.4f of x_end)'
          % (par[N_CST+1], par[N_CST+1]/x_pick))
    print('  equivalent trailing-edge slope %+.3f, mean-line rise %.4f'
          % (np.gradient(zc, xs)[-1], zc[-1]))
    print('  |v_n|/Vinf: mean %.4f  max %.4f   (NACA 4-digit: mean %.4f max %.4f)'
          % (np.nanmean(np.abs(vnc)), np.nanmax(np.abs(vnc)),
             np.nanmean(np.abs(vnn)), np.nanmax(np.abs(vnn))))
    print('  slot gap in x: fore TE %.4f -> flap LE %.2f  = %.4f'
          % (x_pick, S.FLAP_LE[0], S.FLAP_LE[0] - x_pick))

    np.savez('step1v_kulfan.npz', x_end=x_pick, cst=par, xs=xs, zs=zs, zc=zc,
             x_cpmin=x_cpmin, cp_min=cp_min, n_cst=N_CST)

    # ------------------------------------------------------------ figure ---
    fig, ax = plt.subplots(3, 1, figsize=(9.2, 10.6),
                           gridspec_kw=dict(height_ratios=[1.2, 1.0, 1.0]))
    nd = M.place(M.airfoil_nodes(400, S.FLAP['m'], S.FLAP['p'], S.FLAP['t'],
                                 modified=False),
                 S.FLAP_CHORD, S.FLAP_INC, *S.FLAP_LE)
    ax[0].plot(nd[:, 0], nd[:, 1], '-', color='#1a8a5a', lw=1.5)
    ax[0].fill(nd[:, 0], nd[:, 1], color='#1a8a5a', alpha=.12)
    ax[0].plot(xs_far, zs_far, '-', color='#999', lw=1.0,
               label='viscous streamline, traced past the fore TE')
    ax[0].plot(xs, zs, '-', color='#1f4e9c', lw=2.4,
               label='streamline over the fore element')
    ax[0].plot(xs, zc, ':', color='#000', lw=1.8,
               label='CST mean line (n=%d)' % N_CST)
    ax[0].plot(xs, zn, '-.', color='#c44e52', lw=1.4,
               label='NACA 4-digit, same extent')
    ax[0].axvline(x_cpmin, color='#d97706', ls='--', lw=1.2)
    ax[0].annotate('streamline $C_p$ min\n$x=%.3f$' % x_cpmin,
                   (x_cpmin, 0.005), fontsize=7.5, color='#d97706',
                   ha='center', va='bottom')
    ax[0].set_aspect('equal'); ax[0].set_xlim(-0.03, 1.05)
    ax[0].legend(fontsize=7.5, loc='upper left')
    ax[0].set_ylabel('$z$')
    ax[0].set_title('Fore mean line from the VISCOUS flap field, extent set by '
                    'the streamline pressure minimum', fontsize=10)

    ax[1].plot(xs_far, cp_far, '-', color='#1f4e9c', lw=1.5,
               label='$C_p$ along the streamline')
    ax[1].axvline(x_cpmin, color='#d97706', ls='--', lw=1.2)
    ax[1].axvline(x_pick, color='#000', ls=':', lw=1.2,
                  label='adopted fore TE $x=%.4f$' % x_pick)
    ax[1].invert_yaxis()
    ax[1].set_ylabel('$C_p$'); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.25, lw=.6)

    ax[2].plot(xs, vnc, ':', color='#000', lw=1.8, label='CST')
    ax[2].plot(xs, vnn, '-.', color='#c44e52', lw=1.4, label='NACA 4-digit')
    ax[2].axhline(0, color='0.7', lw=0.8)
    ax[2].set_xlabel('$x$')
    ax[2].set_ylabel('$v_n/V_\\infty$ on the fitted line')
    ax[2].set_title('residual turning at the adopted extent', fontsize=9)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.25, lw=.6)

    fig.tight_layout()
    fig.savefig(out)
    print('\nwrote', out)
