"""PART X (2026-07-29): CANDIDATE Fig 3 (fig:nuhat) under the vg two-branch form.

Mirrors fig03_fs_transport_rows.py but marches the disturbance transport with the
ADOPTED candidate kernel instead of canon:
  rate = a_max*clip( softmax2(a_inv*Om*<I>+, a_visc*Om*<-Z>+/R)/a_inv , 0..1 )
         (a_inv=0.19, a_visc=0.0276)
  gate = 1/2[1+tanh((Re_Om/Re_Om^c - 1)/0.35)],
  Re_Om^c = softmin(A+B/P_I^2, A+B_c/P_curv^2),  B_c=129.74  (NO constant ceiling)
via form 'vg' in fpg_recalibration_study (imported; it monkeypatches
fig04.sphere_rate). c_nu,ai=1/6 UNCHANGED (the recalibration touches only the
rate floor and the gate, not the laminar-diffusion constant).

Rows: canon three (beta=-0.10, 0, +0.10) so canon-vs-candidate is visibly
unchanged where it should be, PLUS the strong-FPG payoff row beta=+1.0 (H=2.216,
stagnation) that the canon model cannot ignite. Right column: candidate marched
envelope (black) vs each profile's Drela-Giles envelope (red dashed), with the
marched N=1 and N=9 crossings annotated vs Drela.

Candidate only: writes figs_explore/fs_nuHat_rows_candidate.png; does NOT touch
figs/fs_nuHat_rows.pdf. No tex/solver edits.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import _saai
from _saai import SIGMA_SA
import fig04_shapefactor as f4
from fig04_shapefactor import C_NU_AI, profile_ints, drela, Re_theta0
import fpg_recalibration_study as S       # patches f4.sphere_rate = sphere_rate_eps
from lib.boundary_layer import FalknerSkanWedge

OUT_DIR = os.path.join('repro', 'analytic', 'figs_explore')

# --- adopt the vg two-branch candidate kernel (a_visc=0.0276, B_c=129.74) ---
S.FORM[0] = 'vg'; S.EPS[0] = 0.1455; S.AC[0] = None; S.BC[0] = 129.74


def cand_rate(u, dudy, yc):
    """candidate dN/ds = a*onset from the vg kernel (patched sphere_rate)."""
    return S.sphere_rate_eps(u, dudy, yc)


def march_field(fs, x_max, nx=1600, ny=1200, beta=0.0):
    m = beta/(2.0 - beta)
    eta99 = np.interp(0.99, np.maximum.accumulate(fs.u), fs.eta)
    y_top = 8.0*eta99*np.sqrt(x_max/fs.inviscid_at(x_max))
    dy = y_top/ny; yc = (np.arange(ny) + 0.5)*dy; dx = x_max/nx
    nu = np.ones(ny); field = [nu.copy()]; xs = [0.0]
    k = (C_NU_AI/SIGMA_SA)/dy**2
    for i in range(nx):
        x = (i + 0.5)*dx
        _, u, dudy, v = fs.at(x, np.arange(ny + 1)*dy, cellCentered=True)
        u = np.maximum(u, 1e-12)
        vp = np.clip(v, 0, None)/dy; vm = np.clip(-v, 0, None)/dy
        di = vp + vm + 2*k; lo = -(vp[1:] + k); up = -(vm[:-1] + k)
        di[0] += k; di[-1] -= k
        b = cand_rate(u, dudy, yc)*np.abs(dudy)
        main = u/dx + di; rhs = u/dx*nu + b*nu; rhs[-1] += vm[-1]
        A = sp.diags([lo, main, up], [-1, 0, 1], format='csc')
        nu = spla.spsolve(A, rhs); xs.append((i + 1)*dx); field.append(nu.copy())
    return np.array(xs), yc, np.array(field)


def size_domain(fs, x0, beta):
    x_max = x0
    for _ in range(14):
        xs, yc, fld = march_field(fs, x_max, nx=200, ny=200, beta=beta)
        N = np.log(np.maximum(fld.max(axis=1), 1e-30))
        if not np.all(np.isfinite(N)) or N[-1] > 60.0:
            x_max *= 0.15; continue
        if N[-1] > 14.0:
            return 1.05*float(np.interp(14.0, N, xs))
        x_max *= 2.5
    return x_max


def main():
    # (beta, x0 domain guess, ylimL); beta=+1 is the new strong-FPG payoff row
    ROWS = [(-0.10, 1.2e6, 8000), (0.0, 4.0e6, 12000),
            (0.10, 3e6, None), (1.0, 3e5, None)]
    fig, axs = plt.subplots(len(ROWS), 2, figsize=(11.2, 2.8*len(ROWS)),
                            layout='constrained')
    rows_out = []
    for irow, (beta, x0, ylimL) in enumerate(ROWS):
        fs = FalknerSkanWedge(beta); I_th, H = profile_ints(fs)
        eta99 = np.interp(0.99, np.maximum.accumulate(fs.u), fs.eta)
        x_max = 6.5e6 if beta == 0.0 else size_domain(fs, x0, beta)
        xs, yc, fld = march_field(fs, x_max, beta=beta)
        Ue = fs.inviscid_at(np.maximum(xs, 1e-12)); Rt = I_th*np.sqrt(xs*Ue)
        N2d = np.log(np.maximum(fld, 1e-30))
        axL, axR = axs[irow]
        Rex = xs*Ue; ReyScale = Ue
        X = np.repeat(Rex[:, None], len(yc), 1); Y = yc[None, :]*ReyScale[:, None]
        lev = np.arange(1, 15, 1)
        if ylimL is None:
            tops = []
            for i in range(len(xs)):
                j = np.where(N2d[i] >= 1.0)[0]
                if len(j):
                    tops.append(yc[j[-1]]*ReyScale[i])
            ylimL = 1.05*float(max(tops)) if tops else float(yc[-1]*ReyScale.max())
        cs = axL.contour(X, Y, N2d, levels=lev, colors='k', linewidths=0.7)
        axL.clabel(cs, levels=lev[::2], fmt='%d', fontsize=6.5, inline_spacing=2)
        th = I_th*np.sqrt(xs/np.maximum(Ue, 1e-30))
        d99 = eta99*np.sqrt(xs/np.maximum(Ue, 1e-30))
        axL.plot(Rex, th*ReyScale, '--', color='0.45', lw=1.1)
        axL.plot(Rex, d99*ReyScale, '-', color='0.45', lw=1.1)
        axL.set_ylim(0, ylimL); axL.set_xlim(0, Rex.max())
        axL.annotate(r'$\delta_{99}$', (0.86*Rex.max(),
                     1.12*float(np.interp(0.86*Rex.max(), Rex, d99*ReyScale))),
                     color='0.35', fontsize=9)
        axL.annotate(r'$\theta$', (0.9*Rex.max(),
                     0.55*float(np.interp(0.9*Rex.max(), Rex, th*ReyScale))),
                     color='0.35', fontsize=9)
        axL.set_ylabel(fr'$\beta={beta:+.2f}$ ($H={H:.3f}$)''\n'r'$Re_y$')
        if irow == len(ROWS)-1:
            axL.set_xlabel(r'$Re_x$')
        axL.text(0.02, 0.95, f'({chr(97+2*irow)})', transform=axL.transAxes,
                 fontsize=11, va='top', fontweight='bold')

        env = fld.max(axis=1); N = np.log(np.maximum(env, 1e-30))
        axR.semilogy(Rt, env, 'k-', lw=1.8)
        Rtc = float(Re_theta0(H)); dr = float(drela(H))
        RtD = np.linspace(0, Rt.max(), 300)
        ND = np.where(RtD > Rtc, dr*(RtD - Rtc), 0.0)
        axR.semilogy(RtD, np.exp(ND), 'r--', lw=1.4)
        axR.set_xlim(0, Rt.max()); axR.set_ylim(0.5, 3e6)
        axR.set_ylabel(r'$\max_y \hat\nu$')
        ax2 = axR.twinx(); ax2.set_ylim(np.log(0.5), np.log(3e6)); ax2.set_ylabel(r'$N$')
        if irow == len(ROWS)-1:
            axR.set_xlabel(r'$Re_\theta$')
        axR.grid(alpha=0.3, which='both')
        axR.text(0.02, 0.95, f'({chr(98+2*irow)})', transform=axR.transAxes,
                 fontsize=11, va='top', fontweight='bold')
        # marched N=1/N=9 crossings vs Drela
        def cross(nl):
            return float(np.interp(nl, N, Rt)) if N[-1] >= nl else float('nan')
        Rt1c, Rt9c = cross(1.0), cross(9.0)
        Rt1D, Rt9D = Rtc + 1.0/dr, Rtc + 9.0/dr
        for RtX, nl, col in [(Rt1c, 1, 'C0'), (Rt9c, 9, 'C2')]:
            if np.isfinite(RtX):
                axR.axvline(RtX, color=col, ls=':', lw=1.0, alpha=0.8)
        for RtX in (Rt1D, Rt9D):
            if RtX <= Rt.max():
                axR.axvline(RtX, color='r', ls=':', lw=0.8, alpha=0.5)
        r1 = Rt1c/Rt1D if np.isfinite(Rt1c) else float('nan')
        r9 = Rt9c/Rt9D if np.isfinite(Rt9c) else float('nan')
        axR.text(0.97, 0.05,
                 f"$N\\!=\\!1$: {Rt1c:.0f} vs DG {Rt1D:.0f} ({r1:.2f}$\\times$)\n"
                 f"$N\\!=\\!9$: {Rt9c:.0f} vs DG {Rt9D:.0f} ({r9:.2f}$\\times$)",
                 transform=axR.transAxes, fontsize=6.8, va='bottom', ha='right',
                 bbox=dict(fc='white', ec='0.7', alpha=0.85, pad=1.5))
        if irow == 0:
            axR.legend([r'candidate (vg two-branch: $a_\mathrm{visc}\!=\!0.0276$, '
                        r'$B_c\!=\!130$, $c_{\nu}\!=\!1/6$)',
                        'Drela--Giles envelope'], fontsize=7.0, loc='lower right')
        rows_out.append(dict(beta=beta, H=H, Rtc=Rtc, drela=dr,
                             Rt1_cand=Rt1c, Rt9_cand=Rt9c,
                             Rt1_DG=Rt1D, Rt9_DG=Rt9D, r1=r1, r9=r9,
                             N_end=float(N[-1])))
        print(f"beta={beta:+.2f} H={H:.3f}: N=1 {Rt1c:.0f}/DG{Rt1D:.0f}={r1:.2f}x  "
              f"N=9 {Rt9c:.0f}/DG{Rt9D:.0f}={r9:.2f}x  N_end={N[-1]:.1f}", flush=True)
    fig.suptitle('CANDIDATE (not canon): Fig 3 disturbance transport under the '
                 'vg two-branch form', fontsize=11)
    fp = os.path.join(OUT_DIR, 'fs_nuHat_rows_candidate.png')
    plt.savefig(fp, dpi=150, facecolor='white')
    print(f'wrote {fp}', flush=True)
    with open(os.path.join(OUT_DIR, 'fs_nuHat_rows_candidate.json'), 'w') as f:
        json.dump(rows_out, f, indent=1)


if __name__ == '__main__':
    main()
