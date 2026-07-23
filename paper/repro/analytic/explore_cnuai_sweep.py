"""Sec. II.D study: choose the laminar-diffusion reduction c_nu,ai on the
parabolized (BL-marched) disturbance transport, BEFORE any SA handover
machinery -- the march has no SA production/destruction, so c_nu,ai is
isolated from the turbulent side entirely.

Stage (a): march several attached Falkner-Skan wedges WITHOUT laminar
viscosity reduction (c_nu,ai = 1) and quantify how much the envelope slope
changes from early to late growth: s_early = 4/(Re_theta(N=5)-Re_theta(N=1))
vs s_late = 4/(Re_theta(N=9)-Re_theta(N=5)). An ideal e^N envelope is
shape-driven (Reynolds-free), so s_early/s_late -> the c_nu,ai->0 limit;
the molecular drain scales like 1/Re_theta and bends the envelope, inflating
the ratio's deviation at low Re_theta (early growth).

Stage (b): sweep c_nu,ai in {1, 1/2, 1/4, 1/6, 1/12, 1/24, 1/48, 0} and
quantify the criterion vs c. (1/12 was calibrated on the previous kernel and
carries no special status here.)

Also recorded per (beta, c): the N=1 crossing Re_theta -- the stage-(c) input
for re-anchoring Re_Omega_crit against the Drela-Giles N=1 station.

Onset kernel = the CURRENT solver canon (softmin_2, absorbed scale):
    Re_Omega_crit = softmin_2(1670, 112 + 1.28/(S_hat*g)^2), ramp w=0.35,
    a = 0.19*clip<S_hat*g>_0^1.
(These will be re-anchored in stage (c) after c_nu,ai is chosen; the slope
criterion of stages (a)-(b) is insensitive to the onset scale, which moves
the crossings but hardly the secants between them.)

Run:  python3 explore_cnuai_sweep.py          (from paper/, ~minutes)
Out:  analytic/figs_explore/cnuai_sweep.png, results_cnuai_sweep.json
"""
import json
import os
import _saai  # noqa: F401  (paths + chdir to paper/)
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import dN_dRe_theta, Re_theta0

SIGMA_SA = 2.0 / 3.0

# --- sphere kernel, CURRENT SOLVER CANON (ModelConstants.h, explore-lambda-v)
A_MAX = 0.19
REOM_CEIL, REOM_A, REOM_B = 1670.0, 112.0, 1.28   # softmin_2, absorbed k=0.64
REOM_N = 2.0
RAMP_W = 0.35


def sphere_rate(u, dudy, yc,
                ceil=REOM_CEIL, a_=REOM_A, b_=REOM_B, n_=REOM_N, w=RAMP_W):
    d2u = np.gradient(dudy, yc)
    X = u; Y = yc*dudy; Z = 0.5*yc**2*d2u
    R = np.sqrt(X*X + Y*Y + Z*Z) + 1e-30
    Shat = Y/np.sqrt(X*X + Y*Y + 1e-30)
    g = (Y - X - Z)/R
    P = Shat*g
    a = A_MAX*np.minimum(1.0, np.clip(P, 0.0, None))
    ReOm = yc**2*np.abs(dudy)
    _pw = a_ + b_*np.maximum(P, 1e-6)**(-2.0)
    reomc = (ceil**(-n_) + _pw**(-n_))**(-1.0/n_)
    onset = 0.5*(1.0 + np.tanh((ReOm/reomc - 1.0)/w))
    return a*onset


def march(fs, x_max, c_nu, nx=800, ny=600, seed=1.0):
    """Parabolized disturbance transport on the FS field with molecular
    diffusion coefficient c_nu (c_nu = 1: full; 0: none). No SA terms."""
    eta99 = np.interp(0.99, np.maximum.accumulate(fs.u), fs.eta)
    y_top = 8.0*eta99*np.sqrt(x_max/fs.inviscid_at(x_max))
    dy = y_top/ny; yc = (np.arange(ny) + 0.5)*dy; dx = x_max/nx
    nu = np.ones(ny)*seed; N = [0.0]; xs = [0.0]
    k = (c_nu/SIGMA_SA)/dy**2
    for i in range(nx):
        x = (i + 0.5)*dx
        _, u, dudy, v = fs.at(x, np.arange(ny + 1)*dy, cellCentered=True)
        u = np.maximum(u, 1e-12)
        vp = np.clip(v, 0, None)/dy; vm = np.clip(-v, 0, None)/dy
        di = vp + vm + 2*k; lo = -(vp[1:] + k); up = -(vm[:-1] + k)
        di[0] += k; di[-1] -= k
        b = sphere_rate(u, dudy, yc)*np.abs(dudy)
        main = u/dx + di; rhs = u/dx*nu + b*nu; rhs[-1] += vm[-1]*seed
        A = sp.diags([lo, main, up], [-1, 0, 1], format='csc')
        nu = spla.spsolve(A, rhs)
        xs.append((i + 1)*dx); N.append(float(np.log(max(nu.max()/seed, 1e-300))))
    return np.array(xs), np.array(N)


def measures(fs, I_th, c_nu, x0):
    """Adaptive-x march to N>=14; return (Rt1, Rt5, Rt9, s_early, s_late)."""
    x_max = x0
    for _ in range(14):
        xs, N = march(fs, x_max, c_nu)
        if not np.all(np.isfinite(N)) or N[-1] > 60.0:
            x_max *= 0.15; continue
        if N[-1] > 14.0:
            x_max = 1.1*float(np.interp(14.0, N, xs))
            xs, N = march(fs, x_max, c_nu); break
        x_max *= 3.0
    if N[-1] < 9.0:
        return (float('nan'),)*5
    Ue = fs.inviscid_at(np.maximum(xs, 1e-12)); Rt = I_th*np.sqrt(xs*Ue)
    Rt1 = float(np.interp(1.0, N, Rt))
    Rt5 = float(np.interp(5.0, N, Rt))
    Rt9 = float(np.interp(9.0, N, Rt))
    if not (Rt1 < Rt5 < Rt9):
        return (float('nan'),)*5
    return Rt1, Rt5, Rt9, 4.0/(Rt5 - Rt1), 4.0/(Rt9 - Rt5)


BETAS = [-0.18, -0.15, -0.12, -0.09, -0.06, -0.03, 0.0, 0.05, 0.10]
C_LIST = [1.0, 0.5, 0.25, 1.0/6.0, 1.0/12.0, 1.0/24.0, 1.0/48.0, 0.0]


def main():
    out = {"betas": BETAS, "c_list": C_LIST, "onset": [REOM_CEIL, REOM_A, REOM_B, REOM_N],
           "rows": []}
    for beta in BETAS:
        fs = FalknerSkanWedge(beta)
        I_th = float(np.trapezoid(fs.u*(1 - fs.u), fs.eta))
        H = float(np.trapezoid(1 - fs.u, fs.eta))/I_th
        d = float(np.asarray(dN_dRe_theta(H)))
        Rtc = float(np.asarray(Re_theta0(H)))
        N1_drela = Rtc + 1.0/d
        x0 = 4e6 if beta == 0.0 else (3e5 if beta > 0 else 1.2e6)
        for c in C_LIST:
            Rt1, Rt5, Rt9, sE, sL = measures(fs, I_th, c, x0)
            row = dict(beta=beta, H=H, c_nu=c, Rt1=Rt1, Rt5=Rt5, Rt9=Rt9,
                       s_early=sE, s_late=sL,
                       ratio=(sE/sL if np.isfinite(sE) and np.isfinite(sL) else float('nan')),
                       drela_slope=d, N1_drela=N1_drela)
            out["rows"].append(row)
            print(f"beta={beta:+.3f} H={H:.3f} c={c:8.5f}  "
                  f"Rt1={Rt1:7.0f} Rt9={Rt9:7.0f}  sE={sE:.4e} sL={sL:.4e} "
                  f"sE/sL={row['ratio']:.3f}  (Drela slope {d:.3e}, N1@{N1_drela:.0f})",
                  flush=True)
    jpath = 'repro/analytic/figs_explore/results_cnuai_sweep.json'
    os.makedirs(os.path.dirname(jpath), exist_ok=True)
    json.dump(out, open(jpath, 'w'), indent=1)
    print(f'wrote {jpath}')

    # ---- figure: 3 panels ----
    rows = out["rows"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.5, 4.4))
    cs = np.array(C_LIST)
    cplot = np.where(cs > 0, cs, 1.0/200.0)   # plot c=0 at a pseudo-position
    cmap = plt.cm.coolwarm
    bspan = max(BETAS) - min(BETAS)
    for beta in BETAS:
        rr = [r for r in rows if r["beta"] == beta]
        col = cmap((beta - min(BETAS))/bspan)
        ratio = np.array([r["ratio"] for r in rr])
        a1.semilogx(cplot, ratio, '-o', color=col, ms=4, label=fr'$\beta$={beta:+.2f}')
        Rt1 = np.array([r["Rt1"] for r in rr])
        a3.loglog(cplot, Rt1, '-o', color=col, ms=4)
        a3.axhline(rr[0]["N1_drela"], color=col, lw=0.8, ls=':')
    a1.axhline(1.0, color='k', lw=0.8, ls='--')
    a1.axvline(1.0/12.0, color='0.5', lw=0.8, ls=':')
    a1.text(1.0/12.0, a1.get_ylim()[0], ' 1/12', fontsize=7, va='bottom')
    a1.set_xlabel(r'$c_{\nu,\mathrm{ai}}$  (0 plotted at 1/200)')
    a1.set_ylabel(r'slope ratio $s_{[1,5]}/s_{[5,9]}$')
    a1.legend(fontsize=6.5, ncol=2); a1.grid(alpha=0.3)
    # panel 2: envelope-bend view at three c values, vs H, with Drela overlay
    Hs = np.array(sorted({round(r["H"], 4) for r in rows}))
    for c, mk in ((1.0, 's'), (1.0/12.0, 'o'), (0.0, '^')):
        rr = sorted([r for r in rows if r["c_nu"] == c], key=lambda r: r["H"])
        H = [r["H"] for r in rr]
        a2.semilogy(H, [r["s_early"] for r in rr], '-'+mk, ms=4,
                    label=fr'$s_{{[1,5]}}$, $c_\nu$={c:.3g}')
        a2.semilogy(H, [r["s_late"] for r in rr], '--'+mk, ms=4, alpha=0.6,
                    label=fr'$s_{{[5,9]}}$, $c_\nu$={c:.3g}')
    Hg = np.linspace(min(Hs), max(Hs), 200)
    a2.semilogy(Hg, np.asarray(dN_dRe_theta(Hg)), 'k--', lw=1.6, label='Drela--Giles')
    a2.set_xlabel(r'$H$'); a2.set_ylabel(r'$dN/dRe_\theta$ secant')
    a2.legend(fontsize=6.0, ncol=2); a2.grid(alpha=0.3)
    a3.set_xlabel(r'$c_{\nu,\mathrm{ai}}$  (0 plotted at 1/200)')
    a3.set_ylabel(r'$N\!=\!1$ crossing $Re_\theta$ (dotted: Drela station)')
    a3.grid(alpha=0.3, which='both')
    plt.tight_layout()
    fpath = 'repro/analytic/figs_explore/cnuai_sweep.png'
    plt.savefig(fpath, dpi=140)
    print(f'wrote {fpath}')


if __name__ == '__main__':
    main()
