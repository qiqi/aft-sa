"""TASK 1 (2026-07-29): mechanism trace for why the C=8000 (green) panel-(a)
rate sits BELOW the canon-C=1851.2 (blue) at low H in
model_calibrate_candidate_zb.png.

Variant B (zb): rate = softmax2(a_inv*Om*<I>+, a_visc*Om*<-Z>+/R); the ONSET
gate sees only the canon un-floored coordinate P=Om*I with threshold
Re_Om^c = softmin2(C, 124.6 + 1.424/P^2). Raising C raises the favorable-side
onset threshold. This script marches FS wedges beta=1 (H=2.216, "H=2.2") and
beta=0.5 (H=2.297, "H=2.3") for C in {1851.2, 8000} and records ALONG the
march: Re_theta, realized dN/dRe_theta, and the onset gate value S at the
production-peak band. It locates where the gate opens (S:0->1), the Re_theta
where S reaches 0.99, and the N in [5,9] late-secant window, so we can see how
the window samples the gate.

Reuses the exact study kernel (fpg_recalibration_study._P_and_thresh) via the
zb form. OFFLINE; writes only figs_explore artifacts. No tex/solver edits.
"""
import os
import json
import numpy as np

import _saai  # noqa: F401  (paths + chdir to paper/)
import fig04_shapefactor as f4
import fpg_recalibration_study as S       # sets FORM=['add'],EPS=[0.0]; kernel patch
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import dN_dRe_theta, Re_theta0

OUT_DIR = os.path.join('repro', 'analytic', 'figs_explore')
A_MAX, RAMP_W = f4.A_MAX, f4.RAMP_W
EPS_R = 0.1455                            # a_visc = 0.19*0.1455 = 0.0276
K_ANCHOR = f4.K_ANCHOR                    # 0.712 ; C values are k-carrying


def march_trace(fs, x_max, C, eps_r=EPS_R, nx=1600, ny=1200, seed=1.0):
    """fig04_shapefactor.march, instrumented: at every station record N,
    Re_theta, and the onset gate S at the production-peak band (argmax b).
    Uses FORM='zb' so the rate floor + canon(C) gate are the study kernel."""
    S.FORM[0] = 'zb'; S.EPS[0] = eps_r; S.CEIL_B[0] = C
    I_th = float(np.trapezoid(fs.u*(1-fs.u), fs.eta))
    eta99 = np.interp(0.99, np.maximum.accumulate(fs.u), fs.eta)
    y_top = 8.0*eta99*np.sqrt(x_max/fs.inviscid_at(x_max))
    dy = y_top/ny; yc = (np.arange(ny)+0.5)*dy; dx = x_max/nx
    nu = np.ones(ny)*seed
    from _saai import SIGMA_SA
    k = (f4.C_NU_AI/SIGMA_SA)/dy**2
    xs = [0.0]; Nl = [0.0]; gate_pk = [0.0]; ratio_pk = [0.0]; a_pk = [0.0]
    for i in range(nx):
        x = (i+0.5)*dx
        _, u, dudy, v = fs.at(x, np.arange(ny+1)*dy, cellCentered=True)
        u = np.maximum(u, 1e-12)
        vp = np.clip(v, 0, None)/dy; vm = np.clip(-v, 0, None)/dy
        di = vp+vm+2*k; lo = -(vp[1:]+k); up = -(vm[:-1]+k)
        di[0] += k; di[-1] -= k
        # kernel fields (mirror sphere_rate_eps so we can read gate & rate)
        d2u = np.gradient(dudy, yc)
        X = u; Y = yc*dudy; Z = 0.5*yc**2*d2u
        R = np.sqrt(X*X+Y*Y+Z*Z)+1e-30
        Shat = Y/np.sqrt(X*X+Y*Y+1e-30)
        g = (Y-X-Z)/R
        P, reomc = S._P_and_thresh(Shat, g, Z/R)
        a = A_MAX*np.minimum(1.0, np.clip(P, 0.0, None))
        ReOm = yc**2*np.abs(dudy)
        onset = 0.5*(1.0+np.tanh((ReOm/reomc - 1.0)/RAMP_W))
        b = a*onset*np.abs(dudy)
        # transport step
        main = u/dx + di; rhs = u/dx*nu + b*nu; rhs[-1] += vm[-1]*seed
        import scipy.sparse as sp, scipy.sparse.linalg as spla
        A_ = sp.diags([lo, main, up], [-1, 0, 1], format='csc')
        nu = spla.spsolve(A_, rhs)
        jpk = int(np.argmax(b)) if np.max(b) > 0 else int(np.argmax(a*onset))
        xs.append((i+1)*dx); Nl.append(float(np.log(max(nu.max()/seed, 1e-300))))
        gate_pk.append(float(onset[jpk]))
        ratio_pk.append(float(ReOm[jpk]/reomc[jpk]))
        a_pk.append(float(a[jpk]))
    xs = np.array(xs); Nl = np.array(Nl)
    Ue = fs.inviscid_at(np.maximum(xs, 1e-12))
    Rt = I_th*np.sqrt(xs*Ue)
    return dict(x=xs, N=Nl, Rt=Rt, gate=np.array(gate_pk),
                ratio=np.array(ratio_pk), a=np.array(a_pk), I_th=I_th)


def adaptive(fs, C, beta):
    """grow x_max until N reaches ~10-12 (so N in [5,9] is well inside)."""
    x_max = 3e5 if beta > 0 else 1.2e6
    for _ in range(14):
        tr = march_trace(fs, x_max, C)
        if not np.all(np.isfinite(tr['N'])) or tr['N'][-1] > 60:
            x_max *= 0.2; continue
        if tr['N'][-1] > 11.0:
            x_max = 1.15*float(np.interp(11.0, tr['N'], tr['x']))
            return march_trace(fs, x_max, C)
        x_max *= 2.5
    return tr


def crossing(N, arr, level):
    return float(np.interp(level, N, arr)) if N[-1] >= level else float('nan')


def gate_reaches(Rt, gate, level):
    m = gate >= level
    if not m.any():
        return float('nan')
    return float(Rt[np.argmax(m)])


def main():
    cases = [(1.0, 'H=2.216 ("H=2.2")'), (0.5, 'H=2.297 ("H=2.3")')]
    Cs = [(1851.2, 'canon C=1851.2 (blue)'), (8000.0, 'C=8000 (green)')]
    results = {}
    summary_lines = []
    for beta, hlab in cases:
        fs = FalknerSkanWedge(beta)
        results[hlab] = {}
        for C, clab in Cs:
            tr = adaptive(fs, C, beta)
            N, Rt, gate = tr['N'], tr['Rt'], tr['gate']
            Rt1 = crossing(N, Rt, 1.0)
            Rt5 = crossing(N, Rt, 5.0)
            Rt9 = crossing(N, Rt, 9.0)
            s_late = 4.0/(Rt9-Rt5) if np.isfinite(Rt9) and Rt9 > Rt5 else float('nan')
            s_early = 4.0/(Rt5-Rt1) if np.isfinite(Rt5) and Rt5 > Rt1 else float('nan')
            # gate value at the [5,9] window endpoints
            g5 = float(np.interp(Rt5, Rt, gate)) if np.isfinite(Rt5) else float('nan')
            g9 = float(np.interp(Rt9, Rt, gate)) if np.isfinite(Rt9) else float('nan')
            Rt_g50 = gate_reaches(Rt, gate, 0.50)
            Rt_g99 = gate_reaches(Rt, gate, 0.99)
            dDG = float(dN_dRe_theta(np.trapezoid(fs.u*(1-fs.u), fs.eta) and 0 or 0))  # placeholder
            results[hlab][clab] = dict(
                Rt=Rt.tolist(), N=N.tolist(), gate=gate.tolist(),
                ratio=tr['ratio'].tolist(),
                Rt1=Rt1, Rt5=Rt5, Rt9=Rt9, s_late=s_late, s_early=s_early,
                gate_at_Rt5=g5, gate_at_Rt9=g9, Rt_gate50=Rt_g50, Rt_gate99=Rt_g99)
            line = (f"{hlab:24s} {clab:22s}: Rt1={Rt1:7.0f} Rt5={Rt5:7.0f} "
                    f"Rt9={Rt9:7.0f} s_late={s_late:.3e} | gate(Rt5)={g5:.3f} "
                    f"gate(Rt9)={g9:.3f} Rt[S=.5]={Rt_g50:.0f} Rt[S=.99]={Rt_g99:.0f}")
            print(line, flush=True)
            summary_lines.append(line)
    # DG reference slopes at the two H
    for beta, hlab in cases:
        fs = FalknerSkanWedge(beta)
        I_th = np.trapezoid(fs.u*(1-fs.u), fs.eta)
        H = np.trapezoid(1-fs.u, fs.eta)/I_th
        print(f"  DG dN/dRe_theta at {hlab} (H={H:.3f}) = "
              f"{float(dN_dRe_theta(H)):.3e}", flush=True)
    with open(os.path.join(OUT_DIR, 'zb_rate_lowH_trace.json'), 'w') as f:
        json.dump(results, f, indent=1)
    plot(results, cases, Cs)
    print('\n'.join(summary_lines))


def plot(results, cases, Cs):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))
    colmap = {'canon C=1851.2 (blue)': 'C0', 'C=8000 (green)': 'C2'}
    for row, (beta, hlab) in enumerate(cases):
        fs = FalknerSkanWedge(beta)
        H = float(np.trapezoid(1-fs.u, fs.eta)/np.trapezoid(fs.u*(1-fs.u), fs.eta))
        dg = float(dN_dRe_theta(H))
        axr, axg = axes[row]
        for C, clab in Cs:
            d = results[hlab][clab]
            Rt = np.array(d['Rt']); N = np.array(d['N']); gate = np.array(d['gate'])
            col = colmap[clab]
            # dN/dRe_theta from marched N(Rt)
            mo = np.argsort(Rt); Rt_s, N_s = Rt[mo], N[mo]
            dNdRt = np.gradient(N_s, Rt_s)
            good = (N_s > 0.2) & (N_s < 11)
            axr.plot(Rt_s[good], dNdRt[good], '-', color=col, lw=1.6, label=clab)
            # mark [5,9] window
            for lv, ls in [(d['Rt5'], ':'), (d['Rt9'], ':')]:
                if np.isfinite(lv):
                    axr.axvline(lv, color=col, ls=ls, lw=0.8, alpha=0.6)
            axr.axhspan(0, 0, color='none')
            # gate panel
            axg.plot(Rt_s[good], gate[mo][good], '-', color=col, lw=1.6, label=clab)
            if np.isfinite(d['Rt5']):
                axg.axvspan(d['Rt5'], d['Rt9'], color=col, alpha=0.08)
            if np.isfinite(d['Rt_gate99']):
                axg.axvline(d['Rt_gate99'], color=col, ls='--', lw=0.9, alpha=0.7)
        axr.axhline(dg, color='k', ls='--', lw=1.4, label=f'Drela-Giles ({dg:.2e})')
        axr.set_title(f'(a) rate {hlab}'); axr.set_xlabel(r'$Re_\theta$')
        axr.set_ylabel(r'$dN/dRe_\theta$'); axr.set_xscale('log')
        axr.grid(alpha=0.3, which='both'); axr.legend(fontsize=8)
        axr.set_ylim(0, dg*1.6)
        axg.set_title(f'(b) onset gate S {hlab}'); axg.set_xlabel(r'$Re_\theta$')
        axg.set_ylabel(r'gate $S$ at production peak'); axg.set_xscale('log')
        axg.grid(alpha=0.3, which='both'); axg.legend(fontsize=8)
        axg.set_ylim(-0.03, 1.05)
        axg.text(0.02, 0.9, 'shaded = each C\'s $N\\in[5,9]$ window;\n'
                 'dashed = $Re_\\theta$ where $S=0.99$',
                 transform=axg.transAxes, fontsize=7.5, va='top')
    fig.suptitle('zb variant-B: rate & onset-gate along the march, canon C '
                 'vs C=8000 (rate constants identical, a_visc=0.0276)',
                 fontsize=11)
    plt.tight_layout()
    fp = os.path.join(OUT_DIR, 'zb_rate_lowH_trace.png')
    plt.savefig(fp, dpi=150, facecolor='white')
    print(f'wrote {fp}', flush=True)


if __name__ == '__main__':
    main()
