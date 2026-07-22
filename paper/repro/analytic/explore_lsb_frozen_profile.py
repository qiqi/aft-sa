"""LSB amplification over-prediction: frozen separated-profile instrument.

The Daedalus/Eppler bubbles show implied-N growth well beyond the 1.2-1.5x
of the marching lower-branch family comparison. This instrument isolates the
mechanism on ANALYTICAL cases: a separated (reversed-flow Stewartson branch)
Falkner-Skan profile held at CONSTANT PRESSURE with FROZEN shape and
thickness, at typical LSB Reynolds numbers Re_theta ~ 100-1600.

Frozen is self-consistent: at dp/dx=0 the momentum integral gives
dtheta/dx = Cf/2 = f''(0)*Theta_eta/Re_theta, and reversed-flow profiles sit
near Cf ~ 0 (checked and printed below).

With a frozen parallel profile both growth measures are clean eigenproblems
(no marching, no transient):

  TEMPORAL:  sigma_t = max eig [ diag(b) + (c_nu/sigma) (1/Re_th) D2 ],
             b = a(Re_Omega, Gamma) * Q * |omega|   [U_e/theta units]
             -> a_eff = sigma_t / omega_peak, vs the profile's own inviscid
             Rayleigh ceiling omega_i,max/omega_peak (temporal Rayleigh on
             the wall-bounded profile) and vs a_max = 0.19.

  SPATIAL:   dN/dx = max eig of the generalized problem
             [diag(b) + diff] v = sigma_x diag(max(u, ufrac)) v
             == the parabolic march's asymptotic slope, swept over the
             advection floor ufrac. Reference: the Drela-Giles explicit
             spatial envelope rate dN/dx = dN/dRe_theta(H) * (m+1)/2 * l(H)
             / theta (well-defined at frozen theta; l, m are H-correlations).

Mechanism probe: at the u=0 crossing of a separated profile Gamma = 2, the
band gate is wide open (Q4 = Q_v = 1) and S -> 1, while |u| -> 0: e^N growth
per unit length is bounded by the wave speed (c ~ 0.4 U_e), the model's only
by the advection floor. The ufrac sweep quantifies this.

Gates: committed Q4 (canonical constants) and Lambda_v cV=1 (explore branch).
Outputs: printed tables, results_lsb_frozen.npz, figs_explore/lsb_frozen.png.
"""
import os
import sys
import numpy as np
from scipy.linalg import eig, eigh_tridiagonal
from scipy.optimize import minimize_scalar

sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/analytic")
import _saai  # noqa: F401
from _saai import C_NU_AI, SIGMA_SA
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import dN_dRe_theta
from lib.aft_sources import compute_aft_amplification_rate

FIGD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs_explore')
ETA_MAX, NGRID = 24.0, 1600
RE_THETAS = (100.0, 200.0, 400.0, 800.0, 1600.0)
UFRACS = (0.01, 0.03, 0.10, 0.30)
PROFILES = [(-0.1988, None), (-0.19, -0.03), (-0.17, -0.06),
            (-0.15, -0.08), (-0.12, -0.10)]
GATES = {'q4': dict(gw=4.0, gc=1.005, s=11.0, floor=254.0),
         'lv': dict(gw=1.0, gc=0.9676, s=13.62, floor=177.5)}


def build_profile(beta, guess):
    """Uniform eta grid to ETA_MAX (padded with u=1), analytic f'''."""
    fs = FalknerSkanWedge(beta, guess=guess)
    m = beta/(2.0 - beta)
    eta = np.linspace(0.0, ETA_MAX, NGRID)
    fp = np.interp(eta, fs.eta, fs.u, right=1.0)
    fpp = np.interp(eta, fs.eta, fs.dudeta, right=0.0)
    f = np.concatenate([[0.0], np.cumsum(0.5*(fp[1:] + fp[:-1])*np.diff(eta))])
    fppp = -0.5*(m + 1)*f*fpp - m*(1.0 - fp**2)
    fppp[fpp == 0.0] = 0.0
    I_th = np.trapezoid(fp*(1 - fp), eta)
    H = np.trapezoid(1 - fp, eta)/I_th
    return dict(beta=beta, eta=eta, u=fp, up=fpp, upp=fppp,
                Theta=I_th, H=H, fpp0=fs.dudeta[0])


def kernel_b(pr, Re_th, gate):
    """Amplification source b(y) = rate*Q*|omega| in U_e/theta units, on the
    theta-scaled grid yh = eta/Theta. Interior nodes only (Dirichlet)."""
    g = GATES[gate]
    Th = pr['Theta']
    yh = pr['eta'][1:-1]/Th
    u = pr['u'][1:-1]
    w = Th*pr['up'][1:-1]                 # signed du/dyh
    dwdy = Th*Th*pr['upp'][1:-1]
    absw = np.abs(w)
    ReO = absw*yh**2*Re_th
    Gam = 2*(w*yh)**2/(u**2 + (w*yh)**2 + 1e-300)
    rate = np.asarray(compute_aft_amplification_rate(
        ReO, Gam, lambda_p=0.0, sigmoid_center=g['gc'], sigmoid_slope=g['s'],
        re_omega_floor=g['floor'], barrier_power=4.0))
    num = 2.0*absw*yh*np.abs(u)
    if gate == 'q4':
        den = g['gw']*(dwdy*yh**2)**2 + (w*yh)**2 + u**2 + 1e-300
    else:
        den = g['gw']*np.abs(w*dwdy)*yh**3 + (w*yh)**2 + u**2 + 1e-300
    Q = 1.0 - num/den
    return yh, u, absw, rate*Q*absw


def sigma_temporal(pr, Re_th, gate):
    yh, u, absw, b = kernel_b(pr, Re_th, gate)
    h = yh[1] - yh[0]
    k = (C_NU_AI/SIGMA_SA)/Re_th
    d = b - 2.0*k/h**2
    e = np.full(len(yh) - 1, k/h**2)
    lam, vec = eigh_tridiagonal(d, e, select='i',
                                select_range=(len(yh)-1, len(yh)-1))
    v = np.abs(vec[:, 0])
    j = int(np.argmax(v))
    return float(lam[0]), float(yh[j]), float(u[j]), float(np.max(absw))


def sigma_spatial(pr, Re_th, gate, ufrac, want_vec=False):
    """Generalized problem: [diag(b)+diff] v = sig_x diag(max(u,ufrac)) v.
    B diagonal positive -> similarity transform keeps a symmetric tridiag."""
    yh, u, absw, b = kernel_b(pr, Re_th, gate)
    h = yh[1] - yh[0]
    k = (C_NU_AI/SIGMA_SA)/Re_th
    uf = np.maximum(u, ufrac)
    d = (b - 2.0*k/h**2)/uf
    e = (k/h**2)/np.sqrt(uf[:-1]*uf[1:])
    lam, vec = eigh_tridiagonal(d, e, select='i',
                                select_range=(len(yh)-1, len(yh)-1))
    if not want_vec:
        return float(lam[0])
    v = np.abs(vec[:, 0])/np.sqrt(uf)     # undo the similarity transform
    j = int(np.argmax(v))
    return float(lam[0]), float(yh[j]), float(u[j])


def rayleigh_ceiling(pr, N=700, alo=0.02, ahi=1.2):
    """Temporal Rayleigh on the wall-bounded frozen profile (theta units):
    max over alpha of alpha*Im(c); also return the phase speed Re(c) there."""
    Th = pr['Theta']
    yh = np.linspace(0.0, ETA_MAX/Th, N)
    u = np.interp(yh*Th, pr['eta'], pr['u'], right=1.0)
    upp = Th*Th*np.interp(yh*Th, pr['eta'], pr['upp'], right=0.0)
    h = yh[1] - yh[0]
    D2 = (np.diag(np.ones(N-1), -1) - 2*np.eye(N)
          + np.diag(np.ones(N-1), 1))/h**2

    def wi(alpha, want_c=False):
        B = D2 - alpha**2*np.eye(N)
        A = np.diag(u) @ B - np.diag(upp)
        c = eig(A[1:-1, 1:-1], B[1:-1, 1:-1], right=False)
        i = int(np.argmax(np.imag(c)))
        return (alpha*float(np.imag(c[i])), float(np.real(c[i]))) if want_c \
            else alpha*float(np.imag(c[i]))

    # omega_i(alpha) is multimodal (several stable branches): coarse grid
    # scan first, THEN refine around the best -- a bounded scalar minimizer
    # alone lands on stable branches for the thicker profiles.
    agrid = np.geomspace(alo, ahi, 24)
    wg = [wi(a) for a in agrid]
    j = int(np.argmax(wg))
    lo = agrid[max(j - 1, 0)]
    hi = agrid[min(j + 1, len(agrid) - 1)]
    r = minimize_scalar(lambda a: -wi(a), bounds=(lo, hi), method='bounded',
                        options={'xatol': 1e-3})
    wmax, cph = wi(float(r.x), want_c=True)
    if wmax < max(wg):
        wmax, cph = wi(float(agrid[j]), want_c=True)
        r.x = agrid[j]
    return wmax, cph, float(r.x)


def drela_spatial(H):
    """Drela-Giles explicit spatial envelope rate, theta units:
    dN/dxh = dN/dRe_theta(H) * (m(H)+1)/2 * l(H)."""
    ell = (6.54*H - 14.07)/H**2
    m = (0.058*(H - 4.0)**2/(H - 1.0) - 0.068)/ell
    return float(dN_dRe_theta(H))*0.5*(m + 1.0)*ell


def main():
    os.makedirs(FIGD, exist_ok=True)
    out = {}
    print("frozen-assumption check (dtheta/dx = Cf/2 = f''(0)*Theta/Re_th):")
    profs = []
    for beta, guess in PROFILES:
        pr = build_profile(beta, guess)
        profs.append(pr)
        dthdx = pr['fpp0']*pr['Theta']
        print(f"  beta={beta:+.4f} H={pr['H']:6.3f} f''(0)={pr['fpp0']:+.4f} "
              f"-> dtheta/dx = {dthdx:+.2e}/Re_th "
              f"(over 100 theta at Re_th=200: {100*dthdx/200*100:+.2f}%)")

    print("\ninviscid Rayleigh ceiling (temporal, wall-bounded profile):")
    for pr in profs:
        wmax, cph, amax_alpha = rayleigh_ceiling(pr)
        wpk = float(np.max(pr['Theta']*np.abs(pr['up'])))
        pr['ray'] = (wmax, cph, wpk)
        print(f"  H={pr['H']:6.3f}: omega_i,max={wmax:.4f} (U_e/theta), "
              f"a_ray = {wmax/wpk:.4f}, c_phase = {cph:+.3f} U_e, "
              f"alpha*theta = {amax_alpha:.3f}")

    hdr = (f"\n{'H':>7} {'Re_th':>6} | {'a_eff q4':>8} {'a_eff lv':>8} "
           f"{'a_ray':>7} | {'u@t-pk':>6} {'u@x-pk':>6} | "
           f"{'R_x q4':>7} {'R_x lv':>7} "
           f"(ufrac=0.03; u@pk: temporal / spatial mode; R_x vs Drela)")
    print(hdr)
    rows = []
    for pr in profs:
        dS = drela_spatial(pr['H'])
        for Re_th in RE_THETAS:
            st = {}
            for gate in ('q4', 'lv'):
                sig, ypk, upk, wpk = sigma_temporal(pr, Re_th, gate)
                sx = {uf: sigma_spatial(pr, Re_th, gate, uf) for uf in UFRACS}
                _, yxp, uxp = sigma_spatial(pr, Re_th, gate, 0.03,
                                            want_vec=True)
                st[gate] = dict(sig=sig, ypk=ypk, upk=upk, wpk=wpk, sx=sx,
                                yxp=yxp, uxp=uxp)
            q, l_ = st['q4'], st['lv']
            rows.append(dict(H=pr['H'], Re_th=Re_th, q4=q, lv=l_,
                             ray=pr['ray'], drela=dS))
            print(f"{pr['H']:7.3f} {Re_th:6.0f} | "
                  f"{q['sig']/q['wpk']:8.4f} {l_['sig']/l_['wpk']:8.4f} "
                  f"{pr['ray'][0]/pr['ray'][2]:7.4f} | "
                  f"{q['upk']:+6.3f} {q['uxp']:+6.3f} | "
                  f"{q['sx'][0.03]/dS:7.2f} {l_['sx'][0.03]/dS:7.2f}")

    print("\nadvection-floor sensitivity of dN/dx (Q4 gate, ratio to Drela):")
    print(f"{'H':>7} {'Re_th':>6} | " +
          " ".join(f"uf={uf:<5}" for uf in UFRACS))
    for r in rows:
        print(f"{r['H']:7.3f} {r['Re_th']:6.0f} | " +
              " ".join(f"{r['q4']['sx'][uf]/r['drela']:8.2f}"
                       for uf in UFRACS))

    np.save(os.path.join(FIGD, 'results_lsb_frozen.npy'),
            {'rows': rows}, allow_pickle=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    Hs = sorted({round(r['H'], 3) for r in rows})
    cmap = plt.cm.plasma
    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.4))
    for i, Hv in enumerate(Hs):
        rr = [r for r in rows if round(r['H'], 3) == Hv]
        Re = [r['Re_th'] for r in rr]
        c = cmap(i/(len(Hs) - 1))
        axs[0].plot(Re, [r['q4']['sig']/r['q4']['wpk'] for r in rr], 'o-',
                    color=c, label=f'H={Hv:.2f}')
        axs[0].axhline(rr[0]['ray'][0]/rr[0]['ray'][2], color=c, ls=':',
                       lw=0.9)
        axs[1].plot(Re, [r['q4']['sx'][0.03]/r['drela'] for r in rr], 'o-',
                    color=c)
        axs[2].plot(UFRACS, [rr[1]['q4']['sx'][uf]/rr[1]['drela']
                             for uf in UFRACS], 'o-', color=c,
                    label=f'H={Hv:.2f}')
    axs[0].axhline(0.19, color='k', lw=0.8)
    axs[0].text(110, 0.192, r'$a_{\max}$', fontsize=8)
    axs[0].set_xlabel(r'$Re_\theta$')
    axs[0].set_ylabel(r'$a_{\rm eff}=\sigma_t/\omega_{\rm peak}$ '
                      r'(dots: Rayleigh ceiling)')
    axs[0].legend(fontsize=7)
    axs[1].set_xlabel(r'$Re_\theta$')
    axs[1].set_ylabel(r'$dN/dx$ model / Drela  (ufrac=0.03)')
    axs[1].axhline(1, color='0.5', lw=0.8)
    axs[2].set_xscale('log')
    axs[2].set_xlabel('advection floor ufrac')
    axs[2].set_ylabel(r'$dN/dx$ model / Drela  ($Re_\theta=200$)')
    axs[2].axhline(1, color='0.5', lw=0.8)
    axs[2].legend(fontsize=7)
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGD, 'lsb_frozen.png'), dpi=140)
    print('wrote', os.path.join(FIGD, 'lsb_frozen.png'))


if __name__ == '__main__':
    main()
