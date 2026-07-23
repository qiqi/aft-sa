"""Green-equivalent wall-bounded tanh family: OS onset + rates vs the kernel.

PREVIEW instrument (not yet wired into the paper): the two-parameter
reversed-flow family Dini, Selig & Maughmer (AIAA J 30(11), 1992) used in
place of Stewartson profiles for the laminar part of separation bubbles --
Green's (AGARD CP-4, 1966) profiles, OS-computed via their hyperbolic-
tangent stand-in:

    u_free(y) = (1-G)/2 + (1+G)/2 * tanh(y - h),      y in b_t units,

with G the backflow fraction (LDA bound: G <= 0.15, Fitzgerald & Mueller
AIAA J 1990) and h = h_t/b_t the shear-layer standoff. We add a thin wall
layer, u = u_free * tanh(y/dw), so the no-slip wall exists and the kernel
indicators (which measure the wall distance) are well defined; dw
sensitivity is printed.

Outputs (figs_explore/):
  green_tanh_graze.{png,pdf}  -- the onset-graze plane (S_hat*g, Re_Omega)
      with the attached FS family + Stewartson (as in fig:onsetgraze) PLUS
      each tanh profile traced at its OS-critical Re_theta.
  green_tanh_rates.{png,pdf}  -- per tanh profile: theta*dN/dx vs Re_theta
      from (i) OS via Gaster transform (modern reference), (ii) the model's
      frozen-profile eigenvalue at canonical c_nu,ai=1/6, (iii) its
      inviscid limit, (iv) Drela's s_DG at the profile's H (extrapolation).

Run from paper/: python3 repro/analytic/explore_green_tanh_family.py
"""
import os
import numpy as np
from numpy.linalg import eig as dense_eig
from scipy.linalg import eigh_tridiagonal
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _saai  # noqa: F401
from _saai import A_MAX
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import dN_dRe_theta, Re_theta0

FIGD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs_explore')
SIGMA_SA, C_NU_AI = 2.0/3.0, 1.0/6.0
CEIL, A_, B_, N_ = 2600.0, 175.0, 2.0, 2.0     # graze SHAPE (k-free, as fig02)

# (h_t/b_t, G_t) pairs along the Dini locus, LDA-bounded G <= 0.15;
# (1.0, 0.025) is their laminar-separation point.
PAIRS = [(1.0, 0.025), (1.5, 0.05), (2.0, 0.10), (3.0, 0.15)]
DW_FRAC = 1.0/3.0     # wall-layer scale, dw = DW_FRAC*h


def tanh_profile(h, G, dw_frac=DW_FRAC, ymax_pad=14.0, N=4000):
    y = np.linspace(0.0, h + ymax_pad, N)
    ufree = 0.5*(1.0 - G) + 0.5*(1.0 + G)*np.tanh(y - h)
    u = ufree*np.tanh(y/(dw_frac*h))
    up = np.gradient(u, y)
    upp = np.gradient(up, y)
    th = float(np.trapezoid(u*(1.0 - u), y))
    ds = float(np.trapezoid(1.0 - u, y))
    return dict(y=y, u=u, up=up, upp=upp, th=th, H=ds/th,
                umin=float(u.min()), h=h, G=G)


def kernel_curve(pr, Re_th):
    """(S_hat*g, Re_Omega) trace of the profile at Re_theta (graze plane)."""
    y, u, up, upp = pr['y'], pr['u'], pr['up'], pr['upp']
    X, Y, Z = u, y*up, 0.5*y**2*upp
    R = np.sqrt(X*X + Y*Y + Z*Z) + 1e-30
    P = (Y/np.sqrt(X*X + Y*Y + 1e-30))*(Y - X - Z)/R
    nu = pr['th']/Re_th
    ReOm = y**2*np.abs(up)/nu
    m = (P > 1e-4) & (ReOm > 1.0)
    return P[m], ReOm[m]


def softmin(P):
    pw = A_ + B_/np.maximum(P, 1e-9)**2
    return (CEIL**(-N_) + pw**(-N_))**(-1.0/N_)


# ---------------- temporal Orr-Sommerfeld (theta units) ----------------
class OS:
    def __init__(self, pr, N=260, ymax_th=None):
        th = pr['th']
        ym = ymax_th if ymax_th else float(pr['y'][-1])/th
        self.yh = np.linspace(0.0, ym, N)
        hh = self.yh[1] - self.yh[0]
        self.U = np.interp(self.yh*th, pr['y'], pr['u'], right=1.0)
        self.Upp = th*th*np.interp(self.yh*th, pr['y'], pr['upp'], right=0.0)
        n = N - 2
        I = np.eye(n)
        self.I = I
        self.D2 = (np.diag(np.ones(n-1), -1) - 2*I + np.diag(np.ones(n-1), 1))/hh**2
        D4 = (np.diag(np.ones(n-2), -2) - 4*np.diag(np.ones(n-1), -1) + 6*I
              - 4*np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-2), 2))/hh**4
        D4[0, 0] = 7.0/hh**4
        D4[-1, -1] = 7.0/hh**4
        self.D4 = D4
        self.Ui, self.Uppi = self.U[1:-1], self.Upp[1:-1]

    def c_of(self, al, Re):
        from scipy.linalg import eig as geig
        L2 = self.D2 - al**2*self.I
        A = (np.diag(self.Ui) @ L2 - np.diag(self.Uppi)
             - (self.D4 - 2*al**2*self.D2 + al**4*self.I)/(1j*al*Re))
        c = geig(A, L2, right=False)
        ok = np.isfinite(c) & (np.real(c) > -0.2) & (np.real(c) < 0.98)
        if not ok.any():
            return complex(0.0, -1.0)
        return complex(c[ok][int(np.argmax(np.imag(c[ok])))])

    def wi_max(self, Re, alo=0.03, ahi=1.3, nsc=8, want_alpha=False):
        ag = np.geomspace(alo, ahi, nsc)
        w = [ag[i]*self.c_of(ag[i], Re).imag for i in range(nsc)]
        j = int(np.argmax(w))
        lo, hi = ag[max(j-1, 0)], ag[min(j+1, nsc-1)]
        best_a, best_w = ag[j], w[j]
        for a in np.linspace(lo, hi, 6)[1:-1]:
            wv = a*self.c_of(a, Re).imag
            if wv > best_w:
                best_w, best_a = wv, a
        return (best_w, best_a) if want_alpha else best_w

    def crit_Re(self, lo=8.0, hi=6000.0, iters=11):
        if self.wi_max(lo) > 0:
            return lo
        if self.wi_max(hi) < 0:
            return np.inf
        for _ in range(iters):
            mid = np.sqrt(lo*hi)
            if self.wi_max(mid) > 0:
                hi = mid
            else:
                lo = mid
        return float(np.sqrt(lo*hi))

    def spatial_rate(self, Re):
        """theta*dN/dx = max_omega(-alpha_i) via the Gaster transform."""
        wmax, astar = self.wi_max(Re, want_alpha=True)
        if wmax <= 0:
            return 0.0
        da = 0.02*astar
        c1 = self.c_of(astar - da, Re)
        c2 = self.c_of(astar + da, Re)
        cg = ((astar + da)*c2.real - (astar - da)*c1.real)/(2*da)
        cg = max(cg, 1e-3)
        return wmax/cg


# ------------- model frozen-profile eigenvalue on the profile -------------
def s_model(pr, Re_th, cnu=C_NU_AI, N=6000, ufloor=0.02):
    y = np.linspace(0.0, float(pr['y'][-1]), N)
    u = np.interp(y, pr['y'], pr['u'])
    up = np.interp(y, pr['y'], pr['up'])
    upp = np.gradient(up, y)
    X, Y, Z = u, y*up, 0.5*y**2*upp
    R = np.sqrt(X*X + Y*Y + Z*Z) + 1e-30
    P = (Y/np.sqrt(X*X + Y*Y + 1e-30))*(Y - X - Z)/R
    b = A_MAX*np.clip(P, 0.0, 1.0)*np.abs(up)
    if Re_th is None:      # inviscid supremum
        return float(np.max(b/np.maximum(u, ufloor)))*pr['th']
    D = cnu*pr['th']/(SIGMA_SA*Re_th)
    hh = y[1] - y[0]
    uf = np.maximum(u, ufloor)[1:-1]
    d = (b[1:-1] - 2.0*D/hh**2)/uf
    e = (D/hh**2)/np.sqrt(uf[:-1]*uf[1:])
    w = eigh_tridiagonal(d, e, select='i',
                         select_range=(len(d)-1, len(d)-1))[0]
    return float(w[0])*pr['th']


def drela_sDG(H):
    ell = (6.54*H - 14.07)/H**2
    m = (0.058*(H - 4.0)**2/(H - 1.0) - 0.068)/ell
    return float(dN_dRe_theta(H))*0.5*(m + 1.0)*ell


def fs_curve(beta, guess, Rtc=None):
    fs = FalknerSkanWedge(beta, guess=guess) if guess is not None \
        else FalknerSkanWedge(beta)
    eta, u, up = fs.eta, fs.u, fs.dudeta
    upp = np.gradient(up, eta)
    I_th = float(np.trapezoid(u*(1 - u), eta))
    H = float(np.trapezoid(1 - u, eta))/I_th
    if Rtc is None:
        Rtc = float(np.asarray(Re_theta0(H)))
    sq = Rtc/I_th
    X, Y, Z = u, eta*up, 0.5*eta**2*upp
    R = np.sqrt(X*X + Y*Y + Z*Z) + 1e-30
    P = (Y/np.sqrt(X*X + Y*Y + 1e-30))*(Y - X - Z)/R
    ReOm = eta**2*np.abs(up)*sq
    m = (P > 1e-4) & (ReOm > 1.0)
    return P[m], ReOm[m]


def prep_profile(pair):
    h, G = pair
    pr = tanh_profile(h, G)
    pr['Rc'] = OS(pr).crit_Re()
    for f in (0.2, 0.5):
        pr[f'Rc_dw{f}'] = OS(tanh_profile(h, G, dw_frac=f), N=200).crit_Re(iters=7)
    return pr


def main():
    os.makedirs(FIGD, exist_ok=True)
    from multiprocessing import Pool
    with Pool(len(PAIRS)) as pool:
        results = pool.map(prep_profile, PAIRS)
    profs = []
    for pr in results:
        pr['os'] = OS(pr)
        profs.append(pr)
        print(f"(h={pr['h']}, G={pr['G']}): theta={pr['th']:.3f}b, "
              f"H={pr['H']:.2f}, backflow={-pr['umin']*100:.1f}%U, "
              f"OS Re_theta,crit={pr['Rc']:.0f} "
              f"[dw=0.2h: {pr['Rc_dw0.2']:.0f}, dw=0.5h: {pr['Rc_dw0.5']:.0f}]",
              flush=True)

    # ---------------- Figure 1: enriched graze ----------------
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    Pg = np.geomspace(3e-3, 1.2, 400)
    ax.loglog(Pg, softmin(Pg), 'k--', lw=2.0, zorder=5)
    cmap = plt.cm.coolwarm
    FS_BETAS = [0.35, 0.15, 0.05, 0.0, -0.05, -0.10, -0.15, -0.1988]
    for j, beta in enumerate(FS_BETAS):
        P, ReOm = fs_curve(beta, None)
        ax.loglog(P, ReOm, '-', color=cmap(j/(len(FS_BETAS)-1)), lw=1.3,
                  alpha=0.75)
    P, ReOm = fs_curve(-0.12, -0.10)
    ax.loglog(P, ReOm, ':', color='purple', lw=1.6, alpha=0.9)
    gcol = plt.cm.viridis(np.linspace(0.25, 0.85, len(profs)))
    for pr, col in zip(profs, gcol):
        P, ReOm = kernel_curve(pr, pr['Rc'])
        ax.loglog(P, ReOm, '-.', color=col, lw=2.0, zorder=6)
        ratio = ReOm/softmin(P)
        i = int(np.argmax(ratio))
        ax.plot(P[i], ReOm[i], 's', color=col, ms=6.5, mec='k', mew=0.6,
                zorder=7)
        print(f"  graze (h={pr['h']}, G={pr['G']}): closest approach "
              f"max ReOm/threshold = {ratio[i]:.2f} at Shg={P[i]:.3f}",
              flush=True)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color='k', ls='--', lw=2, label='soft-min shape (2600, 175, 2)'),
        Line2D([], [], color=cmap(0.8), lw=1.3, label='attached FS at DG critical'),
        Line2D([], [], color='purple', ls=':', lw=1.6, label='Stewartson (reversed FS)'),
        Line2D([], [], color=gcol[1], ls='-.', lw=2,
               label='Green/tanh family at OS critical')],
        fontsize=8, loc='upper right')
    ax.set_xlabel(r'$\hat S g$')
    ax.set_ylabel(r'$Re_\Omega$ at the profile-family critical $Re_\theta$')
    ax.set_xlim(3e-3, 1.3); ax.set_ylim(30, 6e3)
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(f'{FIGD}/green_tanh_graze.{ext}',
                    dpi=140 if ext == 'png' else None)
    plt.close(fig)
    print(f'wrote {FIGD}/green_tanh_graze.png', flush=True)

    # ---------------- Figure 2: rates vs Re_theta ----------------
    RTS = np.array([200.0, 400.0, 800.0, 1600.0])
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex=True)
    for k, (pr, col) in enumerate(zip(profs, gcol)):
        ax = axs.flat[k]
        osr = []
        for Rt in RTS:
            r = pr['os'].spatial_rate(Rt) if Rt > pr['Rc'] else np.nan
            osr.append(r)
            print(f"  rate (h={pr['h']}, G={pr['G']}) Rt={Rt:.0f}: "
                  f"OS={r if r==r else float('nan'):.4f}", flush=True)
        mod = [s_model(pr, Rt) for Rt in RTS]
        inv = s_model(pr, None)
        sdg = drela_sDG(pr['H'])
        ax.semilogx(RTS, osr, 'o-', color='k', lw=1.8, ms=5,
                    label='Orr--Sommerfeld (Gaster)')
        ax.semilogx(RTS, mod, 's--', color='C3', lw=1.6, ms=5,
                    label=r'model eigenvalue ($c_{\nu,\mathrm{ai}}\!=\!1/6$)')
        ax.axhline(inv, color='C3', ls=':', lw=1.2,
                   label='model inviscid limit')
        ax.axhline(sdg, color='0.5', ls='-.', lw=1.4,
                   label=f'Drela at H={pr["H"]:.1f} (extrapolated)')
        ax.axvline(pr['Rc'], color='0.7', lw=0.8)
        ax.set_title(f"$h_t/b_t={pr['h']}$, $G_t={pr['G']}$ "
                     f"(H={pr['H']:.1f}, backflow {-pr['umin']*100:.0f}%)",
                     fontsize=10)
        ax.grid(alpha=0.3, which='both')
        if k == 0:
            ax.legend(fontsize=8)
        if k >= 2:
            ax.set_xlabel(r'$Re_\theta$')
        if k % 2 == 0:
            ax.set_ylabel(r'$\theta\, dN/dx$')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(f'{FIGD}/green_tanh_rates.{ext}',
                    dpi=140 if ext == 'png' else None)
    print(f'wrote {FIGD}/green_tanh_rates.png', flush=True)


if __name__ == '__main__':
    main()
