"""Where the physical wavepacket lives on the (Lambda_v, Gamma) plane, as
compact REGIONS, across the Falkner-Skan family incl. separated profiles.

Region = where the Reynolds-stress PRODUCTION density of the most-amplified
Orr-Sommerfeld mode is large: p(y) = (alpha/2)|Im(phi' phi*)| |U'|, the
critical-layer concentration where the wave extracts energy from the mean
shear -- i.e. exactly where the model OUGHT to place its growth. Thick =
p>0.5 max (contiguous core about the peak), thin = p>0.2; star = peak.
Faint contours = the current gate Q1, to see the mismatch.
"""
import os
import sys
import numpy as np
from scipy.linalg import eig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/analytic")
import _saai  # noqa
from _saai import C_V
from explore_lsb_frozen_profile import build_profile, ETA_MAX
from explore_lambda_v_absmargin_plane import indicators_at
from lib.correlations import Re_theta0

FIGD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs_explore')
ATTACHED = [(-0.1988, None), (-0.15, None), (-0.09, None), (0.0, None),
            (0.30, None), (1.0, None)]
REVERSED = [(-0.19, -0.03), (-0.17, -0.06), (-0.15, -0.08), (-0.12, -0.10)]


def q1(GA, LV):
    return 1.0 - np.sqrt(np.clip(GA*(2-GA), 0, None))/np.sqrt(1+(C_V*LV)**2)


def os_mode(pr, Re_th, N=420, alo=0.03, ahi=1.4):
    """Most-amplified OS eigenfunction phi(y); return (yh, phi, alpha, c)."""
    Th = pr['Theta']
    ymax = min(ETA_MAX/Th, 30.0)
    yh = np.linspace(0.0, ymax, N)
    h = yh[1]-yh[0]
    U = np.interp(yh*Th, pr['eta'], pr['u'], right=1.0)
    Upp = Th*Th*np.interp(yh*Th, pr['eta'], pr['upp'], right=0.0)
    n = N-2
    I = np.eye(n)
    D2 = (np.diag(np.ones(n-1), -1)-2*I+np.diag(np.ones(n-1), 1))/h**2
    D4 = (np.diag(np.ones(n-2), -2)-4*np.diag(np.ones(n-1), -1)+6*I
          - 4*np.diag(np.ones(n-1), 1)+np.diag(np.ones(n-2), 2))/h**4
    D4[0, 0] = 7.0/h**4
    D4[-1, -1] = 7.0/h**4
    Ui, Uppi = U[1:-1], Upp[1:-1]

    def solve(al, vec=False):
        L2 = D2-al**2*I
        A = (np.diag(Ui)@L2 - np.diag(Uppi)
             - (D4-2*al**2*D2+al**4*I)/(1j*al*Re_th))
        c, V = eig(A, L2, right=True)
        ok = np.isfinite(c) & (np.real(c) > 0.02) & (np.real(c) < 0.98)
        idx = np.where(ok)[0][int(np.argmax(np.imag(c[ok])))]
        return (complex(c[idx]), V[:, idx]) if vec else \
            al*float(np.imag(c[idx]))
    ag = np.geomspace(alo, ahi, 16)
    wg = [solve(a) for a in ag]
    from scipy.optimize import minimize_scalar
    j = int(np.argmax(wg))
    r = minimize_scalar(lambda a: -solve(a),
                        bounds=(ag[max(j-1, 0)], ag[min(j+1, 15)]),
                        method='bounded', options={'xatol': 2e-3})
    astar = float(r.x)
    c, v = solve(astar, vec=True)
    phi = np.zeros(N, complex)
    phi[1:-1] = v
    return yh, phi, astar, c, astar*c.imag > 0


def production(yh, phi, alpha, Uprime):
    dphi = np.gradient(phi, yh[1]-yh[0])
    p = 0.5*alpha*np.abs(np.imag(dphi*np.conj(phi)))*np.abs(Uprime)
    return p


def contiguous(mask, jp):
    """largest contiguous True run containing index jp."""
    out = np.zeros_like(mask)
    if not mask[jp]:
        return out
    i = jp
    while i >= 0 and mask[i]:
        out[i] = True; i -= 1
    i = jp+1
    while i < len(mask) and mask[i]:
        out[i] = True; i += 1
    return out


def main():
    os.makedirs(FIGD, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    lv = np.linspace(-2.6, 1.4, 400); ga = np.linspace(0, 2, 300)
    LV, GA = np.meshgrid(lv, ga)
    cs = ax.contour(LV, GA, q1(GA, LV), levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                    colors='0.8', linewidths=0.6, zorder=0)
    ax.clabel(cs, fmt='%.1f', fontsize=6, colors='0.6')

    cmap = plt.cm.turbo
    fam = [(b, g) for b, g in ATTACHED] + [(b, g) for b, g in REVERSED]
    data = []
    for beta, guess in fam:
        pr = build_profile(beta, guess)
        Re_th = (200.0 if guess else
                 float(min(2.0*Re_theta0(np.clip(pr['H'], 2.2, None)), 6000.0)))
        try:
            yh, phi, al, c, unst = os_mode(pr, Re_th)
        except Exception as e:
            print(f"  beta={beta}: OS failed ({e})"); continue
        Th = pr['Theta']
        Up = Th*np.interp(yh*Th, pr['eta'], pr['up'], right=0.0)
        p = production(yh, phi, al, Up)
        p /= p.max()
        G, L = indicators_at(pr, yh)
        jp = int(np.argmax(p))
        data.append((pr['H'], G, L, p, jp))
        core = contiguous(p > 0.5, jp)
        print(f"beta={beta:+.3f} H={pr['H']:6.2f} c={c.real:+.2f}"
              f"{'(ABS)' if unst else '(conv)'}: peak (Lv={L[jp]:+.2f},"
              f"Gamma={G[jp]:.2f}); prod-core Gamma"
              f"[{G[core].min():.2f},{G[core].max():.2f}] "
              f"Lv[{L[core].min():+.2f},{L[core].max():+.2f}]", flush=True)

    Hmin = min(d[0] for d in data); Hmax = max(d[0] for d in data)
    for H, G, L, p, jp in data:
        col = cmap((np.log(H)-np.log(Hmin))/(np.log(Hmax)-np.log(Hmin)))
        wide = contiguous(p > 0.2, jp)
        core = contiguous(p > 0.5, jp)
        ow = np.argsort(G[wide])
        ax.plot(L[wide][ow], G[wide][ow], '-', color=col, lw=1.2, alpha=0.45)
        oc = np.argsort(G[core])
        ax.plot(L[core][oc], G[core][oc], '-', color=col, lw=3.4, alpha=0.9,
                solid_capstyle='round')
        ax.plot(L[jp], G[jp], '*', ms=12, mfc='white', mec=col, mew=1.4,
                zorder=6)
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.matplotlib.colors.LogNorm(Hmin, Hmax))
    plt.colorbar(sm, ax=ax, label='shape factor $H$')
    ax.axhline(1, color='0.6', lw=0.6, ls=':')
    ax.axvline(0, color='0.6', lw=0.6, ls=':')
    ax.set_xlim(-2.6, 1.4); ax.set_ylim(0, 2.02)
    ax.set_xlabel(r'$\Lambda_v$'); ax.set_ylabel(r'$\Gamma$')
    ax.set_title('Wavepacket energy-PRODUCTION core (Orr–Sommerfeld);\n'
                 r'thick $p>0.5$, thin $p>0.2$, $\star$ peak; faint = current '
                 '$Q_1$', fontsize=10)
    plt.tight_layout()
    out = os.path.join(FIGD, 'wavepacket_regions.png')
    plt.savefig(out, dpi=145)
    print('wrote', out)


if __name__ == '__main__':
    main()
