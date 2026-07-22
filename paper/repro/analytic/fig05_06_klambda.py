"""fig:worstpoint + fig:klambda_sc -> paper/figs/klambda_profiles.pdf,
paper/figs/klambda_selfconsistent.pdf.

Derivation of K_lambda from the most-amplified-point self-consistency. All kernel constants
(s, g_c, p, floor, c_A) and K_lambda imported; Drela critical Re from src.correlations."""
import _saai
from _saai import S_SLOPE, G_C, P, FLOOR, K_LAMBDA
from lib.aft_sources import compute_composite_gate
from lib.correlations import Re_theta0
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from scipy.integrate import solve_bvp

def fs(beta, n=1400):
    m = beta/(2-beta)
    def ode(x, y): return np.vstack([y[1], y[2], -0.5*(m+1)*y[0]*y[2] - m*(1-y[1]**2)])
    def bc(a, b): return np.array([a[0], a[1], b[1]-1])
    x = np.linspace(0, 10, n); y0 = np.vstack([x-1+np.exp(-x), 1-np.exp(-x), np.exp(-x)])
    s = solve_bvp(ode, bc, x, y0, max_nodes=500000, tol=1e-9); Y = s.sol(x); return x, Y[0], Y[1], Y[2]

def Re_theta_c(H):
    return float(Re_theta0(np.clip(H, 1.05, None)))

def Sz(r, G):
    # Sigmoid-barrier factor S(z) of eq:rate on the clearance ratio
    # r = Re_Omega/Re_Omega^c -- algebraically identical to lib.aft_sources'
    # kernel (constants imported from it); restated on similarity variables.
    r = np.asarray(r, float)
    bar = np.where(r > 1, np.maximum(1-(1/np.maximum(r, 1+1e-9))**P, 1e-300), 1e-300)
    return np.where(r > 1, 1/(1+np.exp(-(S_SLOPE*(G-G_C)+np.log(bar)))), 0.0)

def prof(beta):
    eta, f, fp, fpp = fs(beta); m = beta/(2-beta)
    fppp = -0.5*(m+1)*f*fpp - m*(1-fp**2)
    th = np.trapezoid(fp*(1-fp), eta); H = np.trapezoid(1-fp, eta)/th
    Rc = Re_theta_c(H); sqRex = Rc/th
    ReO = eta**2*fpp*sqRex
    G = 2*(eta*fpp)**2/(fp**2 + (eta*fpp)**2 + 1e-30)
    lam = m*eta**2/np.maximum(fp, 1e-6)
    # FINAL composite gate on similarity variables (scale-invariant; analytic
    # fppp replaces the solver's ring-averaged compact Laplacian):
    Q = np.asarray(compute_composite_gate(fpp, fppp, fp, eta))
    return dict(ReO=ReO, G=G, lam=lam, Q=Q, H=H, Rc=Rc)

def worst_lam(pr, KL):
    # Worst (triggering) point = wall-normal location of LARGEST AMPLIFICATION
    # RATE Q*S. With the derived floor the log barrier dominates the rate near
    # onset, so this is essentially the largest-clearance point; the clearance
    # maximum instead moves the fixed point 6.10 -> 6.29 (3%) -- the paper
    # states the rate definition and this gap.
    S = pr['Q']*Sz(pr['ReO']/(FLOOR*np.exp(KL*np.maximum(pr['lam'], 0))), pr['G'])
    if S.max() <= 0:
        r = pr['ReO']/(FLOOR*np.exp(KL*np.maximum(pr['lam'], 0)))
        return int(np.argmax(r)), S
    return int(np.argmax(S)), S


def main():
    KL = K_LAMBDA
    betas = [-0.15, -0.10, 0.0, 0.3, 0.6, 1.0, 1.4]
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    norm = plt.Normalize(-2.0, 2.0)
    for b in betas:
        pr = prof(b); ReO, G, lam = pr['ReO'], pr['G'], pr['lam']
        cliff = FLOOR*np.exp(KL*np.maximum(lam, 0)); r = np.maximum(ReO, 1e-12)/cliff
        c = np.log2(np.maximum(r, 1e-12))
        pts = np.array([ReO, G]).T.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap='coolwarm', norm=norm); lc.set_array(c); lc.set_linewidth(2.6); ax.add_collection(lc)
        i = worst_lam(pr, KL)[0]  # the determination's triggering point (rate max)
        ax.plot(ReO[i], G[i], '*', ms=14, mfc=cm.coolwarm(norm(c[i])), mec='k', mew=0.7)
        j = int(np.argmax(ReO)); ax.annotate(fr'$\beta$={b}', (ReO[j], G[j]), fontsize=7.5, ha='center', va='bottom')
    ax.axvline(FLOOR, color='0.4', ls=':', lw=1.1)
    ax.text(FLOOR*1.07, 1.45, fr'floor $Re_\Omega^{{\mathrm{{f}}}}={FLOOR:.0f}$', fontsize=8, color='0.3', rotation=90, va='top')
    ax.axhline(1, color='0.5', ls='--', lw=0.8); ax.text(1.1e4, 1.02, r'$\Gamma=1$', fontsize=8, color='0.4')
    ax.set_xscale('log'); ax.set_xlim(10, 2e4); ax.set_ylim(0, 1.6)
    ax.set_xlabel(r'$Re_\Omega$ (each profile at its Drela critical $Re_\theta$)'); ax.set_ylabel(r'$\Gamma$')
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap='coolwarm'), ax=ax,
                 label=r'cliff clearance $\log_2 [Re_\Omega/Re_\Omega^{\mathrm{c}}(\lambda_p)]$')
    plt.tight_layout(); plt.savefig('figs/klambda_profiles.pdf'); plt.close()

    bs = np.array([0.0, 0.02, 0.05, 0.10]); prs = [prof(b) for b in bs]
    dlnRc_db = np.polyfit(bs, np.log([p['Rc'] for p in prs]), 1)[0]
    Ku = np.linspace(3.5, 8.5, 26); Ki = []
    for K in Ku:
        lw = np.array([prs[k]['lam'][worst_lam(prs[k], K)[0]] for k in range(len(bs))])
        Ki.append(dlnRc_db/np.polyfit(bs, lw, 1)[0])
    Ki = np.array(Ki)
    d = Ki - Ku; s = np.where(np.diff(np.sign(d)))[0][0]
    Kfix = Ku[s] + (Ku[s+1]-Ku[s]) * (-d[s])/(d[s+1]-d[s])
    fig, ax = plt.subplots(figsize=(5.4, 4.7))
    ax.plot(Ku, Ki, 'b-', lw=2.2, label=r'$K_\lambda^{\rm implied}=\dfrac{d\ln Re_{\theta c}/d\beta}{d\lambda_p^{\star}/d\beta}$')
    ax.plot(Ku, Ku, 'k--', lw=1.0, label=r'$K_\lambda^{\rm implied}=K_\lambda^{\rm used}$')
    ax.plot(Kfix, Kfix, 'ro', ms=10, zorder=5)
    ax.annotate(fr'$K_\lambda\approx{Kfix:.1f}$', (Kfix, Kfix), textcoords='offset points', xytext=(8, -16), color='r', fontsize=12)
    ax.text(0.40, 0.93, fr'$d\ln Re_{{\theta c}}/d\beta={dlnRc_db:.1f}$ (Drela $\times$ FS)',
            transform=ax.transAxes, fontsize=8.5, va='top', ha='left')
    ax.set_xlim(3.5, 8.5); ax.set_ylim(4, 9.5)
    ax.set_xlabel(r'$K_\lambda$ used (sets the most-amplified point)'); ax.set_ylabel(r'$K_\lambda$ implied by slope match')
    ax.legend(fontsize=8.5, loc='lower left'); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig('figs/klambda_selfconsistent.pdf')
    print(f"d ln Re_theta_c/d beta = {dlnRc_db:.2f};  self-consistent K_lambda = {Kfix:.2f}")
    assert abs(Kfix - K_LAMBDA) < 0.1, \
        f"fixed point {Kfix:.2f} drifted from canonical K_lambda {K_LAMBDA}"
    print("wrote figs/klambda_profiles.pdf, figs/klambda_selfconsistent.pdf")


if __name__ == '__main__':
    main()
