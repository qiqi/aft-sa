"""Section III: derivation of K_lambda from the worst-point criterion.

Two figures:
 (1) figs/klambda_profiles.pdf -- Falkner-Skan profiles (adverse through strong
     favorable), each at its own Drela-Giles critical Re_theta, in the true
     (Re_Omega, Gamma) plane, colored by the LOCAL amplification rate S the solver
     computes pointwise; the star on each is the WORST point (max S) where growth
     triggers. Adverse profiles arch over Gamma=1 into the hot zone; favorable
     profiles stay Gamma<1 and only ignite weakly, near the wall.
 (2) figs/klambda_selfconsistent.pdf -- K_lambda implied vs K_lambda used, where
        K_lambda = (d ln Re_theta_c / d beta) / (d lambda_p^worst / d beta)|_{beta->0}.
     The worst point moves with K_lambda, so the criterion is self-consistent;
     the fixed point (implied = used) is K_lambda ~ 9.7 ~ 10.

Falkner-Skan via solve_bvp (robust for beta>1). Constants match ModelConstants.h.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from scipy.integrate import solve_bvp
S_SLOPE, G_C, P, FLOOR = 5.263, 1.572, 4.0, 100.0

def fs(beta, n=1400):
    m = beta/(2-beta)
    def ode(x, y): return np.vstack([y[1], y[2], -0.5*(m+1)*y[0]*y[2] - m*(1-y[1]**2)])
    def bc(a, b): return np.array([a[0], a[1], b[1]-1])
    x = np.linspace(0, 10, n); y0 = np.vstack([x-1+np.exp(-x), 1-np.exp(-x), np.exp(-x)])
    s = solve_bvp(ode, bc, x, y0, max_nodes=500000, tol=1e-9); Y = s.sol(x); return x, Y[1], Y[2]

def Re_theta_c(H):
    Hk = np.clip(H, 1.05, None)
    return 10.0**((1.415/(Hk-1)-0.489)*np.tanh(20.0/(Hk-1)-12.9) + 3.295/(Hk-1)+0.44)

def Sz(r, G):
    r = np.asarray(r, float)
    bar = np.where(r > 1, np.maximum(1-(1/np.maximum(r, 1+1e-9))**P, 1e-300), 1e-300)
    return np.where(r > 1, 1/(1+np.exp(-(S_SLOPE*(G-G_C)+np.log(bar)))), 0.0)

def prof(beta):
    eta, fp, fpp = fs(beta)
    th = np.trapezoid(fp*(1-fp), eta); H = np.trapezoid(1-fp, eta)/th
    Rc = Re_theta_c(H); sqRex = Rc/th; m = beta/(2-beta)
    ReO = eta**2*fpp*sqRex
    G = 2*(eta*fpp)**2/(fp**2 + (eta*fpp)**2 + 1e-30)
    lam = m*eta**2/np.maximum(fp, 1e-6)
    return dict(ReO=ReO, G=G, lam=lam, H=H, Rc=Rc)

def worst_lam(pr, KL):
    S = Sz(pr['ReO']/(FLOOR*np.exp(KL*np.maximum(pr['lam'], 0))), pr['G'])
    return int(np.argmax(S)), S

# ---------- FIG 1: profiles colored by S, worst points starred ----------
KL = 10.0
betas = [-0.15, -0.10, 0.0, 0.3, 0.6, 1.0, 1.4]
fig, ax = plt.subplots(figsize=(7.4, 5.1))
norm = plt.Normalize(0, 0.12)
for b in betas:
    pr = prof(b); ReO, G, lam = pr['ReO'], pr['G'], pr['lam']
    S = Sz(ReO/(FLOOR*np.exp(KL*np.maximum(lam, 0))), G)
    pts = np.array([ReO, G]).T.reshape(-1, 1, 2); segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap='plasma', norm=norm); lc.set_array(S); lc.set_linewidth(2.6); ax.add_collection(lc)
    i = int(np.argmax(S)); ax.plot(ReO[i], G[i], '*', ms=14, mfc=cm.plasma(norm(S[i])), mec='k', mew=0.7)
    j = int(np.argmax(ReO)); ax.annotate(fr'$\beta$={b}', (ReO[j], G[j]), fontsize=7.5, ha='center', va='bottom')
ax.axhline(1, color='0.5', ls='--', lw=0.8); ax.text(1.1e4, 1.02, r'$\Gamma=1$', fontsize=8, color='0.4')
ax.set_xscale('log'); ax.set_xlim(10, 2e4); ax.set_ylim(0, 1.6)
ax.set_xlabel(r'$Re_\Omega$ (each profile at its Drela critical $Re_\theta$)'); ax.set_ylabel(r'$\Gamma$')
plt.colorbar(cm.ScalarMappable(norm=norm, cmap='plasma'), ax=ax, label=r'local amplification $S$  ($a=a_{\max}S$)')
plt.tight_layout(); plt.savefig('figs/klambda_profiles.pdf'); plt.savefig('/tmp/klambda_profiles.png', dpi=140); plt.close()

# ---------- FIG 2: K_used vs K_implied (self-consistent) ----------
bs = np.array([0.0, 0.02, 0.05, 0.10]); prs = [prof(b) for b in bs]
dlnRc_db = np.polyfit(bs, np.log([p['Rc'] for p in prs]), 1)[0]
Ku = np.linspace(5, 11.5, 27); Ki = []       # cap below the K>=12 worst-point degeneracy
for K in Ku:
    lw = np.array([prs[k]['lam'][worst_lam(prs[k], K)[0]] for k in range(len(bs))])
    Ki.append(dlnRc_db/np.polyfit(bs, lw, 1)[0])
Ki = np.array(Ki)
d = Ki - Ku; s = np.where(np.diff(np.sign(d)))[0][0]
Kfix = Ku[s] + (Ku[s+1]-Ku[s]) * (-d[s])/(d[s+1]-d[s])
fig, ax = plt.subplots(figsize=(5.4, 4.7))
ax.plot(Ku, Ki, 'b-', lw=2.2, label=r'$K_\lambda^{\rm implied}=\dfrac{d\ln Re_{\theta c}/d\beta}{d\lambda_p^{\rm worst}/d\beta}$')
ax.plot(Ku, Ku, 'k--', lw=1.0, label=r'$K_\lambda^{\rm implied}=K_\lambda^{\rm used}$')
ax.plot(Kfix, Kfix, 'ro', ms=10, zorder=5)
ax.annotate(fr'$K_\lambda\approx{Kfix:.1f}$', (Kfix, Kfix), textcoords='offset points', xytext=(8, -16), color='r', fontsize=12)
ax.text(0.40, 0.93, fr'$d\ln Re_{{\theta c}}/d\beta={dlnRc_db:.1f}$ (Drela $\times$ FS)',
        transform=ax.transAxes, fontsize=8.5, va='top', ha='left')
ax.set_xlim(5, 11.5); ax.set_ylim(8, 14)
ax.set_xlabel(r'$K_\lambda$ used (sets the worst point)'); ax.set_ylabel(r'$K_\lambda$ implied by slope match')
ax.legend(fontsize=8.5, loc='lower left'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('figs/klambda_selfconsistent.pdf'); plt.savefig('/tmp/klambda_selfconsistent.png', dpi=140)
print(f"d ln Re_theta_c/d beta = {dlnRc_db:.2f};  self-consistent K_lambda = {Kfix:.2f}")
print("wrote figs/klambda_profiles.pdf, figs/klambda_selfconsistent.pdf")
