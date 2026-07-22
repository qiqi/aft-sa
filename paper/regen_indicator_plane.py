"""Section II figure (figs/indicator_plane.pdf), two panels:
(a) boundary-layer profiles in the local-indicator plane (Re_Omega/Re_theta, Gamma);
(b) the band gate Q across the same profiles vs nondimensional wall distance y/d99.

Each Falkner-Skan similarity profile (adverse separation limit through strong
favorable) is traced from the wall outward. Normalizing Re_Omega by Re_theta
removes the Reynolds number, so the family is universal (Blasius peak at
Re_Omega ~ 2.2 Re_theta). Line style encodes the local pressure-gradient
parameter lambda_p: solid where lambda_p < 0.16 (adverse / ZPG / mild favorable),
dotted where lambda_p >= 0.16 (strong favorable; 0.16 ~ 1/K_lambda). Grayscale
shade runs adverse (dark) -> favorable (light). Falkner-Skan via solve_bvp.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
CUT = 0.16

def fs(beta, n=2400):
    m = beta/(2-beta)
    def ode(x, y): return np.vstack([y[1], y[2], -0.5*(m+1)*y[0]*y[2] - m*(1-y[1]**2)])
    def bc(a, b): return np.array([a[0], a[1], b[1]-1])
    x = np.linspace(0, 12, n); y0 = np.vstack([x-1+np.exp(-x), 1-np.exp(-x), np.exp(-x)])
    s = solve_bvp(ode, bc, x, y0, max_nodes=800000, tol=1e-9); Y = s.sol(x)
    return x, Y[0], Y[1], Y[2]

def prof_q(beta, cA=4.0):
    eta, f, fp, fpp = fs(beta)
    m = beta/(2-beta)
    fppp = -0.5*(m+1)*f*fpp - m*(1-fp**2)
    den = fp**2 + (eta*fpp)**2 + 1e-30
    G = 2*(eta*fpp)**2/den
    Gg = (fppp*eta**2)**2/den
    Q = 1.0 - np.sqrt(np.maximum(G*(2.0-G), 0.0))/(1.0 + cA*Gg)
    eta99 = np.interp(0.99, fp, eta)
    return eta/eta99, Q


def prof(beta):
    eta, f, fp, fpp = fs(beta)
    th = np.trapezoid(fp*(1-fp), eta); m = beta/(2-beta)
    x = eta**2*fpp/th                                  # Re_Omega / Re_theta (Re-independent)
    G = 2*(eta*fpp)**2/(fp**2 + (eta*fpp)**2 + 1e-30)
    lam = m*eta**2/np.maximum(fp, 1e-6)
    ipk = int(np.argmax(x)); outer = np.where(G[ipk:] < 0.008)[0]  # extend outward to Gamma->0
    e = (ipk + outer[0]) if len(outer) else len(G)
    return x[1:e], G[1:e], lam[1:e]

betas = [-0.198, -0.10, 0.0, 0.30, 1.00]
bmin, bmax = -0.198, 1.00
shade = lambda b: str(round(0.50*(b-bmin)/(bmax-bmin), 3))          # 0 (adverse) -> 0.5 (favorable)
# label anchor: (target Gamma on the descending branch, vertical align, dy pts)
place = {-0.198: (None, 'center', 0), -0.10: (None, 'center', 0), 0.0: (None, 'center', 0),
         0.30: (None, 'center', 0), 1.00: (0.15, 'center', 0)}

fig, (ax, axQ) = plt.subplots(1, 2, figsize=(11.8, 5.0))
for b in betas:
    x, G, lam = prof(b); c = 'k'
    soft = lam < CUT; strong = lam >= CUT
    ax.plot(np.where(soft, x, np.nan), np.where(soft, G, np.nan), '-', color=c, lw=2.4)
    ax.plot(np.where(strong, x, np.nan), np.where(strong, G, np.nan), ls=':', color=c, lw=2.8)
    tG, va, dy = place[b]
    if tG is None:                                     # label at Re_Omega peak
        i = int(np.argmax(x))
    else:                                              # label where descending branch hits tG
        ipk = int(np.argmax(x)); j = ipk + int(np.argmin(np.abs(G[ipk:]-tG))); i = j
    ax.annotate(fr'$\beta={b}$', (x[i], G[i]), fontsize=8.5, ha='left', va=va, color=c,
                xytext=(5, dy), textcoords='offset points')
ax.axhline(1, color='0.6', ls=(0, (4, 3)), lw=0.8)
ax.set_xscale('log'); ax.set_xlim(1e-3, 6.0); ax.set_ylim(0, 2.0)
ax.set_xlabel(r'$Re_\Omega/Re_\theta$'); ax.set_ylabel(r'$\Gamma$')
ax.text(0.03, 0.97, '(a)', transform=ax.transAxes, fontsize=11, va='top', fontweight='bold')

# --- (b) the band gate Q across the same profiles vs y/delta99 ---
# label each curve on its distinct rising branch, staggered in Q

for b in betas:
    yn, Q = prof_q(b)
    msk = (yn >= 2e-3) & (yn <= 1.45)         # drop the 0/0 point at the wall
    yn, Q = yn[msk], Q[msk]
    axQ.plot(yn, Q, '-', color='k', lw=2.4)
    if b == -0.198:                           # separation limit: label above its inner peak
        j = int(np.argmax(np.where((yn > 0.05) & (yn < 0.4), Q, 0.0)))
        axQ.annotate(fr'$\beta={b}$', (yn[j], Q[j]), fontsize=8.5, ha='center',
                     xytext=(0, 6), textcoords='offset points')
    else:
        lev = {-0.10: 0.62, 0.0: 0.45, 0.30: 0.28, 1.00: 0.12}[b]
        up = np.where((Q[:-1] <= lev) & (Q[1:] > lev) & (yn[:-1] > 0.05) & (yn[:-1] < 1.0))[0]
        j = up[-1]
        axQ.annotate(fr'$\beta={b}$', (yn[j], Q[j]), fontsize=8.5, ha='left', va='top',
                     xytext=(5, -2), textcoords='offset points')
axQ.set_xlim(0, 1.45); axQ.set_ylim(0, 1.02)
axQ.set_xlabel(r'$y/\delta_{99}$'); axQ.set_ylabel(r'band gate $Q$')
axQ.text(0.03, 0.97, '(b)', transform=axQ.transAxes, fontsize=11, va='top', fontweight='bold')
plt.tight_layout()
plt.savefig('figs/indicator_plane.pdf'); plt.savefig('/tmp/indicator_plane.png', dpi=150)
print("wrote figs/indicator_plane.pdf")
