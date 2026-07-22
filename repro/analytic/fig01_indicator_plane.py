"""fig:indicatorplane -> paper/figs/indicator_plane.pdf.

(a) Falkner-Skan profiles in the local-indicator plane (Re_Omega/Re_theta, Gamma);
(b) the band gate Q across the same profiles vs y/delta99. c_A imported canonical."""
import _saai
from _saai import C_A
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

def prof_q(beta, cA=C_A):
    eta, f, fp, fpp = fs(beta); m = beta/(2-beta)
    fppp = -0.5*(m+1)*f*fpp - m*(1-fp**2)
    den = fp**2 + (eta*fpp)**2 + 1e-30
    G = 2*(eta*fpp)**2/den; Gg = (fppp*eta**2)**2/den
    Q = 1.0 - np.sqrt(np.maximum(G*(2.0-G), 0.0))/(1.0 + cA*Gg)
    eta99 = np.interp(0.99, fp, eta)
    return eta/eta99, Q

def prof(beta):
    eta, f, fp, fpp = fs(beta)
    th = np.trapezoid(fp*(1-fp), eta); m = beta/(2-beta)
    x = eta**2*fpp/th                                  # Re_Omega / Re_theta
    G = 2*(eta*fpp)**2/(fp**2 + (eta*fpp)**2 + 1e-30)
    lam = m*eta**2/np.maximum(fp, 1e-6)
    ipk = int(np.argmax(x)); outer = np.where(G[ipk:] < 0.008)[0]
    e = (ipk + outer[0]) if len(outer) else len(G)
    return x[1:e], G[1:e], lam[1:e]

betas = [-0.198, -0.10, 0.0, 0.30, 1.00]
fig, (ax, axQ) = plt.subplots(1, 2, figsize=(11.8, 5.0))
for b in betas:
    x, G, lam = prof(b); soft = lam < CUT; strong = lam >= CUT
    ax.plot(np.where(soft, x, np.nan), np.where(soft, G, np.nan), '-', color='k', lw=2.4)
    ax.plot(np.where(strong, x, np.nan), np.where(strong, G, np.nan), ls=':', color='k', lw=2.8)
    i = int(np.argmax(x)) if b != 1.00 else int(np.argmax(x)) + int(np.argmin(np.abs(G[int(np.argmax(x)):]-0.15)))
    ax.annotate(fr'$\beta={b}$', (x[i], G[i]), fontsize=8.5, ha='left', va='center',
                xytext=(5, 0), textcoords='offset points')
ax.axhline(1, color='0.6', ls=(0, (4, 3)), lw=0.8)
ax.set_xscale('log'); ax.set_xlim(1e-3, 6.0); ax.set_ylim(0, 2.0)
ax.set_xlabel(r'$Re_\Omega/Re_\theta$'); ax.set_ylabel(r'$\Gamma$')
ax.text(0.03, 0.97, '(a)', transform=ax.transAxes, fontsize=11, va='top', fontweight='bold')

for b in betas:
    yn, Q = prof_q(b); msk = (yn >= 2e-3) & (yn <= 1.45); yn, Q = yn[msk], Q[msk]
    axQ.plot(yn, Q, '-', color='k', lw=2.4)
    if b == -0.198:
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
plt.tight_layout(); plt.savefig('figs/indicator_plane.pdf')
print("wrote figs/indicator_plane.pdf")
