"""Fig (sec:calib): the onset cliff anchored to Drela + Falkner-Skan.

Falkner-Skan profiles (Blasius through strong favorable gradient, beta>1), each
placed at its OWN Drela-Giles critical Re_theta(H), traced in the normalized
(Re_Omega/Re_Omega^c, Gamma) plane of the kernel S(z). If the cliff form
Re_Omega^c(lambda_p) is right, every profile ignites just past the cliff
(ratio ~ 1) at a small amplification rate -- i.e. amplification turns on right at
the critical Reynolds number, uniformly across pressure gradients. The
low-Re_theta Blasius case is deliberately the hottest (thin layer -> the reduced
c_nu,aft=1/12 laminar diffusion is relatively stronger, so a bit more
amplification is needed for net growth); high-Re_theta favorable cases need less.

Falkner-Skan solved with solve_bvp (robust for beta>1, unlike shooting).
Constants match src/numerics/aft_sources.py and ModelConstants.h.
-> figs/collapse_cliff.pdf
"""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

S_SLOPE, G_C, P, FLOOR, K_LAM = 5.263, 1.572, 4.0, 100.0, 10.0

def fs_bvp(beta, eta_max=10.0, n=600):
    """Falkner-Skan (Blasius-consistent eta): f''' + (m+1)/2 f f'' + m(1-f'^2)=0."""
    m = beta / (2.0 - beta)
    def ode(x, y):
        return np.vstack([y[1], y[2], -0.5*(m+1)*y[0]*y[2] - m*(1 - y[1]**2)])
    def bc(ya, yb):
        return np.array([ya[0], ya[1], yb[1] - 1.0])
    x = np.linspace(0, eta_max, n)
    y0 = np.vstack([x - 1 + np.exp(-x), 1 - np.exp(-x), np.exp(-x)])
    s = solve_bvp(ode, bc, x, y0, max_nodes=200000, tol=1e-7)
    Y = s.sol(x)
    return x, Y[1], Y[2], s.success            # eta, f'=u/U_e, f''

def Re_theta_c(H):                              # Drela-Giles (1987) Eq.30, M=0
    Hk = np.clip(H, 1.05, None)
    return 10.0**((1.415/(Hk-1)-0.489)*np.tanh(20.0/(Hk-1)-12.9) + 3.295/(Hk-1)+0.44)

def Sz(r, G):
    r = np.asarray(r, float)
    bar = np.where(r > 1, np.maximum(1 - (1/np.maximum(r, 1+1e-9))**P, 1e-300), 1e-300)
    return np.where(r > 1, 1/(1+np.exp(-(S_SLOPE*(G-G_C)+np.log(bar)))), 0.0)

betas = [0.0, 0.3, 0.6, 1.0, 1.4, 1.8]
fig, ax = plt.subplots(figsize=(7.6, 5.2))
rr = np.logspace(-1, 2, 400); gg = np.linspace(0, 2, 320)
RR, GG = np.meshgrid(rr, gg)
cf = ax.contourf(RR, GG, Sz(RR, GG),
                 levels=[0, 1e-3, 1e-2, 3e-2, 1e-1, 0.3, 0.5, 0.7, 0.9],
                 cmap='Oranges', alpha=0.6)
plt.colorbar(cf, label=r'amplification rate $S(z)$', ax=ax)
ax.axvline(1.0, color='k', ls=':', lw=1.1)
ax.text(1.04, 0.05, 'cliff', rotation=90, fontsize=8, va='bottom')
cols = plt.cm.viridis(np.linspace(0, 0.92, len(betas)))
for b, c in zip(betas, cols):
    eta, fp, fpp, ok = fs_bvp(b)
    th = np.trapezoid(fp*(1-fp), eta); H = np.trapezoid(1-fp, eta)/th
    Rc = Re_theta_c(H); sqRex = Rc/th; m = b/(2-b)
    ReO = eta**2*fpp*sqRex
    lam = m*eta**2/np.maximum(fp, 1e-6)
    ratio = ReO/(FLOOR*np.exp(K_LAM*np.maximum(lam, 0)))
    G = 2*(eta*fpp)**2/(fp**2 + (eta*fpp)**2 + 1e-30)
    i = int(np.argmax(Sz(ratio, G)))
    ax.plot(ratio[1:], G[1:], color=c, lw=2.2,
            label=fr'$\beta$={b:.1f}, $H$={H:.2f}, $Re_{{\theta c}}$={Rc:.0f}')
    ax.plot(ratio[i], G[i], '*', color=c, ms=15, mec='k', mew=0.6)
ax.set_xscale('log'); ax.set_xlim(0.3, 30); ax.set_ylim(0, 2)
ax.set_xlabel(r'$Re_\Omega/Re_\Omega^{\mathrm{c}}$'); ax.set_ylabel(r'$\Gamma$')
ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig('figs/collapse_cliff.pdf'); plt.savefig('/tmp/collapse_cliff.png', dpi=140)
print('wrote figs/collapse_cliff.pdf')
