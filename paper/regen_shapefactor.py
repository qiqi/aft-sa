"""Shape-factor dependence of the SA-AI amplification rate (Sec. III.A).

For each Falkner-Skan similarity profile (parametrized by its shape factor H) we
evaluate the model's amplification rate a = a_max * S(Gamma) at the theta-scale
amplifying band (Re_Omega > 1/2 max Re_Omega, band-mean Gamma, as the results use),
and convert it to an envelope slope dN/dRe_theta via the Falkner-Skan geometry
  dN/dRe_theta = 2 a f''(eta_peak) / [(m+1) I_theta],
normalized once to the Drela-Giles envelope at the Blasius point (H=2.59) -- the
single anchor already used to set (s, g_c) in Sec. III.A. The curve is overlaid on
the Drela-Giles envelope-rate correlation dn/dRe_theta(H) to show that the model
reproduces the shape-factor (pressure-gradient) dependence of the amplification
rate, with a deliberate overshoot near separation where the free-shear ceiling
a_max takes over.
-> figs/shapefactor_amplification.pdf
"""
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

A_MAX, S_SLOPE, G_C = 0.15, 5.263, 1.572   # must match ModelConstants.h / Sec. III

def fs(beta, n=1600):
    m = beta/(2-beta)
    def ode(x, y): return np.vstack([y[1], y[2], -0.5*(m+1)*y[0]*y[2] - m*(1-y[1]**2)])
    def bc(a, b): return np.array([a[0], a[1], b[1]-1])
    x = np.linspace(0, 10, n); y0 = np.vstack([x-1+np.exp(-x), 1-np.exp(-x), np.exp(-x)])
    s = solve_bvp(ode, bc, x, y0, max_nodes=500000, tol=1e-9); Y = s.sol(x)
    return x, Y[1], Y[2]

def drela(H):
    """Drela-Giles (1987) envelope amplification rate dn/dRe_theta(H)."""
    return 0.01*np.sqrt((2.4*H - 3.7 + 2.5*np.tanh(1.5*H - 4.65))**2 + 0.25)

def model_point(beta):
    m = beta/(2-beta); eta, fp, fpp = fs(beta)
    th = np.trapezoid(fp*(1-fp), eta); H = np.trapezoid(1-fp, eta)/th
    ReO = eta**2*fpp                                  # Re_Omega shape (Re-independent)
    r = eta*fpp/np.maximum(fp, 1e-9); G = 2*r**2/(1+r**2)
    ipk = int(np.argmax(ReO)); band = ReO > 0.5*ReO[ipk]
    Gb = float(np.mean(G[band]))                      # band-mean Gamma, theta-scale layer
    a = A_MAX/(1.0+np.exp(-S_SLOPE*(Gb - G_C)))       # rate (barrier off, post-onset)
    return H, a*2*fpp[ipk]/((m+1)*th)                 # H, raw dN/dRe_theta (unnormalized)

betas = np.concatenate([np.linspace(1.0, -0.15, 34), np.linspace(-0.16, -0.198, 8)])
H = np.array([model_point(b)[0] for b in betas])
raw = np.array([model_point(b)[1] for b in betas])
C = drela(2.591)/raw[np.argmin(np.abs(H-2.591))]      # one-point Blasius normalization
model = C*raw

Hg = np.linspace(H.min(), H.max(), 300)
fig, ax = plt.subplots(figsize=(5.4, 4.2))
ax.semilogy(Hg, drela(Hg), 'k--', lw=1.8, label=r'Drela--Giles envelope $dn/dRe_\theta(H)$')
ax.semilogy(H, model, '-', color='C0', lw=2.0, label='SA-AI implied rate')
iB = int(np.argmin(np.abs(H-2.591)))
ax.plot(2.591, model[iB], 'o', color='C3', ms=8, zorder=5)
ax.annotate('Blasius\n(anchor)', (2.591, model[iB]), (2.62, model[iB]*0.42), fontsize=8, color='C3')
ax.axvspan(3.5, H.max(), color='0.85', zorder=0)
ax.text(3.55, 0.09, 'separation\n(free-shear\nceiling)', fontsize=7.5, va='top')
ax.set_xlabel(r'shape factor $H=\delta^*/\theta$')
ax.set_ylabel(r'$dN/dRe_\theta$')
ax.set_xlim(H.min(), H.max()); ax.set_ylim(4e-3, 2e-1)
ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8.5, loc='upper left')
plt.tight_layout()
plt.savefig('figs/shapefactor_amplification.pdf'); plt.savefig('/tmp/shapefactor.png', dpi=140)
print('wrote shapefactor_amplification.pdf; Blasius norm C=%.1f' % C)
print('H range %.2f-%.2f' % (H.min(), H.max()))
