"""fig:kernel -> paper/figs/kernel_maps.pdf.

(a) universal kernel S = rate/a_max over (Re_Omega/Re_Omega^c, Gamma), evaluated by
the SAME src amplification-rate function, with the OPERATIVE Gamma of the two
anchor profiles printed (the map itself is left unannotated): the most-amplifying Gamma (argmax of the gated product
Q*S at Re far above onset) is 0.865 on Blasius -- the gate shuts the Gamma->1
wall, so an attached layer amplifies in the band, not at Gamma=1 -- and 1.575
on the separation-limit profile (saturated shoulder);
(b) cliff Re_Omega^c(lambda_p). All constants imported."""
import _saai
from _saai import A_MAX, G_C, FLOOR, K_LAMBDA, S_SLOPE
from lib.aft_sources import compute_composite_gate
from lib.aft_sources import compute_aft_amplification_rate
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp


def most_amplifying_gamma(beta, n=1400):
    """argmax over the profile of the gated rate Q*S at Re >> onset (barrier=1);
    returns (Gamma, Q*S) there. NOTE the map of panel (a) shows S ALONE: at the
    Blasius operating point S = 0.18 but Q = 0.34, so the realized fraction of
    the ceiling is Q*S = 0.06 -- the TS-vs-KH gap is carried jointly by the
    gate and the sigmoid tail."""
    m = beta/(2-beta)
    def ode(x, y): return np.vstack([y[1], y[2], -0.5*(m+1)*y[0]*y[2] - m*(1-y[1]**2)])
    def bc(a, b): return np.array([a[0], a[1], b[1]-1])
    x = np.linspace(0, 10, n); y0 = np.vstack([x-1+np.exp(-x), 1-np.exp(-x), np.exp(-x)])
    s = solve_bvp(ode, bc, x, y0, max_nodes=500000, tol=1e-9)
    eta = x; f, fp, fpp = s.sol(eta)
    fppp = -0.5*(m+1)*f*fpp - m*(1-fp**2)
    den = fp**2 + (eta*fpp)**2 + 1e-30
    G = 2*(eta*fpp)**2/den
    Q = np.asarray(compute_composite_gate(fpp, fppp, fp, eta))
    S = 1.0/(1.0 + np.exp(-S_SLOPE*(G - G_C)))
    i = int(np.argmax(Q*S))
    return float(G[i]), float(Q[i]*S[i])

def S_of_ratio(r, G):
    """S(z)=rate/a_max via the canonical kernel; lambda_p=0 so cliff=floor, ratio r=Re_Omega/floor."""
    r = np.asarray(r, float); G = np.asarray(G, float)
    rate = np.asarray(compute_aft_amplification_rate(r*FLOOR, G, lambda_p=0.0))
    return rate/A_MAX

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.3))
r = np.logspace(0, 1, 700); G = np.linspace(0, 2, 400)
RR, GG = np.meshgrid(r, G); Sv = S_of_ratio(RR, GG)
levels = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 0.5, 0.7, 0.9]
cs = axL.contour(RR, GG, Sv, levels=levels, colors='k', linewidths=0.8)
axL.clabel(cs, fontsize=7, fmt='%g')
axL.axvline(1.0, color='0.5', ls=':', lw=1.1)
axL.text(1.015, 0.10, 'cliff', color='0.35', fontsize=8, rotation=90, va='bottom')
axL.axhline(1.0, color='0.7', ls='-', lw=0.6)
axL.text(3.2, 1.04, r'$\Gamma=1$ (attached)', color='0.5', fontsize=7.5)
axL.text(3.2, 1.90, r'$\Gamma\to2$ (inflectional)', color='0.5', fontsize=7.5)
G_BL, QS_BL = most_amplifying_gamma(0.0)
G_SEP, QS_SEP = most_amplifying_gamma(-0.1988)
print(f"most-amplifying Gamma: Blasius {G_BL:.3f} (QS={QS_BL:.3f}), "
      f"separation limit {G_SEP:.3f} (QS={QS_SEP:.3f})")
axL.set_xscale('log'); axL.set_xlim(1, 10); axL.set_ylim(0, 2)
axL.set_xlabel(r'$Re_\Omega/Re_\Omega^{\mathrm{c}}$'); axL.set_ylabel(r'$\Gamma$')
axL.text(0.03, 0.97, '(a)', transform=axL.transAxes, fontsize=11, va='top', fontweight='bold')

lam = np.linspace(-0.5, 1.3, 500)
cliff = FLOOR*np.exp(K_LAMBDA*np.maximum(0.0, lam))
axR.plot(lam, cliff, 'k-', lw=1.6)
axR.axhline(FLOOR, color='0.6', ls=':', lw=0.9)
axR.text(0.4, FLOOR*1.15, r'floor $Re_\Omega^{\mathrm{f}}$', color='0.35', fontsize=8)
axR.axvline(0.0, color='0.7', ls='-', lw=0.6)
axR.set_yscale('log'); axR.set_xlim(-0.5, 1.3); axR.set_ylim(100, 3e6)
axR.set_xlabel(r'$\lambda_p$'); axR.set_ylabel(r'$Re_\Omega^{\mathrm{c}}(\lambda_p)$')
axR.text(0.03, 0.97, '(b)', transform=axR.transAxes, fontsize=11, va='top', fontweight='bold')
axR.text(-0.46, 900, r'ZPG / adverse' '\n' r'($\lambda_p\leq0$): floor', color='0.4', fontsize=7.5, va='bottom')
axR.annotate('favorable ($\\lambda_p>0$):\nonset delay',
             xy=(0.85, FLOOR*np.exp(K_LAMBDA*0.85)), xytext=(0.05, 8e4),
             color='0.4', fontsize=7.5, arrowprops=dict(arrowstyle='->', color='0.5', lw=1.0))
plt.tight_layout(); plt.savefig('figs/kernel_maps.pdf')
print('wrote figs/kernel_maps.pdf')
