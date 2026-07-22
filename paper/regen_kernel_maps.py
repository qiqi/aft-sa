"""Fig 1: the amplification-rate kernel of Eq.(rate), shown completely.

(a) S(z) over the NORMALIZED vorticity Reynolds number Re_Omega/Re_Omega^c and
    Gamma. Because the log-barrier depends only on the ratio Re_Omega/Re_Omega^c,
    this single map is UNIVERSAL -- it holds for every pressure gradient, with the
    onset cliff always at ratio = 1. This is the complete content of Eq.(rate).
(b) the cliff location Re_Omega^c(lambda_p) = floor*exp(K_lambda*max(0,lambda_p)):
    favorable pressure gradient is the ONLY thing that moves the cliff (onset
    delay); zero/adverse gradients sit at the floor.

Constants match src/numerics/aft_sources.py and ModelConstants.h.
-> figs/kernel_maps.pdf
"""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S_SLOPE, G_C, P, FLOOR, K_LAMBDA = 11.0, 1.005, 4.0, 254.0, 6.1

def S_of_ratio(r, G):
    """S(z), z = s(G - g_c) + ln(1 - (1/r)^p), for r = Re_Omega/Re_Omega^c."""
    r = np.asarray(r, float); G = np.asarray(G, float)
    safe = np.maximum(r, 1.0 + 1e-12)
    bar = np.maximum(1.0 - (1.0 / safe) ** P, 1e-300)
    z = S_SLOPE * (G - G_C) + np.log(bar)
    return np.where(r > 1.0, 1.0 / (1.0 + np.exp(-z)), 0.0)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.3))

# --- (a) universal kernel over (Re_Omega/Re_Omega^c, Gamma) ---
r = np.logspace(0, 1, 700)          # 1 .. 10
G = np.linspace(0, 2, 400)
RR, GG = np.meshgrid(r, G)
Sv = S_of_ratio(RR, GG)
levels = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 0.5, 0.7, 0.9]
cs = axL.contour(RR, GG, Sv, levels=levels, colors='k', linewidths=0.8)
axL.clabel(cs, fontsize=7, fmt='%g')
axL.axhline(G_C, color='0.45', ls='--', lw=0.9)
axL.text(1.06, G_C + 0.05, r'$g_c$', color='0.3', fontsize=9)
axL.axvline(1.0, color='0.5', ls=':', lw=1.1)                       # cliff at ratio = 1
axL.text(1.015, 0.10, 'cliff', color='0.35', fontsize=8, rotation=90, va='bottom')
axL.axhline(1.0, color='0.7', ls='-', lw=0.6)                       # attached asymptote
axL.text(3.2, 1.04, r'$\Gamma=1$ (attached)', color='0.5', fontsize=7.5)
axL.text(3.2, 1.90, r'$\Gamma\to2$ (inflectional)', color='0.5', fontsize=7.5)
axL.set_xscale('log'); axL.set_xlim(1, 10); axL.set_ylim(0, 2)
axL.set_xlabel(r'$Re_\Omega/Re_\Omega^{\mathrm{c}}$'); axL.set_ylabel(r'$\Gamma$')
axL.text(0.03, 0.97, '(a)', transform=axL.transAxes, fontsize=11, va='top', fontweight='bold')

# --- (b) cliff location vs lambda_p ---
lam = np.linspace(-0.5, 1.3, 500)
cliff = FLOOR * np.exp(K_LAMBDA * np.maximum(0.0, lam))
axR.plot(lam, cliff, 'k-', lw=1.6)
axR.axhline(FLOOR, color='0.6', ls=':', lw=0.9)
axR.text(0.4, FLOOR * 1.15, r'floor $Re_\Omega^{\mathrm{f}}=254$', color='0.35', fontsize=8)
axR.axvline(0.0, color='0.7', ls='-', lw=0.6)
axR.set_yscale('log'); axR.set_xlim(-0.5, 1.3); axR.set_ylim(100, 3e6)
axR.set_xlabel(r'$\lambda_p$'); axR.set_ylabel(r'$Re_\Omega^{\mathrm{c}}(\lambda_p)$')
axR.text(0.03, 0.97, '(b)', transform=axR.transAxes, fontsize=11, va='top', fontweight='bold')
axR.text(-0.46, 900, r'ZPG / adverse' '\n' r'($\lambda_p\leq0$): floor', color='0.4', fontsize=7.5, va='bottom')
axR.annotate('favorable ($\\lambda_p>0$):\nonset delay',
             xy=(0.85, FLOOR * np.exp(K_LAMBDA * 0.85)), xytext=(0.05, 8e4),
             color='0.4', fontsize=7.5,
             arrowprops=dict(arrowstyle='->', color='0.5', lw=1.0))

plt.tight_layout()
plt.savefig('figs/kernel_maps.pdf'); plt.savefig('/tmp/kernel_maps.png', dpi=140)
print('wrote figs/kernel_maps.pdf')
