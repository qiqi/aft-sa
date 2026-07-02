"""Kernel map for Eq. (rate): S(z) over (Re_Omega [log], Gamma [0..2]) at the
baseline cliff (lambda_p<=0, cliff=floor=100). Favorable PG shifts the cliff
right to Re_Omega^c(lambda_p) = floor*exp(K_lambda*lambda_p), delaying growth
onset (illustrated) -- there is NO separate sigma_FPG rate factor.
Constants match src/numerics/aft_sources.py and ModelConstants.h.
-> figs/kernel_maps.pdf
"""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S_SLOPE, G_C, P, FLOOR, K_LAMBDA = 5.263, 1.572, 4.0, 100.0, 10.0

def S_of(ReO, G, cliff=FLOOR):
    ReO = np.asarray(ReO, float); G = np.asarray(G, float)
    safe = np.maximum(ReO, cliff + 1e-9)
    bar = np.maximum(1.0 - (cliff / safe) ** P, 1e-300)
    z = S_SLOPE * (G - G_C) + np.log(bar)
    return np.where(ReO > cliff, 1.0 / (1.0 + np.exp(-z)), 0.0)

fig, ax = plt.subplots(figsize=(6.2, 4.6))
ReO = np.logspace(2, 3, 600); G = np.linspace(0, 2, 400)
RG, GG = np.meshgrid(ReO, G)
Sv = S_of(RG, GG)
# exponentially spaced levels so the small (attached-profile) amplifications show
levels = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 0.5, 0.7, 0.9]
cs = ax.contour(RG, GG, Sv, levels=levels, colors='k', linewidths=0.8)
ax.clabel(cs, fontsize=7, fmt='%g')
ax.axhline(G_C, color='0.45', ls='--', lw=0.9)
ax.text(104, G_C + 0.05, r'$g_c=1.572$', color='0.3', fontsize=8)
# baseline cliff floor (lambda_p <= 0)
ax.axvline(FLOOR, color='0.5', ls=':', lw=1.0)
ax.text(FLOOR * 1.03, 1.75, r'$Re_\Omega^{\mathrm{f}}=100$', color='0.35', fontsize=7.5)
# FPG-shifted cliff for a sample lambda_p>0 (onset delay replaces sigma_FPG)
cl = FLOOR * np.exp(K_LAMBDA * 0.2)   # lambda_p=0.2 -> ~739
ax.axvline(cl, color='0.5', ls='-.', lw=1.0)
ax.text(cl * 1.04, 1.75, r'$Re_\Omega^{\mathrm{c}}(\lambda_p{=}0.2)$', color='0.35', fontsize=7.5)
ax.annotate('', xy=(cl, 0.42), xytext=(FLOOR, 0.42),
            arrowprops=dict(arrowstyle='->', color='0.4', lw=1.1))
ax.text(np.sqrt(FLOOR * cl), 0.50, 'FPG onset delay', color='0.4', fontsize=7.5, ha='center')
ax.set_xscale('log'); ax.set_xlim(100, 1000); ax.set_ylim(0, 2)
ax.set_xlabel(r'$Re_\Omega$'); ax.set_ylabel(r'$\Gamma$')
ax.set_title(r'Amplification-rate kernel $S(z)$', fontsize=10)
plt.tight_layout()
plt.savefig('figs/kernel_maps.pdf'); plt.savefig('/tmp/kernel_maps.png', dpi=140)
print('wrote figs/kernel_maps.pdf')
