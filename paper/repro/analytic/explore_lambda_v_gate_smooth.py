"""Contour comparison: current kinked gate vs the smooth form, plus the
proposed two-pinch product on the smooth basis.

  Q1 (current):  1 - sqrt(G(2-G)) / (1 + cV |Lv|)          (kink at Lv=0)
  Q1s (smooth):  1 - sqrt(G(2-G)) / sqrt(1 + (cV Lv)^2)
  Q1s*Q2:        second pinch at the recirculation zero-advection locus,
                 Q2 = 1 - (G-1)_+^2 / (1 + c2 (Lv - L0)^2), L0 = -2.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs_explore')
CV, C2, L0 = 1.0, 4.0, -2.0

lv = np.linspace(-3.2, 2.2, 801)
ga = np.linspace(0.0, 2.0, 401)
LV, GA = np.meshgrid(lv, ga)
band = np.sqrt(np.clip(GA*(2 - GA), 0, None))
Q1 = 1.0 - band/(1 + CV*np.abs(LV))
Q1s = 1.0 - band/np.sqrt(1 + (CV*LV)**2)
Q2 = 1.0 - np.clip(GA - 1, 0, None)**2/(1 + C2*(LV - L0)**2)

fig, axs = plt.subplots(1, 3, figsize=(15.6, 4.6), sharey=True)
levels = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
for ax, Q, t in ((axs[0], Q1, r'current: $1+c_V|\Lambda_v|$ (kinked)'),
                 (axs[1], Q1s, r'smooth: $\sqrt{1+(c_V\Lambda_v)^2}$'),
                 (axs[2], Q1s*Q2, 'smooth $Q_1 \\cdot Q_2$ '
                  f'(2nd pinch at ({L0}, 2), $c_2$={C2})')):
    cs = ax.contour(LV, GA, Q, levels=levels, colors='0.25',
                    linewidths=0.9)
    ax.clabel(cs, fmt='%.2g', fontsize=7)
    ax.plot(0, 1, 'ko', ms=6)
    ax.set_title(t, fontsize=10)
    ax.set_xlabel(r'$\Lambda_v$')
    ax.grid(alpha=0.25)
axs[2].plot(L0, 2, 'ks', ms=6)
axs[0].set_ylabel(r'$\Gamma$')
plt.tight_layout()
out = os.path.join(FIGD, 'lambda_v_gate_smooth.png')
plt.savefig(out, dpi=140)
print('wrote', out)
