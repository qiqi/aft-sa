"""The universal thin-bubble parabola on the gate plane (user derivation).

Any quadratic velocity profile with a wall zero, u = a y (y - y0) --
arbitrary curvature and crossing height, INCLUDING arbitrary wall shear
(u = y(b + a y) is the same family) -- traces ONE parameter-free curve in
(Gamma, Lambda_v) space:

    Gamma(t)    = 2 / (1 + ((t-1)/(2t-1))^2),          t = y/y0,
    Lambda_v(t) = -2 t (2t-1) / ((2t-1)^2 + (t-1)^2),

with landmarks: wall t->0 -> (1, 0) [= the Q1 pinch]; velocity minimum
t=1/2 -> (0, 0); ZERO CROSSING t=1 -> (2, -2) [= the Q2 anchor];
t->inf -> (8/5, -4/5).

Significance: the parabola is the near-wall Taylor profile at vanishing
wall shear under a pressure gradient -- the ideal THIN laminar separation
bubble (viscosity holds curvature constant = p_x/mu). Constant curvature
means no interior inflection: such a bubble is far from absolute
instability, so in-place growth at its crossing is maximally unphysical --
the gate must pin it. Thickening toward absolute instability = curvature
decays toward the bubble edge, the inflection descends, and the crossing
coordinate rises from -2 toward 0 -- walking OUT of the pocket. Both gate
pinches are thus the two stagnation points of one universal profile.

The pocket CENTER is offset to L0 = -1.8 (from the anchor -2) because the
amplifying eigenmode is smeared by diffusion from the crossing toward the
inflection (frozen-instrument measurement; see results_qmod_*).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs_explore')
CV, L0, C2 = 4.0, -1.8, 2.0

t = np.concatenate([np.linspace(1e-4, 0.499, 400),
                    np.linspace(0.501, 12.0, 1200)])
r = (t - 1)/(2*t - 1)
G = 2/(1 + r*r)
L = -2*t*(2*t - 1)/((2*t - 1)**2 + (t - 1)**2)

lv = np.linspace(-3.2, 1.6, 801)
ga = np.linspace(0.0, 2.0, 401)
LV, GA = np.meshgrid(lv, ga)
band = np.sqrt(np.clip(GA*(2 - GA), 0, None))
Q1 = 1.0 - band/np.sqrt(1 + (CV*LV)**2)
Q2 = 1.0 - np.clip(GA - 1, 0, None)**2/(1 + C2*(LV - L0)**2)

fig, ax = plt.subplots(figsize=(8.6, 6.0))
cs = ax.contour(LV, GA, Q1*Q2, levels=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                colors='0.35', linewidths=0.8, linestyles=':')
ax.clabel(cs, fmt='%.2g', fontsize=7)
ax.plot(L, G, 'C3-', lw=2.2, label='universal thin-bubble parabola '
        r'$u = a\,y(y-y_0)$ (all $a$, $y_0$)')
ax.plot(0, 1, 'ko', ms=9, mfc='w', mew=1.8)
ax.annotate('wall zero $(1,0)$ = $Q_1$ pinch', (0, 1), fontsize=9,
            textcoords='offset points', xytext=(10, 6))
ax.plot(-2, 2, 'ks', ms=9, mfc='w', mew=1.8)
ax.annotate('crossing $(2,-2)$ = $Q_2$ anchor', (-2, 2), fontsize=9,
            textcoords='offset points', xytext=(8, -14))
ax.plot(0, 0, 'k^', ms=8, mfc='w', mew=1.5)
ax.annotate('velocity minimum $(0,0)$', (0, 0), fontsize=8.5,
            textcoords='offset points', xytext=(8, 4))
ax.plot(-0.8, 1.6, 'kd', ms=7, mfc='w', mew=1.2)
ax.annotate(r'$t\to\infty$: $(8/5, -4/5)$', (-0.8, 1.6), fontsize=8.5,
            textcoords='offset points', xytext=(8, 4))
ax.axvline(L0, color='C0', lw=1.0, ls='--')
ax.text(L0 + 0.04, 0.25, f'pocket center $\\Lambda_0$={L0}\n(mode offset '
        'from the $-2$ anchor)', fontsize=8, color='C0')
ax.set_xlim(-3.2, 1.6)
ax.set_ylim(0, 2.02)
ax.set_xlabel(r'$\Lambda_v$')
ax.set_ylabel(r'$\Gamma$')
ax.set_title('Both gate pinches are the two stagnation points of the '
             'universal thin-bubble parabola', fontsize=10.5)
ax.legend(fontsize=8.5, loc='lower left')
ax.grid(alpha=0.25)
plt.tight_layout()
out = os.path.join(FIGD, 'parabola_universal.png')
plt.savefig(out, dpi=140)
print('wrote', out)
