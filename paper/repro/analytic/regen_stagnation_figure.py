"""Figure for the attachment-anchored-branch appendix: standard SA on the
frozen Hiemenz field with EXACTLY zero freestream seed.

Single panel (the former left panel is now Table~\\ref{t:stagbistab} in the
whitepaper): line contours of log10 chi for the sustained L=3000 wedge
(flat-plate figure conventions: dashed = laminar levels chi<1, solid =
1, c_v1, 30), with velocity-magnitude contours |u|/u_e overlaid (steel
blue) so the thin Hiemenz momentum layer is visible against the much
taller sustained chi layer.

Inputs: data/stagnation_field_L3000.npz (x, y, chi; written by
stagnation_bistability.py --field). The Hiemenz velocity field is
recomputed here from the similarity solution on the loaded grid (no
re-solve): u = x f'(y), v = -f(y), |u| = sqrt(u^2+v^2), u_e = x.
-> figs/stagnation_bistability.pdf
Run from paper/: python3 repro/analytic/regen_stagnation_figure.py
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

_H = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(_H, '..', '..'))

fld = np.load(f'{PAPER}/data/stagnation_field_L3000.npz')
x, y, chi = fld['x'], fld['y'], fld['chi']
X, Y = np.meshgrid(x, y, indexing='ij')


def hiemenz():
    """f, f', f'' on a dense eta grid (shooting; f''(0)=1.232588)."""
    def rhs(t, yv):
        return [yv[1], yv[2], -(yv[0]*yv[2] + 1.0 - yv[1]**2)]
    sol = solve_ivp(rhs, [0, 20], [0.0, 0.0, 1.2325876568], dense_output=True,
                    rtol=1e-10, atol=1e-12)
    return sol.sol


# frozen Hiemenz velocity on the loaded grid (matches stagnation_bistability.py)
F = hiemenz()
fy = F(np.minimum(y, 20.0))
f, fp = fy[0], fy[1]
U = np.outer(x, fp)          # u(x,y) = x f'(y)
V = -np.tile(f, (len(x), 1))  # v(y)   = -f(y)
umag = np.sqrt(U**2 + V**2)  # ACTUAL magnitude: grows ~ u_e(x)=x away from stag
Uref = float(umag.max())     # edge speed at the downstream boundary (x=L)
umag_n = umag/Uref           # 0..1; contours are ~vertical outside the BL

fig, axf = plt.subplots(1, 1, figsize=(7.0, 3.2))

lg = np.log10(np.maximum(chi, 1e-12))
# chi contours (flat-plate conventions)
axf.contour(X, Y, lg, levels=[-6, -5, -4, -3, -2, -1], colors='0.55',
            linewidths=0.5, linestyles='dashed')   # per decade, chi<1
axf.contour(X, Y, lg, levels=[0.0], colors='k', linewidths=1.3)
axf.contour(X, Y, lg, levels=[np.log10(7.1), np.log10(30.0)], colors='k',
            linewidths=0.7)

# velocity-magnitude overlay |u| (actual, normalized by the x=L edge speed)
vl = [0.1, 0.2, 0.4, 0.6, 0.8]
cv = axf.contour(X, Y, umag_n, levels=vl, colors='#2166ac',
                 linewidths=0.9, linestyles='solid')
axf.clabel(cv, fmt='%.1f', fontsize=7, inline=True)

axf.set_xlabel('$x/\\delta$')
axf.set_ylabel('$y/\\delta$')
axf.set_ylim(0, 20)
axf.set_xlim(0, x.max())

# tiny legend proxies
from matplotlib.lines import Line2D
axf.legend([Line2D([0], [0], color='k', lw=1.3),
            Line2D([0], [0], color='0.55', lw=0.5, ls='--'),
            Line2D([0], [0], color='#2166ac', lw=0.9)],
           ['$\\chi=1,\\,c_{v1},\\,30$', '$\\chi<1$',
            '$|u|/u_{e,L}$'],
           fontsize=7, loc='upper left', frameon=False, ncol=3,
           handlelength=1.6, columnspacing=1.2)

fig.tight_layout()
out = f'{PAPER}/figs/stagnation_bistability.pdf'
fig.savefig(out)
fig.savefig('/tmp/stagnation_bistability.png', dpi=130)
print('wrote', out)
