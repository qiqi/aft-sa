"""Indicator sphere viewed down the Y=Z (velocity=shear) diagonal:
  vertical  v = X          (curvature) -> curvature poles at (0, +/-1)
  horizontal h = (Y-Z)/2^.5 (shear on the right, velocity on the left)
  wall (0, 1/2^.5, 1/2^.5) projects to the center (0,0).
The front hemisphere (depth=(Y+Z)/2^.5 >= 0) fills the unit disk; the back
hemisphere is folded on by the (Y,Z)->(-Y,-Z) symmetry of the rate (Q1 is
even in X,Y,Z separately once Q2 is dropped), so a profile crossing the rim
reappears reflected -- the "wrap".

Background: the current gate Q1 (should be ~1 at the curvature poles = free
shear). Overlaid: canonical profile trajectories + special points.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/analytic")
import _saai  # noqa
from _saai import C_V
from explore_lsb_frozen_profile import build_profile

FIGD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs_explore')
S2 = np.sqrt(2.0)


def q1_of(X, Y, Z):
    den = Y*Y + Z*Z + 1e-30
    G = 2*Y*Y/den
    Lv = -X*Y/den
    return 1.0 - np.sqrt(np.clip(G*(2-G), 0, None))/np.sqrt(1+(C_V*Lv)**2)


def project(X, Y, Z):
    """to (h, v); fold back hemisphere (Y+Z<0) onto front via (Y,Z)->(-Y,-Z)."""
    X, Y, Z = map(np.asarray, (X, Y, Z))
    back = (Y + Z) < 0
    Yf = np.where(back, -Y, Y)
    Zf = np.where(back, -Z, Z)
    Xf = X.copy()
    return (Yf - Zf)/S2, Xf


def fs_xyz(beta, guess=None):
    pr = build_profile(beta, guess)
    Th = pr['Theta']
    d = pr['eta']/Th
    m = (pr['eta'] < np.interp(0.999, np.maximum.accumulate(pr['u']),
                               pr['eta'])) & (d > 0.03)
    X = d[m]**2*(Th*Th*pr['upp'][m]); Y = d[m]*(Th*pr['up'][m]); Z = pr['u'][m]
    R = np.sqrt(X*X+Y*Y+Z*Z)+1e-30
    return X/R, Y/R, Z/R


def free_xyz(uf, upf, uppf, H=12.0):
    y = np.linspace(0.3, H+8, 500)
    X = y**2*uppf(y); Y = y*upf(y); Z = uf(y)
    R = np.sqrt(X*X+Y*Y+Z*Z)+1e-30
    return X/R, Y/R, Z/R


def main():
    os.makedirs(FIGD, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 7.4))
    # Q1 background on the front-hemisphere disk
    n = 400
    hh = np.linspace(-1, 1, n); vv = np.linspace(-1, 1, n)
    Hh, Vv = np.meshgrid(hh, vv)
    r2 = Hh**2 + Vv**2
    dep = np.sqrt(np.clip(1 - r2, 0, None))
    X = Vv; Y = (Hh + dep)/S2*S2/S2  # Y=(h+depth)/sqrt2? recompute below
    Y = (Hh + dep)/1.0/S2 * S2       # placeholder; set exactly:
    Y = (Hh*S2 + dep*S2)/2.0
    Z = (dep*S2 - Hh*S2)/2.0
    Q = np.where(r2 <= 1, q1_of(X, Y, Z), np.nan)
    pc = ax.contourf(Hh, Vv, Q, levels=np.linspace(0, 1, 11), cmap='Blues',
                     alpha=0.85)
    cs = ax.contour(Hh, Vv, Q, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors='0.4',
                    linewidths=0.6)
    ax.clabel(cs, fmt='%.1f', fontsize=6)
    plt.colorbar(pc, ax=ax, label=r'current gate $Q_1$', shrink=0.8)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color='k', lw=1.0))

    prof = [('Blasius', 'C0', fs_xyz(0.0)),
            ('sep limit', 'C1', fs_xyz(-0.1988)),
            ('Stewartson LSB', 'C3', fs_xyz(-0.15, -0.08)),
            ('mixing layer', 'C2',
             free_xyz(lambda y: np.tanh(y-12), lambda y: 1/np.cosh(y-12)**2,
                      lambda y: -2*np.tanh(y-12)/np.cosh(y-12)**2)),
            ('jet edge', 'C4',
             free_xyz(lambda y: 1/np.cosh(y-12)**2,
                      lambda y: -2*np.sinh(y-12)/np.cosh(y-12)**3,
                      lambda y: (-2+4*np.sinh(y-12)**2/np.cosh(y-12)**2)/np.cosh(y-12)**2)),
            ('wake', 'C5',
             free_xyz(lambda y: 1-0.7/np.cosh(y-12)**2,
                      lambda y: 1.4*np.sinh(y-12)/np.cosh(y-12)**3,
                      lambda y: -0.7*(-2+4*np.sinh(y-12)**2/np.cosh(y-12)**2)/np.cosh(y-12)**2))]
    for lab, col, (X, Y, Z) in prof:
        h, v = project(X, Y, Z)
        ax.plot(h, v, '.', color=col, ms=2.2, label=lab)

    def mark(G, Lv, col, mk, lab):
        X2 = Lv**2/(Lv**2+G/2); rem = 1-X2
        Y = np.sqrt(max(G*rem/2, 0)); Z = np.sqrt(max(rem-Y**2, 0))
        X = -Lv*rem/Y if Y else np.sqrt(X2)
        h, v = project(np.array([X]), np.array([Y]), np.array([Z]))
        ax.plot(h, v, mk, color=col, ms=13, mec='k', mew=1.0, label=lab)
    mark(1.0, 0.0, 'k', 'o', 'wall')
    mark(1.999, -2.0, 'k', 'X', 'dead-air crossing')
    ax.plot(0, 1, '^', color='0.2', ms=12, mec='k'); ax.plot(0, -1, 'v',
            color='0.2', ms=12, mec='k')
    ax.text(0.02, 1.04, 'curvature pole (+)', fontsize=8, ha='center')
    ax.text(0.02, -1.08, 'curvature pole (-)', fontsize=8, ha='center')
    ax.text(0.72, 0.02, 'shear', fontsize=8); ax.text(-0.86, 0.02,
            'velocity', fontsize=8)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.2, 1.2); ax.set_aspect('equal')
    ax.set_xlabel(r'$(Y-Z)/\sqrt{2}$  (shear $\leftrightarrow$ velocity)')
    ax.set_ylabel(r"$X=d^2u''/R$  (curvature)")
    ax.set_title('Indicator sphere down the $Y{=}Z$ diagonal; background = '
                 'current $Q_1$\n(open at the curvature poles = free shear)',
                 fontsize=10)
    ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.28, 0.5))
    plt.tight_layout()
    out = os.path.join(FIGD, 'sphere_curvview.png')
    plt.savefig(out, dpi=145)
    print('wrote', out)


if __name__ == '__main__':
    main()
