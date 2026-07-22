"""SYSTEMATIC placement of the Q2 pocket (user program, replacing eyeball):

For LSB-like profiles the design data is the ZERO-VELOCITY-CROSSING
coordinate on the gate plane: at u=0, Gamma = 2 exactly and
Lambda_v* = -u'' d / u' (d = crossing height). Profiles that are NOT
absolutely unstable must have their crossing PROTECTED (damped); profiles
that ARE absolutely unstable must not be over-protected. The label comes
from the literature-validated Avanci-Rodriguez-Alves geometric criterion
(absolute <=> inflection below the zero-mass-flux line, y_i < y_b), which
collapsed the convective/absolute boundary across profile families.

Families (both literature-standard for LSB local analysis):
  * reversed Falkner-Skan (Stewartson branch) -- the classical LSB local
    model (Alam & Sandham's threshold was derived on these); spans the
    convective side (u_rev up to ~10%).
  * displaced-tanh over a wall (Hammond-Redekopp / Diwan-Ramesh spirit)
    with independent reverse-flow depth u_r and shear-layer height h --
    crosses the threshold into the absolute side.

Output: Lambda_v* vs the Avanci margin (y_b - y_i)/theta for both families
(+ FS-vs-tanh similarity check at matched u_rev), with the current pocket
(L0 = -1.8, half-width 1/sqrt(c2) = 0.71) overlaid.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/analytic")
import _saai  # noqa: F401
from explore_lsb_frozen_profile import build_profile

FIGD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs_explore')
L0, C2 = -1.8, 2.0

FS_REV = [(-0.19, -0.03), (-0.17, -0.06), (-0.15, -0.08), (-0.135, -0.09),
          (-0.12, -0.10), (-0.10, -0.105), (-0.08, -0.108)]
TANH_H = (3.0, 4.0, 6.0, 8.0)
TANH_UR = (0.03, 0.06, 0.10, 0.14, 0.18, 0.22, 0.28, 0.35)
DW = 1.0


def tanh_profile(h, ur, N=1600, ymax=None):
    ymax = ymax or max(h + 14.0, 24.0)
    y = np.linspace(0.0, ymax, N)
    u = (0.5*(1 - ur) + 0.5*(1 + ur)*np.tanh(y - h)) \
        * (1.0 - np.exp(-(y/DW)**2))
    dy = y[1] - y[0]
    up = np.gradient(u, dy)
    upp = np.gradient(up, dy)
    th = np.trapezoid(np.clip(u, 0, None)*(1 - np.clip(u, 0, None)), y)
    return dict(y=y, u=u, up=up, upp=upp, theta=max(th, 1e-9))


def crossing_and_label(y, u, up, upp, theta):
    """(u_rev, Lambda_v* at the u=0 crossing, Avanci margin (y_b-y_i)/theta).
    Returns None if there is no recirculation."""
    jm = int(np.argmin(u))
    if u[jm] >= 0:
        return None
    urev = -float(u[jm])
    above = jm + np.where(u[jm:] > 0.0)[0]
    if not len(above):
        return None
    j0 = int(above[0])
    # crossing height and local derivatives (linear interp around j0)
    f = -u[j0-1]/(u[j0] - u[j0-1])
    y0 = float(y[j0-1] + f*(y[j0] - y[j0-1]))
    up0 = float(up[j0-1] + f*(up[j0] - up[j0-1]))
    upp0 = float(upp[j0-1] + f*(upp[j0] - upp[j0-1]))
    lv_star = -upp0*y0/max(up0, 1e-12)
    # Avanci: shear-layer inflection (max du/dy above the minimum) vs the
    # zero net-mass-flux height
    j_i = jm + int(np.argmax(up[jm:]))
    cum = np.concatenate([[0.0], np.cumsum(0.5*(u[1:] + u[:-1])*np.diff(y))])
    ab = np.where(cum[jm:] > 0.0)[0]
    j_b = jm + (int(ab[0]) if len(ab) else len(y) - 1 - jm)
    margin = (float(y[j_b]) - float(y[j_i]))/theta
    return urev, lv_star, margin


def main():
    os.makedirs(FIGD, exist_ok=True)
    rows_fs, rows_th = [], []
    print("reversed FS (Stewartson branch):")
    for beta, guess in FS_REV:
        try:
            pr = build_profile(beta, guess)
        except Exception as e:
            print(f"  beta={beta}: FS solve failed ({e})")
            continue
        r = crossing_and_label(pr['eta']/pr['Theta'], pr['u'],
                               pr['Theta']*pr['up'],
                               pr['Theta']**2*pr['upp'], 1.0)
        if r is None:
            print(f"  beta={beta}: no recirculation")
            continue
        urev, lvs, marg = r
        rows_fs.append((urev, lvs, marg, pr['H']))
        print(f"  beta={beta:+.3f} H={pr['H']:6.2f}: u_rev={urev*100:5.1f}%  "
              f"Lv*={lvs:+.2f}  margin={marg:+.2f} "
              f"({'ABS' if marg > 0 else 'conv'})", flush=True)

    print("displaced-tanh family:")
    for h in TANH_H:
        for ur in TANH_UR:
            p = tanh_profile(h, ur)
            r = crossing_and_label(p['y']/p['theta'], p['u'],
                                   p['theta']*p['up'],
                                   p['theta']**2*p['upp'], 1.0)
            if r is None:
                continue
            urev, lvs, marg = r
            rows_th.append((urev, lvs, marg, h))
            print(f"  h={h:4.1f} ur={ur:4.2f}: u_rev={urev*100:5.1f}%  "
                  f"Lv*={lvs:+.2f}  margin={marg:+.2f} "
                  f"({'ABS' if marg > 0 else 'conv'})", flush=True)

    fs = np.array(rows_fs)
    th = np.array(rows_th)
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.8))
    ax = axs[0]
    ax.scatter(fs[:, 2], fs[:, 1], c='C3', marker='s', s=45,
               label='reversed FS (Stewartson)')
    for hval, mk in zip(TANH_H, ('o', '^', 'v', 'D')):
        sel = th[th[:, 3] == hval]
        ax.scatter(sel[:, 2], sel[:, 1], marker=mk, s=38,
                   c=np.where(sel[:, 2] > 0, 'C2', 'C0'),
                   label=f'tanh h={hval:g} (green=ABS)')
    ax.axvline(0, color='r', ls='--', lw=1.0)
    ax.text(0.02, 0.95, 'Avanci threshold', transform=ax.transAxes,
            fontsize=8, color='r')
    ax.axhline(L0, color='0.3', lw=1.0)
    ax.axhspan(L0 - 1/np.sqrt(C2), L0 + 1/np.sqrt(C2), color='0.8',
               alpha=0.5)
    ax.text(ax.get_xlim()[0]*0.95, L0 + 0.1,
            f'current pocket $\\Lambda_0$={L0}, half-width '
            f'{1/np.sqrt(C2):.2f}', fontsize=8, color='0.3')
    ax.set_xlabel(r'Avanci margin $(y_b - y_i)/\theta$   '
                  '(>0: absolutely unstable)')
    ax.set_ylabel(r'$\Lambda_v^*$ at the $u=0$ crossing')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # similarity check: FS vs tanh at matched u_rev (convective side)
    ax = axs[1]
    ax.plot(fs[:, 0]*100, fs[:, 1], 'C3s-', label='reversed FS')
    for hval, mk in zip(TANH_H, ('o', '^', 'v', 'D')):
        sel = th[th[:, 3] == hval]
        ax.plot(sel[:, 0]*100, sel[:, 1], marker=mk, ms=5, lw=1.0,
                label=f'tanh h={hval:g}')
    ax.set_xlabel(r'peak reverse flow $u_{rev}/U_e$ [%]')
    ax.set_ylabel(r'$\Lambda_v^*$')
    ax.axhline(L0, color='0.3', lw=1.0)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FIGD, 'q2_systematic.png')
    plt.savefig(out, dpi=140)
    print('wrote', out)


if __name__ == '__main__':
    main()
