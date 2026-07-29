"""TASK 2 (2026-07-29): indicator-sphere view of the low-H Falkner-Skan
family, to show WHY the onset gate cannot set distinct onsets at low H.

Reuses fig01_indicator_sphere machinery (sphere_coords, band_eta) and the
same RP^2 orthographic layout. Plots FS trajectories for H = 2.216 ("2.2",
beta=1), 2.30, 2.40, 2.50, 2.59 (Blasius) as thin curves; the wall-normal
amplifying band (OS Reynolds-stress production > half peak) as thick segments
where it exists. A low-H companion panel zooms the near-wall / neutral-locus
region. Quantifies the sphere-space separation between the H=2.216 and H=2.30
trajectories (max & mean geodesic + Euclidean along the curve) and the
difference in their profile-max P = Omega_hat*I_hat, against the Drela
critical Re_theta0 ratio (~1.8x).

OFFLINE; writes figs_explore/indicator_sphere_lowH.png + a JSON table.
No tex/solver/canon-figure edits.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _saai  # noqa: F401
import fig01_indicator_sphere as f1
from explore_lsb_frozen_profile import build_profile
from explore_wavepacket_regions import os_mode, production, contiguous
from lib.boundary_layer import FalknerSkanWedge
from lib.correlations import Re_theta0
from scipy.optimize import brentq


def fast_band(pr, Re_th, N=120):
    """band_eta (fig01) with reduced OS alpha-resolution for speed."""
    try:
        yh, phi, al, c, unstable = os_mode(pr, Re_th, N=N)
    except Exception:
        return None
    if not unstable:
        return None
    Th = pr['Theta']
    Up = np.interp(yh*Th, pr['eta'], pr['up'])*Th
    p = production(yh, phi, al, Up)
    jp = int(np.argmax(p))
    mask = contiguous(p > 0.5*p[jp], jp)
    et = yh[mask]*Th
    return float(et.min()), float(et.max())

SQ2 = np.sqrt(2.0)
OUT_DIR = os.path.join('repro', 'analytic', 'figs_explore')


def H_of(beta):
    fs = FalknerSkanWedge(beta)
    I = np.trapezoid(fs.u*(1-fs.u), fs.eta)
    return float(np.trapezoid(1-fs.u, fs.eta)/I)


def beta_for_H(Ht):
    if Ht <= 2.2164:
        return 1.0
    return float(brentq(lambda x: H_of(x)-Ht, 1e-4, 1.0))


# target family
TARGETS = [(2.216, 'C3', r'$H=2.216\;(\beta=1,$ "2.2"$)$'),
           (2.30,  'C1', r'$H=2.30$'),
           (2.40,  'C0', r'$H=2.40$'),
           (2.50,  'C2', r'$H=2.50$'),
           (2.59,  '0.35', r'$H=2.59$ (Blasius)')]


def unit_vecs(pr):
    """(X,Y,Z)/R unit vectors and P=Om*I along the profile, plus shear mask."""
    eta, u, up, upp = pr['eta'], pr['u'], pr['up'], pr['upp']
    X0 = u; Y0 = eta*up; Z0 = 0.5*eta**2*upp
    R = np.sqrt(X0*X0 + Y0*Y0 + Z0*Z0) + 1e-30
    n = np.vstack([X0/R, Y0/R, Z0/R])       # 3 x npts unit vectors
    Shat = Y0/np.sqrt(X0*X0 + Y0*Y0 + 1e-30)
    g = (Y0 - X0 - Z0)/R
    P = Shat*g
    shear = up > 1e-3*np.max(up)
    return eta, n, P, shear


def separation(prA, prB):
    """geodesic + Euclidean sphere-space separation between two profiles,
    sampled on a common eta grid over the shared shear region."""
    eA, nA, PA, shA = unit_vecs(prA)
    eB, nB, PB, shB = unit_vecs(prB)
    e0 = max(eA[shA].min(), eB[shB].min())
    e1 = min(eA[shA].max(), eB[shB].max())
    eg = np.linspace(e0, e1, 400)
    nAi = np.vstack([np.interp(eg, eA, nA[k]) for k in range(3)])
    nBi = np.vstack([np.interp(eg, eB, nB[k]) for k in range(3)])
    # renormalize after interpolation
    nAi /= np.linalg.norm(nAi, axis=0) + 1e-30
    nBi /= np.linalg.norm(nBi, axis=0) + 1e-30
    dot = np.clip(np.sum(nAi*nBi, axis=0), -1.0, 1.0)
    geo = np.arccos(dot)                    # radians
    euc = np.linalg.norm(nAi - nBi, axis=0)
    return dict(geo_max=float(geo.max()), geo_mean=float(geo.mean()),
                geo_max_deg=float(np.degrees(geo.max())),
                geo_mean_deg=float(np.degrees(geo.mean())),
                euc_max=float(euc.max()), euc_mean=float(euc.mean()),
                Pmax_A=float(PA[shA].max()), Pmax_B=float(PB[shB].max()))


def draw(ax, zoom=None):
    tt = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(tt), np.sin(tt), '-', color='0.8', lw=1.0, zorder=1)
    for xy, mk in [((-1/SQ2, 0), '<'), ((1/SQ2, 0), '>'),
                   ((0, 1), '^'), ((0, -1), 'v')]:
        ax.plot([xy[0]], [xy[1]], marker=mk, color='0.4', ms=8, zorder=6)
    # Om*I contours
    hg = np.linspace(-0.999, 0.999, 601); vg = hg.copy()
    Hm, Vm = np.meshgrid(hg, vg)
    disk = Hm**2 + Vm**2 < 1.0
    Wm = np.sqrt(np.clip(1.0 - Hm**2 - Vm**2, 0, None))
    X = (Wm - Hm)/SQ2; Y = (Wm + Hm)/SQ2; Z = Vm
    Om = Y/np.sqrt(X*X + Y*Y + 1e-30)
    G = Y - X - Z
    Psg = np.where(disk, Om*G, np.nan)
    ax.contour(Hm, Vm, Psg, levels=[0.0], colors='0.55', linewidths=1.2,
               linestyles='--', zorder=2)
    ax.contour(Hm, Vm, Psg, levels=[0.025, 0.2, 0.4, 0.6, 0.8],
               colors='0.75', linewidths=0.7, zorder=2)
    ax.plot([0], [0], 'k.', ms=5, zorder=6)
    ax.set_aspect('equal')
    if zoom is None:
        ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.2, 1.15)
    else:
        ax.set_xlim(*zoom[0]); ax.set_ylim(*zoom[1])
    ax.axis('off')


def main():
    # 1) build profiles (cheap) and compute+SAVE the quantitative separation
    #    table FIRST, so the load-bearing deliverable survives even if the OS
    #    band solves are slow.
    profs = []
    for Ht, col, lab in TARGETS:
        b = beta_for_H(Ht)
        pr = build_profile(b, None)
        profs.append([Ht, b, col, lab, pr, None])
        print(f"H={Ht} beta={b:.4f} (H_act={pr['H']:.3f}) built; "
              f"Re_theta0={float(Re_theta0(Ht)):.0f}", flush=True)

    prmap = {round(p[0], 3): p[4] for p in profs}
    table = {}
    pairs = [(2.216, 2.30), (2.30, 2.40), (2.216, 2.59)]
    for a, bb in pairs:
        sep = separation(prmap[round(a, 3)], prmap[round(bb, 3)])
        sep['Re_theta0_ratio'] = float(Re_theta0(a))/float(Re_theta0(bb))
        sep['Re_theta0_A'] = float(Re_theta0(a))
        sep['Re_theta0_B'] = float(Re_theta0(bb))
        table[f'H{a}_vs_H{bb}'] = sep
        print(f"\n=== H={a} vs H={bb} ===")
        print(f"  geodesic  max={sep['geo_max_deg']:.2f} deg  "
              f"mean={sep['geo_mean_deg']:.2f} deg")
        print(f"  Euclidean max={sep['euc_max']:.4f}  mean={sep['euc_mean']:.4f}")
        print(f"  profile-max P=Om*I: A={sep['Pmax_A']:.2e}  B={sep['Pmax_B']:.2e}"
              f"  |diff|={abs(sep['Pmax_A']-sep['Pmax_B']):.2e}")
        print(f"  Drela Re_theta0: A={sep['Re_theta0_A']:.0f} "
              f"B={sep['Re_theta0_B']:.0f}  ratio={sep['Re_theta0_ratio']:.2f}x")
    with open(os.path.join(OUT_DIR, 'indicator_sphere_lowH.json'), 'w') as f:
        json.dump({'targets': [(p[0], p[1], p[4]['H']) for p in profs],
                   'separation': table}, f, indent=1)
    print('wrote indicator_sphere_lowH.json (separation table)', flush=True)

    # 2) OS amplifying bands (reduced resolution for speed)
    for p in profs:
        Ht, b = p[0], p[1]
        p[5] = fast_band(p[4], max(2.0*float(Re_theta0(Ht)), 600.0))
        print(f"  band H={Ht}: {p[5]}", flush=True)

    fig = plt.figure(figsize=(13.2, 6.6))
    axm = fig.add_axes([0.02, 0.05, 0.46, 0.9])
    axz = fig.add_axes([0.52, 0.05, 0.46, 0.9])
    draw(axm)
    # zoom on the near-wall / neutral cluster: wall point is at (h,v)=(0,0),
    # strong-FPG curves ride the Om*I=0 dashed locus near there
    draw(axz, zoom=((-0.28, 0.42), (-0.35, 0.35)))
    for Ht, b, col, lab, pr, band in profs:
        eta, u, up, upp = pr['eta'], pr['u'], pr['up'], pr['upp']
        h, v, w = f1.sphere_coords(u, up, upp, eta)
        for ax in (axm, axz):
            f1.draw_segments(ax, h, v, w, col, 1.6, band=band, eta=eta)
        axm.plot([], [], '-', color=col, lw=2.0, label=lab)
    axm.legend(fontsize=8.5, loc='lower left', framealpha=0.9)
    axm.set_title('RP$^2$ indicator sphere: low-H Falkner-Skan family\n'
                  '(thin = trajectory, thick = OS amplifying band; '
                  'dashed = neutral locus $\\hat\\Omega\\hat I=0$)',
                  fontsize=10)
    axz.set_title('near-wall / neutral-locus zoom\n'
                  '(strong-FPG curves bunch on the $\\hat\\Omega\\hat I=0$ '
                  'locus — the onset-resolution limit)', fontsize=10)
    fp = os.path.join(OUT_DIR, 'indicator_sphere_lowH.png')
    plt.savefig(fp, dpi=150, facecolor='white')
    print(f'wrote {fp}', flush=True)


if __name__ == '__main__':
    main()
