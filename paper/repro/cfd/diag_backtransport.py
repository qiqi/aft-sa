"""Reverse-flow back-transport test, deep Eppler bubble (60k, alpha=5).

Per x-bin along the upper surface, split the near-wall band at the dividing
streamline (u=0): report chi in the REVERSE-FLOW layer (below, u<0) vs the
SHEAR layer (above, u>0), the reverse-layer peak speed, and where each peak
sits. Back-transport signature: chi in the reverse layer is comparable to /
ahead of the shear layer at stations UPSTREAM of shear-layer transition
(chi_shear<1) -- nuHat present at the wall before local growth made it.
Also writes a 2-D chi-contour + u=0 line figure.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regen_eppler_v2 as R
from vtk.util.numpy_support import vtk_to_numpy

CASE = "/home/qiqi/flexcompute/sa-ai/flow360_v2/sweep_Re60k_a5"
NU = 0.1/60000.0
CHI_INF = R.C_V1*np.exp(-9.0)


def load():
    g, pd, arrs = R.load_slice(CASE)
    p = vtk_to_numpy(g.GetPoints().GetData())
    nh = vtk_to_numpy(pd.GetArray('nuHat'))
    wd = vtk_to_numpy(pd.GetArray('wallDistance'))
    vel = vtk_to_numpy(pd.GetArray('velocity'))
    z_up, z_lo, _ = R.airfoil_surfaces(CASE)
    up, _ = R._side_mask(p, z_up, z_lo)
    return p, nh, wd, vel, up


def main():
    p, nh, wd, vel, up = load()
    x, u = p[:, 0], vel[:, 0]
    band = up & (wd > 1e-5) & (wd <= 0.03)
    print("deep Eppler bubble Re=6e4 alpha=5 (upper); chi = nuHat/muRef\n")
    print(f"{'x/c':>5} {'y0(u=0)':>8} | {'REVERSE layer (u<0)':>26} | "
          f"{'SHEAR layer (u>0)':>24}")
    print(f"{'':5} {'':8} | {'chi_rev':>9}{'y_rev':>8}{'u_rev':>9} | "
          f"{'chi_sh':>9}{'y_sh':>8}{'N_sh':>7}")
    rows = []
    for xc in np.arange(0.24, 0.66, 0.03):
        m = band & (x >= xc-0.015) & (x < xc+0.015)
        if m.sum() < 5:
            continue
        idx = np.where(m)[0]
        dd, uu, cc = wd[idx], u[idx], nh[idx]/NU
        o = np.argsort(dd); dd, uu, cc = dd[o], uu[o], cc[o]
        # dividing streamline: first u=0 crossing going up
        y0 = np.nan
        for k in range(1, len(uu)):
            if uu[k-1] < 0 <= uu[k]:
                y0 = dd[k]; break
        rev = uu < 0
        shear = uu >= 0
        crev = cc[rev].max() if rev.any() else np.nan
        yrev = dd[rev][np.argmax(cc[rev])] if rev.any() else np.nan
        urev = uu.min()
        csh = cc[shear].max() if shear.any() else np.nan
        ysh = dd[shear][np.argmax(cc[shear])] if shear.any() else np.nan
        Nsh = np.log(max(csh/CHI_INF, 1e-30))
        rows.append((xc, y0, crev, urev, csh))
        print(f"{xc:5.2f} {y0:8.4f} | {crev:9.2f}{yrev:8.4f}{urev:+9.3f} | "
              f"{csh:9.2f}{ysh:8.4f}{Nsh:7.2f}", flush=True)

    # 2-D chi contour with the u=0 dividing line
    fig, ax = plt.subplots(figsize=(9, 4))
    sel = band & (x > 0.2) & (x < 0.75)
    xi = np.linspace(0.2, 0.75, 220); yi = np.linspace(0, 0.03, 120)
    from scipy.interpolate import griddata
    XI, YI = np.meshgrid(xi, yi)
    pts = np.column_stack([x[sel], wd[sel]])
    Lchi = griddata(pts, np.log10(np.clip(nh[sel]/NU, 1e-4, None)),
                    (XI, YI), method='linear')
    U = griddata(pts, u[sel], (XI, YI), method='linear')
    cs = ax.contour(XI, YI, Lchi, levels=range(-3, 3), colors='k',
                    linewidths=0.7)
    ax.clabel(cs, fmt=lambda v: f'$10^{{{int(v)}}}$', fontsize=7)
    ax.contour(XI, YI, U, levels=[0.0], colors='r', linewidths=2.0)
    ax.contourf(XI, YI, (U < 0).astype(float), levels=[0.5, 1.5],
                colors=['#ffcccc'], alpha=0.5)
    ax.set_xlabel('x/c'); ax.set_ylabel('d/c')
    ax.set_title('60k Eppler bubble: log$_{10}\\chi$ contours; red = $u$=0 '
                 'dividing line; pink = reverse flow')
    plt.tight_layout()
    out = "/tmp/backtransport_60k.png"
    plt.savefig(out, dpi=130)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
