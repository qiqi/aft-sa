"""Is the thick Re=1e5 bubble under-resolved? (mesh-sensitivity probe)

At Re=1e5 the two families' L2 solutions still differ (str 0.769/0.0444 vs
cav 0.859/0.0341), so this measures, on the center-span slice of each case:

  per upper-surface station x/c = 0.5 .. 0.95:
    h_rev   wall-normal height of the reverse-flow region (u_x < 0),
    d_sl    thickness of the separated shear layer (u_x: 0 -> 0.5 U_e,loc),
    dn      local mesh spacing at the shear-layer height (median
            nearest-neighbor distance of slice points there; for stretched
            cells this is the SMALL dimension, so N_sl below is optimistic),
    N_sl    d_sl / dn -- points across the separated shear layer;

  per wake station x/c = 1.01 .. 1.10 (the merging of the upper separated
  layer with the lower-surface boundary layer past the trailing edge):
    w_def   vertical extent of the wake deficit (u_x < 0.9 U_inf),
    u_min   minimum u_x in the deficit (reverse flow persists past the TE?),
    dz_wk   local spacing at the deficit center, and N_wk = w_def / dz_wk;
    d_lbl   thickness of the LOWER-surface boundary layer arriving at the TE
            (u_x: 0 -> 0.9 U_e at x/c=0.99 below the camber) and N_lbl,
            the points across it -- the thin high-momentum layer whose
            entrainment closes (or fails to close) the dead-air region.

Run: SAAI_CFD_ROOT=... python3 repro/cfd/analyze_bubble_resolution.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial import cKDTree
from vtk.util.numpy_support import vtk_to_numpy
import regen_eppler_v2 as R

B = os.environ.get("SAAI_CFD_ROOT", "/home/qiqi/flexcompute/sa-ai/flow360_fr")
UINF = 0.1  # Mach (velocity in sound-speed units)

CASES = [
    ('str L1 1e5', f'{B}/sweep_str_Re100k_a5'),
    ('str L2 1e5', f'{B}/sweep_strL2_Re100k_a5'),
    ('cav L1 1e5', f'{B}/sweep_Re100k_a5'),
    ('cav L2 1e5', f'{B}/sweep_cavL2_Re100k_a5'),
    ('str L2 2e5', f'{B}/strL2prop_eppler387_Re200k_a5'),
    ('cav L2 2e5', f'{B}/cavL2prop_eppler387_Re200k_a5'),
]
X_SURF = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
X_WAKE = (1.01, 1.03, 1.06, 1.10)


def load(d):
    g, pd, arrs = R.load_slice(d)
    P = vtk_to_numpy(g.GetPoints().GetData())
    V = vtk_to_numpy(pd.GetArray('velocity'))
    WD = vtk_to_numpy(pd.GetArray('wallDistance'))
    _, _, zc = R.airfoil_surfaces(d)
    return P[:, [0, 2]], V[:, 0], WD, zc


def col(P, x0, w=0.008):
    return np.abs(P[:, 0] - x0) < w


def spacing_at(tree, pts_2d, x0, z0, k=8):
    dd, _ = tree.query([x0, z0], k=k)
    return float(np.median(dd[1:]))


def profile(dsel, usel, nb=400):
    """(d, u) profile by binning wall distance; returns bin centers + mean u."""
    o = np.argsort(dsel)
    d, u = dsel[o], usel[o]
    return d, u


def analyze(name, d):
    P, ux, wd, zc = load(d)
    tree = cKDTree(P)
    print(f"-- {name} ({os.path.basename(d)})")
    print(f"   {'x/c':>5} {'h_rev':>7} {'d_sl':>7} {'dn@sl':>8} {'N_sl':>5}")
    for x0 in X_SURF:
        m = col(P, x0) & (P[:, 1] > zc(x0)) & (wd < 0.12)
        if m.sum() < 10:
            print(f"   {x0:5.2f}   (no pts)"); continue
        dsel, usel, zsel = wd[m], ux[m], P[m, 1]
        rev = usel < 0
        h_rev = float(dsel[rev].max()) if rev.any() else 0.0
        ue = float(np.percentile(usel, 98))
        above = dsel > h_rev
        # shear layer: first d beyond h_rev where u exceeds 0.5*ue
        d_half = None
        oo = np.argsort(dsel)
        for i in oo:
            if dsel[i] > h_rev and usel[i] > 0.5*ue:
                d_half = float(dsel[i]); break
        d_sl = (d_half - h_rev) if d_half else float('nan')
        z_sl = float(np.median(zsel[np.abs(dsel - h_rev) < 0.25*max(h_rev, 1e-3)])) \
            if h_rev > 0 else zc(x0)
        dn = spacing_at(tree, P, x0, z_sl if h_rev > 0 else zc(x0) + 0.005)
        nsl = d_sl/dn if (d_sl == d_sl and dn > 0) else float('nan')
        print(f"   {x0:5.2f} {h_rev:7.4f} {d_sl:7.4f} {dn:8.5f} {nsl:5.1f}")
    # lower BL at the TE
    x0 = 0.99
    m = col(P, x0, 0.005) & (P[:, 1] < zc(x0))
    dsel, usel = wd[m], ux[m]
    oo = np.argsort(dsel)
    d_lbl = float('nan')
    for i in oo:
        if usel[i] > 0.9*UINF:
            d_lbl = float(dsel[i]); break
    dn_l = spacing_at(tree, P, x0, zc(x0) - max(d_lbl, 1e-3))
    print(f"   lower BL at 0.99c: d_lbl={d_lbl:.4f}  dn={dn_l:.5f}  "
          f"N_lbl={d_lbl/dn_l:.1f}")
    print(f"   {'x/c':>5} {'w_def':>7} {'u_min/U':>8} {'dz_wk':>8} {'N_wk':>5}")
    for x0 in X_WAKE:
        m = col(P, x0, 0.01) & (np.abs(P[:, 1] - zc(0.99)) < 0.15)
        if m.sum() < 10:
            print(f"   {x0:5.2f}   (no pts)"); continue
        zsel, usel = P[m, 1], ux[m]
        dm = usel < 0.9*UINF
        if not dm.any():
            print(f"   {x0:5.2f}   (no deficit)"); continue
        zlo, zhi = float(zsel[dm].min()), float(zsel[dm].max())
        w_def = zhi - zlo
        umin = float(usel.min())/UINF
        zctr = 0.5*(zlo + zhi)
        dz = spacing_at(tree, P, x0, zctr)
        print(f"   {x0:5.2f} {w_def:7.4f} {umin:8.3f} {dz:8.5f} {w_def/dz:5.1f}")


for name, d in CASES:
    if os.path.isdir(d):
        analyze(name, d)
    else:
        print(f"-- {name}: MISSING {d}")
