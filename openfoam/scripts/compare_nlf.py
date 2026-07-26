#!/usr/bin/env python3
"""NLF(1)-0416 strL0 alpha=0: OpenFOAM SA-AI vs Flow360 SA-AI on the identical
mesh. Extracts Cp(x), Cf(x) on the airfoil and near-wall max-chi(x) per side;
reports transition locations (chi = 1 and chi = c_v1 crossings, and the Cf
rise) for both codes.

Conventions: Flow360 surface Cf/Cp are already freestream-normalized; its
slice nuHat is in c_inf*L units -> chi = nuHat/(M/Re) = nuHat/2.5e-8.
OpenFOAM p is kinematic (Cp = 2p), wallShearStress kinematic (Cf = 2|tau|),
chi = nuTilda/2.5e-7.

Run with the compute venv python (vtk).
"""
import glob
import os

import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

RE = 4.0e6
MACH = 0.1
NU_OF = 1.0 / RE            # OpenFOAM kinematic viscosity
NU_F360 = MACH / RE         # Flow360 c_inf*L units
C_V1 = 7.1

OF_CASE = "/local_data/qiqi/openfoam-sa-ai/cases/nlf_strL0_a0"
F360_CASE = "/home/qiqi/flexcompute/sa-ai/flow360_fr/strL0prop_nlf0416_Re4M_a0"
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_vtu(path):
    if path.endswith(".pvtu"):
        r = vtk.vtkXMLPUnstructuredGridReader()
    elif path.endswith(".vtp"):
        r = vtk.vtkXMLPolyDataReader()
    else:
        r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()


def arrays(obj, names, cell=False):
    d = obj.GetCellData() if cell else obj.GetPointData()
    return {n: vtk_to_numpy(d.GetArray(n)) for n in names
            if d.GetArray(n) is not None}


# ---------------- Flow360 ----------------

def f360_surface():
    g = read_vtu(f"{F360_CASE}/surface_fluid_nlf0416.pvtu")
    pts = vtk_to_numpy(g.GetPoints().GetData())
    a = arrays(g, ["Cf", "Cp"])
    # one span slice
    ys = np.unique(pts[:, 1])
    m = np.abs(pts[:, 1] - ys[0]) < 1e-9
    x, z = pts[m, 0], pts[m, 2]
    cf = a["Cf"][m]
    if cf.ndim > 1:
        cf = np.linalg.norm(cf, axis=1)
    cf = np.abs(cf)
    cp = a["Cp"][m]
    return x, z, cp, cf


def f360_chi():
    g = read_vtu(f"{F360_CASE}/slice_centerSpan.pvtu")
    pts = vtk_to_numpy(g.GetPoints().GetData())
    a = arrays(g, ["nuHat", "wallDistance"])
    chi = a["nuHat"] / NU_F360
    return pts[:, 0], pts[:, 2], chi, a["wallDistance"]


# ---------------- OpenFOAM ----------------

def of_surface():
    vd = glob.glob(f"{OF_CASE}/VTK/nlf_strL0_a0_75000/boundary/nlf0416.vtp")
    g = read_vtu(sorted(vd)[-1])
    # cell (face) data is authoritative on patches
    cc = vtk.vtkCellCenters()
    cc.SetInputData(g)
    cc.Update()
    pts = vtk_to_numpy(cc.GetOutput().GetPoints().GetData())
    a = arrays(g, ["p", "wallShearStress"], cell=True)
    m = np.abs(pts[:, 1] - pts[:, 1].min()) < 1e6  # all (single span cell)
    cp = 2.0 * a["p"]
    cf = 2.0 * np.linalg.norm(a["wallShearStress"], axis=1)
    return pts[:, 0], pts[:, 2], cp, cf


def of_chi(surf_xz):
    from scipy.spatial import cKDTree
    vd = glob.glob(f"{OF_CASE}/VTK/nlf_strL0_a0_75000/internal.vtu")
    g = read_vtu(sorted(vd)[-1])
    cc = vtk.vtkCellCenters()
    cc.SetInputData(g)
    cc.Update()
    pts = vtk_to_numpy(cc.GetOutput().GetPoints().GetData())
    a = arrays(g, ["nuTilda"], cell=True)
    chi = a["nuTilda"] / NU_OF
    tree = cKDTree(surf_xz)
    wd, _ = tree.query(pts[:, [0, 2]])
    return pts[:, 0], pts[:, 2], chi, wd


# ---------------- analysis ----------------

def side_of(x, z, surf_x, surf_z):
    """upper if z above local chord-line midpoint of the airfoil at x."""
    # midcamber approx: for each x, mean of surface z at that x (upper+lower)
    order = np.argsort(surf_x)
    sx, sz = surf_x[order], surf_z[order]
    # crude camber: for query x, take mean of sz within a small window
    zc = np.interp(x, np.linspace(0, 1, 200),
                   [sz[(np.abs(sx - xi) < 0.02)].mean() if
                    (np.abs(sx - xi) < 0.02).any() else 0.0
                    for xi in np.linspace(0, 1, 200)])
    return z >= zc


def maxchi_vs_x(x, z, chi, wd, upper, nbins=100, wd_max=0.02):
    m = (wd < wd_max) & (x >= 0) & (x <= 1.0) & upper
    xb = np.linspace(0, 1, nbins + 1)
    xc = 0.5 * (xb[:-1] + xb[1:])
    out = np.full(nbins, np.nan)
    for i in range(nbins):
        s = m & (x >= xb[i]) & (x < xb[i + 1])
        if s.any():
            out[i] = chi[s].max()
    return xc, out


def crossing(xc, mchi, level):
    v = np.isfinite(mchi)
    xs, cs = xc[v], mchi[v]
    ix = np.where(cs >= level)[0]
    if not len(ix) or ix[0] == 0:
        return np.nan
    i = ix[0]
    w = (level - cs[i - 1]) / (cs[i] - cs[i - 1])
    return xs[i - 1] + w * (xs[i] - xs[i - 1])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fx, fz, fcp, fcf = f360_surface()
    ox, oz, ocp, ocf = of_surface()
    fX, fZ, fchi, fwd = f360_chi()
    oX, oZ, ochi, owd = of_chi(np.column_stack([ox, oz]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for (x, z, cp, cf, lab, mk) in ((fx, fz, fcp, fcf, "Flow360", "."),
                                    (ox, oz, ocp, ocf, "OpenFOAM", "x")):
        up = side_of(x, z, np.concatenate([fx, fx]), np.concatenate([fz, fz])) if False else z >= np.interp(x, np.sort(fx), fz[np.argsort(fx)]*0)  # placeholder
    # simpler side split: z>=0 approx fails near LE lower; use sign of z relative
    # to chord line z=0 shifted by camber sign; NLF(1)-0416 lower surface dips
    # below z=0 for x<~0.9 -- use z>=0.0 with LE care. Good enough for x>0.03.
    for (x, z, cp, cf, lab, ls) in ((fx, fz, fcp, fcf, "Flow360", "-"),
                                    (ox, oz, ocp, ocf, "OpenFOAM", "--")):
        up = z >= 0
        for m, side in ((up, "upper"), (~up, "lower")):
            o = np.argsort(x[m])
            axes[0].plot(x[m][o], -cp[m][o], ls, lw=1,
                         label=f"{lab} {side}" if side == "upper" else None)
            axes[1].semilogy(x[m][o], np.maximum(cf[m][o], 1e-6), ls, lw=1,
                             label=f"{lab} {side}")
    axes[0].set_xlabel("x/c"); axes[0].set_ylabel(r"$-C_p$")
    axes[0].set_ylim(-1, 1.6); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("x/c"); axes[1].set_ylabel(r"$C_f$")
    axes[1].set_ylim(1e-5, 2e-2); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

    print(f"{'side':>6} {'metric':>10} {'OpenFOAM':>9} {'Flow360':>9}")
    for upper_flag, side in ((True, "upper"), (False, "lower")):
        fxc, fmc = maxchi_vs_x(fX, fZ, fchi, fwd, (fZ >= 0) == upper_flag)
        oxc, omc = maxchi_vs_x(oX, oZ, ochi, owd, (oZ >= 0) == upper_flag)
        axes[2].semilogy(fxc, fmc, "-", lw=1, label=f"F360 {side}")
        axes[2].semilogy(oxc, omc, "--", lw=1, label=f"OF {side}")
        for level, name in ((1.0, "chi=1"), (C_V1, "chi=cv1")):
            xo = crossing(oxc, omc, level)
            xf = crossing(fxc, fmc, level)
            print(f"{side:>6} {name:>10} {xo:9.4f} {xf:9.4f}")
    axes[2].axhline(1, color="k", lw=0.5); axes[2].axhline(C_V1, color="k", lw=0.5, ls=":")
    axes[2].set_xlabel("x/c"); axes[2].set_ylabel(r"max $\chi$ (wd<0.02)")
    axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)
    fig.suptitle("NLF(1)-0416 Re=4e6 alpha=0, strL0 (identical mesh): OpenFOAM vs Flow360 SA-AI")
    fig.tight_layout()
    fig.savefig(f"{OUT}/nlf_strL0_compare.png", dpi=110)
    fig.savefig(f"{OUT}/nlf_strL0_compare.pdf")
    print(f"wrote {OUT}/nlf_strL0_compare.png")


if __name__ == "__main__":
    main()
