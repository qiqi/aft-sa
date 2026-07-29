"""fig:dragcrisisfields -> paper/figs/dragcrisis_fields.pdf.

Center-span fields of three drag-crisis matrix cases spanning the traverse
at the middle seed (Tu 0.2%, up-ladder): subcritical Re_D = 1e5, mid-crisis
5e5 (steady limit cycle; the plotted field is the final snapshot), and
supercritical 2e6. Rows = cases, columns = velocity magnitude |u|/U_inf and
log10(chi); the chi = 1 contour (the paper's transition-front level) is
overdrawn thick blue on the chi panels. Line contours (not filled), the
paper-wide convention; Re per row lives in the caption, not the axes.
chi = nuHat/muRef (freestream check: the far field reads
chi_inf = 1.0835e-2 exactly). Fields are probed from
slice_centerSpan.pvtu onto a uniform Cartesian window; contour levels are
shared by all rows.

Run from anywhere: python3 repro/cfd/regen_dragcrisis_fields.py
  [--root /local_data/qiqi/sa-ai/dragcrisis_matrix]
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..', '..', 'figs'))
PREV = os.path.join(HERE, 'figs_explore')

MACH = 0.1
# (case dir, Re label for the caption); low/high extremes carry mesh suffixes
CASES = [("cyl_Re10_Tu0.2_up_lowre",   r"$Re_D=10$"),
         ("cyl_Re1000_Tu0.2_up_lowre", r"$10^3$"),
         ("cyl_Re100000_Tu0.2_up",     r"$10^5$"),
         ("cyl_Re500000_Tu0.2_up",     r"$5\times10^5$"),
         ("cyl_Re2000000_Tu0.2_up",    r"$2\times10^6$")]
# window in D around the cylinder (center x=0.5, z=0)
XW = (-1.0, 4.0)
ZW = (-1.35, 1.35)
NX, NZ = 1100, 682
# labeled line-contour levels (no colorbar; e-ink grayscale)
ULEV = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
CHI_SUB = [-6.0, -4.0, -2.0, -1.0]                  # chi<1 (dashed)
CHI_SOL = [0.0, np.log10(7.1), np.log10(30.0)]      # chi=1, c_v1, 30 (solid)


def probe(case_dir):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(os.path.join(case_dir, "slice_centerSpan.pvtu"))
    r.Update()
    g = r.GetOutput()
    y0 = vtk_to_numpy(g.GetPoints().GetData())[0, 1]
    xs = np.linspace(*XW, NX)
    zs = np.linspace(*ZW, NZ)
    X, Z = np.meshgrid(xs, zs)
    pts = np.column_stack([X.ravel(), np.full(X.size, y0), Z.ravel()])
    vp = vtk.vtkPoints()
    vp.SetData(numpy_to_vtk(pts, deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vp)
    pr = vtk.vtkProbeFilter()
    pr.SetInputData(poly)
    pr.SetSourceData(g)
    pr.Update()
    pd = pr.GetOutput().GetPointData()
    valid = np.zeros(X.size, bool)
    valid[vtk_to_numpy(pr.GetValidPoints())] = True
    v = vtk_to_numpy(pd.GetArray("velocity"))
    u = np.linalg.norm(v, axis=1) / MACH
    mu = json.load(open(os.path.join(case_dir, "Flow360.json"))
                   )["freestream"]["muRef"]
    chi = vtk_to_numpy(pd.GetArray("nuHat")) / mu
    u = np.where(valid, u, np.nan).reshape(NZ, NX)
    lchi = np.where(valid, np.log10(np.clip(chi, 1e-8, None)),
                    np.nan).reshape(NZ, NX)
    return xs, zs, u, lchi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/local_data/qiqi/sa-ai/dragcrisis_matrix")
    args = ap.parse_args()

    plt.rcParams.update({"font.size": 8})
    n = len(CASES)
    fig, axs = plt.subplots(n, 2, figsize=(6.6, 8.7), sharex=True,
                            sharey=True, gridspec_kw=dict(hspace=0.05,
                                                          wspace=0.04))

    def chifmt(L):
        return f"${10**L:.0f}$"

    for row, (case, _lbl) in enumerate(CASES):
        xs, zs, u, lchi = probe(os.path.join(args.root, case))
        axU, axC = axs[row]
        cu = axU.contour(xs, zs, u, levels=ULEV, colors="k",
                         linewidths=0.5, zorder=1)
        axU.clabel(cu, fmt="%.1f", fontsize=5.5, inline=True,
                   inline_spacing=1)
        axC.contour(xs, zs, lchi, levels=CHI_SUB, colors="0.55",
                    linewidths=0.45, linestyles="dashed", zorder=1)
        cs = axC.contour(xs, zs, lchi, levels=CHI_SOL, colors="k",
                         linewidths=[1.2, 0.6, 0.6], zorder=2)
        axC.clabel(cs, fmt=chifmt, fontsize=5.5, inline=True,
                   inline_spacing=1)
        for ax in (axU, axC):
            ax.add_patch(plt.Circle((0.5, 0.0), 0.5, fc="0.85", ec="k",
                                    lw=0.6, zorder=5))
            ax.set_aspect("equal")
            ax.set_xlim(*XW)
            ax.set_ylim(*ZW)
            ax.tick_params(labelsize=6)
        axU.set_ylabel("$z/D$", fontsize=7)
        print(f"  {case}: probed", flush=True)
    for ax in axs[-1]:
        ax.set_xlabel("$x/D$", fontsize=7)
    os.makedirs(PREV, exist_ok=True)
    fig.savefig(os.path.join(OUT, "dragcrisis_fields.pdf"),
                bbox_inches="tight", dpi=220)
    fig.savefig(os.path.join(PREV, "dragcrisis_fields.png"), dpi=130,
                bbox_inches="tight")
    print(f"wrote {OUT}/dragcrisis_fields.pdf")


if __name__ == "__main__":
    main()
