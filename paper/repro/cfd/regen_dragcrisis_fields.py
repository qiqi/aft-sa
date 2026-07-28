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
CASES = ["cyl_Re100000_Tu0.2_up",
         "cyl_Re500000_Tu0.2_up",
         "cyl_Re2000000_Tu0.2_up"]
# window in D around the cylinder (center x=0.5, z=0)
XW = (-1.0, 4.0)
ZW = (-1.55, 1.55)
NX, NZ = 1100, 682
ULEV = np.linspace(0.1, 1.9, 13)
CLEV = np.arange(-2.0, 4.51, 0.5)


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

    plt.rcParams.update({"font.size": 11})
    fig, axs = plt.subplots(3, 2, figsize=(7.4, 6.9), sharex=True,
                            sharey=True, gridspec_kw=dict(hspace=0.06,
                                                          wspace=0.04))
    imu = imc = None
    for row, case in enumerate(CASES):
        xs, zs, u, lchi = probe(os.path.join(args.root, case))
        axU, axC = axs[row]
        imu = axU.contour(xs, zs, u, levels=ULEV, cmap="viridis",
                          linewidths=0.55, zorder=1)
        imc = axC.contour(xs, zs, lchi, levels=CLEV, cmap="magma",
                          linewidths=0.55, zorder=1)
        axC.contour(xs, zs, lchi, levels=[0.0], colors="tab:blue",
                    linewidths=1.4, zorder=2)
        for ax in (axU, axC):
            ax.add_patch(plt.Circle((0.5, 0.0), 0.5, fc="0.85", ec="k",
                                    lw=0.6, zorder=5))
            ax.set_aspect("equal")
            ax.set_xlim(*XW)
            ax.set_ylim(*ZW)
        axU.set_ylabel("$z/D$")
        print(f"  {case}: probed", flush=True)
    for ax in axs[-1]:
        ax.set_xlabel("$x/D$")
    # continuous gradient bars (a ContourSet colorbar renders line
    # contours as sparse ticks on white)
    smu = plt.cm.ScalarMappable(plt.Normalize(ULEV[0], ULEV[-1]), "viridis")
    smc = plt.cm.ScalarMappable(plt.Normalize(CLEV[0], CLEV[-1]), "magma")
    cbu = fig.colorbar(smu, ax=axs[:, 0], location="bottom", fraction=0.05,
                       pad=0.11, aspect=34, ticks=[0.5, 1.0, 1.5])
    cbu.set_label(r"$|\mathbf{u}|/U_\infty$")
    cbc = fig.colorbar(smc, ax=axs[:, 1], location="bottom", fraction=0.05,
                       pad=0.11, aspect=34, ticks=[-2, -1, 0, 1, 2, 3, 4])
    cbc.set_label(r"$\log_{10}\chi$ (blue: $\chi=1$)")
    os.makedirs(PREV, exist_ok=True)
    fig.savefig(os.path.join(OUT, "dragcrisis_fields.pdf"),
                bbox_inches="tight", dpi=220)
    fig.savefig(os.path.join(PREV, "dragcrisis_fields.png"), dpi=130,
                bbox_inches="tight")
    print(f"wrote {OUT}/dragcrisis_fields.pdf")


if __name__ == "__main__":
    main()
