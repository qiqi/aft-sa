"""Cross-section views of the TWO spheroid mesh families at the run
condition (re65/re72 ladder, level L1):

  O-grid (structured half-model)      /local_data/.../mesh_re65_L1.cgns
  unstructured (Flynn360 cavity fam.) /local_data/.../mesh_unstr_L1.cgns

Figures (exploratory, figs_explore/, house style: no in-figure titles,
captions in spheroid_mesh_xsec_captions.md):
  spheroid_mesh_xsec_ogrid.png    meridian-plane (y=0) wireframe of the
                                  O-grid: full domain / body / nose zoom
                                  (wall-normal growth + pole clustering
                                  + far-field blend)
  spheroid_mesh_xsec_unstr.png    the same meridian cut of the FULL-BODY
                                  unstructured mesh: full domain (octree
                                  hex far field + tet glue) / body
                                  (prism shell) / nose zoom (BL prisms)
  spheroid_mesh_xsec_transverse.png  y-z cut at mid-body (x = 0, i.e.
                                  x/L = 0.5 from the nose), O-grid vs
                                  unstructured, domain + body zoom
  spheroid_mesh_xsec_spacing.png  wall-spacing comparison strip:
                                  meridional ds, circumferential arc,
                                  and the (identical) first-cell
                                  height/growth of the two families.

Wireframes are TRUE plane cuts of the run CGNS files (vtkCutter +
vtkExtractEdges), not analytic re-drawings.

Run: python3 paper/repro/cfd/regen_spheroid_mesh_xsec.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIGD = os.path.join(HERE, "figs_explore")
sys.path.insert(0, os.path.join(REPO, "spheroid"))
import ogrid_spheroid as og                       # noqa: E402
import unstruct_spheroid as us                    # noqa: E402

MESHD = "/local_data/qiqi/sa-ai/spheroid_meshes"
OGRID = os.path.join(MESHD, "mesh_re65_L1.cgns")
UNSTR = os.path.join(MESHD, "mesh_unstr_L1.cgns")
LEVEL = 1

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 15, "axes.grid": False,
    "legend.frameon": False, "legend.fontsize": 11.5,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "figure.dpi": 110,
    "savefig.dpi": 170})


def cut_segments(cgns, origin, normal):
    """Plane cut of every zone of a CGNS mesh -> line segments (N,2,2)
    in the two in-plane coordinates (the two axes not aligned w/ normal)."""
    r = vtk.vtkCGNSReader()
    r.SetFileName(cgns)
    r.UpdateInformation()
    r.EnableAllBases()
    r.Update()
    plane = vtk.vtkPlane()
    plane.SetOrigin(*origin)
    plane.SetNormal(*normal)
    keep = [i for i in range(3) if abs(normal[i]) < 0.5]
    segs = []
    it = r.GetOutput().NewIterator()
    it.InitTraversal()
    while not it.IsDoneWithTraversal():
        obj = it.GetCurrentDataObject()
        it.GoToNextItem()
        if obj is None or obj.GetNumberOfPoints() == 0:
            continue
        cut = vtk.vtkCutter()
        cut.SetInputData(obj)
        cut.SetCutFunction(plane)
        cut.Update()
        ed = vtk.vtkExtractEdges()
        ed.SetInputData(cut.GetOutput())
        ed.Update()
        poly = ed.GetOutput()
        if poly.GetNumberOfLines() == 0:
            continue
        P = vtk_to_numpy(poly.GetPoints().GetData())[:, keep]
        lines = vtk_to_numpy(poly.GetLines().GetData()).reshape(-1, 3)[:, 1:]
        segs.append(P[lines])
    return np.concatenate(segs) if segs else np.zeros((0, 2, 2))


def panel(ax, segs, xl, yl, lw=0.25):
    pad = 0.02 * (xl[1] - xl[0])
    m = ((segs[..., 0].min(1) < xl[1] + pad)
         & (segs[..., 0].max(1) > xl[0] - pad)
         & (segs[..., 1].min(1) < yl[1] + pad)
         & (segs[..., 1].max(1) > yl[0] - pad))
    ax.add_collection(LineCollection(segs[m], linewidths=lw, colors="k"))
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_aspect("equal")


def meridian_figure(cgns, out, y_cut=0.0):
    """Full-domain / body / nose panels of the y = y_cut meridian cut.
    For the half-model O-grid, y = 0 is the symmetry BOUNDARY (a plane cut
    there intersects no cell interior): pass a small offset y_cut so the
    cut catches the first circumferential cell layer, whose cross-section
    is the meridian half-plane grid."""
    segs = cut_segments(cgns, (0.0, y_cut, 0.0), (0.0, 1.0, 0.0))
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.6))
    panel(axs[0], segs, (-33, 33), (-33, 33), lw=0.35)
    panel(axs[1], segs, (-0.62, 0.62), (-0.42, 0.42), lw=0.22)
    panel(axs[2], segs, (-0.5045, -0.485), (-0.0075, 0.0075), lw=0.3)
    axs[2].set_xticks([-0.504, -0.498, -0.492, -0.486])
    for ax, (xlab, ylab) in zip(axs, [("x/L", "z/L")] * 3):
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGD, out), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, f"({len(segs):,} segments)")


def transverse_figure(out):
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 11))
    for row, cgns in enumerate((OGRID, UNSTR)):
        segs = cut_segments(cgns, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        panel(axs[row, 0], segs, (-33, 33), (-33, 33), lw=0.35)
        panel(axs[row, 1], segs, (-0.30, 0.30), (-0.30, 0.30), lw=0.22)
        for ax in axs[row]:
            ax.set_xlabel("y/L")
            ax.set_ylabel("z/L")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGD, out), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def spacing_figure(out):
    # O-grid ladder spacings (analytic, = the mesh generator's)
    cfg = og.LEVELS[LEVEL]
    ds_pole = cfg["DS_POLE_OVER_D"] * og.D
    P, _, s, _ = og.meridian_points(cfg["N_M"], ds_pole)
    ds_og = np.diff(s)
    x_og = 0.5 * (P[1:, 0] + P[:-1, 0]) + 0.5      # x/L from nose
    r_og = 0.5 * (P[1:, 1] + P[:-1, 1])
    dc_og = r_og * np.pi / cfg["N_C"]
    # unstructured rings
    pts, rings, i_tail, s_u, ds_u, arc_max, _ = us.build_rings(LEVEL)
    x_u, dc_u, ds_uu = [], [], []
    for k, (i0, ni, phi) in enumerate(rings):
        x_u.append(pts[i0][0] + 0.5)
        r = np.hypot(pts[i0][1], pts[i0][2])
        dc_u.append(2.0 * np.pi * r / ni)
        ds_uu.append(0.5 * (ds_u[k] + ds_u[k + 1]))
    h0, growth = us.BL_LADDER[LEVEL]

    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.6))
    axs[0].semilogy(x_og, ds_og, color="#3b6bb5", lw=2,
                    label="O-grid (meridian)")
    axs[0].semilogy(x_u, ds_uu, "--", color="#e0821f", lw=2,
                    label="unstructured (meridian)")
    axs[0].set_xlabel("x/L")
    axs[0].set_ylabel(r"$\Delta s_\parallel/L$")
    axs[0].legend()
    axs[1].semilogy(x_og, dc_og, color="#3b6bb5", lw=2,
                    label="O-grid (circumferential)")
    axs[1].semilogy(x_u, dc_u, "--", color="#e0821f", lw=2,
                    label="unstructured (circumferential)")
    axs[1].axhline(arc_max, color="0.55", lw=1, ls=":")
    axs[1].set_xlabel("x/L")
    axs[1].set_ylabel(r"$\Delta s_\phi/L$")
    axs[1].legend()
    n = np.arange(0, 51)
    axs[2].semilogy(n, h0 * growth**n, color="#3b6bb5", lw=2,
                    label=f"both families: $h_0$={h0:.2e} L, g={growth:.4f}")
    axs[2].axhline(arc_max, color="0.55", lw=1, ls=":")
    axs[2].text(1, arc_max * 1.35, "isotropic stop (unstructured)",
                fontsize=10.5, color="0.35")
    axs[2].set_xlabel("wall-normal layer index")
    axs[2].set_ylabel("layer height /L")
    axs[2].legend()
    for ax in axs:
        ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGD, out), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    os.makedirs(FIGD, exist_ok=True)
    meridian_figure(OGRID, "spheroid_mesh_xsec_ogrid.png", y_cut=1e-5)
    meridian_figure(UNSTR, "spheroid_mesh_xsec_unstr.png")
    transverse_figure("spheroid_mesh_xsec_transverse.png")
    spacing_figure("spheroid_mesh_xsec_spacing.png")


if __name__ == "__main__":
    main()
