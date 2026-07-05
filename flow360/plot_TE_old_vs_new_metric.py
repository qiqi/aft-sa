"""Compare TE region: old metric (hwall everywhere) vs new metric (local contour spacing).
Each column shows the actual mesh cells at the same h_TE=h_0 contour input.
"""
import vtk, numpy as np, os
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def load_mesh_edges(path):
    r = vtk.vtkUnstructuredGridReader(); r.SetFileName(path); r.Update()
    g = r.GetOutput()
    pts = vtk_to_numpy(g.GetPoints().GetData())[:, [0, 1]]
    edges = set()
    for i in range(g.GetNumberOfCells()):
        c = g.GetCell(i); npts = c.GetNumberOfPoints()
        ids = [c.GetPointId(j) for j in range(npts)]
        for j in range(npts):
            a, b = ids[j], ids[(j+1) % npts]
            edges.add((min(a, b), max(a, b)))
    return pts, pts[np.array(list(edges), dtype=int)]

cases = [
    ('/home/qiqi/flexcompute/aft-sa/flow360/proper_cav_L0_TEprop/mesh2d.vtk',
     'L0 ($h_0{=}14\\mu$m, $r{=}2.0$)', 14e-6),
    ('/home/qiqi/flexcompute/aft-sa/flow360/proper_cav_L1_TEprop/mesh2d.vtk',
     'L1 ($h_0{=}7\\mu$m, $r{=}1.5$)', 7e-6),
    ('/home/qiqi/flexcompute/aft-sa/flow360/proper_cav_L2_TEprop/mesh2d.vtk',
     'L2 ($h_0{=}3.5\\mu$m, $r{=}1.25$)', 3.5e-6),
]

fig, axs = plt.subplots(3, len(cases), figsize=(12, 11))

for col, (path, lbl, h0) in enumerate(cases):
    pts, segs = load_mesh_edges(path)
    print(f"{lbl}: {len(pts)} pts, {len(segs)} edges")
    for row, w_factor in enumerate([100, 30, 8]):
        ax = axs[row, col]
        w = w_factor * h0
        xlim = (1 - w, 1 + w); ylim = (-w, w)
        in_box = ((segs[:,:,0] >= xlim[0] - 1e-5) & (segs[:,:,0] <= xlim[1] + 1e-5) &
                  (segs[:,:,1] >= ylim[0] - 1e-5) & (segs[:,:,1] <= ylim[1] + 1e-5))
        keep = in_box.any(axis=1)
        lc = LineCollection(segs[keep], colors='k', linewidths=0.4, rasterized=True)
        ax.add_collection(lc)
        ax.plot(1, 0, 'r.', ms=6, zorder=10)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal')
        ax.grid(alpha=0.2, lw=0.3); ax.tick_params(labelsize=7)
        if row == 0:
            ax.set_title(lbl, fontsize=10)
        ax.set_ylabel(f"$\\pm {w_factor} h_0$\n({2*w*1e6:.0f}μm)", fontsize=8)
        if row == 2: ax.set_xlabel("x/c", fontsize=9)

plt.suptitle("TE-region mesh with new (local-segment) metric and closed-contour fix: L0-L3 at α=4° refinement series. Rows: ±100 h_0, ±30 h_0, ±8 h_0. Red dot = (1, 0) shared TE node.", fontsize=10)
plt.tight_layout(pad=0.5)
plt.savefig('/tmp/TE_old_vs_new_metric.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/TE_mesh_zoom_L3.pdf', dpi=200)
print("wrote /tmp/TE_old_vs_new_metric.png + paper figs/TE_mesh_zoom_L3.pdf")
