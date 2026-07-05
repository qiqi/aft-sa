"""Visualize the mesh near the trailing edge for cavity L1/L2 and O-grid L2.
Uses the center-span slice (.pvtu) which already contains the 2D cut through the volume mesh.
Plots cell edges in a zoomed view of the TE region.
"""
import vtk, numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def load_slice_cells(d):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f"{d}/slice_centerSpan.pvtu")
    r.Update()
    g = r.GetOutput()
    pts = vtk_to_numpy(g.GetPoints().GetData())
    # 2D plane: discard y component (span-aligned)
    pts2 = pts[:, [0, 2]]
    n_cells = g.GetNumberOfCells()
    edges = set()
    for i in range(n_cells):
        c = g.GetCell(i)
        npts = c.GetNumberOfPoints()
        ids = [c.GetPointId(j) for j in range(npts)]
        for j in range(npts):
            a, b = ids[j], ids[(j+1) % npts]
            edges.add((min(a,b), max(a,b)))
    edge_arr = np.array(list(edges), dtype=int)
    segs = pts2[edge_arr]  # (Nedges, 2, 2)
    return pts2, segs

cases = [('proper_cav_L1', 'cavity L1', 96950),
         ('proper_cav_L2', 'cavity L2', 351314),
         ('proper_str_L2', 'O-grid L2', 254881)]

zooms = [
    ('full',      (-0.05, 1.10),  (-0.20,  0.20)),
    ('te_wide',   ( 0.85, 1.05),  (-0.05,  0.05)),
    ('te_close',  ( 0.98, 1.02),  (-0.01,  0.01)),
    ('te_micro',  ( 0.995, 1.005),(-0.002, 0.002)),
]

# 4 zooms x 3 cases
fig, axs = plt.subplots(len(zooms), len(cases), figsize=(15, 16))

for col, (d, label, ncell) in enumerate(cases):
    pts2, segs = load_slice_cells(f"/home/qiqi/flexcompute/aft-sa/flow360/{d}")
    print(f"{label}: {len(pts2)} pts, {len(segs)} edges in slice")
    for row, (zname, xlim, zlim) in enumerate(zooms):
        ax = axs[row, col]
        # filter segments that touch the zoom box (any endpoint inside)
        in_box = ((segs[:,:,0] >= xlim[0]-0.05) & (segs[:,:,0] <= xlim[1]+0.05) &
                  (segs[:,:,1] >= zlim[0]-0.02) & (segs[:,:,1] <= zlim[1]+0.02))
        keep = in_box.any(axis=1)
        lc = LineCollection(segs[keep], colors='k', linewidths=0.3)
        ax.add_collection(lc)
        ax.set_xlim(*xlim); ax.set_ylim(*zlim)
        ax.set_aspect('equal')
        ax.grid(alpha=0.2, lw=0.3)
        if row == 0:
            ax.set_title(f"{label}\n(Ncell={ncell})", fontsize=10)
        if col == 0:
            ax.set_ylabel(f"{zname}\nz/c", fontsize=9)
        if row == len(zooms)-1:
            ax.set_xlabel("x/c", fontsize=9)
        ax.tick_params(labelsize=7)

plt.tight_layout(pad=0.5)
plt.savefig('/tmp/TE_mesh_zoom.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/TE_mesh_zoom.pdf')
print("wrote /tmp/TE_mesh_zoom.png + paper figs/TE_mesh_zoom.pdf")
