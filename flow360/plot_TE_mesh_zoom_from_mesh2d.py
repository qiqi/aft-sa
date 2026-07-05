"""Visualize the TE region directly from the 2D mesh (mesh2d.vtk) — no need to wait
for the solver to run. Compare original cavity L1, the new properly-refined TE
cavity L0/L1/L3 (with r_TE refining per level), and reference O-grid.
"""
import vtk, numpy as np, os
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def load_mesh_edges(path):
    """Read a VTK file (2D mesh) and return points + edges in the (x,y) plane.
    Output: (Nedges, 2, 2) array of segments.
    """
    if path.endswith('.vtk'):
        r = vtk.vtkUnstructuredGridReader(); r.SetFileName(path); r.Update()
    else:
        r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(path); r.Update()
    g = r.GetOutput()
    pts = vtk_to_numpy(g.GetPoints().GetData())
    # mesh2d.vtk: 2D mesh in (x, y, 0) — z is zero. For slice .pvtu in 3D, y is span.
    if path.endswith('.vtk'):
        pts2 = pts[:, [0, 1]]   # (x, y) since mesh2d has y as the wall-normal coord
    else:
        pts2 = pts[:, [0, 2]]   # (x, z)
    n_cells = g.GetNumberOfCells()
    edges = set()
    for i in range(n_cells):
        c = g.GetCell(i); npts = c.GetNumberOfPoints()
        ids = [c.GetPointId(j) for j in range(npts)]
        for j in range(npts):
            a, b = ids[j], ids[(j+1) % npts]
            edges.add((min(a,b), max(a,b)))
    return pts2, pts2[np.array(list(edges), dtype=int)]

F = "/home/qiqi/flexcompute/aft-sa/flow360"

# Use mesh2d.vtk directly — no solver run needed
cases = [
    (f"{F}/proper_cav_L1/mesh2d.vtk",         "cavity L1\n(no $h_\\mathrm{TE}$ control)"),
    (f"{F}/proper_cav_L1_TEprop/mesh2d.vtk",  "cavity L1\n($h_\\mathrm{TE}{=}h_0$, $r_\\mathrm{TE}{=}1.5$)"),
    (f"{F}/proper_cav_L3_TEprop/mesh2d.vtk",  "cavity L3\n($h_\\mathrm{TE}{=}h_0$, $r_\\mathrm{TE}{=}1.125$)"),
    (f"{F}/proper_str_L2/slice_centerSpan.pvtu", "O-grid L2"),
]

zooms = [
    ('te_wide',  (0.85, 1.05),  (-0.05, 0.05)),
    ('te_close', (0.98, 1.02),  (-0.01, 0.01)),
    ('te_mid',   (0.995, 1.005),(-0.002, 0.002)),
    ('te_micro', (0.9995,1.0005),(-0.0003,0.0003)),
]

fig, axs = plt.subplots(len(zooms), len(cases), figsize=(15, 17))
for col, (path, lbl) in enumerate(cases):
    if not os.path.exists(path):
        print(f"SKIP {path}"); continue
    pts2, segs = load_mesh_edges(path)
    print(f"{lbl}: {len(pts2)} pts, {len(segs)} edges")
    for row, (zname, xlim, zlim) in enumerate(zooms):
        ax = axs[row, col]
        in_box = ((segs[:,:,0] >= xlim[0]-0.02) & (segs[:,:,0] <= xlim[1]+0.02) &
                  (segs[:,:,1] >= zlim[0]-0.01) & (segs[:,:,1] <= zlim[1]+0.01))
        keep = in_box.any(axis=1)
        lc = LineCollection(segs[keep], colors='k', linewidths=0.3, rasterized=True)
        ax.add_collection(lc)
        ax.set_xlim(*xlim); ax.set_ylim(*zlim); ax.set_aspect('equal')
        ax.grid(alpha=0.2, lw=0.3); ax.tick_params(labelsize=7)
        if row == 0: ax.set_title(lbl, fontsize=10)
        if col == 0: ax.set_ylabel(zname, fontsize=9)
        if row == len(zooms)-1: ax.set_xlabel("x/c", fontsize=9)
plt.tight_layout(pad=0.5)
plt.savefig('/tmp/TE_mesh_zoom_iterate.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/TE_mesh_zoom_L3.pdf', dpi=200)
print("wrote /tmp/TE_mesh_zoom_iterate.png + paper figs/TE_mesh_zoom_L3.pdf")
