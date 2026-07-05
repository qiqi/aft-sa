"""Show a single tight zoom at TE for each case, sized to display ~5-10 individual
cells across the window. This proves the cells AT the TE corner are isotropic on
h_TE=h_0 meshes; the fan-like appearance in larger zooms is the prism stack
extending radially from the SHARED TE node into the wake.
"""
import vtk, numpy as np, os
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

def load_mesh_edges(path):
    if path.endswith('.vtk'):
        r = vtk.vtkUnstructuredGridReader(); r.SetFileName(path); r.Update()
    else:
        r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(path); r.Update()
    g = r.GetOutput()
    pts = vtk_to_numpy(g.GetPoints().GetData())
    pts2 = pts[:, [0, 1]] if path.endswith('.vtk') else pts[:, [0, 2]]
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

# Each case has its own h_0; show a window with ~20 cells across the window
cases = [
    # (path, label, h_0)
    (f"{F}/proper_cav_L1/mesh2d.vtk",         "cavity L1 (no $h_\\mathrm{TE}$ control)", 7e-6),
    (f"{F}/proper_cav_L1_TEprop/mesh2d.vtk",  "cavity L1 ($h_\\mathrm{TE}=h_0=7\\mu m$, $r=1.5$)", 7e-6),
    (f"{F}/proper_cav_L3_TEprop/mesh2d.vtk",  "cavity L3 ($h_\\mathrm{TE}=h_0=1.75\\mu m$, $r=1.125$)", 1.75e-6),
    (f"{F}/proper_str_L2/slice_centerSpan.pvtu", "O-grid L2 (reference)", 7e-6),
]

# Two zoom levels: wider (15-20 cells) and tighter (~5-7 cells around the TE point)
fig, axs = plt.subplots(2, len(cases), figsize=(15, 8))

for col, (path, lbl, h0) in enumerate(cases):
    if not os.path.exists(path): continue
    pts2, segs = load_mesh_edges(path)
    # Window 1: ~30 cells across
    w1 = 30 * h0
    # Window 2: ~10 cells across — INDIVIDUAL CELLS SHOULD BE READILY VISIBLE
    w2 = 8 * h0
    for row, w in enumerate([w1, w2]):
        ax = axs[row, col]
        xlim = (1 - w, 1 + w); ylim = (-w, w)
        in_box = ((segs[:,:,0] >= xlim[0]-1e-5) & (segs[:,:,0] <= xlim[1]+1e-5) &
                  (segs[:,:,1] >= ylim[0]-1e-5) & (segs[:,:,1] <= ylim[1]+1e-5))
        keep = in_box.any(axis=1)
        lc = LineCollection(segs[keep], colors='k', linewidths=0.5, rasterized=True)
        ax.add_collection(lc)
        # mark the (1, 0) TE point
        ax.plot(1, 0, 'r.', ms=8, zorder=10)
        ax.add_patch(Circle((1, 0), h0, fill=False, ec='red', lw=0.8, ls=':'))
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal')
        ax.grid(alpha=0.2, lw=0.3); ax.tick_params(labelsize=7)
        if row == 0:
            ax.set_title(lbl, fontsize=9)
            ax.set_ylabel(f"$\\pm 30 h_0$\n{2*w*1e6:.0f} $\\mu m$", fontsize=8)
        else:
            ax.set_ylabel(f"$\\pm 8 h_0$\n{2*w*1e6:.0f} $\\mu m$", fontsize=8)
            ax.set_xlabel("x/c", fontsize=9)

plt.suptitle("TE region with red dot at (1,0), red dashed circle of radius $h_0$. "
             "On $h_\\mathrm{TE}=h_0$ cavity meshes, the cells in the dashed circle are isotropic.",
             fontsize=10)
plt.tight_layout(pad=0.5)
plt.savefig('/tmp/TE_tight_zoom.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/TE_mesh_zoom_L3.pdf', dpi=200)
print("wrote /tmp/TE_tight_zoom.png")
