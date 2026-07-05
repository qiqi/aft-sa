"""Overlay the airfoil contour points (input to the mesher) on the mesh cells.
This shows directly whether the mesher is respecting our requested wall-tangential
spacing at the TE.
"""
import vtk, numpy as np, os
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def load_mesh_edges(path):
    if path.endswith('.vtk'):
        r = vtk.vtkUnstructuredGridReader(); r.SetFileName(path); r.Update()
    else:
        r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(path); r.Update()
    g = r.GetOutput()
    pts = vtk_to_numpy(g.GetPoints().GetData())
    pts2 = pts[:, [0, 1]] if path.endswith('.vtk') else pts[:, [0, 2]]
    edges = set()
    for i in range(g.GetNumberOfCells()):
        c = g.GetCell(i); npts = c.GetNumberOfPoints()
        ids = [c.GetPointId(j) for j in range(npts)]
        for j in range(npts):
            a, b = ids[j], ids[(j+1) % npts]
            edges.add((min(a,b), max(a,b)))
    return pts2, pts2[np.array(list(edges), dtype=int)]

def load_contour_pts(case_dir):
    """Parse contours.txt to recover the airfoil contour points we asked for."""
    p = f"{case_dir}/contours.txt"
    if not os.path.exists(p): return None, None
    lines = open(p).readlines()
    # Format: H0, GROWTH, HWALL, HMAX, NPOINTS N, then N rows of "x z", then NCURVES, then curves
    npoints = None; data_start = None
    for i, ln in enumerate(lines):
        if ln.startswith("NPOINTS"):
            npoints = int(ln.split()[1]); data_start = i+1; break
    if npoints is None: return None, None
    coords = np.array([list(map(float, ln.split())) for ln in lines[data_start:data_start+npoints]])
    # The first curve is the farfield, the second is the airfoil. Skip the farfield (NCURVES line then curves)
    # Read NCURVES section
    after_data = data_start + npoints
    ncurves_line = lines[after_data]   # "NCURVES 2"
    # Then NCURVES lines like: "0 481 0 1 2 ... 480 0"  (closed=0, N+1 indices, indices)
    curve_lines = lines[after_data+1 : after_data+1+int(ncurves_line.split()[1])]
    # The AIRFOIL is the second curve (curve index 1)
    parts = curve_lines[1].split()
    npts_curve = int(parts[1])
    indices = [int(x) for x in parts[2:2+npts_curve]]
    airfoil_pts = coords[indices[:-1]]   # drop the closing duplicate
    return coords, airfoil_pts

F = "/home/qiqi/flexcompute/aft-sa/flow360"

cases = [
    (f"{F}/proper_cav_L1",         "cavity L1\n(no $h_\\mathrm{TE}$ control)",       7e-6),
    (f"{F}/proper_cav_L1_TEprop",  "cavity L1\n($h_\\mathrm{TE}=h_0=7\\mu m$, $r{=}1.5$)", 7e-6),
    (f"{F}/proper_cav_L3_TEprop",  "cavity L3\n($h_\\mathrm{TE}=h_0=1.75\\mu m$, $r{=}1.125$)", 1.75e-6),
]

fig, axs = plt.subplots(2, len(cases), figsize=(13, 8))

for col, (case_dir, lbl, h0) in enumerate(cases):
    mesh_path = f"{case_dir}/mesh2d.vtk"
    if not os.path.exists(mesh_path): continue
    pts2, segs = load_mesh_edges(mesh_path)
    _, ctour = load_contour_pts(case_dir)
    print(f"{lbl}: {len(pts2)} mesh pts, {len(segs)} edges, {len(ctour)} contour pts")
    # show two zoom levels
    for row, w_factor in enumerate([60, 15]):
        ax = axs[row, col]
        w = w_factor * h0
        xlim = (1 - w, 1 + w); ylim = (-w, w)
        # mesh edges
        in_box = ((segs[:,:,0] >= xlim[0]-1e-5) & (segs[:,:,0] <= xlim[1]+1e-5) &
                  (segs[:,:,1] >= ylim[0]-1e-5) & (segs[:,:,1] <= ylim[1]+1e-5))
        keep = in_box.any(axis=1)
        lc = LineCollection(segs[keep], colors='0.4', linewidths=0.4, rasterized=True)
        ax.add_collection(lc)
        # contour points overlay — only those in window
        in_win = ((ctour[:,0] >= xlim[0]) & (ctour[:,0] <= xlim[1]) &
                  (ctour[:,1] >= ylim[0]) & (ctour[:,1] <= ylim[1]))
        cwin = ctour[in_win]
        ax.scatter(cwin[:,0], cwin[:,1], color='red', s=12, zorder=10,
                   marker='o', edgecolors='darkred', linewidth=0.4)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal')
        ax.grid(alpha=0.2, lw=0.3); ax.tick_params(labelsize=7)
        if row == 0:
            ax.set_title(f"{lbl}\n[{len(cwin)} contour pts in window]", fontsize=9)
        ax.set_ylabel(f"$\\pm{w_factor}h_0$ window\n({2*w*1e6:.0f}μm)", fontsize=8)
        if row == 1: ax.set_xlabel("x/c", fontsize=9)

plt.suptitle("Airfoil contour points (red) overlaid on the mesh edges (gray). "
             "Demonstrates whether the mesher respects our requested tangential spacing.",
             fontsize=10)
plt.tight_layout(pad=0.4)
plt.savefig('/tmp/TE_mesh_with_contour.png', dpi=140)
print("wrote /tmp/TE_mesh_with_contour.png")
