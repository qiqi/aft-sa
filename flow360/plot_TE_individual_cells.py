"""Zoom in tight enough at TE to see INDIVIDUAL CELLS, so the isotropy is visible.
Each cell is ~h_0 in size; at L3, h_0=1.75e-6, so we need a window ~30*h_0 ≈ 50 micron wide.
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

cases = [
    (f"{F}/proper_cav_L1/mesh2d.vtk",         "cavity L1\n(no $h_\\mathrm{TE}$ control)\n$h_0=7\\!\\times\\!10^{-6}$"),
    (f"{F}/proper_cav_L1_TEprop/mesh2d.vtk",  "cavity L1\n($h_\\mathrm{TE}{=}h_0$, $r{=}1.5$)\n$h_0=7\\!\\times\\!10^{-6}$"),
    (f"{F}/proper_cav_L3_TEprop/mesh2d.vtk",  "cavity L3\n($h_\\mathrm{TE}{=}h_0$, $r{=}1.125$)\n$h_0=1.75\\!\\times\\!10^{-6}$"),
    (f"{F}/proper_str_L2/slice_centerSpan.pvtu", "O-grid L2\n(reference)"),
]

# Per-case zooms tuned to show ~10-30 individual cells across the window.
# Each case has its own h_0; we set the window proportional to h_0 to make cells visible.
zooms_per_case = [
    [(0.85, 1.05, 0.05),  (0.99, 1.01, 0.01),    (0.997, 1.003, 0.003), (0.9995, 1.0005, 0.0005)],
    [(0.85, 1.05, 0.05),  (0.99, 1.01, 0.01),    (0.997, 1.003, 0.003), (0.9995, 1.0005, 0.0005)],
    [(0.85, 1.05, 0.05),  (0.99, 1.01, 0.01),    (0.997, 1.003, 0.003), (0.9999, 1.0001, 0.0001)],
    [(0.85, 1.05, 0.05),  (0.99, 1.01, 0.01),    (0.997, 1.003, 0.003), (0.9995, 1.0005, 0.0005)],
]

row_labels = ['te_wide\n($\\pm0.05c$)', 'te_close\n($\\pm0.01c$)',
              'te_mid\n($\\pm0.003c$)', 'te_micro\n(per-case tight zoom)']

fig, axs = plt.subplots(4, len(cases), figsize=(15, 16))
for col, (path, lbl) in enumerate(cases):
    if not os.path.exists(path):
        print(f"SKIP {path}"); continue
    pts2, segs = load_mesh_edges(path)
    print(f"{lbl}: {len(pts2)} pts, {len(segs)} edges")
    for row, (x0, x1, half) in enumerate(zooms_per_case[col]):
        ax = axs[row, col]
        ylim = (-half, half)
        in_box = ((segs[:,:,0] >= x0-0.02) & (segs[:,:,0] <= x1+0.02) &
                  (segs[:,:,1] >= ylim[0]-0.01) & (segs[:,:,1] <= ylim[1]+0.01))
        keep = in_box.any(axis=1)
        lc = LineCollection(segs[keep], colors='k', linewidths=0.35, rasterized=True)
        ax.add_collection(lc)
        ax.set_xlim(x0, x1); ax.set_ylim(*ylim); ax.set_aspect('equal')
        ax.grid(alpha=0.2, lw=0.3); ax.tick_params(labelsize=7)
        if row == 0: ax.set_title(lbl, fontsize=10)
        if col == 0: ax.set_ylabel(row_labels[row], fontsize=9)
        if row == 3: ax.set_xlabel("x/c", fontsize=9)
plt.tight_layout(pad=0.5)
plt.savefig('/tmp/TE_mesh_individual.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/TE_mesh_zoom_L3.pdf', dpi=200)
print("wrote /tmp/TE_mesh_individual.png")
