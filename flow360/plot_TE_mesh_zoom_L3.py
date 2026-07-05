"""TE mesh zoom: cavity L1, L2, L3 + O-grid L2."""
import vtk, numpy as np, os
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def load_slice_cells(d):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f"{d}/slice_centerSpan.pvtu")
    r.Update()
    g = r.GetOutput()
    pts = vtk_to_numpy(g.GetPoints().GetData())
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
    segs = pts2[edge_arr]
    return pts2, segs

import json
def get_ncell(d):
    p = f"{d}/mesh.cgns.json"
    if os.path.exists(p):
        m = json.load(open(p))
        if 'nNodes' in m: return int(m['nNodes'])
    return None

cases = [('proper_cav_L1',         'cavity L1\n(no $h_\\mathrm{TE}$ control)'),
         ('proper_cav_L1_TEtight', 'cavity L1\n($h_\\mathrm{TE}{=}h_0$, isotropic)'),
         ('proper_cav_L3_TEtight', 'cavity L3\n($h_\\mathrm{TE}{=}h_0$, isotropic)'),
         ('proper_str_L2',         'O-grid L2')]

zooms = [
    ('full',      (-0.05, 1.10),  (-0.20,  0.20)),
    ('te_wide',   ( 0.85, 1.05),  (-0.05,  0.05)),
    ('te_close',  ( 0.98, 1.02),  (-0.01,  0.01)),
    ('te_micro',  ( 0.995, 1.005),(-0.002, 0.002)),
]

fig, axs = plt.subplots(len(zooms), len(cases), figsize=(18, 16))

for col, (d, label) in enumerate(cases):
    full = f"/home/qiqi/flexcompute/aft-sa/flow360/{d}"
    if not os.path.exists(f"{full}/slice_centerSpan.pvtu"):
        print(f"SKIP {d}: no slice file")
        continue
    pts2, segs = load_slice_cells(full)
    nc = get_ncell(full)
    print(f"{label}: {len(pts2)} slice pts, {len(segs)} edges, ncell≈{nc}")
    for row, (zname, xlim, zlim) in enumerate(zooms):
        ax = axs[row, col]
        in_box = ((segs[:,:,0] >= xlim[0]-0.05) & (segs[:,:,0] <= xlim[1]+0.05) &
                  (segs[:,:,1] >= zlim[0]-0.02) & (segs[:,:,1] <= zlim[1]+0.02))
        keep = in_box.any(axis=1)
        lc = LineCollection(segs[keep], colors='k', linewidths=0.3, rasterized=True)
        ax.add_collection(lc)
        ax.set_xlim(*xlim); ax.set_ylim(*zlim)
        ax.set_aspect('equal'); ax.grid(alpha=0.2, lw=0.3)
        if row == 0:
            ax.set_title(f"{label}\n(Ncell={nc})", fontsize=10)
        if col == 0:
            ax.set_ylabel(f"{zname}\nz/c", fontsize=9)
        if row == len(zooms)-1:
            ax.set_xlabel("x/c", fontsize=9)
        ax.tick_params(labelsize=7)

plt.tight_layout(pad=0.5)
plt.savefig('/tmp/TE_mesh_zoom_L3.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/TE_mesh_zoom_L3.pdf', dpi=200)
print("wrote /tmp/TE_mesh_zoom_L3.png + paper figs/TE_mesh_zoom_L3.pdf")
