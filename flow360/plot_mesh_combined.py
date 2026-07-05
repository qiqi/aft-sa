"""Combined Fig 5: LE-detail (top row) and TE-detail (bottom row) for unstructured
(L1 TEprop, left column) vs structured O-grid (L1, right column).
Replaces the previous separate mesh.pdf (LE only) and TE_mesh_zoom_L3.pdf.
"""
import vtk, numpy as np, os
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

UNSTRUCT = '/home/qiqi/flexcompute/aft-sa/flow360/proper_cav_L1_TEprop/mesh2d.vtk'
STRUCT_MSH = '/home/qiqi/flexcompute/aft-sa/flow360/proper_str_L1/mesh.msh'

def load_vtk_edges(path):
    r = vtk.vtkUnstructuredGridReader(); r.SetFileName(path); r.Update()
    g = r.GetOutput()
    pts = vtk_to_numpy(g.GetPoints().GetData())[:, [0, 1]]
    edges = set()
    for i in range(g.GetNumberOfCells()):
        c = g.GetCell(i); n = c.GetNumberOfPoints()
        ids = [c.GetPointId(j) for j in range(n)]
        for j in range(n):
            a, b = ids[j], ids[(j+1) % n]
            edges.add((min(a, b), max(a, b)))
    return pts, pts[np.array(list(edges), dtype=int)]

def load_msh_symmetry_edges(path):
    """Read .msh v2, return the (x,z) edges of the symmetry-plane quads (tag 4)."""
    lines = open(path).read().splitlines()
    n_idx = lines.index('$Nodes')
    e_idx = lines.index('$Elements')
    n_count = int(lines[n_idx + 1])
    nodes = {}
    for i in range(n_count):
        toks = lines[n_idx + 2 + i].split()
        nid = int(toks[0]); x = float(toks[1]); y = float(toks[2]); z = float(toks[3])
        nodes[nid] = (x, y, z)
    e_count = int(lines[e_idx + 1])
    quads = []
    z_vals = set()
    for i in range(e_count):
        toks = lines[e_idx + 2 + i].split()
        etype = int(toks[1])
        ntags = int(toks[2])
        ptag  = int(toks[3])
        if etype == 3 and ptag == 4:   # quad on symmetry1
            ids = [int(t) for t in toks[3 + ntags : 3 + ntags + 4]]
            quads.append(ids)
            for nid in ids:
                z_vals.add(nodes[nid][1])
    # Find which span position symmetry1 sits at
    print(f"  symmetry1 y-values: {sorted(z_vals)[:3]}...")
    # Each quad: 4 ids — gather (x, z) coords (we use Y is span, X-Z is airfoil plane)
    pts_xz = []
    pts_idx = {}
    edges = set()
    for q in quads:
        for nid in q:
            if nid not in pts_idx:
                pts_idx[nid] = len(pts_xz)
                pts_xz.append((nodes[nid][0], nodes[nid][2]))
        # 4 edges of quad
        for j in range(4):
            a, b = pts_idx[q[j]], pts_idx[q[(j+1) % 4]]
            edges.add((min(a, b), max(a, b)))
    pts_xz = np.array(pts_xz)
    edges_arr = pts_xz[np.array(list(edges), dtype=int)]
    return pts_xz, edges_arr

print("loading unstructured...")
u_pts, u_seg = load_vtk_edges(UNSTRUCT)
print(f"  {len(u_pts)} pts, {len(u_seg)} edges")
print("loading structured...")
s_pts, s_seg = load_msh_symmetry_edges(STRUCT_MSH)
print(f"  {len(s_pts)} pts, {len(s_seg)} edges")

# Filter window helper
def filter_edges(seg, xlim, ylim):
    in_box = ((seg[:,:,0] >= xlim[0] - 1e-5) & (seg[:,:,0] <= xlim[1] + 1e-5) &
              (seg[:,:,1] >= ylim[0] - 1e-5) & (seg[:,:,1] <= ylim[1] + 1e-5))
    return seg[in_box.any(axis=1)]

fig, axs = plt.subplots(2, 2, figsize=(11, 8))

# Row 0: leading edge detail (zoom ~3% chord around LE)
# Row 1: trailing edge detail (zoom ~3% chord around TE)
panels = [
    (0, 0, u_seg, 'Unstructured (Delaunay, anisotropic Spalding metric)', (-0.02, 0.04), (-0.03, 0.03)),
    (0, 1, s_seg, 'Structured O-grid (Construct2D)',                       (-0.02, 0.04), (-0.03, 0.03)),
    (1, 0, u_seg, '',                                                       ( 0.96, 1.02), (-0.02, 0.02)),
    (1, 1, s_seg, '',                                                       ( 0.96, 1.02), (-0.02, 0.02)),
]

for row, col, seg, title, xlim, ylim in panels:
    ax = axs[row, col]
    f = filter_edges(seg, xlim, ylim)
    lc = LineCollection(f, colors='k', linewidths=0.35, rasterized=True)
    ax.add_collection(lc)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal')
    ax.grid(alpha=0.2, lw=0.3); ax.tick_params(labelsize=8)
    if title: ax.set_title(title, fontsize=11)
    if row == 1: ax.set_xlabel('$x/c$', fontsize=10)
    if col == 0:
        if row == 0: ax.set_ylabel('LE detail\n$z/c$', fontsize=10)
        else:         ax.set_ylabel('TE detail\n$z/c$', fontsize=10)

plt.tight_layout(pad=0.5)
plt.savefig('/tmp/mesh_combined.png', dpi=140)
plt.savefig('/home/qiqi/flexcompute/aft-sa/paper/figs/mesh.pdf', dpi=200)
print('wrote /tmp/mesh_combined.png + paper figs/mesh.pdf')
