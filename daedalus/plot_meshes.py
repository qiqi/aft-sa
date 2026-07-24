"""Verification views of the two Daedalus L0 wing meshes."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import ogrid_wing as og
from wing_geometry import SectionFamily, chord, HALF_SPAN, C_ROOT, XQC


def plane_segments(nodes):
    """Grid-line segments of one (N_s, N_j+1, 2) plane."""
    segs = []
    N, M, _ = nodes.shape
    for i in range(N):
        i1 = (i + 1) % N
        segs += [[nodes[i, j], nodes[i1, j]] for j in range(M)]
    for i in range(N):
        segs += [[nodes[i, j], nodes[i, j + 1]] for j in range(M - 1)]
    return segs


fam = SectionFamily(og.N_PER_SIDE)
growth = og.solve_growth(og.H0, og.N_J, og.R_FF)
d = og.H0 * (growth**np.arange(og.N_J + 1) - 1) / (growth - 1)
j_bl = int(np.argmin(np.abs(d - 0.12)))
P0 = fam.contour(0.0)
tt = np.roll(P0, -1, axis=0) - np.roll(P0, 1, axis=0)
itop = int(np.argmax(P0[:, 1]))
rot = 1.0 if -tt[itop, 0] > 0 else -1.0
root = og.march_plane(fam.contour(0.0), d, rot, j_bl)
c_tip = chord(1.0)
slit = og.slit_plane(XQC - 0.25 * c_tip, XQC + 0.75 * c_tip, d, 2 * og.N_PER_SIDE, d[j_bl])

fig, axs = plt.subplots(2, 2, figsize=(13, 11))
views = [(axs[0, 0], root, (-0.1, 1.1), (-0.35, 0.35), 'root section (DAE-11), near field'),
         (axs[0, 1], root, (-95, 95), (-95, 95), 'root section, full domain (R=100 c_root)'),
         (axs[1, 0], slit, (0.05, 0.60), (-0.16, 0.16), 'beyond-tip slit plane: ray fan + bent end rays')]
for ax, pl, xl, zl, title in views:
    lc = LineCollection(plane_segments(pl), linewidths=0.25, colors='k')
    ax.add_collection(lc)
    ax.set_xlim(*xl); ax.set_ylim(*zl); ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)

# planform view of the wall quads (x vs y)
ax = axs[1, 1]
t = np.linspace(0, 1, og.N_WING + 1)
ys = (HALF_SPAN - og.TIP_GAP) * np.sin(0.5 * np.pi * t)
for y in ys:
    c = chord(min(y / HALF_SPAN, 1.0))
    x_le = XQC - 0.25 * c
    ax.plot([x_le, x_le + c], [y, y], 'k-', lw=0.4)
etas = np.linspace(0, 1, 200)
cs = chord(etas)
ax.plot(XQC - 0.25 * cs, etas * HALF_SPAN, 'b-', lw=1)
ax.plot(XQC + 0.75 * cs, etas * HALF_SPAN, 'b-', lw=1)
ax.axhline(HALF_SPAN, color='r', lw=0.8, ls='--', label='pinch (tip)')
ax.set_aspect('equal'); ax.legend(fontsize=8)
ax.set_title('planform: wing stations + tip pinch', fontsize=10)
plt.tight_layout()
plt.savefig('mesh_ogrid_views.png', dpi=130, bbox_inches='tight')
print('wrote mesh_ogrid_views.png')

# ---- cavity mesh: node cloud on a thin center-span slab -------------------
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
r = vtk.vtkCGNSReader()
r.SetFileName('cavity_L0/wing_cavity_L0.cgns')
r.UpdateInformation(); r.EnableAllBases(); r.Update()
ds = r.GetOutput()
it = ds.NewIterator(); it.InitTraversal()
pts = []
while not it.IsDoneWithTraversal():
    obj = it.GetCurrentDataObject()
    if obj is not None and obj.GetNumberOfPoints() > 0:
        pts.append(vtk_to_numpy(obj.GetPoints().GetData()))
    it.GoToNextItem()
P = np.vstack(pts)
X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
m = np.abs(Y) < 0.10
fig, axs = plt.subplots(1, 3, figsize=(16, 5.5))
for ax, xl, zl, title in [
        (axs[0], (-0.2, 1.2), (-0.5, 0.5), 'cavity: center-span slab |y|<0.1, near field'),
        (axs[1], (0.10, 0.40), (-0.02, 0.10), 'zoom: BL prism layering (upper surface)'),
        (axs[2], (-60, 60), (-60, 60), 'far field')]:
    ax.plot(X[m], Z[m], '.', ms=0.5, alpha=0.5)
    ax.set_xlim(*xl); ax.set_ylim(*zl); ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
plt.tight_layout()
plt.savefig('mesh_cavity_views.png', dpi=130, bbox_inches='tight')
print('wrote mesh_cavity_views.png')
