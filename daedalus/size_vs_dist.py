"""Cell size vs wall distance for the cavity mesh: quantifies the BL-to-tet
size transition. size = cbrt(cell volume); distance = cell center to the wing
surface (implicit distance on the extracted wall). Cells within 2 m of the
section only. Prints binned medians; optional PNG.

Usage: python3 size_vs_dist.py <mesh.cgns> [out.png]
"""
import sys
import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import viz_meshes as V

cgns = sys.argv[1]
png = sys.argv[2] if len(sys.argv) > 2 else None

mb = V.read_cgns(cgns)
blocks = V.leaf_blocks(mb)
vols = V.volume_blocks(blocks)
wall = V.wall_from_volume(vols, (-V.XT, V.XT))

# cell centers + volumes
sizes, dists = [], []
imp = vtk.vtkImplicitPolyDataDistance()
imp.SetInput(wall)
for ds in vols:
    cs = vtk.vtkCellSizeFilter()
    cs.SetInputData(ds)
    cs.ComputeVolumeOn(); cs.ComputeAreaOff(); cs.ComputeLengthOff()
    cs.ComputeVertexCountOff()
    cs.Update()
    out = cs.GetOutput()
    vol = vtk_to_numpy(out.GetCellData().GetArray('Volume'))
    cc = vtk.vtkCellCenters(); cc.SetInputData(ds); cc.Update()
    ctr = vtk_to_numpy(cc.GetOutput().GetPoints().GetData())
    r_xz = np.hypot(ctr[:, 0] - V.XQC, ctr[:, 2])
    keep = (r_xz < 2.5) & (np.abs(ctr[:, 1]) < V.XT + 1.0) & (vol > 0)
    idx = np.where(keep)[0]
    # subsample for the implicit-distance query
    if len(idx) > 400000:
        idx = np.random.default_rng(0).choice(idx, 400000, replace=False)
    d = np.array([abs(imp.EvaluateFunction(ctr[i])) for i in idx])
    sizes.append(np.cbrt(vol[idx]))
    dists.append(d)
sizes = np.concatenate(sizes); dists = np.concatenate(dists)

bins = np.geomspace(1e-4, 2.0, 25)
print(f'{"wall dist [m]":>16} {"median size [m]":>16} {"p90 size":>10} {"n":>8}')
med = []
for a, b in zip(bins, bins[1:]):
    m = (dists >= a) & (dists < b)
    if m.sum() < 10:
        med.append(np.nan); continue
    md = np.median(sizes[m]); med.append(md)
    print(f'{np.sqrt(a*b):16.4f} {md:16.4f} {np.quantile(sizes[m],0.9):10.4f} {m.sum():8d}')
med = np.array(med)
r = med[1:] / med[:-1]
print(f'max adjacent-bin size ratio: {np.nanmax(r):.2f}')

if png:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(dists, sizes, '.', ms=0.5, alpha=0.15)
    ax.loglog(np.sqrt(bins[:-1] * bins[1:]), med, 'r-o', ms=3, label='median')
    ax.set_xlabel('distance from wing surface [m]')
    ax.set_ylabel('cell size = cbrt(volume) [m]')
    ax.grid(alpha=0.3, which='both'); ax.legend()
    fig.savefig(png, dpi=130, bbox_inches='tight')
    print('wrote', png)
