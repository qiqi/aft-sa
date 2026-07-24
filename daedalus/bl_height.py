"""Measure the prism (BL) shell height of the cavity mesh vs chord position,
in spanwise bands, and compare with the local surface edge scales.

Usage: python3 bl_height.py <mesh.cgns>
"""
import sys
import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import viz_meshes as V

mb = V.read_cgns(sys.argv[1])
blocks = V.leaf_blocks(mb)
vols = V.volume_blocks(blocks)
wall = V.wall_from_volume(vols, (-V.XT, V.XT))
imp = vtk.vtkImplicitPolyDataDistance()
imp.SetInput(wall)

BANDS = [('root', 0.0, 0.4), ('mid', 8.5, 0.4), ('neartip', 16.9, 0.1)]
for ds in vols:
    types = vtk_to_numpy(ds.GetCellTypesArray()) if hasattr(ds, 'GetCellTypesArray') \
        else np.array([ds.GetCellType(i) for i in range(ds.GetNumberOfCells())])
    cc = vtk.vtkCellCenters(); cc.SetInputData(ds); cc.Update()
    ctr = vtk_to_numpy(cc.GetOutput().GetPoints().GetData())
    is_prism = (types == vtk.VTK_WEDGE) | (types == vtk.VTK_HEXAHEDRON)
    for name, y0, hw in BANDS:
        m = is_prism & (np.abs(ctr[:, 1] - y0) < hw) & \
            (np.hypot(ctr[:, 0] - V.XQC, ctr[:, 2]) < 1.5) & (ctr[:, 2] > 0)
        idx = np.where(m)[0]
        if len(idx) > 60000:
            idx = np.random.default_rng(0).choice(idx, 60000, replace=False)
        x = ctr[idx, 0]
        dist = np.array([abs(imp.EvaluateFunction(ctr[i])) for i in idx])
        print(f'\n== band {name} (y ~ {y0}) upper surface: prism shell height '
              f'by chord station ==')
        print(f'{"x [m]":>8} {"shell h [m]":>12}')
        for a, b in [(0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5),
                     (0.5, 0.7), (0.7, 0.85), (0.85, 0.95)]:
            mm = (x >= a + 0.0) & (x < b)
            if mm.sum() < 5:
                continue
            print(f'{0.5*(a+b):8.3f} {np.quantile(dist[mm], 0.98):12.3f}')

# local surface scales at the root for reference
from wing_geometry import SectionFamily, chord, HALF_SPAN
import wing_geometry as wg
n_per_side, n_span = 64, 80
fam = SectionFamily(n_per_side)
t = np.linspace(0, 1, n_span + 1)
ys = HALF_SPAN * np.sin(0.5 * np.pi * t)
print('\n== surface edge scales (root band) ==')
dy_root = ys[1] - ys[0]
r = fam.contour(0.0)
seg = np.linalg.norm(np.roll(r, -1, axis=0) - r, axis=1)
xs = r[:, 0]
print(f'{"x [m]":>8} {"chordwise ds":>13} {"spanwise dy":>12} '
      f'{"sqrt(ds*dy)":>12} {"6.6*sqrt(2A)":>13}')
for a, b in [(0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5),
             (0.5, 0.7), (0.7, 0.85), (0.85, 0.95)]:
    mm = (xs >= a) & (xs < b)
    if mm.sum() < 2:
        continue
    ds = np.median(seg[mm])
    gm = np.sqrt(ds * dy_root)
    # stop when next increment ~ sqrt(2A); cumulative = increment*g/(g-1), g=1.2
    print(f'{0.5*(a+b):8.3f} {ds:13.4f} {dy_root:12.3f} {gm:12.3f} {6.0*gm:13.3f}')
