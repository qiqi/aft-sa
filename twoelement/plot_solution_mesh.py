"""Six-panel view of a solved level: Mach line contours and the mesh, at three
zooms -- whole configuration, flap only, and the flap LE / slot region.

Reads the quasi-2D volume output and the 2D triangle mesh written by the
mesher, takes the mid-span plane, and draws:

    row 1  Mach LINE contours (not filled)
    row 2  the mesh edges

Run:  python3 plot_solution_mesh.py case_L1 [outfile.pdf]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def read_vtu(path):
    """Points + triangles + point arrays from a (legacy or XML) VTK file."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    if path.endswith('.vtk'):
        r = vtk.vtkUnstructuredGridReader()
        r.SetFileName(path); r.ReadAllScalarsOn(); r.ReadAllVectorsOn()
    elif path.endswith('.pvtu'):
        r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(path)
    else:
        r = vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(path)
    r.Update()
    g = r.GetOutput()
    pts = vtk_to_numpy(g.GetPoints().GetData())
    pd = g.GetPointData()
    arr = {pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i))
           for i in range(pd.GetNumberOfArrays())}
    return g, pts, arr


def midplane_tris(g, pts, span_axis=1):
    """Nodes + triangles on one span plane of the quasi-2D WEDGE extrusion.

    The mesher extrudes in Y (2 planes, 1 cell), so the airfoil plane is X-Z
    and each cell is a wedge whose first three nodes are the bottom triangle.
    """
    from vtk.util.numpy_support import vtk_to_numpy
    s_ = pts[:, span_axis]
    keep = np.isclose(s_, np.unique(np.round(s_, 9))[0], atol=1e-9)
    idx = np.where(keep)[0]
    remap = -np.ones(len(pts), int)
    remap[idx] = np.arange(len(idx))
    cells = vtk_to_numpy(g.GetCells().GetConnectivityArray())
    offs = vtk_to_numpy(g.GetCells().GetOffsetsArray())
    tris = []
    for a_, b_ in zip(offs[:-1], offs[1:]):
        c = cells[a_:b_]
        cs = [v for v in c if keep[v]]
        if len(cs) == 3:
            tris.append([remap[v] for v in cs])
    plane = np.delete(pts[idx], span_axis, axis=1)      # -> (x, z)
    return idx, np.asarray(tris, int), plane


ZOOMS = [('overall', None),
         ('flap', (0.62, 1.06, -0.09, 0.19)),
         ('flap LE / slot', (0.655, 0.795, 0.005, 0.115))]


if __name__ == '__main__':
    case = sys.argv[1] if len(sys.argv) > 1 else 'case_L1'
    out = sys.argv[2] if len(sys.argv) > 2 else '%s_solution_mesh.pdf' % case

    import os
    vf = '%s/volume.pvtu' % case
    if not os.path.exists(vf):
        vf = '%s/volume_proc0.vtu' % case
    g, pts, arr = read_vtu(vf)
    print('volume arrays:', sorted(arr))
    idx, tris, P2 = midplane_tris(g, pts)
    if 'Mach' in arr:
        mach = arr['Mach'][idx]
    else:
        v = arr['velocity'][idx]
        mach = np.linalg.norm(v, axis=1)*0.1        # fall back on |u|/Uinf * M
    print('mid-plane: %d nodes, %d tris ; Mach %.4f .. %.4f'
          % (len(P2), len(tris), mach.min(), mach.max()))
    T = mtri.Triangulation(P2[:, 0], P2[:, 1], tris)

    gm, pm, _ = read_vtu('%s/mesh2d.vtk' % case)
    # the 2D mesh may also carry its plane in x-z; pick the two varying axes
    nun = [len(np.unique(np.round(pm[:, i], 9))) for i in range(3)]
    pm = pm[:, [i for i in range(3) if nun[i] > 2][:2]]
    from vtk.util.numpy_support import vtk_to_numpy
    cm = vtk_to_numpy(gm.GetCells().GetConnectivityArray())
    om = vtk_to_numpy(gm.GetCells().GetOffsetsArray())
    mt = np.array([cm[a:b] for a, b in zip(om[:-1], om[1:]) if b - a == 3])
    Tm = mtri.Triangulation(pm[:, 0], pm[:, 1], mt)
    print('mesh2d: %d nodes, %d tris' % (len(pm), len(mt)))

    lv = np.linspace(max(mach.min(), 1e-4), mach.max(), 26)
    fig, ax = plt.subplots(2, 3, figsize=(16.0, 7.6))
    for j, (name, box) in enumerate(ZOOMS):
        a0, a1 = ax[0, j], ax[1, j]
        a0.tricontour(T, mach, levels=lv, linewidths=0.6, colors='#1f4e9c')
        a1.triplot(Tm, color='0.35', lw=0.25)
        for a in (a0, a1):
            if box:
                a.set_xlim(box[0], box[1]); a.set_ylim(box[2], box[3])
            else:
                a.set_xlim(-0.12, 1.12); a.set_ylim(-0.16, 0.26)
            a.set_aspect('equal')
            a.tick_params(labelsize=7)
        a0.set_title('Mach contours - %s' % name, fontsize=9.5)
        a1.set_title('mesh - %s' % name, fontsize=9.5)
    fig.suptitle('%s : SA-AI, Re=1e6, M=0.1, alpha=-1 deg' % case, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out)
    print('wrote', out)
