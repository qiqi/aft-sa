"""Comprehensive visualization set for the two Daedalus L0 wing meshes.

Surface views (VTK offscreen GL, run under xvfb-run): upper/lower global +
tip zoom, isometric tip, front/back zoomed to root and tip, side view from
the tip. Slice views (vtkCutter -> matplotlib line work): spanwise cuts
(root, mid, very close to tip, exactly on the tip), quarter-chord x-cut
(full wing + tip zoom), far-field global.

Output: /tmp/daedalus_mesh_views/{ogrid,cavity}/*.png
Usage:  xvfb-run -a python3 viz_meshes.py [ogrid|cavity|all]
"""
import os
import sys
import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = '/tmp/daedalus_mesh_views'
XT = 17.07              # tip y
XQC = 0.22925           # quarter chord x

MESHES = {
    'ogrid':  dict(cgns=f'{HERE}/wing_ogrid_L0.cgns',
                   wall_names=('wing',), span=(0.0, XT), y_root_view=0.35),
    'cavity': dict(cgns=f'{HERE}/cavity_L0/wing_cavity_L0.cgns',
                   wall_names=('body',), span=(-XT, XT), y_root_view=0.0),
}


def read_cgns(path):
    r = vtk.vtkCGNSReader()
    r.SetFileName(path)
    r.UpdateInformation()
    r.EnableAllBases()
    try:
        r.LoadBndPatchOn()
    except AttributeError:
        r.SetLoadBndPatch(1)
    r.Update()
    return r.GetOutput()


def leaf_blocks(mb):
    """[(name, dataset)] leaves of the multiblock tree."""
    out = []
    it = mb.NewIterator()
    it.InitTraversal()
    while not it.IsDoneWithTraversal():
        obj = it.GetCurrentDataObject()
        name = it.GetCurrentMetaData().Get(vtk.vtkCompositeDataSet.NAME()) \
            if it.GetCurrentMetaData() else ''
        if obj is not None and obj.GetNumberOfPoints() > 0:
            out.append((name or '', obj))
        it.GoToNextItem()
    return out


def surface_polys(blocks, names):
    """Surface polydata of the blocks whose name matches (2D patches)."""
    app = vtk.vtkAppendPolyData()
    n = 0
    for name, ds in blocks:
        if not any(w in name.lower() for w in names):
            continue
        gf = vtk.vtkGeometryFilter()
        gf.SetInputData(ds)
        gf.Update()
        if gf.GetOutput().GetNumberOfCells() > 0:
            app.AddInputData(gf.GetOutput())
            n += 1
    if n == 0:
        return None
    app.Update()
    return app.GetOutput()


def wall_from_volume(vols, span):
    """Wing wall surface extracted geometrically from the volume blocks'
    external surface (gmshtocgns writes BCs as UserDefined, which the VTK
    CGNS reader skips): keep external cells whose center is within 2 m of
    the section in (x, z) and strictly inside the wing span (this excludes
    the symmetry plane at y=0, the far cylinder/sphere, and the end caps)."""
    app = vtk.vtkAppendPolyData()
    y0, y1 = span
    for ds in vols:
        sf = vtk.vtkDataSetSurfaceFilter()
        sf.SetInputData(ds)
        sf.Update()
        surf = sf.GetOutput()
        cc = vtk.vtkCellCenters()
        cc.SetInputData(surf)
        cc.Update()
        ctr = vtk_to_numpy(cc.GetOutput().GetPoints().GetData())
        r_xz = np.hypot(ctr[:, 0] - XQC, ctr[:, 2])
        eps = 1e-6
        keep = (r_xz < 2.0) & (ctr[:, 1] > y0 + eps) & (ctr[:, 1] < y1 + 0.5) \
            if y0 == 0.0 else \
            (r_xz < 2.0) & (np.abs(ctr[:, 1]) < y1 + 0.5)
        ids = vtk.vtkIdTypeArray()
        for i in np.where(keep)[0]:
            ids.InsertNextValue(int(i))
        sel_node = vtk.vtkSelectionNode()
        sel_node.SetFieldType(vtk.vtkSelectionNode.CELL)
        sel_node.SetContentType(vtk.vtkSelectionNode.INDICES)
        sel_node.SetSelectionList(ids)
        sel = vtk.vtkSelection(); sel.AddNode(sel_node)
        ex = vtk.vtkExtractSelection()
        ex.SetInputData(0, surf); ex.SetInputData(1, sel); ex.Update()
        gf = vtk.vtkGeometryFilter()
        gf.SetInputData(ex.GetOutput()); gf.Update()
        if gf.GetOutput().GetNumberOfCells() > 0:
            app.AddInputData(gf.GetOutput())
    app.Update()
    return app.GetOutput()


def volume_blocks(blocks):
    return [ds for name, ds in blocks
            if ds.GetNumberOfCells() > 0 and
            ds.GetCell(0).GetCellDimension() == 3]


# ---------------------------------------------------------------------------
def render_surface(poly, campos, focal, viewup, scale, fname, size=(1600, 1200),
                   edges=True):
    m = vtk.vtkPolyDataMapper(); m.SetInputData(poly)
    a = vtk.vtkActor(); a.SetMapper(m)
    p = a.GetProperty()
    p.SetColor(0.93, 0.93, 0.96)
    # at global zoom the cells are sub-pixel and edge ink blackens the wing
    if edges:
        p.EdgeVisibilityOn()
    p.SetEdgeColor(0.15, 0.15, 0.15); p.SetLineWidth(1.0)
    p.SetAmbient(0.35); p.SetDiffuse(0.65); p.SetSpecular(0.0)
    ren = vtk.vtkRenderer(); ren.AddActor(a); ren.SetBackground(1, 1, 1)
    ren.TwoSidedLightingOn()
    cam = ren.GetActiveCamera()
    cam.ParallelProjectionOn()
    cam.SetPosition(*campos); cam.SetFocalPoint(*focal); cam.SetViewUp(*viewup)
    cam.SetParallelScale(scale)
    ren.ResetCameraClippingRange()
    rw = vtk.vtkRenderWindow(); rw.SetOffScreenRendering(1)
    rw.AddRenderer(ren); rw.SetSize(*size); rw.Render()
    w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(rw); w2i.Update()
    wr = vtk.vtkPNGWriter(); wr.SetFileName(fname)
    wr.SetInputConnection(w2i.GetOutputPort()); wr.Write()
    print('  wrote', os.path.basename(fname))


def cut_segments(vols, origin, normal, axes):
    """Cut all volume blocks with a plane; return line segments projected on
    the two in-plane axes (axes = (0,2) for a y-cut -> (x,z))."""
    segs = []
    pl = vtk.vtkPlane(); pl.SetOrigin(*origin); pl.SetNormal(*normal)
    for ds in vols:
        cut = vtk.vtkCutter(); cut.SetCutFunction(pl); cut.SetInputData(ds)
        cut.Update()
        ee = vtk.vtkExtractEdges(); ee.SetInputData(cut.GetOutput()); ee.Update()
        out = ee.GetOutput()
        if out.GetNumberOfCells() == 0:
            continue
        pts = vtk_to_numpy(out.GetPoints().GetData())
        lines = vtk_to_numpy(out.GetLines().GetData()).reshape(-1, 3)[:, 1:]
        segs.append(pts[lines][:, :, list(axes)])
    return np.concatenate(segs, axis=0) if segs else np.zeros((0, 2, 2))


def plot_segments(segs, xlim, ylim, labels, title, fname, size=(14, 9), lw=0.3):
    fig, ax = plt.subplots(figsize=size)
    ax.add_collection(LineCollection(segs, linewidths=lw, colors='k'))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal')
    ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
    ax.set_title(title, fontsize=11)
    fig.savefig(fname, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('  wrote', os.path.basename(fname))


# ---------------------------------------------------------------------------
def make_views(tag):
    cfg = MESHES[tag]
    odir = f'{OUT}/{tag}'
    os.makedirs(odir, exist_ok=True)
    print(f'== {tag}: reading {cfg["cgns"]}')
    mb = read_cgns(cfg['cgns'])
    blocks = leaf_blocks(mb)
    print('   blocks:', [n for n, _ in blocks][:12])
    wall = surface_polys(blocks, cfg['wall_names'])
    vols = volume_blocks(blocks)
    if wall is None or wall.GetNumberOfCells() == 0:
        wall = wall_from_volume(vols, cfg['span'])
    print(f'   wall cells: {wall.GetNumberOfCells() if wall else 0}, '
          f'volume blocks: {len(vols)}')

    y0, y1 = cfg['span']
    ymid = 0.5 * (y0 + y1)
    span_w = 0.5 * (y1 - y0) + 1.0
    yr = cfg['y_root_view']
    tipf = (0.32, XT - 0.35, 0.0)           # focal point for tip zooms
    D = 60.0

    S = [  # (name, campos, focal, viewup, scale, size)
        ('surf_upper_global', (0.32, ymid, D), (0.32, ymid, 0), (1, 0, 0),
         span_w / 4.0, (2560, 640)),
        ('surf_upper_tip', (tipf[0], tipf[1], D), tipf, (1, 0, 0),
         0.55, (1600, 1200)),
        ('surf_lower_global', (0.32, ymid, -D), (0.32, ymid, 0), (1, 0, 0),
         span_w / 4.0, (2560, 640)),
        ('surf_lower_tip', (tipf[0], tipf[1], -D), tipf, (1, 0, 0),
         0.55, (1600, 1200)),
        ('surf_iso_tip', (tipf[0] - 2.0, tipf[1] + 2.6, 1.8), tipf, (0, 0, 1),
         0.65, (1600, 1200)),
        ('surf_front_root', (-D, yr, 0), (0.32, yr, 0), (0, 0, 1),
         0.45, (1600, 1200)),
        ('surf_front_tip', (-D, tipf[1], 0), tipf, (0, 0, 1),
         0.45, (1600, 1200)),
        ('surf_back_root', (D, yr, 0), (0.32, yr, 0), (0, 0, 1),
         0.45, (1600, 1200)),
        ('surf_back_tip', (D, tipf[1], 0), tipf, (0, 0, 1),
         0.45, (1600, 1200)),
        ('surf_side_from_tip', (0.32, XT + D, 0), (0.32, XT - 0.2, 0), (0, 0, 1),
         0.55, (1600, 1200)),
    ]
    for name, campos, focal, up, scale, size in S:
        render_surface(wall, campos, focal, up, scale, f'{odir}/{name}.png', size,
                       edges='global' not in name)

    # ---- slices ----
    near = dict(xlim=(-0.35, 1.35), ylim=(-0.55, 0.55), labels=('x [m]', 'z [m]'))
    tipz = dict(xlim=(-0.05, 0.65), ylim=(-0.28, 0.28), labels=('x [m]', 'z [m]'))
    cuts = [
        ('slice_span_root', (0.32, max(y0 + 1e-3, 1e-3), 0), (0, 1, 0), (0, 2),
         near, 'spanwise cut near the root', 0.35),
        ('slice_span_mid', (0.32, 0.5 * XT, 0), (0, 1, 0), (0, 2),
         near, f'spanwise cut y = {0.5*XT:.2f} m (mid-span)', 0.35),
        ('slice_span_neartip', (0.32, XT - 0.12, 0), (0, 1, 0), (0, 2),
         tipz, f'spanwise cut y = {XT-0.12:.2f} m (very close to the tip)', 0.35),
        ('slice_span_on_tip', (0.32, XT + 1e-4, 0), (0, 1, 0), (0, 2),
         tipz, f'spanwise cut exactly on the tip (y = {XT} m + 1e-4)', 0.35),
        ('slice_farfield_global', (0.32, max(y0 + 1e-3, 1e-3), 0), (0, 1, 0), (0, 2),
         dict(xlim=(-100, 100), ylim=(-100, 100), labels=('x [m]', 'z [m]')),
         'far field (spanwise cut near root)', 0.25),
    ]
    for name, org, nrm, axes, w, title, lw in cuts:
        segs = cut_segments(vols, org, nrm, axes)
        plot_segments(segs, w['xlim'], w['ylim'], w['labels'],
                      f'{tag}: {title}', f'{odir}/{name}.png', lw=lw)

    # quarter-chord x-cut: full wing + tip zoom (y horizontal, z vertical)
    segs = cut_segments(vols, (XQC, 0, 0), (1, 0, 0), (1, 2))
    plot_segments(segs, (y0 - 1.0, y1 + 1.5), (-1.2, 1.2), ('y [m]', 'z [m]'),
                  f'{tag}: x-cut at quarter chord (x = {XQC:.3f} m), whole wing',
                  f'{odir}/slice_xcut_qc_full.png', size=(20, 4), lw=0.25)
    plot_segments(segs, (XT - 0.7, XT + 1.1), (-0.45, 0.45), ('y [m]', 'z [m]'),
                  f'{tag}: x-cut at quarter chord, wing-tip zoom',
                  f'{odir}/slice_xcut_qc_tip.png', lw=0.4)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for tag in (['ogrid', 'cavity'] if which == 'all' else [which]):
        make_views(tag)
    print('done ->', OUT)
