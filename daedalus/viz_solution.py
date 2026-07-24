"""Solution visualizations on the SAME diagnostic views as viz_meshes.py:
surface Cf_vec as LIC (vtkSurfaceLICMapper, colored by |Cf|) on the 10 surface
views; velocity LIC (numpy line-integral convolution on resampled cut planes,
colored by |u|/U_inf) on the cut views.

Usage: xvfb-run -a python3 viz_solution.py <case_dir> [outdir_name]
"""
import os
import sys
import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import viz_meshes as V

U_INF_MACH = 0.1


def read_pvtu(path):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()


def find_surface(case):
    for fn in os.listdir(case):
        if fn.startswith('surface_') and fn.endswith('.pvtu'):
            return os.path.join(case, fn)
    raise FileNotFoundError(f'no surface pvtu in {case}')


# ---------------------------------------------------------------- surface LIC
def surface_lic_views(case, odir):
    surf = read_pvtu(find_surface(case))
    gf = vtk.vtkGeometryFilter()
    gf.SetInputData(surf)
    gf.Update()
    pd = gf.GetOutput()
    pdd = pd.GetPointData()
    if pdd.GetArray('CfVec') is None:
        print('  no CfVec on surface; skipping surface LIC')
        return
    pdd.SetActiveVectors('CfVec')
    cf = vtk_to_numpy(pdd.GetArray('CfVec'))
    cfm = np.linalg.norm(cf, axis=1)
    import vtkmodules.util.numpy_support as ns
    arr = ns.numpy_to_vtk(cfm)
    arr.SetName('Cfmag')
    pdd.AddArray(arr)
    pdd.SetActiveScalars('Cfmag')
    vmax = 0.006          # FIXED absolute scale: images comparable across cases

    try:
        from vtkmodules.vtkRenderingLIC import vtkSurfaceLICMapper
    except ImportError:
        vtkSurfaceLICMapper = vtk.vtkSurfaceLICMapper
    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.667, 0.0)          # blue -> red
    lut.SetRange(0.0, vmax)
    lut.Build()

    tag = os.path.basename(case).replace('case_', '')
    mesh = 'ogrid' if 'ogrid' in case else 'cavity'
    cfgspan = V.MESHES[mesh]['span']
    y0, y1 = cfgspan
    ymid = 0.5 * (y0 + y1)
    span_w = 0.5 * (y1 - y0) + 1.0
    yr = V.MESHES[mesh]['y_root_view']
    tipf = (0.32, V.XT - 0.35, 0.0)
    D = 60.0
    views = [
        ('surf_upper_global', (0.32, ymid, D), (0.32, ymid, 0), (1, 0, 0),
         span_w / 4.0, (4096, 1024)),
        ('surf_upper_tip', (tipf[0], tipf[1], D), tipf, (1, 0, 0), 0.55, (1600, 1200)),
        ('surf_lower_global', (0.32, ymid, -D), (0.32, ymid, 0), (1, 0, 0),
         span_w / 4.0, (4096, 1024)),
        ('surf_lower_tip', (tipf[0], tipf[1], -D), tipf, (1, 0, 0), 0.55, (1600, 1200)),
        ('surf_iso_tip', (tipf[0] - 2.0, tipf[1] + 2.6, 1.8), tipf, (0, 0, 1),
         0.65, (1600, 1200)),
        ('surf_front_root', (-D, yr, 0), (0.32, yr, 0), (0, 0, 1), 0.45, (1600, 1200)),
        ('surf_front_tip', (-D, tipf[1], 0), tipf, (0, 0, 1), 0.45, (1600, 1200)),
        ('surf_back_root', (D, yr, 0), (0.32, yr, 0), (0, 0, 1), 0.45, (1600, 1200)),
        ('surf_back_tip', (D, tipf[1], 0), tipf, (0, 0, 1), 0.45, (1600, 1200)),
        ('surf_side_from_tip', (0.32, V.XT + D, 0), (0.32, V.XT - 0.2, 0), (0, 0, 1),
         0.55, (1600, 1200)),
    ]
    for name, campos, focal, up, scale, size in views:
        m = vtkSurfaceLICMapper()
        m.SetInputData(pd)
        m.SetInputArrayToProcess(0, 0, 0, 0, 'CfVec')
        m.SetScalarVisibility(1)
        m.SetLookupTable(lut)
        m.SetScalarRange(0.0, vmax)
        lic = m.GetLICInterface() if hasattr(m, 'GetLICInterface') else m
        try:
            lic.SetColorMode(1)              # multiply LIC into scalar colors
            lic.SetLICIntensity(0.6)
            lic.SetNumberOfSteps(60)
            lic.SetStepSize(0.3)
            lic.SetEnhanceContrast(1)
        except AttributeError:
            pass
        a = vtk.vtkActor(); a.SetMapper(m)
        a.GetProperty().SetAmbient(0.9); a.GetProperty().SetDiffuse(0.1)
        ren = vtk.vtkRenderer(); ren.AddActor(a); ren.SetBackground(1, 1, 1)
        cam = ren.GetActiveCamera()
        cam.ParallelProjectionOn()
        cam.SetPosition(*campos); cam.SetFocalPoint(*focal); cam.SetViewUp(*up)
        cam.SetParallelScale(scale)
        ren.ResetCameraClippingRange()
        rw = vtk.vtkRenderWindow(); rw.SetOffScreenRendering(1)
        rw.AddRenderer(ren); rw.SetSize(*size); rw.Render()
        w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(rw); w2i.Update()
        wr = vtk.vtkPNGWriter(); wr.SetFileName(f'{odir}/{name}.png')
        wr.SetInputConnection(w2i.GetOutputPort()); wr.Write()
        print('  wrote', name, flush=True)


# ------------------------------------------------------------------- cut LIC
def numpy_lic(u, v, nsteps=25, step=0.8, seed=0):
    """Basic LIC: advect through white noise both ways along (u, v) [ny, nx]."""
    ny, nx = u.shape
    rng = np.random.default_rng(seed)
    noise = rng.random((ny, nx))
    mag = np.hypot(u, v) + 1e-30
    un, vn = u / mag, v / mag
    acc = noise.copy()
    cnt = np.ones_like(noise)

    def sample(f, X, Y):
        Xc = np.clip(X, 0, nx - 1.001); Yc = np.clip(Y, 0, ny - 1.001)
        x0 = Xc.astype(int); y0 = Yc.astype(int)
        fx = Xc - x0; fy = Yc - y0
        return (f[y0, x0] * (1 - fx) * (1 - fy) + f[y0, x0 + 1] * fx * (1 - fy)
                + f[y0 + 1, x0] * (1 - fx) * fy + f[y0 + 1, x0 + 1] * fx * fy)

    for sgn in (1.0, -1.0):
        X, Y = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
        for _ in range(nsteps):
            du = sample(un, X, Y); dv = sample(vn, X, Y)
            X = X + sgn * step * du; Y = Y + sgn * step * dv
            acc += sample(noise, X, Y)
            cnt += 1.0
    return acc / cnt


def cut_lic_views(case, odir):
    vol = read_pvtu(os.path.join(case, 'volume.pvtu'))
    pdd = vol.GetPointData()
    names = [pdd.GetArrayName(i) for i in range(pdd.GetNumberOfArrays())]
    vname = next((n for n in ('velocity', 'primitiveVars') if n in names), None)
    if vname is None:
        print('  no velocity array; arrays:', names)
        return
    tag = os.path.basename(case)
    mesh = 'ogrid' if 'ogrid' in case else 'cavity'
    y0s = V.MESHES[mesh]['span'][0]

    cuts = [
        ('slice_span_root', (0.32, max(y0s + 1e-3, 1e-3), 0), (0, 1, 0), (0, 2),
         (-0.35, 1.35), (-0.55, 0.55), ('x [m]', 'z [m]')),
        ('slice_span_mid', (0.32, 0.5 * V.XT, 0), (0, 1, 0), (0, 2),
         (-0.35, 1.35), (-0.55, 0.55), ('x [m]', 'z [m]')),
        ('slice_span_neartip', (0.32, V.XT - 0.12, 0), (0, 1, 0), (0, 2),
         (-0.05, 0.65), (-0.28, 0.28), ('x [m]', 'z [m]')),
        ('slice_span_on_tip', (0.32, V.XT + 1e-4, 0), (0, 1, 0), (0, 2),
         (-0.05, 0.65), (-0.28, 0.28), ('x [m]', 'z [m]')),
        ('slice_xcut_qc_tip', (V.XQC, 0, 0), (1, 0, 0), (1, 2),
         (V.XT - 0.7, V.XT + 1.1), (-0.45, 0.45), ('y [m]', 'z [m]')),
        ('slice_farfield_global', (0.32, max(y0s + 1e-3, 1e-3), 0), (0, 1, 0), (0, 2),
         (-40, 40), (-40, 40), ('x [m]', 'z [m]')),
    ]
    for name, org, nrm, axes, xlim, zlim, labels in cuts:
        NX, NY = 1100, 850
        xs = np.linspace(*xlim, NX); zs = np.linspace(*zlim, NY)
        pts = np.zeros((NX * NY, 3))
        Xg, Zg = np.meshgrid(xs, zs)
        ax0, ax1 = axes
        pts[:, ax0] = Xg.ravel(); pts[:, ax1] = Zg.ravel()
        fixed_axis = 3 - ax0 - ax1
        pts[:, fixed_axis] = org[fixed_axis]
        vp = vtk.vtkPoints()
        from vtkmodules.util.numpy_support import numpy_to_vtk
        vp.SetData(numpy_to_vtk(pts, deep=True))
        poly = vtk.vtkPolyData(); poly.SetPoints(vp)
        pr = vtk.vtkProbeFilter(); pr.SetInputData(poly); pr.SetSourceData(vol)
        pr.Update()
        out = pr.GetOutput().GetPointData()
        arr = vtk_to_numpy(out.GetArray(vname))
        if vname == 'primitiveVars':
            vel = arr[:, 1:4]
        else:
            vel = arr
        valid = np.zeros(NX * NY, bool)
        valid[vtk_to_numpy(pr.GetValidPoints())] = True
        Uc = np.where(valid, vel[:, ax0], np.nan).reshape(NY, NX)
        Wc = np.where(valid, vel[:, ax1], np.nan).reshape(NY, NX)
        mag = np.hypot(np.where(valid, vel[:, 0], np.nan),
                       np.where(valid, np.linalg.norm(vel[:, 1:], axis=1)
                                if vel.shape[1] > 2 else vel[:, 1], np.nan))
        mag = np.linalg.norm(np.where(valid[:, None], vel, np.nan),
                             axis=1).reshape(NY, NX) / U_INF_MACH
        lic = numpy_lic(np.nan_to_num(Uc), np.nan_to_num(Wc))
        lic = (lic - np.nanmin(lic)) / (np.nanmax(lic) - np.nanmin(lic) + 1e-30)
        cmap = plt.get_cmap('viridis')
        col = cmap(np.clip(mag / 1.4, 0, 1))[:, :, :3]
        shade = (0.55 + 0.9 * lic)[:, :, None]
        img = np.clip(col * shade, 0, 1)
        img[np.isnan(mag)] = (0.85, 0.85, 0.85)
        fig, ax = plt.subplots(figsize=(12, 12 * (zlim[1] - zlim[0]) / (xlim[1] - xlim[0]) + 1))
        ax.imshow(img, origin='lower', extent=(*xlim, *zlim), aspect='equal',
                  interpolation='bilinear')
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
        ax.set_title(f'{tag}: velocity LIC, |u|/U_inf color — {name}', fontsize=11)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1.4))
        fig.colorbar(sm, ax=ax, fraction=0.03, label='|u| / U_inf')
        fig.savefig(f'{odir}/{name}.png', dpi=130, bbox_inches='tight')
        plt.close(fig)
        print('  wrote', name, flush=True)


if __name__ == '__main__':
    case = sys.argv[1].rstrip('/')
    oname = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(case).replace('case_', 'sol_')
    odir = f'{V.OUT}/{oname}'
    os.makedirs(odir, exist_ok=True)
    print(f'== {case} -> {odir}')
    surface_lic_views(case, odir)
    if 'surfonly' not in sys.argv:
        cut_lic_views(case, odir)
    print('done')
