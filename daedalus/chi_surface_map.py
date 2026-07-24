"""Surface map of max chi = nuHat/nu taken over the near-wall band -- the 3D
analog of the 2D "max chi vs x" transition diagnostic.

Method (offline, no ParaView app): KD-tree of wall nodes; every volume node
within a chord-scaled wall band (BAND * local chord) scatter-maxes its chi
onto its nearest wall node (np.maximum.at). ~1.4 s at L0, minutes at L2.

Usage: python3 chi_surface_map.py <case_dir> [surface.pvtu name] [field]
Writes <case>/chi_surface_map.png and chi_surface.npz.
"""
import json
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from wing_geometry import chord, HALF_SPAN, XQC
import sectional_compare as SC

BAND = 0.05          # wall band = BAND * local chord
HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()


def tri_faces(grid):
    """Triangulated connectivity of a VTK surface grid (quads split)."""
    tris = []
    for c in range(grid.GetNumberOfCells()):
        ids = grid.GetCell(c).GetPointIds()
        n = ids.GetNumberOfIds()
        p = [ids.GetId(k) for k in range(n)]
        for k in range(1, n - 1):
            tris.append((p[0], p[k], p[k + 1]))
    return np.array(tris)


def max_chi_on_wall(case, surf_name, field):
    t0 = time.time()
    vol = load(f'{case}/volume.pvtu')
    pts = vtk_to_numpy(vol.GetPoints().GetData())
    pd = vol.GetPointData()
    names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
    fname = field or next(n for n in ('nuHat', 'solutionTurbulence', 'Mach')
                          if n in names)
    nuhat = vtk_to_numpy(pd.GetArray(fname)).astype(float)

    wgrid = load(f'{case}/{surf_name}')
    wall = vtk_to_numpy(wgrid.GetPoints().GetData())

    from scipy.spatial import cKDTree
    dmax_glob = BAND * chord(0.0)
    d, idx = cKDTree(wall).query(pts, distance_upper_bound=dmax_glob,
                                 workers=8)
    m = np.isfinite(d)
    eta_w = np.clip(np.abs(wall[:, 1]) / HALF_SPAN, 0, 1)
    m &= d < BAND * chord(eta_w[np.minimum(idx, len(wall) - 1)])
    smax = np.full(len(wall), np.nan)
    valid = np.zeros(len(wall), bool)
    np.maximum.at(valid, idx[m], True)
    tmp = np.full(len(wall), -np.inf)
    np.maximum.at(tmp, idx[m], nuhat[m])
    smax[valid] = tmp[valid]

    # nuHat (solver nondim: a_inf, L_grid) -> chi = nuHat/nu, nu* = muRef
    # (same convention as flow360/add_derived_to_slice.py in the 2D repro)
    j = json.load(open(f'{case}/Flow360.json'))
    nu_star = j['freestream']['muRef']
    chi = smax / nu_star
    print(f'{case}: field {fname}, {m.sum()} band pts -> {len(wall)} wall '
          f'nodes ({valid.mean() * 100:.1f}% covered), {time.time() - t0:.1f}s',
          flush=True)
    return wgrid, wall, chi


def planform_plot(case, wgrid, wall, chi, out):
    tris = tri_faces(wgrid)
    c_loc = chord(np.clip(np.abs(wall[:, 1]) / HALF_SPAN, 0, 1))
    xc = np.clip((wall[:, 0] - (XQC - 0.25 * c_loc)) / c_loc, 0.0, 1.0)
    upper = wall[:, 2] >= np.interp(xc, SC._CAM_X, SC._CAM_Z) * c_loc
    logchi = np.log10(np.clip(chi, 1e-8, None))
    fig, axs = plt.subplots(2, 1, figsize=(16, 6.2), sharex=True)
    for ax, side, name in ((axs[0], upper, 'upper'), (axs[1], ~upper, 'lower')):
        keep = side[tris].all(axis=1) & np.isfinite(logchi[tris]).all(axis=1)
        t = mtri.Triangulation(wall[:, 1], wall[:, 0], tris[keep])
        pc = ax.tripcolor(t, logchi, shading='gouraud', cmap='inferno',
                          vmin=-6, vmax=2)
        ax.tricontour(t, logchi, levels=[0.0], colors='cyan',
                      linewidths=1.0)
        ax.set_ylabel(f'{name}\nx [m]')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        fig.colorbar(pc, ax=ax, label=r'$\log_{10}\,\max_n \chi$', pad=0.01)
    axs[1].set_xlabel('y [m]')
    fig.suptitle(f'{os.path.basename(case)}: near-wall max '
                 r'$\chi=\tilde{\nu}/\nu$ (cyan: $\chi=1$)', fontsize=11)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out, flush=True)


def contour_plot(case, wgrid, wall, chi, out):
    """Contour LINES of log10 max chi on the top-view planform, one 16:9
    panel per side (span horizontal, chord stretched to fill)."""
    tris = tri_faces(wgrid)
    c_loc = chord(np.clip(np.abs(wall[:, 1]) / HALF_SPAN, 0, 1))
    xc = np.clip((wall[:, 0] - (XQC - 0.25 * c_loc)) / c_loc, 0.0, 1.0)
    upper = wall[:, 2] >= np.interp(xc, SC._CAM_X, SC._CAM_Z) * c_loc
    logchi = np.log10(np.clip(chi, 1e-8, None))
    levels = np.arange(-5, 3)                       # chi = 1e-5 .. 1e2
    fmt = {lv: (rf'$10^{{{lv:d}}}$' if lv else r'$\chi=1$') for lv in levels}

    ee = np.linspace(0, 1, 200)
    x_le = XQC - 0.25 * chord(ee)
    x_te = XQC + 0.75 * chord(ee)

    fig, axs = plt.subplots(2, 1, figsize=(16, 18))   # each panel ~16:9
    for ax, side, name in ((axs[0], upper, 'upper'), (axs[1], ~upper, 'lower')):
        keep = side[tris].all(axis=1) & np.isfinite(logchi[tris]).all(axis=1)
        t = mtri.Triangulation(wall[:, 1], wall[:, 0], tris[keep])
        cs = ax.tricontour(t, logchi, levels=levels, colors='k',
                           linewidths=0.7)
        ax.tricontour(t, logchi, levels=[0.0], colors='k', linewidths=2.0)
        ax.clabel(cs, levels, fmt=fmt, fontsize=8, inline=True)
        ax.plot(ee * HALF_SPAN, x_le, 'k-', lw=1.2)
        ax.plot(ee * HALF_SPAN, x_te, 'k-', lw=1.2)
        ax.plot([0, 0], [x_le[0], x_te[0]], 'k-', lw=1.2)
        ax.set_ylabel(f'{name} surface\nx [m]')
        ax.set_xlim(-0.1, HALF_SPAN + 0.1)
        ax.invert_yaxis()                            # LE at top
    axs[1].set_xlabel('y [m]')
    fig.suptitle(f'{os.path.basename(case)}: contours of near-wall max '
                 r'$\chi=\tilde{\nu}/\nu$ (bold: $\chi=1$)', fontsize=12)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    case = sys.argv[1].rstrip('/')
    surf = sys.argv[2] if len(sys.argv) > 2 else (
        'surface_fluid_wing.pvtu' if 'ogrid' in case
        else 'surface_farfield_body.pvtu')
    field = sys.argv[3] if len(sys.argv) > 3 else None
    wgrid, wall, chi = max_chi_on_wall(case, surf, field)
    np.savez(f'{case}/chi_surface.npz', wall=wall, chi=chi)
    planform_plot(case, wgrid, wall, chi, f'{case}/chi_surface_map.png')
    contour_plot(case, wgrid, wall, chi, f'{case}/chi_contours.png')
