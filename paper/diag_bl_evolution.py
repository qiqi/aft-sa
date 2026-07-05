"""Wall-normal BL profiles of cavL2_a9 at pseudo-steps 2000, 4000, 6000,
8000, 10000, 12000. Shows χ field evolution to see if it's growing,
oscillating, or decaying.

Layout: 6 rows (one per stage) × 4 cols:
  upper Mach | upper log10χ | lower Mach | lower log10χ
"""
import os, sys, numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, '/home/qiqi/flexcompute/aft-sa/paper')
from regen_nlf_v2 import walk_contour_xz

NU = 2.5e-8
MACH = 0.1
SD = "/home/qiqi/flexcompute/aft-sa/flow360/cavL2prop_nlf0416_Re4M_a9/snapshots"
STAGES = [2000, 4000, 6000, 8000, 10000, 12000]

def probe_2d_field(cd_volume_pvtu, surface_pvtu, side, L=0.005, n_probe=200, x_clip=(0.0, 0.5)):
    """Probe wall-normal profile from a SNAPSHOT volume.pvtu, using surface
    contour from the case directory's surface pvtu (geometry is the same)."""
    # Read surface (for contour walk)
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(surface_pvtu); r.Update()
    sg = r.GetOutput()
    Xm = vtk_to_numpy(sg.GetPoints().GetData())
    # We need walk_contour_xz which takes a directory. Use a hack: copy surface to a temp.
    # Or replicate inline:
    # ... actually use the original case dir for the walk:
    case_dir = "/home/qiqi/flexcompute/aft-sa/flow360/cavL2prop_nlf0416_Re4M_a9"
    Xm, Zm, up_idx, lo_idx = walk_contour_xz(case_dir)
    idx = up_idx if side == 'upper' else lo_idx
    xs = Xm[idx]; zs = Zm[idx]
    keep = (xs >= x_clip[0]) & (xs <= x_clip[1])
    xs = xs[keep]; zs = zs[keep]
    tx_raw = np.gradient(xs); tz_raw = np.gradient(zs)
    s = np.sqrt(tx_raw**2 + tz_raw**2) + 1e-30
    tx = tx_raw / s; tz = tz_raw / s
    nx = tz; nz = -tx
    if side == 'upper' and np.mean(nz) < 0: nx, nz = -nx, -nz
    elif side == 'lower' and np.mean(nz) > 0: nx, nz = -nx, -nz
    order = np.argsort(xs)
    xs = xs[order]; zs = zs[order]; nx = nx[order]; nz = nz[order]
    M = len(xs)

    # Read volume snapshot
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(cd_volume_pvtu); r.Update()
    vol = r.GetOutput()
    pts_v = vtk_to_numpy(vol.GetPoints().GetData())
    y0 = float(np.median(pts_v[:,1]))

    eta = np.linspace(0.0, 1.0, n_probe)
    dists = (eta**1.5) * L + 1e-7
    pts_arr = np.empty((M * n_probe, 3))
    for j in range(n_probe):
        d = dists[j]
        pts_arr[j*M:(j+1)*M, 0] = xs + d*nx
        pts_arr[j*M:(j+1)*M, 1] = y0
        pts_arr[j*M:(j+1)*M, 2] = zs + d*nz
    vpts = vtk.vtkPoints()
    vpts.SetData(numpy_to_vtk(pts_arr, deep=True))
    poly = vtk.vtkPolyData(); poly.SetPoints(vpts)
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(poly); probe.SetSourceData(vol); probe.Update()
    out = probe.GetOutput(); pdd = out.GetPointData()
    valid = vtk_to_numpy(probe.GetValidPoints())
    mask = np.zeros(M * n_probe, bool); mask[valid] = True
    vel = vtk_to_numpy(pdd.GetArray('velocity'))
    nuhat = vtk_to_numpy(pdd.GetArray('nuHat'))
    vel = np.where(mask[:,None], vel, np.nan)
    nuhat = np.where(mask, nuhat, np.nan)
    M_local = np.sqrt(vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2)
    chi = nuhat / NU
    return xs, dists, M_local.reshape(n_probe, M), chi.reshape(n_probe, M)


def main():
    n = len(STAGES)
    fig = plt.figure(figsize=(20, 3.0*n))
    gs = gridspec.GridSpec(n, 4, hspace=0.35, wspace=0.10,
                           top=0.97, bottom=0.04, left=0.04, right=0.99)
    y_max_yp = 5000  # wall-distance × Re_inf
    surf_pvtu = f"{SD}/surface_fluid_nlf0416_s{STAGES[-1]}.pvtu"
    for r_idx, step in enumerate(STAGES):
        vol_pvtu = f"{SD}/volume_s{step}.pvtu"
        # Upper
        xs_u, d_u, M_u, chi_u = probe_2d_field(vol_pvtu, surf_pvtu, 'upper', L=0.005,
                                                n_probe=200, x_clip=(0.0, 0.5))
        d_yp_u = d_u / NU
        XX, DD = np.meshgrid(xs_u, d_yp_u)
        # Upper Mach
        ax = fig.add_subplot(gs[r_idx, 0])
        levels_M = np.arange(0.02, 0.36, 0.02)
        cs = ax.contour(XX, DD, M_u, levels_M, colors='k', linewidths=0.5)
        ax.clabel(cs, fontsize=5.5, inline=True, fmt='%.2f', inline_spacing=2)
        ax.set_ylim(0, y_max_yp); ax.set_xlim(0, 0.5)
        if r_idx == 0: ax.set_title(f'Upper Mach', fontsize=9)
        ax.set_ylabel(f'step {step}\n$d \\cdot Re_\\infty$', fontsize=9)
        if r_idx < n-1: ax.set_xticklabels([])
        else: ax.set_xlabel('x/c')

        # Upper χ
        ax = fig.add_subplot(gs[r_idx, 1])
        levels_chi = np.arange(-7.0, 2.01, 0.5)
        with np.errstate(invalid='ignore'):
            log_chi = np.log10(np.maximum(chi_u, 1e-10))
        cs = ax.contour(XX, DD, log_chi, levels_chi, colors='k', linewidths=0.5)
        ax.clabel(cs, fontsize=5.5, inline=True, fmt='%.1f', inline_spacing=2)
        ax.set_ylim(0, y_max_yp); ax.set_xlim(0, 0.5); ax.set_yticklabels([])
        if r_idx == 0: ax.set_title('Upper $\\log_{10}\\chi$', fontsize=9)
        if r_idx < n-1: ax.set_xticklabels([])
        else: ax.set_xlabel('x/c')

        # Lower
        xs_l, d_l, M_l, chi_l = probe_2d_field(vol_pvtu, surf_pvtu, 'lower', L=0.005,
                                                n_probe=200, x_clip=(0.0, 0.5))
        d_yp_l = d_l / NU
        XX, DD = np.meshgrid(xs_l, d_yp_l)
        # Lower Mach
        ax = fig.add_subplot(gs[r_idx, 2])
        cs = ax.contour(XX, DD, M_l, levels_M, colors='k', linewidths=0.5)
        ax.clabel(cs, fontsize=5.5, inline=True, fmt='%.2f', inline_spacing=2)
        ax.set_ylim(0, y_max_yp); ax.set_xlim(0, 0.5); ax.set_yticklabels([])
        if r_idx == 0: ax.set_title('Lower Mach', fontsize=9)
        if r_idx < n-1: ax.set_xticklabels([])
        else: ax.set_xlabel('x/c')

        # Lower χ
        ax = fig.add_subplot(gs[r_idx, 3])
        with np.errstate(invalid='ignore'):
            log_chi = np.log10(np.maximum(chi_l, 1e-10))
        cs = ax.contour(XX, DD, log_chi, levels_chi, colors='k', linewidths=0.5)
        ax.clabel(cs, fontsize=5.5, inline=True, fmt='%.1f', inline_spacing=2)
        ax.set_ylim(0, y_max_yp); ax.set_xlim(0, 0.5); ax.set_yticklabels([])
        if r_idx == 0: ax.set_title('Lower $\\log_{10}\\chi$', fontsize=9)
        if r_idx < n-1: ax.set_xticklabels([])
        else: ax.set_xlabel('x/c')

    fig.suptitle(f'NLF α=9 cavL2 — BL evolution across pseudo-steps 2000…12000', fontsize=11)
    plt.savefig('/tmp/cavL2_a9_BL_evolution.png', dpi=110, bbox_inches='tight')
    print('wrote /tmp/cavL2_a9_BL_evolution.png')


if __name__ == '__main__':
    main()
