"""Generate (x, wall-distance) contour plots for the NLF α=9 case,
analogous to the flat-plate top rows but on an airfoil.

Algorithm (same as wallnormal_max_metrics in regen_nlf_v2.py):
  1. Walk the surface contour by connectivity
  2. Order clockwise; split at LE and TE into upper/lower segments
  3. For each surface point, shoot a wall-normal probe line outward
  4. Sample velocity, nuhat, omega along the line
  5. Plot the resulting (x, wall-distance) field as contours

Layout: for the requested case, 4 panels:
  upper Mach | upper log10(χ)
  lower Mach | lower log10(χ)
"""
import os, sys, numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, '/home/qiqi/flexcompute/aft-sa/paper')
from regen_nlf_v2 import walk_contour_xz, slice_y_plane, load_slice_derived

B = "/home/qiqi/flexcompute/aft-sa/flow360"
NU_UNIT = 1.0 / 4e6
MACH = 0.1

def probe_2d_field(cd, side='upper', L=0.015, n_probe=200, x_clip=(0.0, 0.5)):
    """Probe the wall-normal field at every surface point on `side`.
    Returns (x_surf, d_grid, M_field, chi_field, omega_field) where
        x_surf: 1D array of surface x-coordinates (in walk order)
        d_grid: 1D array of wall distances (probe sample distances)
        M_field, chi_field, omega_field: 2D arrays of shape (n_probe, M_surf)
    """
    Xm, Zm, up_idx, lo_idx = walk_contour_xz(cd)
    idx = up_idx if side == 'upper' else lo_idx
    xs = Xm[idx]; zs = Zm[idx]
    # Apply x clip (drop TE-base flat region for tighter plot)
    keep = (xs >= x_clip[0]) & (xs <= x_clip[1])
    xs = xs[keep]; zs = zs[keep]
    # Tangent + outward normal in walk order
    tx_raw = np.gradient(xs); tz_raw = np.gradient(zs)
    s = np.sqrt(tx_raw**2 + tz_raw**2) + 1e-30
    tx = tx_raw / s; tz = tz_raw / s
    nx = tz; nz = -tx
    if side == 'upper' and np.mean(nz) < 0: nx, nz = -nx, -nz
    elif side == 'lower' and np.mean(nz) > 0: nx, nz = -nx, -nz
    # Sort by x (so contour-plotting works in increasing x)
    order = np.argsort(xs)
    xs = xs[order]; zs = zs[order]
    nx = nx[order]; nz = nz[order]
    M = len(xs)

    slice_g = load_slice_derived(cd)
    y0 = slice_y_plane(cd)
    # Use a stretched probe spacing so near-wall is well resolved
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
    probe.SetInputData(poly); probe.SetSourceData(slice_g); probe.Update()
    out = probe.GetOutput(); pdd = out.GetPointData()
    valid = vtk_to_numpy(probe.GetValidPoints())
    mask = np.zeros(M * n_probe, bool); mask[valid] = True
    vel = vtk_to_numpy(pdd.GetArray('velocity'))
    nuhat = vtk_to_numpy(pdd.GetArray('nuHat'))
    omega = vtk_to_numpy(pdd.GetArray('vorticityMagnitude'))
    vel = np.where(mask[:,None], vel, np.nan)
    nuhat = np.where(mask, nuhat, np.nan)
    omega = np.where(mask, omega, np.nan)
    # Mach number (local): velocity stored as u/c_inf, so |v_stored| = M_local
    M_loc = np.sqrt(vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2)
    chi = nuhat / NU_UNIT
    # Reshape (n_probe, M)
    M_field = M_loc.reshape(n_probe, M)
    chi_field = chi.reshape(n_probe, M)
    omega_field = omega.reshape(n_probe, M)
    return xs, dists, M_field, chi_field, omega_field

def plot_case(cd, case_name, x_clip=(0.0, 0.5), L=0.005, y_max_yp=5000,
              out_path='/tmp/airfoil_bl_contours.png'):
    n_probe = 250
    sides = ['upper', 'lower']
    data = {}
    for side in sides:
        data[side] = probe_2d_field(cd, side=side, L=L, n_probe=n_probe, x_clip=x_clip)
    fig = plt.figure(figsize=(13, 8))
    gs = gridspec.GridSpec(2, 2, hspace=0.30, wspace=0.10,
                           top=0.92, bottom=0.08, left=0.08, right=0.92)
    # Wall-distance axis = wall_distance × Re_unit. For Re_unit=4e6, L_probe=0.005,
    # full y range = 20000.  Restrict to y_max_yp=5000 (≈1.25e-3 chord) to focus on
    # the laminar BL + the kernel-active band.
    for r, side in enumerate(sides):
        xs, dists, M_f, chi_f, _ = data[side]
        dists_yp = dists / NU_UNIT
        XX, DD = np.meshgrid(xs, dists_yp)
        # Mach contours
        ax = fig.add_subplot(gs[r, 0])
        levels_M = np.arange(0.02, 0.32, 0.02)  # M from 0.02 to 0.30 step 0.02
        cs = ax.contour(XX, DD, M_f, levels_M, colors='k', linewidths=0.5)
        ax.clabel(cs, fontsize=6, inline=True, fmt='%.2f', inline_spacing=2)
        ax.set_title(f'Mach number, {side} surface', y=1.01, fontsize=10)
        ax.set_ylim(0, y_max_yp); ax.set_ylabel(r'$d \cdot Re_\infty$')
        if r < 1: ax.set_xticklabels([])
        else: ax.set_xlabel('x/c')
        # log10(chi) contours — uniform 0.5 spacing
        ax = fig.add_subplot(gs[r, 1])
        levels_chi = np.arange(-7.0, 2.01, 0.5)
        with np.errstate(divide='ignore', invalid='ignore'):
            log_chi = np.log10(np.maximum(chi_f, 1e-10))
        cs = ax.contour(XX, DD, log_chi, levels_chi, colors='k', linewidths=0.5)
        ax.clabel(cs, fontsize=6, inline=True, fmt='%.1f', inline_spacing=2)
        ax.set_title(f'$\\log_{{10}} \\chi = \\log_{{10}}\\tilde\\nu/\\nu$, {side} surface',
                     y=1.01, fontsize=10)
        ax.set_ylim(0, y_max_yp); ax.set_yticklabels([])
        if r < 1: ax.set_xticklabels([])
        else: ax.set_xlabel('x/c')
    fig.suptitle(f'NLF α=9, {case_name} — wall-normal BL fields', fontsize=11)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'wrote {out_path}')
    return data

if __name__ == '__main__':
    cases = {
        'cavL1': f"{B}/cavL1prop_nlf0416_Re4M_a9",
        'cavL2': f"{B}/cavL2prop_nlf0416_Re4M_a9",
        'strL1': f"{B}/strL1prop_nlf0416_Re4M_a9",
        'strL2': f"{B}/strL2prop_nlf0416_Re4M_a9",
    }
    for name, cd in cases.items():
        if not os.path.exists(f"{cd}/slice_with_derived.pvtu"):
            print(f"SKIP {name}"); continue
        plot_case(cd, name, x_clip=(0.0, 0.5), L=0.005, y_max_yp=5000,
                  out_path=f'/tmp/airfoil_bl_contours_{name}.png')
