"""Side-by-side (x, wall-distance) χ contours on the lower surface for
all four grids at α=9 — to see clearly that the L1/L2 cavity AND structured
grids grow χ in the BL while L1 cavity stays laminar.
Also overlay the OFFLINE-computed amp_rate (where the kernel says
amplification should happen) to compare against actual χ growth.
"""
import os, sys, numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, '/home/qiqi/flexcompute/aft-sa/paper')
from regen_nlf_v2 import walk_contour_xz, slice_y_plane, load_slice_derived

B = "/home/qiqi/flexcompute/aft-sa/flow360"
NU = 2.5e-8
MACH = 0.1

def probe_2d_field(cd, side='lower', L=0.015, n_probe=250, x_clip=(0.0, 0.5)):
    Xm, Zm, up_idx, lo_idx = walk_contour_xz(cd)
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
    slice_g = load_slice_derived(cd)
    y0 = slice_y_plane(cd)
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
    nuhat = np.where(mask, vtk_to_numpy(pdd.GetArray('nuHat')), np.nan)
    ReO = np.where(mask, vtk_to_numpy(pdd.GetArray('Re_Omega')), np.nan)
    amp = np.where(mask, vtk_to_numpy(pdd.GetArray('amp_rate')), np.nan)
    lp = np.where(mask, vtk_to_numpy(pdd.GetArray('lambda_p')), np.nan)
    chi = nuhat / NU
    return xs, dists, chi.reshape(n_probe, M), ReO.reshape(n_probe, M), amp.reshape(n_probe, M), lp.reshape(n_probe, M)

if __name__ == '__main__':
    cases = [('cavL1', f"{B}/cavL1prop_nlf0416_Re4M_a9"),
             ('cavL2', f"{B}/cavL2prop_nlf0416_Re4M_a9"),
             ('strL1', f"{B}/strL1prop_nlf0416_Re4M_a9"),
             ('strL2', f"{B}/strL2prop_nlf0416_Re4M_a9")]
    L = 0.015
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(4, 3, hspace=0.30, wspace=0.10,
                           top=0.95, bottom=0.05, left=0.06, right=0.97)
    y_max = L / NU  # in wall units (chord units · Re_unit). NU=2.5e-8 → y_max=600000. Too big.
    # use chord units instead: y axis in chord-fraction × 1e3
    for r, (name, cd) in enumerate(cases):
        xs, dists, chi_f, ReO_f, amp_f, lp_f = probe_2d_field(cd, side='lower',
                                                              L=L, n_probe=250,
                                                              x_clip=(0.0, 0.5))
        d_yp = dists * 1e3  # in 0.001 chord units
        XX, DD = np.meshgrid(xs, d_yp)
        # log10(χ) contours
        ax = fig.add_subplot(gs[r, 0])
        levels = np.arange(-6, 2.01, 0.5)
        with np.errstate(invalid='ignore'):
            log_chi = np.log10(np.maximum(chi_f, 1e-10))
        cs = ax.contour(XX, DD, log_chi, levels, colors='k', linewidths=0.5)
        ax.clabel(cs, fontsize=6, inline=True, fmt='%.1f')
        ax.set_title(f'{name}: $\\log_{{10}}\\chi$', y=1.01, fontsize=10)
        ax.set_ylim(0, L*1e3); ax.set_ylabel('d (10⁻³ chord)')
        if r < 3: ax.set_xticklabels([])
        else: ax.set_xlabel('x/c')
        # log10(Re_Ω) contours
        ax = fig.add_subplot(gs[r, 1])
        levels = np.arange(0, 5.01, 0.5)
        with np.errstate(invalid='ignore'):
            log_R = np.log10(np.maximum(ReO_f, 1e-2))
        cs = ax.contour(XX, DD, log_R, levels, colors='b', linewidths=0.5)
        ax.clabel(cs, fontsize=6, inline=True, fmt='%.1f')
        ax.set_title(f'{name}: $\\log_{{10}} Re_\\Omega$', y=1.01, fontsize=10)
        ax.set_ylim(0, L*1e3); ax.set_yticklabels([])
        if r < 3: ax.set_xticklabels([])
        else: ax.set_xlabel('x/c')
        # amp_rate contours (offline-computed kernel rate)
        ax = fig.add_subplot(gs[r, 2])
        # Linear levels, 1e-3 spacing
        levels = [1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
        cs = ax.contour(XX, DD, amp_f, levels, colors='r', linewidths=0.5)
        ax.clabel(cs, fontsize=6, inline=True, fmt='%g')
        ax.set_title(f'{name}: offline amp_rate (K=10, $\\sigma_{{FPG}}$)', y=1.01, fontsize=10)
        ax.set_ylim(0, L*1e3); ax.set_yticklabels([])
        if r < 3: ax.set_xticklabels([])
        else: ax.set_xlabel('x/c')
    fig.suptitle('NLF α=9 lower surface: χ growth vs offline-computed amp rate',
                 fontsize=11)
    plt.savefig('/tmp/diag_bl_compare_lower.png', dpi=130, bbox_inches='tight')
    print('wrote /tmp/diag_bl_compare_lower.png')
