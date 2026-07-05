"""Focus: lower-side χ evolution at pseudo-steps 2000, 4000, 6000, 8000, 10000, 12000.
3 cols (steps grouped 2/2/2), with Mach and χ stacked per stage."""
import os, sys, numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/qiqi/flexcompute/aft-sa/paper')
from regen_nlf_v2 import walk_contour_xz

NU = 2.5e-8
SD = "/home/qiqi/flexcompute/aft-sa/flow360/cavL2prop_nlf0416_Re4M_a9/snapshots"
STAGES = [2000, 4000, 6000, 8000, 10000, 12000]

def probe(vol_pvtu, side='lower', L=0.005, n_probe=200, x_clip=(0.0, 0.5)):
    case_dir = "/home/qiqi/flexcompute/aft-sa/flow360/cavL2prop_nlf0416_Re4M_a9"
    Xm, Zm, up_idx, lo_idx = walk_contour_xz(case_dir)
    idx = lo_idx if side == 'lower' else up_idx
    xs = Xm[idx]; zs = Zm[idx]
    keep = (xs >= x_clip[0]) & (xs <= x_clip[1])
    xs = xs[keep]; zs = zs[keep]
    tx_raw = np.gradient(xs); tz_raw = np.gradient(zs)
    s = np.sqrt(tx_raw**2 + tz_raw**2) + 1e-30
    tx = tx_raw / s; tz = tz_raw / s
    nx = tz; nz = -tx
    if side == 'lower' and np.mean(nz) > 0: nx, nz = -nx, -nz
    elif side == 'upper' and np.mean(nz) < 0: nx, nz = -nx, -nz
    order = np.argsort(xs); xs = xs[order]; zs = zs[order]; nx = nx[order]; nz = nz[order]
    M = len(xs)
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(vol_pvtu); r.Update()
    vol = r.GetOutput()
    pts_v = vtk_to_numpy(vol.GetPoints().GetData())
    y0 = float(np.median(pts_v[:,1]))
    eta = np.linspace(0.0, 1.0, n_probe)
    dists = (eta**1.5) * L + 1e-7
    pts_arr = np.empty((M * n_probe, 3))
    for j in range(n_probe):
        d = dists[j]
        pts_arr[j*M:(j+1)*M, 0] = xs + d*nx; pts_arr[j*M:(j+1)*M, 1] = y0
        pts_arr[j*M:(j+1)*M, 2] = zs + d*nz
    vpts = vtk.vtkPoints(); vpts.SetData(numpy_to_vtk(pts_arr, deep=True))
    poly = vtk.vtkPolyData(); poly.SetPoints(vpts)
    probef = vtk.vtkProbeFilter()
    probef.SetInputData(poly); probef.SetSourceData(vol); probef.Update()
    out = probef.GetOutput(); pdd = out.GetPointData()
    valid = vtk_to_numpy(probef.GetValidPoints())
    mask = np.zeros(M * n_probe, bool); mask[valid] = True
    vel = vtk_to_numpy(pdd.GetArray('velocity'))
    nuhat = vtk_to_numpy(pdd.GetArray('nuHat'))
    vel = np.where(mask[:,None], vel, np.nan)
    nuhat = np.where(mask, nuhat, np.nan)
    M_loc = np.sqrt(vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2)
    chi = nuhat / NU
    return xs, dists, M_loc.reshape(n_probe, M), chi.reshape(n_probe, M)

fig, axes = plt.subplots(len(STAGES), 1, figsize=(14, 3.2*len(STAGES)), sharex=True)
y_max = 5000
for ax, step in zip(axes, STAGES):
    vol_pvtu = f"{SD}/volume_s{step}.pvtu"
    xs, d, M_f, chi = probe(vol_pvtu, side='lower', L=0.005, n_probe=200, x_clip=(0.0, 0.5))
    d_yp = d / NU
    XX, DD = np.meshgrid(xs, d_yp)
    # Mach contours (light)
    cs1 = ax.contour(XX, DD, M_f, np.arange(0.02, 0.30, 0.04), colors='blue',
                      linewidths=0.5, alpha=0.5)
    ax.clabel(cs1, fontsize=6, inline=True, fmt='%.2f', inline_spacing=2)
    # χ contours (bold)
    with np.errstate(invalid='ignore'):
        log_chi = np.log10(np.maximum(chi, 1e-10))
    cs2 = ax.contour(XX, DD, log_chi, np.arange(-7.0, 2.01, 0.5),
                      colors='k', linewidths=0.6)
    ax.clabel(cs2, fontsize=6, inline=True, fmt='%.1f', inline_spacing=2)
    # Highlight χ=1 line in red
    cs3 = ax.contour(XX, DD, log_chi, [0.0], colors='red', linewidths=1.5)
    try: ax.clabel(cs3, fontsize=8, fmt='χ=1', inline=False)
    except Exception: pass
    ax.set_ylim(0, y_max)
    ax.set_ylabel(f'step {step}\n$d \\cdot Re_\\infty$')
    ax.grid(alpha=0.3)
axes[-1].set_xlabel('x/c')
fig.suptitle('NLF α=9 cavL2 lower surface — χ contours (black) + Mach contours (blue, faint), χ=1 in RED', fontsize=11)
plt.tight_layout()
plt.savefig('/tmp/cavL2_a9_lower_evolution.png', dpi=130, bbox_inches='tight')
print('wrote /tmp/cavL2_a9_lower_evolution.png')
