"""Wall-anchored log10(chi) contour sheets: 6 grids x 2 surfaces per (airfoil, alpha).

Rows: cavity/O-grid interleaved by level (both L0, both L1, both L2);
left column upper surface, right column lower. Line contours in the
flat-plate-figure style (dashed = chi < 1 laminar levels, solid = turbulent).
x-axis: wall-anchor x/c; y-axis: wall-normal distance (each probe scan is one
vertical line of the sheet). chi = nuHat * Re/M -- the center-span slice's
nuHat is a*L-normalized (freestream check: chi_inf*M/Re = 2.19e-11 exactly).

Usage: python3 plot_chi_contour_maps.py [nlf0416|eppler387] [alpha] [out.png]
"""
import sys, os
sys.path.insert(0, '/home/qiqi/flexcompute/sa-ai/paper/repro/cfd')
os.environ.setdefault('SAAI_CFD_ROOT', '/home/qiqi/flexcompute/sa-ai/flow360_tie')
import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import regen_nlf_v2 as m

AF = sys.argv[1] if len(sys.argv) > 1 else 'nlf0416'
ALPHA = sys.argv[2] if len(sys.argv) > 2 else '4'
OUT = sys.argv[3] if len(sys.argv) > 3 else f'chi_contour_6x2_{AF}_a{ALPHA}.png'
ROOT = os.environ['SAAI_CFD_ROOT']
if AF == 'nlf0416':
    RE, CASETAG, L_UP, L_LO = 4e6, 'nlf0416_Re4M', 0.0015, 0.0025
else:
    scale = np.sqrt(4e6/2e5)
    RE, CASETAG, L_UP, L_LO = 2e5, 'eppler387_Re200k', 0.0015*scale, 0.0025*scale
MACH = 0.1
ROWS = [('cavL0prop', 'cavity L0'), ('strL0prop', 'O-grid L0'),
        ('cavL1prop', 'cavity L1'), ('strL1prop', 'O-grid L1'),
        ('cavL2prop', 'cavity L2'), ('strL2prop', 'O-grid L2')]


def chi_scan(case_d, side, L_probe, n_probe=140):
    """(x_anchors, dists, chi[n_probe, M]) probed along outward wall normals."""
    Xm, Zm, up_idx, lo_idx = m.walk_contour_xz(case_d, af=AF)
    idx = up_idx if side == 'upper' else lo_idx
    xs = Xm[idx]; zs = Zm[idx]
    tx_raw = np.gradient(xs); tz_raw = np.gradient(zs)
    s = np.sqrt(tx_raw**2 + tz_raw**2) + 1e-30
    tx, tz = tx_raw/s, tz_raw/s
    nx, nz = tz, -tx
    if side == 'upper' and np.mean(nz) < 0:
        nx, nz = -nx, -nz
    elif side == 'lower' and np.mean(nz) > 0:
        nx, nz = -nx, -nz
    M = len(xs)
    dists = np.linspace(1e-6, L_probe, n_probe)
    y0 = m.slice_y_plane(case_d)
    pts = np.empty((M*n_probe, 3))
    for j, d in enumerate(dists):
        pts[j*M:(j+1)*M, 0] = xs + d*nx
        pts[j*M:(j+1)*M, 1] = y0
        pts[j*M:(j+1)*M, 2] = zs + d*nz
    vp = vtk.vtkPoints(); vp.SetData(numpy_to_vtk(pts, deep=True))
    poly = vtk.vtkPolyData(); poly.SetPoints(vp)
    g, _, _ = m.load_slice(case_d)
    pr = vtk.vtkProbeFilter(); pr.SetInputData(poly); pr.SetSourceData(g); pr.Update()
    nu = vtk_to_numpy(pr.GetOutput().GetPointData().GetArray('nuHat')) * (RE/MACH)
    valid = vtk_to_numpy(pr.GetValidPoints())
    mask = np.zeros(M*n_probe, bool); mask[valid] = True
    chi = np.where(mask, nu, np.nan).reshape(n_probe, M)
    o = np.argsort(xs)
    return xs[o], dists, chi[:, o]


MAJOR = [-3, -2, -1, 0, np.log10(7.1)]
MINOR = [-2.5, -1.5, -0.5]
FMT = {-3: '-3', -2: '-2', -1: '-1', 0: r'$\chi{=}1$', np.log10(7.1): r'$c_{v1}$'}

fig, axes = plt.subplots(6, 2, figsize=(11.5, 12.5), sharex=True)
for r, (gname, glabel) in enumerate(ROWS):
    case = f"{ROOT}/{gname}_{CASETAG}_a{ALPHA}"
    for c, (side, L) in enumerate((('upper', L_UP), ('lower', L_LO))):
        ax = axes[r, c]
        x, d, chi = chi_scan(case, side, L)
        logchi = np.log10(np.clip(chi, 1e-8, None))
        ax.contour(x, d*1e3, logchi, levels=MINOR, colors='k', linewidths=0.4)
        cs = ax.contour(x, d*1e3, logchi, levels=MAJOR, colors='k', linewidths=0.8)
        ax.clabel(cs, fmt=FMT, fontsize=6.5, inline_spacing=2)
        ax.set_xlim(0, 1); ax.set_ylim(0, L*1e3)
        if c == 0:
            ax.set_ylabel(f'{glabel}\n' + r'$d/c\times10^3$', fontsize=9)
    print(f"{gname}: both sides scanned", flush=True)
axes[0, 0].set_title(f'upper surface (L={L_UP:.4f}c)', fontsize=10)
axes[0, 1].set_title(f'lower surface (L={L_LO:.4f}c)', fontsize=10)
for c in range(2):
    axes[-1, c].set_xlabel('wall-anchor x/c')
fig.suptitle(rf'{AF}, $\alpha={ALPHA}^\circ$: $\log_{{10}}\chi$ contours, all six grids',
             y=0.995, fontsize=12)
plt.tight_layout()
plt.savefig(OUT, dpi=130, bbox_inches='tight')
print('wrote', OUT)
