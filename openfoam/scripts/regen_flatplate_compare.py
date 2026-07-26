#!/usr/bin/env python3
"""Generate the paper's flat_plate_batch_flow360 figure layout from BOTH
solvers, using the SAME extraction machinery (copied verbatim-in-spirit from
sa-ai/paper/regen_flatplate_flow360.py -- that script is left untouched):

  flatplate_fig_flow360.pdf   from flow360_fr/flatplate_sphere_Tu*  (pvtu)
  flatplate_fig_openfoam.pdf  from openfoam cases/flatplate_Tu*     (foam)

Also prints the paper's onset-vs-AGS diagnostic (chi=1 crossing converted to
Re_theta via the laminar relation 0.664 sqrt(Re_x)) for both solvers.

Conventions identical to the paper script: x-binned profiles (121 bins),
theta normalized by LOCAL edge velocity u_e = max(u) with the integral cut at
u/u_e >= 0.999, Cf from the first off-wall point against FREESTREAM q.

Run with the compute venv python (vtk):
  /home/qiqi/flexcompute/compute/.venv/bin/python regen_flatplate_compare.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from extract_transition import (cell_coords, read_scalar_field,
                                read_vector_field, latest_time, NX, NZ)

NU = 1.0e-6
MACH = 0.1
TU_LIST = [0.04, 0.08, 0.16, 0.30, 0.60]
SYMBOLS = ['o', 's', '^', 'v', 'D']
PLATE_END_X = 6.0
OUTLET_MARGIN = 0.5
CONTOUR_X_MAX = 4.0

F360_ROOT = "/home/qiqi/flexcompute/sa-ai/flow360_fr"
OF_ROOT = "/local_data/qiqi/openfoam-sa-ai/cases"


def AGS_Reth(tu_pct):
    return 163.0 + np.exp(6.91 - tu_pct)


SS_BAND = [(0.026, 2.78, 3.82), (0.04, 2.80, 3.85), (0.08, 2.80, 3.88),
           (0.12, 2.62, 3.72), (0.16, 2.10, 3.25), (0.20, 1.82, 3.00),
           (0.24, 1.66, 2.90), (0.28, 1.55, 2.83), (0.32, 1.47, 2.76),
           (0.342, 1.42, 2.70)]


# ---------------- readers: both return (pts_xz, u, chi, wd) ----------------

def extract_volume_f360(cd):
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(f"{cd}/volume.pvtu")
    r.Update()
    g = r.GetOutput()
    pd = g.GetPointData()
    pts = vtk_to_numpy(g.GetPoints().GetData())
    u = vtk_to_numpy(pd.GetArray('velocity'))[:, 0] / MACH
    chi = vtk_to_numpy(pd.GetArray('nuHat')) / NU
    wd = vtk_to_numpy(pd.GetArray('wallDistance'))
    y_unique = np.unique(pts[:, 1])
    m = np.abs(pts[:, 1] - y_unique[0]) < 1e-5
    return pts[m][:, [0, 2]], u[m], chi[m], wd[m]


def extract_volume_openfoam(cd):
    t = latest_time(cd)
    n = NX * NZ
    U = read_vector_field(f"{cd}/{t}/U", n)
    nuT = read_scalar_field(f"{cd}/{t}/nuTilda", n)
    xc, zc, _ = cell_coords()
    X = np.tile(xc, NZ)
    Z = np.repeat(zc, NX)
    return np.column_stack([X, Z]), U[:, 0], nuT / NU, Z.copy()


# ------------- paper-script machinery (adapted, same numerics) -------------

def regrid_to_xy(x, z, vals, x_grid, z_grid):
    from scipy.interpolate import griddata
    XG, ZG = np.meshgrid(x_grid, z_grid)
    V = griddata(np.column_stack([x, z]), vals, (XG, ZG), method='linear')
    return XG, ZG, V


def cf_and_retheta(pts_v, u, chi, wd):
    z_clip = 0.05
    x_max_data = PLATE_END_X - OUTLET_MARGIN
    x_bins = np.linspace(0.0, x_max_data, 121)
    x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
    Re_theta_arr = np.full(len(x_centers), np.nan)
    cf_volume_arr = np.full(len(x_centers), np.nan)
    chi_max_arr = np.full(len(x_centers), np.nan)
    for i in range(len(x_centers)):
        m = ((pts_v[:, 0] >= x_bins[i]) & (pts_v[:, 0] < x_bins[i+1])
             & (wd >= 0) & (wd <= z_clip))
        if m.sum() < 5:
            continue
        wd_loc = wd[m]; u_loc = u[m]
        order = np.argsort(wd_loc)
        wd_s, u_s = wd_loc[order], u_loc[order]
        keep = np.concatenate(([True], np.diff(wd_s) > 1e-10))
        wd_s, u_s = wd_s[keep], u_s[keep]
        if len(wd_s) >= 2:
            u_e = u_s.max()
            r = u_s / u_e
            above = np.where(r >= 0.999)[0]
            edge = above[0] if len(above) else len(u_s) - 1
            if edge >= 1:
                rr = r[:edge+1]
                theta = np.trapezoid(rr * (1.0 - rr), wd_s[:edge+1])
                Re_theta_arr[i] = theta / NU
            nz = np.where(wd_s > 1e-9)[0]
            if len(nz):
                j = nz[0]
                if wd_s[j] < 1e-3:
                    cf_volume_arr[i] = 2.0 * NU * u_s[j] / wd_s[j]
        chi_max_arr[i] = chi[m].max() if m.sum() else np.nan
    return x_centers, Re_theta_arr, cf_volume_arr, chi_max_arr


def onset_vs_ags(label, results):
    print(f"--- {label}: chi=1 onset (Blasius-converted Re_theta) vs AGS ---")
    out = {}
    for tu in TU_LIST:
        if tu not in results:
            continue
        xs, reth, cf, chimax = results[tu]
        m = np.isfinite(chimax)
        xs2, cs = xs[m], chimax[m]
        ix = np.where(cs > 1.0)[0]
        if not len(ix) or ix[0] == 0:
            continue
        i = ix[0]
        f = (1.0 - cs[i-1])/(cs[i] - cs[i-1])
        rex = (xs2[i-1] + f*(xs2[i]-xs2[i-1]))*1e6
        rth = 0.664*np.sqrt(rex)
        ags = AGS_Reth(tu)
        out[tu] = rth
        print(f"Tu={tu:5.2f}%  AGS={ags:6.0f}  chi=1 onset={rth:6.0f}  "
              f"({(rth/ags-1)*100:+5.1f}%)")
    return out


def make_figure(source_name, extract, root, case_fmt, outname):
    n = len(TU_LIST)
    fig = plt.figure(figsize=(9, 11.0))
    gs = gridspec.GridSpec(n + 1, 2, height_ratios=[1]*n + [1.8],
                           hspace=0.32, wspace=0.10,
                           top=0.985, bottom=0.06, left=0.07, right=0.97)
    cf_results = {}

    for row, tu in enumerate(TU_LIST):
        cd = os.path.join(root, case_fmt.format(int(round(tu*1000))))
        try:
            pts_v, u, chi, wd = extract(cd)
        except Exception as e:
            print(f"missing {cd}: {e}")
            continue
        x_grid = np.linspace(0.0, CONTOUR_X_MAX, 200)
        z_phys_max = 0.020
        z_grid_stretch = np.linspace(0.0, 1.0, 200)**1.6 * z_phys_max
        XG, ZG, UG = regrid_to_xy(pts_v[:, 0], wd, u, x_grid, z_grid_stretch)
        _, _, NG = regrid_to_xy(pts_v[:, 0], wd,
                                np.log10(np.maximum(chi, 1e-10)),
                                x_grid, z_grid_stretch)
        ZG_yp = ZG / NU
        y_max_plot = 8000

        ax = fig.add_subplot(gs[row, 0])
        levels_u = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        cs = ax.contour(XG, ZG_yp, UG, levels_u, colors='k', linewidths=0.6)
        ax.clabel(cs, fontsize=6.0, inline=True, fmt='%g', inline_spacing=2)
        ax.set_title(rf'$u/U_\infty$, $Tu={tu:g}$%', y=0.82, fontsize=9)
        ax.set_ylim(0, y_max_plot); ax.set_ylabel('y')
        if row < n-1: ax.set_xticklabels([])
        else: ax.set_xlabel(r'$Re_x/10^6$')

        ax = fig.add_subplot(gs[row, 1])
        levels_n = np.arange(-7.0, 2.01, 0.5)
        cs = ax.contour(XG, ZG_yp, NG, levels_n, colors='k', linewidths=0.6)
        ax.clabel(cs, fontsize=6.0, inline=True, fmt='%g', inline_spacing=2)
        ax.set_title(rf'$\log_{{10}}\hat\nu/\nu$, $Tu={tu:g}$%', y=0.82,
                     fontsize=9)
        ax.set_ylim(0, y_max_plot); ax.set_yticklabels([])
        if row < n-1: ax.set_xticklabels([])
        else: ax.set_xlabel(r'$Re_x/10^6$')

        cf_results[tu] = cf_and_retheta(pts_v, u, chi, wd)

    ax_chi = fig.add_subplot(gs[n, 0])
    ax_cf = fig.add_subplot(gs[n, 1])
    C_V1 = 7.1
    y_bot = 1.3e-5
    Re_unit = 1.0 / NU
    for k, tu in enumerate(TU_LIST):
        if tu not in cf_results:
            continue
        x_centers, Re_th, cf_vol, chi_mx = cf_results[tu]
        valid_chi = np.isfinite(chi_mx) & (x_centers < PLATE_END_X - OUTLET_MARGIN)
        ax_chi.semilogy(x_centers[valid_chi], chi_mx[valid_chi], '-',
                        color='k', lw=1.0, marker=SYMBOLS[k], markevery=12,
                        mfc='w', mec='k', ms=4, label=rf'$Tu={tu:g}$%')
        valid_cf = (np.isfinite(Re_th) & (Re_th > 1)
                    & (x_centers < PLATE_END_X - OUTLET_MARGIN))
        ax_cf.loglog(Re_th[valid_cf], cf_vol[valid_cf], SYMBOLS[k], mfc='w',
                     mec='k', color='k', ms=4, label=rf'SA-AI, $Tu={tu:g}$%')

    ax_chi.axhline(1.0, color='gray', lw=0.6, ls=':', alpha=0.7)
    ax_chi.axhline(C_V1, color='gray', lw=0.8, ls='--', alpha=0.7)
    ax_chi.text(0.02, 1.5, r'$\chi=1$', color='0.4', fontsize=7, va='bottom')
    ax_chi.text(0.02, C_V1*1.5, r'$\chi=c_{v1}$', color='0.4', fontsize=7,
                va='bottom')
    ax_chi.set_xlabel(r'$Re_x / 10^6$')
    ax_chi.set_ylabel(r'$\chi=\tilde\nu/\nu$')
    ax_chi.set_xlim(0, CONTOUR_X_MAX); ax_chi.set_ylim(1e-5, 1e2)
    ax_chi.grid(True, which='major', alpha=0.5)
    ax_chi.grid(True, which='minor', alpha=0.2)
    for k, tu in enumerate(TU_LIST):
        x_ags = (AGS_Reth(tu) / 0.664)**2 / Re_unit
        if x_ags < CONTOUR_X_MAX:
            ax_chi.axvline(x_ags, color='0.5', lw=0.8, ls='-', zorder=1)
            ax_chi.plot(x_ags, y_bot, SYMBOLS[k], mfc='k', mec='k', ms=8,
                        zorder=5)
    ax_chi.legend(loc='lower right', fontsize=7, frameon=False, ncol=2)

    first = next(iter(cf_results))
    Re_th_ref = np.array(cf_results[first][1])
    valid = np.isfinite(Re_th_ref) & (Re_th_ref > 1)
    Re_th_ref = np.sort(Re_th_ref[valid])
    ax_cf.loglog(Re_th_ref, 0.441/Re_th_ref, 'k--', lw=1.2,
                 label=r'laminar, $C_f=0.441/Re_\theta$')
    ax_cf.loglog(Re_th_ref, 2.0*(np.log(Re_th_ref)/0.38 + 3.7)**(-2), 'k:',
                 lw=1.2, label=r'turbulent, $C_f=2[(\ln Re_\theta)/0.38+3.7]^{-2}$')
    ss_tu = np.array([b[0] for b in SS_BAND])
    ss_rb = np.array([0.664*np.sqrt(b[1]*1e6) for b in SS_BAND])
    ss_re = np.array([0.664*np.sqrt(b[2]*1e6) for b in SS_BAND])
    y_min = 1e-4
    for k, tu in enumerate(TU_LIST):
        Re_AGS = AGS_Reth(tu)
        ax_cf.axvline(Re_AGS, color='0.5', lw=0.8, ls='-', zorder=1)
        ax_cf.plot(Re_AGS, y_min*1.15, SYMBOLS[k], mfc='k', mec='k', ms=8,
                   zorder=5)
        if tu <= ss_tu.max():
            rb = float(np.interp(tu, ss_tu, ss_rb))
            re = float(np.interp(tu, ss_tu, ss_re))
            ax_cf.plot([rb, re], [y_min*1.05, y_min*1.05], '-', color='0.5',
                       lw=2.5, alpha=0.5, zorder=2)
    ax_cf.legend(loc='lower left', fontsize=7, frameon=False)
    ax_cf.grid(True, which='major', alpha=0.5)
    ax_cf.grid(True, which='minor', alpha=0.2)
    ax_cf.set_xlim(200, 5000); ax_cf.set_ylim(1e-4, 1e-2)
    ax_cf.set_xlabel(r'$Re_\theta$'); ax_cf.set_ylabel(r'$C_f$')

    out = os.path.join(OUT_DIR, outname)
    plt.savefig(out, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(out.replace('.pdf', '.png'), dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')
    return cf_results


def main():
    r_of = make_figure('openfoam', extract_volume_openfoam, OF_ROOT,
                       'flatplate_Tu{:04d}', 'flatplate_fig_openfoam.pdf')
    onset_vs_ags('OpenFOAM', r_of)
    r_f6 = make_figure('flow360', extract_volume_f360, F360_ROOT,
                       'flatplate_sphere_Tu{:04d}', 'flatplate_fig_flow360.pdf')
    onset_vs_ags('Flow360', r_f6)


if __name__ == '__main__':
    main()
