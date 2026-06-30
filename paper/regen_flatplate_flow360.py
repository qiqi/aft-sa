"""Regenerate flat_plate_batch_flow360.pdf from Flow360 runs.

Layout (matches the archived original + new χ vs Re_θ panel):
  5 rows × 2 cols of contours, one row per Tu:
    left  : u/U_inf contours
    right : log10(nuhat/nu) contours
  Final row spans both columns and holds TWO plots:
    left  : chi (log) vs Re_theta — overlay of all 5 Tu cases
    right : Cf  (log) vs Re_theta — overlay of all 5 Tu cases + S-S markers

All extraction is from volume.pvtu (cell-centered → already nodal in Flow360
output). Span direction (y) is collapsed by selecting a single y-slice.
"""
import os, sys, numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

B = "/home/qiqi/flexcompute/aft-sa/flow360"
NU = 1.0e-6   # Re_unit = 1e6, nu = 1/Re  (=> Re_x = x * 1e6)
MACH = 0.1    # Flow360 stores velocity = u/c_∞; freestream u/U_∞ = Mach/Mach = 1, so divide by MACH
# AGS-calibrated case set (Tu in S-S/AGS natural-transition range; drop the old
# out-of-range Tu=0.85% and the mis-sourced 5-pt set). See run_flatplate_ags.py.
TU_LIST = [0.04, 0.08, 0.16, 0.30, 0.60]
SYMBOLS = ['o', 's', '^', 'v', 'D']

def AGS_Reth(tu_pct):
    """Abu-Ghannam & Shaw (1980) zero-PG transition-onset Re_theta (Tu in %)."""
    return 163.0 + np.exp(6.91 - tu_pct)

# Schubauer-Skramstad (1948) Fig. 7 transition band, digitized (NACA Rep. 909):
# (Tu%, Re_x_begin, Re_x_end) in units of 1e6 -- shown as a secondary reference.
SS_BAND = [(0.026, 2.78, 3.82), (0.04, 2.80, 3.85), (0.08, 2.80, 3.88),
           (0.12, 2.62, 3.72), (0.16, 2.10, 3.25), (0.20, 1.82, 3.00),
           (0.24, 1.66, 2.90), (0.28, 1.55, 2.83), (0.32, 1.47, 2.76),
           (0.342, 1.42, 2.70)]

# Plate extends 0..PLATE_END_X. Last ~10% near outlet has BL corrupted by the
# downstream BC (artificial pressure adjustment) so we clip it out.
PLATE_END_X = 6.0
OUTLET_MARGIN = 0.5         # ignore data within this distance of outlet
CONTOUR_X_MAX = 4.0

def case_dir(tu):
    return f"{B}/flatplate_ags_Tu{int(round(tu*1000)):04d}"

def extract_volume(cd):
    """Return points (x, z), u, chi at all volume nodes. Span (y) handled by
    taking the y closest to a single span slice (y is quasi-2D with thin span)."""
    r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(f"{cd}/volume.pvtu"); r.Update()
    g = r.GetOutput(); pd = g.GetPointData()
    pts = vtk_to_numpy(g.GetPoints().GetData())
    vel = vtk_to_numpy(pd.GetArray('velocity'))
    nuhat = vtk_to_numpy(pd.GetArray('nuHat'))
    wd = vtk_to_numpy(pd.GetArray('wallDistance'))
    # Flow360 stores velocity in units of c_∞ (sound speed). Convert to U_∞ by /Mach.
    u = vel[:, 0] / MACH
    chi = nuhat / NU
    # Quasi-2D mesh: span has only ~2 y-values. Pick the first one.
    y_unique = np.unique(pts[:, 1])
    span_sel = np.abs(pts[:, 1] - y_unique[0]) < 1e-5
    return pts[span_sel], u[span_sel], chi[span_sel], wd[span_sel]

def regrid_to_xy(x, z, vals, x_grid, z_grid):
    """Linear-interpolate scattered (x, z, vals) onto a rectilinear (x_grid, z_grid)."""
    from scipy.interpolate import griddata
    XG, ZG = np.meshgrid(x_grid, z_grid)
    V = griddata(np.column_stack([x, z]), vals, (XG, ZG), method='linear')
    return XG, ZG, V

def cf_and_retheta(cd):
    """Compute Cf and Re_theta along the plate from surface output and wall-normal probe."""
    # Use surface output if present
    surf = None
    for cand in ['surface.pvtu', 'surface_wall.pvtu', 'surface_fluid_plate.pvtu']:
        if os.path.exists(f"{cd}/{cand}"):
            surf = f"{cd}/{cand}"; break
    if surf is None:
        # Fallback: glob for any surface*.pvtu
        import glob
        matches = sorted(glob.glob(f"{cd}/surface*.pvtu"))
        if matches: surf = matches[0]
    if surf and os.path.exists(surf):
        r = vtk.vtkXMLPUnstructuredGridReader(); r.SetFileName(surf); r.Update()
        g = r.GetOutput(); pd = g.GetPointData()
        names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
        pts = vtk_to_numpy(g.GetPoints().GetData())
        if 'Cf' in names:
            Cf = vtk_to_numpy(pd.GetArray('Cf'))
            cf_mag = np.linalg.norm(Cf, axis=1) if Cf.ndim > 1 else np.abs(Cf)
        else:
            cf_mag = None
    else:
        names = []; pts = None; cf_mag = None

    # For Re_theta we need theta = ∫₀^δ u(1-u) dz where δ ≲ BL thickness.
    # Clip to wd < z_clip so far-field doesn't contaminate the integral.
    pts_v, u, chi, wd = extract_volume(cd)
    z_clip = 0.05  # physical BL height bound (laminar δ≈0.012 at x=6, turbulent ~0.04)
    x_max_data = PLATE_END_X - OUTLET_MARGIN
    x_bins = np.linspace(0.0, x_max_data, 121)
    x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
    Re_theta_arr = np.full(len(x_centers), np.nan)
    cf_volume_arr = np.full(len(x_centers), np.nan)
    chi_max_arr = np.full(len(x_centers), np.nan)
    for i in range(len(x_centers)):
        m = (pts_v[:, 0] >= x_bins[i]) & (pts_v[:, 0] < x_bins[i+1]) & (wd >= 0) & (wd <= z_clip)
        if m.sum() < 5: continue
        wd_loc = wd[m]; u_loc = u[m]
        order = np.argsort(wd_loc)
        wd_s, u_s = wd_loc[order], u_loc[order]
        # Deduplicate
        keep = np.concatenate(([True], np.diff(wd_s) > 1e-10))
        wd_s, u_s = wd_s[keep], u_s[keep]
        if len(wd_s) >= 2:
            # Momentum thickness with EDGE-VELOCITY normalization, integral cut
            # at the BL edge. A fixed-height integral (to z_clip) is corrupted by
            # the weak inviscid overshoot over the plate (u/U_inf ~ 1.02-1.04
            # just above the BL): there u(1-u) < 0 and cancels the real momentum
            # deficit, collapsing theta to ~40% of Blasius and shifting the
            # laminar points off the Cf = 0.441/Re_theta line. Normalizing by
            # the local edge velocity u_e = max(u) and integrating only to the
            # edge (first wd where u/u_e >= 0.999) recovers Blasius to within ~5%
            # in the laminar region and the log-law in the turbulent region.
            u_e = u_s.max()
            r = u_s / u_e
            above = np.where(r >= 0.999)[0]
            edge = above[0] if len(above) else len(u_s) - 1
            if edge >= 1:
                rr = r[:edge+1]
                theta = np.trapezoid(rr * (1.0 - rr), wd_s[:edge+1])
                Re_theta_arr[i] = theta / NU
            # Wall shear: skip wd=0 wall node; use first wd>0 node. Cf is
            # referenced to the FREESTREAM dynamic pressure (u_s is u/U_inf),
            # NOT u_e, so it is left unnormalized by the edge velocity.
            nz = np.where(wd_s > 1e-9)[0]
            if len(nz):
                j = nz[0]
                if wd_s[j] < 1e-3:
                    cf_volume_arr[i] = 2.0 * NU * u_s[j] / wd_s[j]
        chi_max_arr[i] = chi[m].max() if m.sum() else np.nan

    return x_centers, Re_theta_arr, cf_volume_arr, chi_max_arr, cf_mag, pts


def main():
    n = len(TU_LIST)
    fig = plt.figure(figsize=(9, 11.0))
    gs = gridspec.GridSpec(n + 1, 2, height_ratios=[1]*n + [1.8],
                           hspace=0.32, wspace=0.10,
                           top=0.985, bottom=0.06, left=0.07, right=0.97)

    cf_results = {}  # Tu -> (Re_theta, cf_volume, chi_max)

    for row, tu in enumerate(TU_LIST):
        cd = case_dir(tu)
        if not os.path.exists(f"{cd}/volume.pvtu"):
            print(f"missing: {cd}"); continue
        pts_v, u, chi, wd = extract_volume(cd)
        print(f"Tu={tu}%: {len(pts_v)} pts, x=[{pts_v[:,0].min():.2f},{pts_v[:,0].max():.2f}]")

        # Build a contour grid. y plotted in wall units (multiply by Re_unit=1/NU).
        # Laminar δ ≈ 5e-3 at x=6 → 5000 wall units; turbulent δ ≈ 0.05x → ~30000 max.
        # Use a logarithmically-stretched z grid so the BL is well-resolved.
        # Clip x range to just past the latest transition (skip outlet-corrupted tail).
        x_grid = np.linspace(0.0, CONTOUR_X_MAX, 200)
        z_phys_max = 0.020  # ≈ 20000 wall units; captures BL with margin
        z_grid_stretch = np.linspace(0.0, 1.0, 200)**1.6 * z_phys_max  # finer near wall
        XG, ZG, UG = regrid_to_xy(pts_v[:,0], wd, u, x_grid, z_grid_stretch)
        _, _, NG = regrid_to_xy(pts_v[:,0], wd, np.log10(np.maximum(chi, 1e-10)),
                                 x_grid, z_grid_stretch)
        ZG_yp = ZG / NU
        y_max_plot = 8000  # focus on the BL; laminar δ≈5000 yp at x=4

        # u/U_inf contours (left)
        ax = fig.add_subplot(gs[row, 0])
        levels_u = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        cs = ax.contour(XG, ZG_yp, UG, levels_u, colors='k', linewidths=0.6)
        ax.clabel(cs, fontsize=6.0, inline=True, fmt='%g', inline_spacing=2)
        ax.set_title(rf'$u/U_\infty$, $Tu={tu:g}$%', y=0.82, fontsize=9)
        ax.set_ylim(0, y_max_plot); ax.set_ylabel('y')
        if row < n-1: ax.set_xticklabels([])
        else: ax.set_xlabel(r'$Re_x/10^6$')

        # log10(nuhat/nu) contours (right) — uniformly spaced so visual spacing
        # matches the χ(x) curve in the bottom-left panel.
        ax = fig.add_subplot(gs[row, 1])
        levels_n = np.arange(-7.0, 2.01, 0.5)
        cs = ax.contour(XG, ZG_yp, NG, levels_n, colors='k', linewidths=0.6)
        ax.clabel(cs, fontsize=6.0, inline=True, fmt='%g', inline_spacing=2)
        ax.set_title(rf'$\log_{{10}}\hat\nu/\nu$, $Tu={tu:g}$%', y=0.82, fontsize=9)
        ax.set_ylim(0, y_max_plot); ax.set_yticklabels([])
        if row < n-1: ax.set_xticklabels([])
        else: ax.set_xlabel(r'$Re_x/10^6$')

        # Wall-shear-based Cf & chi vs (x, Re_theta) from the volume slice
        xc, Re_th, cf_vol, chi_mx, _, _ = cf_and_retheta(cd)
        cf_results[tu] = (xc, Re_th, cf_vol, chi_mx)

    # ===== Bottom row: chi vs x (left) + Cf vs Re_theta (right) =====
    ax_chi = fig.add_subplot(gs[n, 0])
    ax_cf = fig.add_subplot(gs[n, 1])

    # Monochrome with per-Tu symbols (matching the bottom-right Cf panel) so
    # the two bottom panels share one legend convention.
    C_V1 = 7.1
    y_bot = 1.5e-8   # height of the bottom-edge S-S markers (χ-axis bottom)
    Re_unit = 1.0 / NU
    for k, tu in enumerate(TU_LIST):
        if tu not in cf_results: continue
        x_centers, Re_th, cf_vol, chi_mx = cf_results[tu]
        valid_chi = np.isfinite(chi_mx) & (x_centers < PLATE_END_X - OUTLET_MARGIN)
        ax_chi.semilogy(x_centers[valid_chi], chi_mx[valid_chi], '-', color='k',
                        lw=1.0, marker=SYMBOLS[k], markevery=12, mfc='w', mec='k',
                        ms=4, label=rf'$Tu={tu:g}$%')
        valid_cf = np.isfinite(Re_th) & (Re_th > 1) & (x_centers < PLATE_END_X - OUTLET_MARGIN)
        ax_cf.loglog(Re_th[valid_cf], cf_vol[valid_cf], SYMBOLS[k], mfc='w', mec='k',
                     color='k', ms=4, label=rf'SA-AF, $Tu={tu:g}$%')

    # χ reference horizontals: χ=1 (start of σ_t blend) and χ=c_v1=7.1
    # (half-saturation of f_v1 — where Cf actually rises; see body text).
    ax_chi.axhline(1.0, color='gray', lw=0.6, ls=':', alpha=0.7)
    ax_chi.axhline(C_V1, color='gray', lw=0.8, ls='--', alpha=0.7)
    ax_chi.text(0.02, 1.0 * 1.5, r'$\chi=1$', color='0.4', fontsize=7, va='bottom')
    ax_chi.text(0.02, C_V1 * 1.5, r'$\chi=c_{v1}$', color='0.4', fontsize=7, va='bottom')
    # x-axis IS Re_x/1e6 (Re_x = x*Re_unit, Re_unit=1e6).
    ax_chi.set_xlabel(r'$Re_x / 10^6$'); ax_chi.set_ylabel(r'$\chi=\tilde\nu/\nu$')
    ax_chi.set_xlim(0, CONTOUR_X_MAX); ax_chi.set_ylim(1e-8, 1e2)
    ax_chi.grid(True, which='major', alpha=0.5)
    ax_chi.grid(True, which='minor', alpha=0.2)
    # AGS-1980 transition-onset reference, as x=Re_x/1e6 = (AGS_Reth/0.664)^2/Re_unit:
    # solid-gray vertical + filled per-Tu symbol on the bottom edge (same style as
    # the bottom-right panel). The model's χ=1 crossing should sit near these.
    for k, tu in enumerate(TU_LIST):
        x_ags = (AGS_Reth(tu) / 0.664)**2 / Re_unit
        if x_ags < CONTOUR_X_MAX:
            ax_chi.axvline(x_ags, color='0.5', lw=0.8, ls='-', zorder=1)
            ax_chi.plot(x_ags, y_bot, SYMBOLS[k], mfc='k', mec='k', ms=8, zorder=5)
    ax_chi.legend(loc='lower right', fontsize=7, frameon=False, ncol=2)

    # Cf reference correlations (Blasius + Coles-Fernholz)
    Re_th_ref = np.array(cf_results[TU_LIST[0]][1])
    valid = np.isfinite(Re_th_ref) & (Re_th_ref > 1)
    Re_th_ref = Re_th_ref[valid]
    cf_lam = 0.441 / Re_th_ref
    cf_turb = 2.0 * (1.0 / 0.38 * np.log(Re_th_ref) + 3.7)**(-2)
    ax_cf.loglog(Re_th_ref, cf_lam, 'k--', lw=1.2,
                 label=r'laminar, $C_f=0.441/Re_\theta$')
    ax_cf.loglog(Re_th_ref, cf_turb, 'k:', lw=1.2,
                 label=r'turbulent, $C_f=2[(\ln Re_\theta)/0.38+3.7]^{-2}$')

    # AGS-1980 onset markers (calibration target) + S-S Fig.7 band (context).
    # Per Tu: vertical gray line + filled symbol at AGS Re_theta_t; a faint
    # horizontal bar shows the digitized S-S transition band [begin,end] in
    # Re_theta at that Tu (interpolated; skipped beyond S-S's 0.342% range).
    from numpy import interp
    ss_tu = np.array([b[0] for b in SS_BAND])
    ss_rb = np.array([0.664*np.sqrt(b[1]*1e6) for b in SS_BAND])  # begin Re_theta
    ss_re = np.array([0.664*np.sqrt(b[2]*1e6) for b in SS_BAND])  # end   Re_theta
    y_min = 1e-4
    for k, tu in enumerate(TU_LIST):
        Re_AGS = AGS_Reth(tu)
        ax_cf.axvline(Re_AGS, color='0.5', lw=0.8, ls='-', zorder=1)
        ax_cf.plot(Re_AGS, y_min * 1.15, SYMBOLS[k], mfc='k', mec='k', ms=8, zorder=5)
        if tu <= ss_tu.max():   # S-S band only within its measured range
            rb, re = float(interp(tu, ss_tu, ss_rb)), float(interp(tu, ss_tu, ss_re))
            ax_cf.plot([rb, re], [y_min*1.05, y_min*1.05], '-', color='0.5',
                       lw=2.5, alpha=0.5, zorder=2)
    # legend proxies for the two references
    from matplotlib.lines import Line2D
    ref_handles = [Line2D([],[],color='0.5',marker='o',mfc='k',mec='k',ls='-',lw=0.8,
                          ms=6,label='AGS 1980 onset'),
                   Line2D([],[],color='0.5',lw=2.5,alpha=0.5,label='S-S Fig.7 band')]
    ax_cf.legend(loc='lower left', fontsize=7, frameon=False)
    ax_cf.grid(True, which='major', alpha=0.5)
    ax_cf.grid(True, which='minor', alpha=0.2)
    ax_cf.set_xlim(200, 5000); ax_cf.set_ylim(1e-4, 1e-2)
    ax_cf.set_xlabel(r'$Re_\theta$'); ax_cf.set_ylabel(r'$C_f$')

    out = "/home/qiqi/flexcompute/aft-sa/paper/figs/flat_plate_batch_flow360.pdf"
    plt.savefig(out, bbox_inches='tight', pad_inches=0.05)
    plt.savefig('/tmp/flat_plate_batch_flow360.png', dpi=110, bbox_inches='tight')
    print(f'wrote {out}')
    print('wrote /tmp/flat_plate_batch_flow360.png')


if __name__ == '__main__':
    main()
