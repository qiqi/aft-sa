"""Diagnostic v3 — direct mesh-cell query (no probe interpolation).

For each x bin on the lower surface, find the cell where Re_Ω peaks in the
BL band (z below surface, within 0.005 chord). Compute λ_p at that cell
from the local pressure gradient via VTK gradient filter on the slice.

Output a table with mfoil H, mfoil Re_θ, Flow360 (Re_Ω, λ_p, χ), and a
phase plot in (H, λ_p) colored by Re_θ to test Re-independence.
"""
import os, sys, pickle, numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/qiqi/flexcompute/aft-sa/paper')
from regen_nlf_v2 import walk_contour_xz, load_slice_derived

B = "/home/qiqi/flexcompute/aft-sa/flow360"
NU_UNIT = 1.0 / 4e6

CASES = {
    'cavL1_a9':  f"{B}/cavL1prop_nlf0416_Re4M_a9",
    'cavL2_a9':  f"{B}/cavL2prop_nlf0416_Re4M_a9",
    'strL1_a9':  f"{B}/strL1prop_nlf0416_Re4M_a9",
    'strL2_a9':  f"{B}/strL2prop_nlf0416_Re4M_a9",
}

def grad_p_slice(slice_g):
    gf = vtk.vtkGradientFilter()
    gf.SetInputData(slice_g)
    gf.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 'p')
    gf.SetResultArrayName('grad_p')
    gf.Update()
    return gf.GetOutput()

def analyze_lower(cd):
    """Find the kernel-active cell at each x station along the lower BL band.
    Returns rows of (x, H_n/a, Re_Omega, lambda_p, chi, d_active)."""
    g = grad_p_slice(load_slice_derived(cd))
    pd = g.GetPointData()
    pts = vtk_to_numpy(g.GetPoints().GetData())
    Re_O = vtk_to_numpy(pd.GetArray('Re_Omega'))
    nuhat = vtk_to_numpy(pd.GetArray('nuHat'))
    vel = vtk_to_numpy(pd.GetArray('velocity'))
    gp = vtk_to_numpy(pd.GetArray('grad_p'))
    rho = vtk_to_numpy(pd.GetArray('rho'))
    omega = vtk_to_numpy(pd.GetArray('vorticityMagnitude'))

    # Lower-surface contour
    Xm, Zm, up, lo = walk_contour_xz(cd)
    x_lo = Xm[lo]; z_lo = Zm[lo]
    order = np.argsort(x_lo)
    x_lo, z_lo = x_lo[order], z_lo[order]

    chi = nuhat / NU_UNIT
    rows = []
    # 25 x-stations distributed across the lower surface
    x_targets = np.concatenate([
        np.linspace(0.04, 0.50, 24),
        np.linspace(0.55, 0.95, 9)
    ])
    for x_t in x_targets:
        z_surf = float(np.interp(x_t, x_lo, z_lo))
        # Narrow vertical column below the surface; band thickness 0.005 chord
        m = (np.abs(pts[:,0] - x_t) < 0.003) & (pts[:,2] < z_surf) & (pts[:,2] > z_surf - 0.005)
        if m.sum() < 3: continue
        # cell of max Re_Ω in the band → kernel-active location
        Re_O_band = Re_O[m]
        k_local = int(np.argmax(Re_O_band))
        k_global = np.where(m)[0][k_local]
        d_a = z_surf - pts[k_global, 2]   # geometric wall distance (z below surface)
        v_x = vel[k_global, 0]; v_z = vel[k_global, 2]
        gx = gp[k_global, 0]; gz = gp[k_global, 2]
        u_dot_gp = v_x*gx + v_z*gz
        u_sq = v_x*v_x + v_z*v_z + 1e-12
        lambda_p = -d_a*d_a * u_dot_gp / (rho[k_global] * NU_UNIT * u_sq)
        rows.append({
            'x': x_t, 'd_active': d_a,
            'Re_Omega': float(Re_O[k_global]),
            'lambda_p': float(lambda_p),
            'chi': float(chi[k_global]),
            'chi_max_band': float(chi[m].max()),
            'omega': float(omega[k_global]),
        })
    return rows

if __name__ == '__main__':
    with open(f"{B}/mfoil_nlf0416_Re4M.pkl", 'rb') as f: mn = pickle.load(f)
    a = 9.0
    mlo = mn[a]['lower']
    xtr_lo = mn[a]['xtr_lower']
    order = np.argsort(mlo['x'])
    x_m = np.asarray(mlo['x'])[order]
    H_m = np.asarray(mlo['H'])[order]
    Reth_m = np.asarray(mlo['Re_theta'])[order]
    turb_m = np.asarray(mlo['turb'])[order]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig2, axp = plt.subplots(1, 2, figsize=(13, 5))
    colors = {'cavL1_a9':'C0','cavL2_a9':'C1','strL1_a9':'C2','strL2_a9':'C3'}
    all_results = {}
    for name, cd in CASES.items():
        if not os.path.exists(f"{cd}/slice_with_derived.pvtu"):
            print(f"SKIP {name}"); continue
        print(f"analyzing {name}...")
        rows = analyze_lower(cd)
        all_results[name] = rows
        x = np.array([r['x'] for r in rows])
        ReO = np.array([r['Re_Omega'] for r in rows])
        lp = np.array([r['lambda_p'] for r in rows])
        chi = np.array([r['chi'] for r in rows])
        chi_band = np.array([r['chi_max_band'] for r in rows])
        c = colors[name]
        axes[0,0].semilogy(x, ReO, '-o', color=c, ms=3, lw=1, label=name)
        axes[0,1].plot(x, lp, '-o', color=c, ms=3, lw=1)
        axes[1,0].semilogy(x, chi_band, '-o', color=c, ms=3, lw=1)
        axes[1,1].plot(x, np.array([r['d_active'] for r in rows]), '-o', color=c, ms=3, lw=1)
        # phase
        H_at = np.interp(x, x_m, H_m)
        Re_at = np.interp(x, x_m, Reth_m)
        chi_at = chi_band
        # Restrict to mfoil-laminar zone
        m_lam = (H_at > 2.1) & (H_at < 4.0) & (np.interp(x, x_m, turb_m) < 0.5)
        if m_lam.any():
            axp[0].scatter(H_at[m_lam], lp[m_lam],
                           c=np.log10(np.maximum(Re_at[m_lam], 1)),
                           vmin=2.0, vmax=3.5, s=50, cmap='viridis',
                           edgecolors=c, linewidths=1.2, label=name)
            axp[1].scatter(H_at[m_lam], lp[m_lam],
                           c=np.log10(np.maximum(chi_at[m_lam], 1e-4)),
                           vmin=-4, vmax=2, s=50, cmap='coolwarm',
                           edgecolors=c, linewidths=1.2, label=name)

    # mfoil on top of Re_Ω and λ_p panels
    axes[0,0].plot(x_m, Reth_m, 'k-', lw=1.2, label='mfoil $Re_\\theta$')
    for ax in axes.flat:
        ax.axvline(xtr_lo, color='gray', ls='--', alpha=0.6)
        ax.grid(alpha=0.3); ax.set_xlim(0, 1)
    axes[0,0].set_ylabel('$Re_\\Omega$ (peak in BL) and mfoil $Re_\\theta$')
    axes[0,0].set_ylim(10, 1e4); axes[0,0].legend(fontsize=8)
    axes[0,1].set_ylabel('$\\lambda_p$ (at peak $Re_\\Omega$ cell)')
    axes[0,1].axhline(0, color='gray', ls=':')
    axes[0,1].axhline(0.64, color='red', ls=':', alpha=0.6, label='$\\lambda_\\ast=0.64$')
    axes[0,1].legend(fontsize=8)
    axes[1,0].set_ylabel('$\\chi$ max in BL band')
    axes[1,0].set_ylim(1e-4, 1e2)
    axes[1,1].set_ylabel('$d_{active}$ (wall distance)')
    for ax in axes[-1,:]: ax.set_xlabel('x/c')
    fig.suptitle(f'NLF α={a:.0f} lower surface — mfoil $x_{{tr}}={xtr_lo:.2f}$')
    fig.tight_layout()
    fig.savefig('/tmp/diag_lower_v3_alongx.png', dpi=120)
    print('wrote /tmp/diag_lower_v3_alongx.png')

    # Phase plot
    for ax in axp:
        ax.axhline(0, color='gray', ls=':')
        ax.axvline(2.59, color='red', ls='--', alpha=0.5)
        ax.set_xlabel('$H$ (from mfoil)')
        ax.set_ylabel('$\\lambda_p$ (Flow360)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
    cb1 = plt.colorbar(axp[0].collections[0], ax=axp[0])
    cb1.set_label('$\\log_{10} Re_\\theta$ (mfoil)')
    cb2 = plt.colorbar(axp[1].collections[0], ax=axp[1])
    cb2.set_label('$\\log_{10} \\chi$')
    axp[0].set_title('Laminar zone $(H, \\lambda_p)$ — colored by $Re_\\theta$')
    axp[1].set_title('Same — colored by Flow360 $\\chi$')
    fig2.suptitle(f'NLF α={a:.0f} lower: is $(H \\leftrightarrow \\lambda_p)$ Re-independent in the laminar zone?')
    fig2.tight_layout()
    fig2.savefig('/tmp/diag_lower_v3_phase.png', dpi=120)
    print('wrote /tmp/diag_lower_v3_phase.png')

    # Print the table for the laminar zone
    print("\n=== mfoil-laminar zone (turb=0) — comparison ===")
    for name, rows in all_results.items():
        print(f"\n--- {name} ---")
        print(f"{'x':>6} {'H_mfl':>6} {'Reθ_mfl':>8} {'ReΩ_360':>8} {'λp_360':>8} {'χ_360':>10}")
        for r in rows:
            H = float(np.interp(r['x'], x_m, H_m))
            Reth = float(np.interp(r['x'], x_m, Reth_m))
            turb = float(np.interp(r['x'], x_m, turb_m))
            if turb > 0.5: continue
            if not (2.0 < H < 4.0): continue
            print(f"{r['x']:>6.3f} {H:>6.3f} {Reth:>8.0f} {r['Re_Omega']:>8.1f} {r['lambda_p']:>+8.3f} {r['chi_max_band']:>10.4f}")
