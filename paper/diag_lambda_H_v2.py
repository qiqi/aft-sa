"""Diagnostic v2 — investigate why Flow360 grows chi on the lower surface
at high α while mfoil predicts no growth.

Strategy:
  1. mfoil provides ground-truth H(x), Re_θ(x), N(x) on the laminar BL.
  2. Flow360 provides λ_p at the kernel-active wall distance, plus χ_max.
  3. For each x station on the lower surface, plot:
        H (mfoil), Re_θ (mfoil), λ_p (Flow360), χ_max (Flow360)
  4. Then make an (H, λ_p) phase plot colored by Re_θ to see if the
     mapping is Reynolds-independent.
  5. Overlay Drela's Re_θ_crit(H_k) curve from the e^N method.

λ_p extraction: use the SAME formula as the SA-AI kernel:
    λ_p = -d² · (u · ∇p) / (ρ ν |u|²)
computed at the local point at wall distance d_active, where d_active is
chosen so Re_Ω = d² |ω|/ν ≈ 200 (the kernel-active band).
"""
import os, sys, pickle, numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/qiqi/flexcompute/aft-sa/paper')
from regen_nlf_v2 import walk_contour_xz, load_slice_derived

B = "/home/qiqi/flexcompute/aft-sa/flow360"
NU_UNIT = 1.0 / 4e6    # ν in chord units for Re=4M
MACH = 0.1             # Flow360 stores u/c_∞

CASES = {
    'cavL1_a9':  f"{B}/cavL1prop_nlf0416_Re4M_a9",
    'cavL2_a9':  f"{B}/cavL2prop_nlf0416_Re4M_a9",
    'strL1_a9':  f"{B}/strL1prop_nlf0416_Re4M_a9",
    'strL2_a9':  f"{B}/strL2prop_nlf0416_Re4M_a9",
}

def add_grad_p(slice_g):
    """Run a VTK gradient filter on the slice to get ∇p."""
    gf = vtk.vtkGradientFilter()
    gf.SetInputData(slice_g)
    gf.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 'p')
    gf.SetResultArrayName('grad_p')
    gf.Update()
    return gf.GetOutput()

def slice_y_plane(g):
    p = vtk_to_numpy(g.GetPoints().GetData())
    return float(np.median(p[:,1]))

def probe_wallnormal(slice_g, x0, z0, nx, nz, L=0.005, n_probe=200):
    """Probe wall-normal profile at one surface point.

    Returns dict of arrays sampled along the probe line (NaN where outside mesh)."""
    y0 = slice_y_plane(slice_g)
    dists = np.linspace(1e-7, L, n_probe)
    pts_arr = np.empty((n_probe, 3))
    pts_arr[:,0] = x0 + dists * nx
    pts_arr[:,1] = y0
    pts_arr[:,2] = z0 + dists * nz
    vpts = vtk.vtkPoints()
    vpts.SetData(numpy_to_vtk(pts_arr, deep=True))
    poly = vtk.vtkPolyData(); poly.SetPoints(vpts)
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(poly); probe.SetSourceData(slice_g); probe.Update()
    out = probe.GetOutput(); pd = out.GetPointData()
    valid = vtk_to_numpy(probe.GetValidPoints())
    mask = np.zeros(n_probe, bool); mask[valid] = True
    def fld(name, default_dim=1):
        a = pd.GetArray(name)
        if a is None:
            return np.full((n_probe,) if default_dim == 1 else (n_probe, default_dim), np.nan)
        arr = vtk_to_numpy(a)
        if arr.ndim == 1:
            return np.where(mask, arr, np.nan)
        out = arr.copy(); out[~mask] = np.nan
        return out
    return {
        'd': dists, 'mask': mask,
        'velocity': fld('velocity', 3),
        'p': fld('p'),
        'rho': fld('rho'),
        'omega': fld('vorticityMagnitude'),
        'nuhat': fld('nuHat'),
        'grad_p': fld('grad_p', 3),
        'Re_Omega': fld('Re_Omega'),
        'Gamma': fld('Gamma'),
    }

def kernel_active_idx(d, omega):
    """Index of probe sample where Re_Ω = d²|ω|/ν is maximum in the BL.
    That is the cell where the SA-AI amplification kernel is most active."""
    Re_O = d**2 * np.abs(omega) / NU_UNIT
    valid = np.isfinite(Re_O) & (d > 1e-6)
    if valid.sum() < 3: return None
    # Mask invalid as -inf so argmax ignores them
    Re_O_m = np.where(valid, Re_O, -np.inf)
    return int(np.argmax(Re_O_m))

def analyze(name, cd, side='lower', x_stations=None):
    if x_stations is None:
        # Dense sampling in the LAMINAR / transitioning region (x<0.5)
        x_stations = np.concatenate([
            np.linspace(0.05, 0.50, 31),
            np.linspace(0.55, 0.95, 9)
        ])
    Xm, Zm, up_idx, lo_idx = walk_contour_xz(cd)
    idx = lo_idx if side == 'lower' else up_idx
    xs = Xm[idx]; zs = Zm[idx]
    tx_raw = np.gradient(xs); tz_raw = np.gradient(zs)
    s = np.sqrt(tx_raw**2 + tz_raw**2) + 1e-30
    tx = tx_raw / s; tz = tz_raw / s
    nx = tz; nz = -tx
    if side == 'upper' and np.mean(nz) < 0: nx, nz = -nx, -nz
    elif side == 'lower' and np.mean(nz) > 0: nx, nz = -nx, -nz
    slice_g = add_grad_p(load_slice_derived(cd))
    results = []
    for x_t in x_stations:
        i = int(np.argmin(np.abs(xs - x_t)))
        prof = probe_wallnormal(slice_g, xs[i], zs[i], nx[i], nz[i], L=0.005, n_probe=200)
        ka = kernel_active_idx(prof['d'], prof['omega'])
        if ka is None: continue
        d_a = prof['d'][ka]
        vel = prof['velocity'][ka]  # (vx, vy, vz) at this point, in u/c_∞ units
        if not np.isfinite(vel).all(): continue
        u_mag_sq = vel[0]**2 + vel[2]**2 + 1e-12  # in (u/c)² ≡ M² when at U_∞
        gp = prof['grad_p'][ka]
        if not np.isfinite(gp).all(): continue
        rho = prof['rho'][ka]
        # λ_p = -d² · (u · ∇p) / (ρ ν |u|²)   — same formula as kernel
        u_dot_gp = vel[0] * gp[0] + vel[2] * gp[2]
        lambda_p = -d_a**2 * u_dot_gp / (rho * NU_UNIT * u_mag_sq)
        chi_max = np.nanmax(prof['nuhat'] / NU_UNIT) if np.isfinite(prof['nuhat']).any() else np.nan
        Re_O_peak = float(d_a**2 * np.abs(prof['omega'][ka]) / NU_UNIT)
        results.append({
            'x': xs[i], 'd_active': float(d_a), 'lambda_p': float(lambda_p),
            'chi_max': float(chi_max), 'Re_Omega_peak': Re_O_peak,
            'U_mag_local_Minf': float(np.sqrt(u_mag_sq)),
        })
    return results

if __name__ == '__main__':
    with open(f"{B}/mfoil_nlf0416_Re4M.pkl", 'rb') as f:
        mn = pickle.load(f)
    a = 9.0
    mlo = mn[a]['lower']
    xtr_lo = mn[a]['xtr_lower']
    print(f"mfoil α={a} lower: xtr={xtr_lo:.3f}")
    # Sort mfoil arrays by x
    order = np.argsort(mlo['x'])
    x_m = np.asarray(mlo['x'])[order]
    H_m = np.asarray(mlo['H'])[order]
    Reth_m = np.asarray(mlo['Re_theta'])[order]
    n_m = np.asarray(mlo['n'])[order]
    turb_m = np.asarray(mlo['turb'])[order]

    # Run Flow360 analysis on each case
    f360 = {}
    for name, cd in CASES.items():
        if not os.path.exists(f"{cd}/slice_with_derived.pvtu"):
            print(f"SKIP {name}"); continue
        print(f"analyzing {name}...")
        f360[name] = analyze(name, cd, side='lower')

    # ===== Multi-panel along x =====
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colors = {'cavL1_a9':'C0','cavL2_a9':'C1','strL1_a9':'C2','strL2_a9':'C3'}
    # Top-left: mfoil H along lower surface
    axes[0,0].plot(x_m, H_m, 'k-', lw=1.5, label='mfoil H')
    axes[0,0].axvline(xtr_lo, color='k', ls='--', alpha=0.6, label=f'$x_{{tr}}$={xtr_lo:.2f}')
    axes[0,0].axhline(2.59, color='red', ls=':', alpha=0.5, label='Blasius H=2.59')
    axes[0,0].set_ylabel('mfoil $H$ (laminar BL)')
    axes[0,0].set_ylim(2.0, 4.0); axes[0,0].legend(fontsize=8); axes[0,0].grid(alpha=0.3)
    # Top-right: mfoil Re_θ + Drela's Re_θ_crit(H)
    axes[0,1].semilogy(x_m, Reth_m, 'k-', lw=1.5, label='mfoil $Re_\\theta$')
    def drela_Reth_crit(Hk):
        return 10**((1.415/(Hk-1) - 0.489)*np.tanh(20/(Hk-1) - 12.9) + 3.295/(Hk-1) + 0.44)
    valid = (H_m > 2.1) & (H_m < 6) & ~np.asarray(turb_m, bool)
    Re_crit = np.where(valid, drela_Reth_crit(np.clip(H_m, 2.1, 6.0)), np.nan)
    axes[0,1].semilogy(x_m, Re_crit, 'k:', lw=1.2, label='Drela $Re_{\\theta,crit}(H)$')
    axes[0,1].axvline(xtr_lo, color='k', ls='--', alpha=0.6)
    axes[0,1].set_ylabel('$Re_\\theta$')
    axes[0,1].set_ylim(100, 1e4); axes[0,1].legend(fontsize=8); axes[0,1].grid(alpha=0.3, which='both')
    # Bottom-left: Flow360 λ_p along lower surface
    for name, res in f360.items():
        if not res: continue
        x = [r['x'] for r in res]; lp = [r['lambda_p'] for r in res]
        axes[1,0].plot(x, lp, '-o', color=colors[name], label=name, ms=3, lw=1)
    axes[1,0].axvline(xtr_lo, color='k', ls='--', alpha=0.6)
    axes[1,0].axhline(0, color='gray', ls=':', alpha=0.5)
    axes[1,0].set_ylabel('Flow360 $\\lambda_p$ (at $Re_\\Omega=200$)')
    axes[1,0].legend(fontsize=8); axes[1,0].grid(alpha=0.3)
    # Bottom-right: Flow360 χ_max (the actual amplification result)
    for name, res in f360.items():
        if not res: continue
        x = [r['x'] for r in res]; chi = [r['chi_max'] for r in res]
        axes[1,1].semilogy(x, chi, '-o', color=colors[name], label=name, ms=3, lw=1)
    axes[1,1].axvline(xtr_lo, color='k', ls='--', alpha=0.6)
    axes[1,1].set_ylabel('Flow360 $\\chi_{max}$ in BL')
    axes[1,1].set_ylim(1e-4, 1e2); axes[1,1].legend(fontsize=8); axes[1,1].grid(alpha=0.3, which='both')
    for ax in axes[-1,:]: ax.set_xlabel('x/c')
    for ax in axes.flat: ax.set_xlim(0, 1)
    fig.suptitle(f'NLF α={a:.0f} lower surface — Flow360 vs mfoil ($x_{{tr,mfoil}}={xtr_lo:.2f}$)')
    fig.tight_layout()
    fig.savefig('/tmp/diag_lower_v2_alongx.png', dpi=120)
    print('wrote /tmp/diag_lower_v2_alongx.png')

    # ===== (H, λ_p) phase plot — colored by Re_θ =====
    fig2, axp = plt.subplots(1, 2, figsize=(12, 5))
    # Interpolate mfoil H, Re_θ onto Flow360 x stations for each case
    for name, res in f360.items():
        if not res: continue
        x_f = np.array([r['x'] for r in res])
        lp_f = np.array([r['lambda_p'] for r in res])
        chi_f = np.array([r['chi_max'] for r in res])
        H_at_f = np.interp(x_f, x_m, H_m)
        Reth_at_f = np.interp(x_f, x_m, Reth_m)
        sc = axp[0].scatter(H_at_f, lp_f, c=np.log10(np.maximum(Reth_at_f, 1)),
                             vmin=2.0, vmax=3.7, s=40, cmap='viridis',
                             edgecolors=colors[name], linewidths=1.0, label=name)
        axp[1].scatter(H_at_f, lp_f, c=np.log10(np.maximum(chi_f, 1e-4)),
                       vmin=-4, vmax=2, s=40, cmap='coolwarm',
                       edgecolors=colors[name], linewidths=1.0, label=name)
    for ax in axp:
        ax.axhline(0, color='gray', ls=':', alpha=0.5)
        ax.axvline(2.59, color='red', ls='--', alpha=0.5)
        ax.set_xlabel('$H$ (from mfoil)')
        ax.set_ylabel('$\\lambda_p$ (Flow360)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='lower left')
    cb1 = plt.colorbar(axp[0].collections[0], ax=axp[0])
    cb1.set_label('$\\log_{10} Re_\\theta$')
    cb2 = plt.colorbar(axp[1].collections[0], ax=axp[1])
    cb2.set_label('$\\log_{10}\\chi_{max}$')
    axp[0].set_title('phase $(H, \\lambda_p)$ colored by $Re_\\theta$')
    axp[1].set_title('phase $(H, \\lambda_p)$ colored by $\\chi_{max}$')
    fig2.suptitle(f'α={a:.0f} lower-surface phase map — is $(H \\leftrightarrow \\lambda_p)$ Re-independent?')
    fig2.tight_layout()
    fig2.savefig('/tmp/diag_lower_v2_phase.png', dpi=120)
    print('wrote /tmp/diag_lower_v2_phase.png')
