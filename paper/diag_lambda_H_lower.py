"""Diagnostic: on the NLF α=9 lower surface (mfoil predicts laminar until
x=0.66 but our model lets chi grow), extract along the BL:
   - λ_p (local PG sensor used by σ_FPG / FPG-cliff)
   - H = δ*/θ (shape factor used by Drela's Re_θ_crit(H))
   - Re_θ
Then plot λ_p vs H, colored by Re_θ, to see if the (λ_p, H) mapping is
Reynolds-independent. If yes, we can directly translate Drela's H-cutoff
to a λ_p-cutoff. If not, the cliff must depend on the triplet.
"""
import os, sys, pickle, numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/qiqi/flexcompute/aft-sa/paper')
from regen_nlf_v2 import walk_contour_xz, slice_y_plane, load_slice_derived

B = "/home/qiqi/flexcompute/aft-sa/flow360"
NU = 1.0 / 4e6   # Re=4M, ν=1/Re in chord units
MACH = 0.1

# Cases to analyze (α=9 lower surface is the main target; α=15 for cross-check)
CASES = {
    'cavL1_a9': f"{B}/cavL1prop_nlf0416_Re4M_a9",
    'cavL2_a9': f"{B}/cavL2prop_nlf0416_Re4M_a9",
    'strL1_a9': f"{B}/strL1prop_nlf0416_Re4M_a9",
    'strL2_a9': f"{B}/strL2prop_nlf0416_Re4M_a9",
}

def extract_profile_at(slice_g, x_target, z_target, nx, nz, L=0.05, n_probe=200):
    """Sample a wall-normal profile at (x,z) along outward direction (nx,nz).
    Returns (d, u_t, p, omega, nuhat, Re_O) along the line."""
    y0 = slice_y_plane_from_g(slice_g)
    dists = np.linspace(1e-7, L, n_probe)
    pts_arr = np.empty((n_probe, 3))
    pts_arr[:,0] = x_target + dists*nx
    pts_arr[:,1] = y0
    pts_arr[:,2] = z_target + dists*nz
    vpts = vtk.vtkPoints(); vpts.SetData(vtk.util.numpy_support.numpy_to_vtk(pts_arr, deep=True))
    poly = vtk.vtkPolyData(); poly.SetPoints(vpts)
    probe = vtk.vtkProbeFilter()
    probe.SetInputData(poly); probe.SetSourceData(slice_g); probe.Update()
    out = probe.GetOutput(); pd = out.GetPointData()
    valid = vtk_to_numpy(probe.GetValidPoints())
    mask = np.zeros(n_probe, bool); mask[valid] = True
    def fld(name):
        a = pd.GetArray(name)
        if a is None: return np.full(n_probe, np.nan)
        return np.where(mask, vtk_to_numpy(a), np.nan)
    vel = pd.GetArray('velocity')
    if vel is None:
        return None
    vel_arr = np.where(mask[:,None], vtk_to_numpy(vel), np.nan)
    # tangential velocity = projection of velocity onto tangent direction (-nz, nx) (perp to outward normal)
    tx, tz = -nz, nx
    u_t = vel_arr[:,0]*tx + vel_arr[:,2]*tz
    p = fld('p')
    omega = fld('vorticityMagnitude')
    nuhat = fld('nuHat')
    return dists, u_t, p, omega, nuhat

def slice_y_plane_from_g(g):
    p = vtk_to_numpy(g.GetPoints().GetData())
    return float(np.median(p[:,1]))

def compute_bl_integrals(d, u_t, U_e=None):
    """θ = ∫ (u/U_e)(1-u/U_e) dd, δ* = ∫ (1-u/U_e) dd, H = δ*/θ.
    Edge velocity U_e = max |u_t| in the BL (a robust definition that
    handles BL-edge oscillation)."""
    valid = np.isfinite(u_t) & (d > 0)
    if valid.sum() < 5: return np.nan, np.nan, np.nan
    d_v = d[valid]; u_v = np.abs(u_t[valid])
    if U_e is None:
        # Use 99% of the asymptotic tail value
        # Take median of upper half of probe samples
        U_e = np.median(u_v[len(u_v)//2:])
    if U_e < 1e-6: return np.nan, np.nan, np.nan
    u_norm = np.clip(u_v / U_e, 0.0, 1.5)
    # Clip integration to where u_norm crosses 0.99 (BL edge)
    idx_edge = np.argmax(u_norm >= 0.99) if (u_norm >= 0.99).any() else len(u_norm)-1
    if idx_edge < 3: return np.nan, np.nan, np.nan
    d_int = d_v[:idx_edge+1]; u_int = u_norm[:idx_edge+1]
    theta = np.trapezoid(u_int * (1.0 - u_int), d_int)
    delta_star = np.trapezoid(1.0 - u_int, d_int)
    H = delta_star / theta if theta > 0 else np.nan
    return theta, delta_star, H

def find_kernel_active_d(d, omega, target_ReO=200.0):
    """Find wall distance where Re_Ω = d²·|ω|/ν ≈ target. Returns that d."""
    Re_O = d**2 * np.abs(omega) / NU
    valid = np.isfinite(Re_O)
    if valid.sum() < 3: return np.nan
    d_v, R_v = d[valid], Re_O[valid]
    # Find first crossing of target
    cross = np.where(R_v > target_ReO)[0]
    if len(cross) == 0: return d_v[-1]  # never reaches target
    return d_v[cross[0]]

def analyze_case(name, cd, side='lower'):
    """Walk lower surface, compute BL integrals, λ_p at kernel-active band."""
    Xm, Zm, up_idx, lo_idx = walk_contour_xz(cd)
    idx = lo_idx if side == 'lower' else up_idx
    xs = Xm[idx]; zs = Zm[idx]
    # Tangent + normal via central diffs
    tx_raw = np.gradient(xs); tz_raw = np.gradient(zs)
    s = np.sqrt(tx_raw**2 + tz_raw**2) + 1e-30
    tx = tx_raw / s; tz = tz_raw / s
    # CW walk → right perp = (tz, -tx) is outward
    nx_arr = tz; nz_arr = -tx
    # Fix orientation per side
    if side == 'upper' and np.mean(nz_arr) < 0:
        nx_arr, nz_arr = -nx_arr, -nz_arr
    elif side == 'lower' and np.mean(nz_arr) > 0:
        nx_arr, nz_arr = -nx_arr, -nz_arr
    slice_g = load_slice_derived(cd)

    # Sample at evenly-spaced x stations across the lower surface
    target_xs = np.linspace(0.05, 0.95, 19)
    results = []
    for x_t in target_xs:
        # Find the surface point closest to x_t
        i = int(np.argmin(np.abs(xs - x_t)))
        x0, z0, nx0, nz0 = xs[i], zs[i], nx_arr[i], nz_arr[i]
        out = extract_profile_at(slice_g, x0, z0, nx0, nz0, L=0.05, n_probe=200)
        if out is None: continue
        d, u_t, p, omega, nuhat = out
        # Edge velocity from velocity magnitude tail
        valid = np.isfinite(u_t) & (d > 1e-5)
        if valid.sum() < 50: continue
        # Use velocity at the far end as U_e (already in u/c, convert to u/U_∞)
        U_e_stored = np.nanmedian(u_t[len(u_t)*3//4:])
        if abs(U_e_stored) < 1e-6: continue
        U_e_normalized = abs(U_e_stored) / MACH  # → 0..~1 in U_∞ units
        # BL integrals (normalize d to chord; u_t/U_e_stored is dimensionless)
        theta, delta_s, H = compute_bl_integrals(d, u_t / U_e_stored if U_e_stored > 0 else -u_t / abs(U_e_stored))
        if np.isnan(H): continue
        Re_th = U_e_normalized * theta / NU
        # λ_p at the kernel-active wall distance (d where Re_Ω ≈ 200)
        d_active = find_kernel_active_d(d, omega, target_ReO=200.0)
        if np.isnan(d_active): continue
        # Sample p, omega, velocity at d_active
        valid_d = np.isfinite(d) & np.isfinite(p)
        if valid_d.sum() < 5: continue
        i_a = int(np.argmin(np.abs(d - d_active)))
        # Local λ_p: -(d_active)² (u·∇p)/(ρν|u|²)
        # Use centered finite difference along streamwise direction in slice plane
        # Simpler: use Bernoulli surrogate λ_p ≈ d² · (dU/dx) / ν  with U=U_e
        # Compute dU/dx by sampling slice at x±Δ
        dx = 0.02
        out_p = extract_profile_at(slice_g, x0 + dx*tx[i], z0 + dx*tz[i], nx0, nz0, L=0.005, n_probe=20)
        out_m = extract_profile_at(slice_g, x0 - dx*tx[i], z0 - dx*tz[i], nx0, nz0, L=0.005, n_probe=20)
        if out_p is None or out_m is None: continue
        # Edge velocity at x+dx and x-dx
        Ue_p = np.nanmedian(out_p[1][len(out_p[1])*3//4:]) / MACH
        Ue_m = np.nanmedian(out_m[1][len(out_m[1])*3//4:]) / MACH
        if not (np.isfinite(Ue_p) and np.isfinite(Ue_m)): continue
        dUe_dx = (Ue_p - Ue_m) / (2*dx)
        lambda_p = d_active**2 * dUe_dx / NU
        chi_max = np.nanmax(nuhat / NU) if np.isfinite(nuhat).any() else np.nan
        results.append({
            'x': x0, 'H': H, 'theta': theta, 'Re_theta': Re_th,
            'lambda_p': lambda_p, 'd_active': d_active,
            'U_e': U_e_normalized, 'dUe_dx': dUe_dx, 'chi_max': chi_max,
        })
    return results

if __name__ == '__main__':
    # Load mfoil for α=9 lower
    with open(f"{B}/mfoil_nlf0416_Re4M.pkl",'rb') as f: mn = pickle.load(f)
    mlo = mn[9.0]['lower']
    xtr_lo = mn[9.0]['xtr_lower']
    print(f"mfoil α=9 lower: xtr={xtr_lo:.3f}, surface points: {len(mlo['x'])}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 5))
    colors = {'cavL1_a9':'C0','cavL2_a9':'C1','strL1_a9':'C2','strL2_a9':'C3'}

    all_results = {}
    for name, cd in CASES.items():
        if not os.path.exists(f"{cd}/slice_with_derived.pvtu"):
            print(f"SKIP {name}: no slice_with_derived"); continue
        print(f"analyzing {name}...")
        try:
            results = analyze_case(name, cd, side='lower')
        except Exception as e:
            print(f"  failed: {e}"); continue
        all_results[name] = results
        if not results: continue
        x = np.array([r['x'] for r in results])
        H = np.array([r['H'] for r in results])
        Re_th = np.array([r['Re_theta'] for r in results])
        lp = np.array([r['lambda_p'] for r in results])
        chi = np.array([r['chi_max'] for r in results])
        c = colors[name]
        axes[0,0].plot(x, H, '-o', color=c, label=name, ms=3, lw=1)
        axes[0,1].plot(x, Re_th, '-o', color=c, label=name, ms=3, lw=1)
        axes[1,0].plot(x, lp, '-o', color=c, label=name, ms=3, lw=1)
        axes[1,1].semilogy(x, chi, '-o', color=c, label=name, ms=3, lw=1)
        # Scatter for the H ↔ λ_p map
        ax2[0].scatter(H, lp, c=Re_th, vmin=200, vmax=2000, s=30, cmap='viridis',
                       edgecolors=c, linewidths=0.8, label=name)
        # Scatter colored by χ to flag bad cells
        ax2[1].scatter(H, lp, c=np.log10(np.maximum(chi, 1e-4)), vmin=-4, vmax=2,
                       s=30, cmap='coolwarm', edgecolors=c, linewidths=0.8, label=name)

    # Overlay Drela's Re_θ,crit(H_k) trace as a reference
    H_grid = np.linspace(2.2, 4.0, 50)
    # Drela's e^N onset correlation (used in XFOIL/mfoil)
    def drela_Reth_crit(Hk):
        return 10**((1.415/(Hk-1) - 0.489)*np.tanh(20/(Hk-1) - 12.9) + 3.295/(Hk-1) + 0.44)
    axes[0,1].plot(np.full_like(H_grid, 0.5), drela_Reth_crit(H_grid), 'k:', alpha=0.6)  # dummy
    # Show xtr_lo as vertical line
    for ax in axes.flat:
        ax.axvline(xtr_lo, color='gray', ls='--', alpha=0.6, label='mfoil $x_{tr}$')
        ax.grid(alpha=0.3); ax.set_xlim(0, 1)
    axes[0,0].set_ylabel('$H = \\delta^*/\\theta$'); axes[0,0].set_ylim(2.0, 4.5)
    axes[0,1].set_ylabel('$Re_\\theta$')
    axes[0,1].set_yscale('log'); axes[0,1].set_ylim(50, 5e3)
    axes[1,0].set_ylabel('$\\lambda_p$ (at $Re_\\Omega=200$)')
    axes[1,0].axhline(0, color='gray', ls=':', alpha=0.5)
    axes[1,1].set_ylabel('$\\chi_{max}$ in BL'); axes[1,1].set_ylim(1e-4, 1e2)
    for ax in axes[-1,:]: ax.set_xlabel('x/c')
    axes[0,0].legend(fontsize=7, loc='upper left')
    fig.suptitle('NLF α=9 lower surface — Flow360 vs mfoil ($x_{tr}=0.66$)')
    fig.tight_layout()
    fig.savefig('/tmp/diag_lower_H_lambda.png', dpi=120)
    print("wrote /tmp/diag_lower_H_lambda.png")

    # (H, λ_p) phase plot
    ax2[0].set_xlabel('$H$'); ax2[0].set_ylabel('$\\lambda_p$')
    ax2[0].set_title('$(H, \\lambda_p)$ colored by $Re_\\theta$')
    ax2[0].axhline(0, color='gray', ls=':')
    ax2[0].grid(alpha=0.3)
    plt.colorbar(ax2[0].collections[0], ax=ax2[0], label='$Re_\\theta$')
    ax2[1].set_xlabel('$H$'); ax2[1].set_ylabel('$\\lambda_p$')
    ax2[1].set_title('$(H, \\lambda_p)$ colored by $\\log_{10}\\chi_{max}$')
    ax2[1].axhline(0, color='gray', ls=':')
    ax2[1].grid(alpha=0.3)
    plt.colorbar(ax2[1].collections[0], ax=ax2[1], label='$\\log_{10}\\chi$')
    # Mark Blasius
    for ax in ax2:
        ax.axvline(2.59, color='red', ls='--', alpha=0.5)
        ax.text(2.59, ax.get_ylim()[1]*0.9, 'Blasius', color='red', fontsize=8)
    fig2.suptitle('NLF α=9 lower-surface phase: $(H, \\lambda_p)$ at $Re_\\Omega=200$ band')
    fig2.tight_layout()
    fig2.savefig('/tmp/diag_lower_H_lambda_phase.png', dpi=120)
    print("wrote /tmp/diag_lower_H_lambda_phase.png")
