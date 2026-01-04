#!/usr/bin/env python
"""
Debug SA turbulence model terms at specific chord locations.

Dumps all terms in the SA equation:
- Production: P = cb1 * S_tilde * nuHat
- Destruction: D = cw1 * fw * (nuHat/d)^2
- cb2 term: (cb2/sigma) * |grad_nuHat|^2
- Diffusion: (1/sigma) * div((nu + nuHat) * grad_nuHat)
- Transport: u * d(nuHat)/dx + v * d(nuHat)/dy

Compares against expected values for a Spalding profile and
computes boundary layer integral quantities (theta, delta*).
"""

import json
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import load_yaml as load_config
from src.grid.loader import load_or_generate_grid
from src.solvers.rans_solver import RANSSolver
from src.constants import NGHOST


# SA model constants
CB1 = 0.1355
CB2 = 0.622
SIGMA = 2.0 / 3.0
KAPPA = 0.41
CV1 = 7.1
CW1 = CB1 / (KAPPA ** 2) + (1.0 + CB2) / SIGMA
CW2 = 0.3
CW3 = 2.0


def spalding_u_plus(y_plus, kappa=0.41, B=5.0, max_iter=50):
    """Compute u+ from y+ using Spalding's law (inverted numerically)."""
    u_plus = np.zeros_like(y_plus, dtype=float)
    for i, yp in enumerate(y_plus):
        if yp < 1e-10:
            u_plus[i] = yp
            continue
        up = min(yp, np.sqrt(yp))  # Initial guess
        for _ in range(max_iter):
            e_term = np.exp(-kappa * B)
            kup = kappa * up
            exp_kup = np.exp(min(kup, 50))  # Prevent overflow
            f = up + e_term * (exp_kup - 1 - kup - 0.5*kup**2 - (1/6)*kup**3) - yp
            df = 1 + e_term * kappa * (exp_kup - 1 - kup - 0.5*kup**2)
            if abs(df) < 1e-14:
                break
            up_new = up - f / df
            if abs(up_new - up) < 1e-10:
                up = up_new
                break
            up = max(0, up_new)
        u_plus[i] = up
    return u_plus


def compute_sa_functions(chi):
    """Compute SA model functions fv1, fv2, fw given chi = nuHat/nu_laminar."""
    chi = np.maximum(chi, 1e-10)
    chi3 = chi ** 3
    fv1 = chi3 / (chi3 + CV1 ** 3)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)
    return fv1, fv2


def compute_fw(r):
    """Compute fw function."""
    r = np.clip(r, 0, 10)
    g = r + CW2 * (r**6 - r)
    c6 = CW3 ** 6
    fw = g * ((1.0 + c6) / (g**6 + c6)) ** (1.0/6.0)
    return fw


def compute_bl_integrals(y, u, u_edge):
    """
    Compute boundary layer integral quantities.
    
    Parameters
    ----------
    y : array
        Wall-normal distance
    u : array
        Streamwise velocity
    u_edge : float
        Edge velocity
        
    Returns
    -------
    delta_star : float
        Displacement thickness
    theta : float
        Momentum thickness
    H : float
        Shape factor H = delta*/theta
    """
    # Displacement thickness: delta* = integral(1 - u/u_e) dy
    integrand_ds = 1.0 - u / u_edge
    delta_star = np.trapz(integrand_ds, y)
    
    # Momentum thickness: theta = integral(u/u_e * (1 - u/u_e)) dy
    integrand_th = (u / u_edge) * (1.0 - u / u_edge)
    theta = np.trapz(integrand_th, y)
    
    # Shape factor
    H = delta_star / theta if theta > 0 else 0
    
    return delta_star, theta, H


def extract_sa_terms(solver, i_idx):
    """
    Extract all SA equation terms at a given i-index.
    """
    metrics = solver.metrics
    xc = metrics.xc
    yc = metrics.yc
    vol = metrics.volume
    Sj_x = metrics.Sj_x
    Sj_y = metrics.Sj_y
    wall_dist = np.asarray(solver.wall_dist_jax)
    
    Q = np.asarray(solver.Q_jax)
    NI = Q.shape[0] - 2 * NGHOST
    NJ = Q.shape[1] - 2 * NGHOST
    
    # Interior solution at this i
    p = Q[NGHOST + i_idx, NGHOST:NGHOST+NJ, 0]
    u = Q[NGHOST + i_idx, NGHOST:NGHOST+NJ, 1]
    v = Q[NGHOST + i_idx, NGHOST:NGHOST+NJ, 2]
    nuHat = Q[NGHOST + i_idx, NGHOST:NGHOST+NJ, 3]
    
    # Wall distance for this i-line
    d = wall_dist[i_idx, :]
    
    # Physical quantities
    nu_laminar = 1.0 / solver.config.reynolds
    V_inf = np.sqrt(solver.freestream.u_inf**2 + solver.freestream.v_inf**2)
    
    # Chi = nuHat / nu_laminar
    chi = nuHat / nu_laminar
    
    # SA functions
    fv1, fv2 = compute_sa_functions(chi)
    
    # Effective viscosities
    mu_t = nuHat * fv1  # Turbulent viscosity (kinematic)
    nu_eff = nu_laminar + np.maximum(nuHat, 0)  # For SA diffusion
    
    # Compute wall normal direction from face normal at wall
    Sx = Sj_x[i_idx, 0]
    Sy = Sj_y[i_idx, 0]
    S_mag = np.sqrt(Sx**2 + Sy**2)
    nx = Sx / (S_mag + 1e-14)
    ny = Sy / (S_mag + 1e-14)
    
    # Tangential direction
    tx = -ny
    ty = nx
    
    # Tangential velocity
    u_tan = u * tx + v * ty
    if np.mean(u_tan) < 0:
        tx, ty = -tx, -ty
        u_tan = -u_tan
    
    vel_mag = np.sqrt(u**2 + v**2)
    
    # Wall distance for each cell
    y_dist = d.copy()
    
    # Estimate gradients using finite differences
    # du/dy (wall-normal gradient of streamwise velocity)
    dudy = np.zeros_like(u)
    dnuHat_dy = np.zeros_like(nuHat)
    for j in range(1, len(u)-1):
        dy = y_dist[j+1] - y_dist[j-1]
        if dy > 1e-14:
            dudy[j] = (u_tan[j+1] - u_tan[j-1]) / dy
            dnuHat_dy[j] = (nuHat[j+1] - nuHat[j-1]) / dy
    # Boundary values
    if y_dist[0] > 1e-14:
        dudy[0] = u_tan[0] / y_dist[0]  # Linear at wall
        dnuHat_dy[0] = nuHat[0] / y_dist[0]
    dudy[-1] = dudy[-2]
    dnuHat_dy[-1] = dnuHat_dy[-2]
    
    # Vorticity magnitude (approximated as |du/dy| for BL)
    omega = np.abs(dudy)
    
    # S_tilde = omega + (nuHat / kappa^2 / d^2) * fv2
    inv_k2d2 = 1.0 / (KAPPA**2 * d**2 + 1e-20)
    S_tilde = omega + nuHat * inv_k2d2 * fv2
    S_tilde = np.maximum(S_tilde, 1e-16)
    
    # Production: P = cb1 * S_tilde * nuHat
    P = CB1 * S_tilde * nuHat
    
    # r and fw for destruction
    r = nuHat / (S_tilde * KAPPA**2 * d**2 + 1e-20)
    r = np.clip(r, 0, 10)
    fw = compute_fw(r)
    
    # Destruction: D = cw1 * fw * (nuHat/d)^2
    D = CW1 * fw * (nuHat / d)**2
    
    # cb2 term: (cb2/sigma) * |grad_nuHat|^2
    # Approximating grad_nuHat magnitude as |dnuHat/dy| (dominant in BL)
    grad_nuHat_sq = dnuHat_dy**2
    cb2_term = (CB2 / SIGMA) * grad_nuHat_sq
    
    # Wall shear for Cf
    vol_wall = vol[i_idx, 0]
    area_wall = np.sqrt(Sj_x[i_idx, 0]**2 + Sj_y[i_idx, 0]**2)
    dy_wall = vol_wall / (area_wall + 1e-14) / 2.0  # Wall-to-center distance
    du_dy_wall = u_tan[0] / dy_wall
    mu_wall = nu_laminar + np.maximum(nuHat[0], 0) * fv1[0]
    tau_wall = mu_wall * du_dy_wall
    u_tau = np.sqrt(np.abs(tau_wall))
    q_inf = 0.5 * V_inf**2
    Cf_local = tau_wall / q_inf
    
    # y+ and u+
    y_plus = y_dist * u_tau / nu_laminar
    u_plus = u_tan / (u_tau + 1e-14)
    u_plus_spalding = spalding_u_plus(y_plus)
    
    # Boundary layer integrals
    # Find edge velocity (take value at y+ ~ 500 or last point)
    j_edge = np.argmax(y_plus > 500) if np.any(y_plus > 500) else len(y_plus) - 1
    u_edge = u_tan[j_edge] if j_edge > 0 else u_tan[-1]
    delta_star, theta, H = compute_bl_integrals(y_dist[:j_edge+1], u_tan[:j_edge+1], u_edge)
    
    # Re_theta = theta * u_edge / nu_laminar
    Re_theta = theta * u_edge / nu_laminar
    
    # Expected chi for equilibrium turbulent BL
    # In log layer: chi ~ kappa * y+ for fully turbulent
    chi_expected = KAPPA * y_plus  # Simple estimate
    
    return {
        'i_idx': int(i_idx),
        'x_chord': float(xc[i_idx, 0]),
        'nu_laminar': float(nu_laminar),
        'V_inf': float(V_inf),
        'u_tau': float(u_tau),
        'Cf': float(Cf_local),
        # BL integrals
        'delta_star': float(delta_star),
        'theta': float(theta),
        'H': float(H),
        'Re_theta': float(Re_theta),
        # Profiles
        'y_dist': y_dist.tolist(),
        'y_plus': y_plus.tolist(),
        'u_tan': u_tan.tolist(),
        'u_plus': u_plus.tolist(),
        'u_plus_spalding': u_plus_spalding.tolist(),
        'nuHat': nuHat.tolist(),
        'chi': chi.tolist(),
        'chi_expected': chi_expected.tolist(),
        # SA functions
        'fv1': fv1.tolist(),
        'fv2': fv2.tolist(),
        'fw': fw.tolist(),
        'r': r.tolist(),
        # SA terms (per unit volume)
        'omega': omega.tolist(),
        'S_tilde': S_tilde.tolist(),
        'Production': P.tolist(),
        'Destruction': D.tolist(),
        'cb2_term': cb2_term.tolist(),
        'P_minus_D': (P - D).tolist(),
        'net_source': (P - D + cb2_term).tolist(),
    }


def find_chord_indices(metrics, chord_fracs, n_wake):
    """Find i-indices for chord locations on upper/lower surfaces."""
    xc = metrics.xc
    NI = xc.shape[0]
    x_surface = xc[:, 0]
    le_idx = np.argmin(x_surface[n_wake:NI-n_wake]) + n_wake
    
    result = {}
    for cf in chord_fracs:
        # Upper surface (from LE going down in i)
        for i in range(le_idx, n_wake, -1):
            if x_surface[i] >= cf:
                result[f'upper_{cf}'] = i
                break
        # Lower surface (from LE going up in i)
        for i in range(le_idx, NI - n_wake):
            if x_surface[i] >= cf:
                result[f'lower_{cf}'] = i
                break
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Debug SA terms in boundary layer')
    parser.add_argument('--config', default='config/examples/naca0012_re1m.yaml')
    parser.add_argument('--max-iter', type=int, default=30000)
    parser.add_argument('--output', default='sa_terms_debug.json')
    args = parser.parse_args()
    
    # Load config and grid
    sim_config = load_config(args.config)
    sim_config.solver.max_iter = args.max_iter
    config = sim_config.to_solver_config()
    
    X, Y = load_or_generate_grid(
        sim_config.grid.airfoil,
        n_surface=sim_config.grid.n_surface,
        n_normal=sim_config.grid.n_normal,
        n_wake=sim_config.grid.n_wake,
        y_plus=sim_config.grid.y_plus,
        reynolds=sim_config.flow.reynolds,
        verbose=True,
    )
    n_wake = sim_config.grid.n_wake
    
    # Create and run solver
    from collections import deque
    solver = RANSSolver.__new__(RANSSolver)
    solver.config = config
    solver.X = X
    solver.Y = Y
    solver.NI = X.shape[0] - 1
    solver.NJ = X.shape[1] - 1
    solver.iteration = 0
    solver.residual_history = []
    solver.iteration_history = []
    solver.converged = False
    solver._divergence_buffer = None
    solver._compute_metrics()
    solver._initialize_state()
    solver._initialize_output()
    
    print(f"\nRunning {args.max_iter} iterations...")
    solver.run_steady_state()
    
    # Extract SA terms
    chord_fracs = [0.05, 0.25, 0.50, 0.75]
    indices = find_chord_indices(solver.metrics, chord_fracs, n_wake)
    
    print(f"\nExtracting SA terms at chord locations...")
    profiles = {}
    for name, i_idx in indices.items():
        print(f"  {name} (i={i_idx})...")
        profiles[name] = extract_sa_terms(solver, i_idx)
    
    # Save to JSON
    output = {
        'config': {
            'reynolds': config.reynolds,
            'alpha': config.alpha,
            'max_iter': args.max_iter,
        },
        'profiles': profiles,
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {args.output}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SA TERMS DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    for name, prof in profiles.items():
        print(f"\n{name} (x/c = {prof['x_chord']:.3f}):")
        print(f"  BL integrals: theta={prof['theta']:.6f}, delta*={prof['delta_star']:.6f}, H={prof['H']:.2f}")
        print(f"  Re_theta = {prof['Re_theta']:.0f}")
        print(f"  Cf = {prof['Cf']:.5f} (expected ~0.005 for turbulent)")
        
        y_plus = np.array(prof['y_plus'])
        chi = np.array(prof['chi'])
        P = np.array(prof['Production'])
        D = np.array(prof['Destruction'])
        cb2 = np.array(prof['cb2_term'])
        
        # Log layer stats (y+ ~ 30-100)
        mask = (y_plus > 30) & (y_plus < 100)
        if np.any(mask):
            print(f"  Log layer (30 < y+ < 100):")
            print(f"    chi: mean={chi[mask].mean():.1f}, max={chi[mask].max():.1f}")
            print(f"    P/D ratio: mean={(P[mask]/(D[mask]+1e-30)).mean():.2f}")
            print(f"    cb2/P ratio: mean={(cb2[mask]/(P[mask]+1e-30)).mean():.2f}")
        
        # Near wall stats (y+ ~ 1-10)
        mask_wall = (y_plus > 1) & (y_plus < 10)
        if np.any(mask_wall):
            print(f"  Near wall (1 < y+ < 10):")
            print(f"    chi: mean={chi[mask_wall].mean():.2f}")
            print(f"    P/D ratio: mean={(P[mask_wall]/(D[mask_wall]+1e-30)).mean():.2f}")
    
    # Print table for one location
    print("\n" + "-" * 80)
    print("Detailed SA terms at upper_0.25:")
    print("-" * 80)
    prof = profiles.get('upper_0.25', list(profiles.values())[0])
    y_plus = np.array(prof['y_plus'])
    chi = np.array(prof['chi'])
    chi_exp = np.array(prof['chi_expected'])
    P = np.array(prof['Production'])
    D = np.array(prof['Destruction'])
    cb2 = np.array(prof['cb2_term'])
    fv2 = np.array(prof['fv2'])
    
    print(f"{'j':>3} | {'y+':>8} | {'chi':>8} | {'chi_exp':>8} | {'P':>10} | {'D':>10} | {'P-D':>10} | {'cb2':>10} | {'fv2':>6}")
    print("-" * 95)
    for j in range(min(25, len(y_plus))):
        if y_plus[j] > 300:
            break
        print(f"{j:3d} | {y_plus[j]:8.1f} | {chi[j]:8.2f} | {chi_exp[j]:8.2f} | {P[j]:10.2e} | {D[j]:10.2e} | {P[j]-D[j]:10.2e} | {cb2[j]:10.2e} | {fv2[j]:6.2f}")


if __name__ == '__main__':
    main()
