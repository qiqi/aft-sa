#!/usr/bin/env python
"""
Debug SA turbulence model - compute ALL terms in the SA equation.

SA equation (in physical units):
  D(nuHat)/Dt = P - D + (1/sigma)*div((nu+nuHat)*grad(nuHat)) + (cb2/sigma)*|grad(nuHat)|^2

where:
  D(nuHat)/Dt = d(nuHat)/dt + u*d(nuHat)/dx + v*d(nuHat)/dy  (material derivative)
  P = cb1 * S_tilde * nuHat                                   (production)
  D = cw1 * fw * (nuHat/d)^2                                  (destruction)

At steady state: d(nuHat)/dt = 0, so:
  0 = P - D + Diffusion + cb2_term - Convection
  
where Convection = u*d(nuHat)/dx + v*d(nuHat)/dy
"""

import json
import numpy as np
import sys
from pathlib import Path

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


def compute_fw(r):
    r = np.clip(r, 0, 10)
    g = r + CW2 * (r**6 - r)
    c6 = CW3 ** 6
    fw = g * ((1.0 + c6) / (g**6 + c6)) ** (1.0/6.0)
    return fw


def extract_all_sa_terms(solver, i_idx):
    """
    Extract ALL SA equation terms at a given i-index.
    Computes: Production, Destruction, Convection, Diffusion, cb2 term
    """
    metrics = solver.metrics
    xc = metrics.xc
    yc = metrics.yc
    vol = metrics.volume
    Sj_x = metrics.Sj_x
    Sj_y = metrics.Sj_y
    Si_x = metrics.Si_x
    Si_y = metrics.Si_y
    wall_dist = np.asarray(solver.wall_dist_jax)
    
    Q = np.asarray(solver.Q_jax)
    NI = Q.shape[0] - 2 * NGHOST
    NJ = Q.shape[1] - 2 * NGHOST
    
    # Get interior solution (remove ghost cells)
    Q_int = Q[NGHOST:NGHOST+NI, NGHOST:NGHOST+NJ, :]
    
    # Solution at this i-line
    p = Q_int[i_idx, :, 0]
    u = Q_int[i_idx, :, 1]
    v = Q_int[i_idx, :, 2]
    nuHat = Q_int[i_idx, :, 3]
    
    # Neighbors for gradient computation
    if i_idx > 0:
        nuHat_im1 = Q_int[i_idx-1, :, 3]
        u_im1 = Q_int[i_idx-1, :, 1]
        v_im1 = Q_int[i_idx-1, :, 2]
    else:
        nuHat_im1 = nuHat
        u_im1 = u
        v_im1 = v
        
    if i_idx < NI - 1:
        nuHat_ip1 = Q_int[i_idx+1, :, 3]
        u_ip1 = Q_int[i_idx+1, :, 1]
        v_ip1 = Q_int[i_idx+1, :, 2]
    else:
        nuHat_ip1 = nuHat
        u_ip1 = u
        v_ip1 = v
    
    # Wall distance for this i-line
    d = wall_dist[i_idx, :]
    
    # Physical quantities
    nu_lam = 1.0 / solver.config.reynolds
    V_inf = np.sqrt(solver.freestream.u_inf**2 + solver.freestream.v_inf**2)
    
    # Chi and SA functions
    chi = nuHat / nu_lam
    chi3 = chi ** 3
    fv1 = chi3 / (chi3 + CV1**3 + 1e-30)
    fv2 = 1.0 - chi / (1.0 + chi * fv1 + 1e-30)
    
    # Wall normal direction from j-face normal at wall
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
    
    # Wall distance array
    y_dist = d.copy()
    
    # Compute gradients using finite differences
    # d/dy (wall-normal)
    dudy = np.zeros_like(u)
    dnuHat_dy = np.zeros_like(nuHat)
    for j in range(1, len(u)-1):
        dy = y_dist[j+1] - y_dist[j-1]
        if dy > 1e-14:
            dudy[j] = (u_tan[j+1] - u_tan[j-1]) / dy
            dnuHat_dy[j] = (nuHat[j+1] - nuHat[j-1]) / dy
    if y_dist[0] > 1e-14:
        dudy[0] = u_tan[0] / y_dist[0]
        dnuHat_dy[0] = nuHat[0] / y_dist[0]
    dudy[-1] = dudy[-2]
    dnuHat_dy[-1] = dnuHat_dy[-2]
    
    # d/dx (streamwise) - use i-neighbors
    # Need to account for grid metrics
    dx_i = xc[i_idx+1, :] - xc[i_idx-1, :] if i_idx > 0 and i_idx < NI-1 else np.ones_like(xc[i_idx, :]) * 0.01
    dx_i = np.maximum(np.abs(dx_i), 1e-10)
    
    dnuHat_dx = (nuHat_ip1 - nuHat_im1) / (2 * dx_i[:len(nuHat)])
    du_dx = (u_ip1 - u_im1) / (2 * dx_i[:len(u)])
    dv_dx = (v_ip1 - v_im1) / (2 * dx_i[:len(v)])
    
    # Vorticity magnitude
    omega = np.abs(dudy)  # Dominant term in BL
    
    # S_tilde
    inv_k2d2 = 1.0 / (KAPPA**2 * d**2 + 1e-20)
    S_tilde = omega + nuHat * inv_k2d2 * fv2
    S_tilde = np.maximum(S_tilde, 1e-16)
    
    # === PRODUCTION ===
    P = CB1 * S_tilde * nuHat
    
    # === DESTRUCTION ===
    r = nuHat / (S_tilde * KAPPA**2 * d**2 + 1e-20)
    r = np.clip(r, 0, 10)
    fw = compute_fw(r)
    D = CW1 * fw * (nuHat / d)**2
    
    # === CONVECTION: u*dnuHat/dx + v*dnuHat/dy ===
    # v is mostly wall-normal in BL, u is mostly streamwise
    Conv = u * dnuHat_dx + v * dnuHat_dy
    
    # === cb2 term: (cb2/sigma) * |grad(nuHat)|^2 ===
    grad_nuHat_sq = dnuHat_dx**2 + dnuHat_dy**2
    cb2_term = (CB2 / SIGMA) * grad_nuHat_sq
    
    # === DIFFUSION: (1/sigma) * div((nu+nuHat)*grad(nuHat)) ===
    # This is complex - approximate using second derivatives
    nu_eff = nu_lam + np.maximum(nuHat, 0)
    
    # d2(nuHat)/dy2
    d2nuHat_dy2 = np.zeros_like(nuHat)
    for j in range(1, len(nuHat)-1):
        dy = (y_dist[j+1] - y_dist[j-1]) / 2
        if dy > 1e-14:
            d2nuHat_dy2[j] = (nuHat[j+1] - 2*nuHat[j] + nuHat[j-1]) / dy**2
    
    # d2(nuHat)/dx2
    d2nuHat_dx2 = (nuHat_ip1 - 2*nuHat + nuHat_im1) / (dx_i[:len(nuHat)]**2 + 1e-20)
    
    # Simplified diffusion (dominant y-direction term)
    # div((nu+nuHat)*grad(nuHat)) ≈ (nu+nuHat)*laplacian(nuHat) + grad(nu+nuHat)·grad(nuHat)
    # ≈ nu_eff * d2nuHat/dy2 + dnuHat/dy * dnuHat/dy (since d(nu_eff)/dy = dnuHat/dy)
    Diff = (1.0/SIGMA) * (nu_eff * (d2nuHat_dy2 + d2nuHat_dx2) + dnuHat_dy**2 + dnuHat_dx**2)
    
    # Wall shear for u_tau
    vol_wall = vol[i_idx, 0]
    area_wall = np.sqrt(Sj_x[i_idx, 0]**2 + Sj_y[i_idx, 0]**2)
    dy_wall = vol_wall / (area_wall + 1e-14) / 2.0
    du_dy_wall = u_tan[0] / dy_wall
    mu_wall = nu_lam + np.maximum(nuHat[0], 0) * fv1[0]
    tau_wall = mu_wall * du_dy_wall
    u_tau = np.sqrt(np.abs(tau_wall))
    
    # y+ and u+
    y_plus = y_dist * u_tau / nu_lam
    
    # Steady-state balance: P - D + Diff + cb2 - Conv = 0
    # So: Conv = P - D + Diff + cb2 (what Conv should be to balance)
    residual = P - D + Diff + cb2_term - Conv
    
    return {
        'i_idx': int(i_idx),
        'x_chord': float(xc[i_idx, 0]),
        'nu_lam': float(nu_lam),
        'u_tau': float(u_tau),
        # Profiles
        'y_plus': y_plus.tolist(),
        'chi': chi.tolist(),
        'nuHat': nuHat.tolist(),
        # All SA terms
        'Production': P.tolist(),
        'Destruction': D.tolist(),
        'Convection': Conv.tolist(),
        'Diffusion': Diff.tolist(),
        'cb2_term': cb2_term.tolist(),
        'Residual': residual.tolist(),
        # Gradient components
        'dnuHat_dx': dnuHat_dx.tolist(),
        'dnuHat_dy': dnuHat_dy.tolist(),
        'u': u.tolist(),
        'v': v.tolist(),
    }


def find_chord_indices(metrics, chord_fracs, n_wake):
    xc = metrics.xc
    NI = xc.shape[0]
    x_surface = xc[:, 0]
    le_idx = np.argmin(x_surface[n_wake:NI-n_wake]) + n_wake
    
    result = {}
    for cf in chord_fracs:
        for i in range(le_idx, n_wake, -1):
            if x_surface[i] >= cf:
                result[f'upper_{cf}'] = i
                break
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/examples/naca0012_re1m.yaml')
    parser.add_argument('--max-iter', type=int, default=30000)
    parser.add_argument('--output', default='sa_balance_debug.json')
    args = parser.parse_args()
    
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
    
    chord_fracs = [0.05, 0.25, 0.50]
    indices = find_chord_indices(solver.metrics, chord_fracs, n_wake)
    
    print(f"\nExtracting all SA terms...")
    profiles = {}
    for name, i_idx in indices.items():
        print(f"  {name} (i={i_idx})...")
        profiles[name] = extract_all_sa_terms(solver, i_idx)
    
    with open(args.output, 'w') as f:
        json.dump(profiles, f, indent=2)
    print(f"\nSaved to: {args.output}")
    
    # Print detailed balance
    print("\n" + "=" * 100)
    print("SA EQUATION BALANCE (all terms)")
    print("=" * 100)
    print("Equation: d(nuHat)/dt = P - D + Diff + cb2 - Conv")
    print("At steady state: 0 = P - D + Diff + cb2 - Conv")
    print()
    
    for name, prof in profiles.items():
        y_plus = np.array(prof['y_plus'])
        P = np.array(prof['Production'])
        D = np.array(prof['Destruction'])
        Conv = np.array(prof['Convection'])
        Diff = np.array(prof['Diffusion'])
        cb2 = np.array(prof['cb2_term'])
        Res = np.array(prof['Residual'])
        chi = np.array(prof['chi'])
        
        print(f"\n{name} (x/c = {prof['x_chord']:.3f}):")
        print("-" * 100)
        print(f"{'j':>3} | {'y+':>7} | {'chi':>6} | {'P':>10} | {'D':>10} | {'Conv':>10} | {'Diff':>10} | {'cb2':>10} | {'Residual':>10}")
        print("-" * 100)
        
        for j in range(min(25, len(y_plus))):
            if y_plus[j] > 300:
                break
            print(f"{j:3d} | {y_plus[j]:7.1f} | {chi[j]:6.2f} | {P[j]:10.2e} | {D[j]:10.2e} | {Conv[j]:10.2e} | {Diff[j]:10.2e} | {cb2[j]:10.2e} | {Res[j]:10.2e}")
        
        # Summary statistics
        mask_inner = (y_plus > 5) & (y_plus < 30)
        mask_log = (y_plus > 30) & (y_plus < 100)
        
        if np.any(mask_inner):
            print(f"\nInner layer (5 < y+ < 30) averages:")
            print(f"  P={P[mask_inner].mean():.2e}, D={D[mask_inner].mean():.2e}, Conv={Conv[mask_inner].mean():.2e}, Diff={Diff[mask_inner].mean():.2e}")
            print(f"  P/D = {(P[mask_inner]/D[mask_inner]).mean():.2f}")
            print(f"  Conv/P = {(Conv[mask_inner]/(P[mask_inner]+1e-30)).mean():.2f}")
            
        if np.any(mask_log):
            print(f"\nLog layer (30 < y+ < 100) averages:")
            print(f"  P={P[mask_log].mean():.2e}, D={D[mask_log].mean():.2e}, Conv={Conv[mask_log].mean():.2e}, Diff={Diff[mask_log].mean():.2e}")
            print(f"  P/D = {(P[mask_log]/D[mask_log]).mean():.2f}")
            print(f"  Conv/P = {(Conv[mask_log]/(P[mask_log]+1e-30)).mean():.2f}")


if __name__ == '__main__':
    main()
