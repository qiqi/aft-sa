#!/usr/bin/env python
"""
Debug boundary layer profiles at specific chord locations.

Compares computed profiles against law of the wall (Spalding profile).
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


def spalding_profile(y_plus: np.ndarray, kappa: float = 0.41, B: float = 5.0) -> np.ndarray:
    """
    Spalding's law of the wall: y+ = u+ + exp(-kappa*B) * [exp(kappa*u+) - 1 - kappa*u+ - ...]
    
    Inverted numerically to get u+ from y+.
    """
    u_plus = np.zeros_like(y_plus)
    for i, yp in enumerate(y_plus):
        if yp < 1e-10:
            u_plus[i] = yp  # Linear sublayer
            continue
        
        # Newton iteration to solve Spalding equation
        up = min(yp, yp**0.5)  # Initial guess
        for _ in range(50):
            e_term = np.exp(-kappa * B)
            kup = kappa * up
            exp_kup = np.exp(kup)
            f = up + e_term * (exp_kup - 1 - kup - 0.5*kup**2 - (1/6)*kup**3) - yp
            df = 1 + e_term * kappa * (exp_kup - 1 - kup - 0.5*kup**2)
            if abs(df) < 1e-14:
                break
            up_new = up - f / df
            if abs(up_new - up) < 1e-10:
                up = up_new
                break
            up = up_new
        u_plus[i] = up
    return u_plus


def extract_bl_profile(solver, i_idx: int):
    """
    Extract boundary layer profile at a given i-index.
    
    Returns dict with y+, u+, nuHat, chi, etc.
    """
    # Grid info
    metrics = solver.metrics
    xc = metrics.xc  # Cell center x
    yc = metrics.yc  # Cell center y
    vol = metrics.volume
    Sj_x = metrics.Sj_x
    Sj_y = metrics.Sj_y
    wall_dist = np.asarray(solver.wall_dist_jax)
    
    # Solution (sync from JAX)
    Q = np.asarray(solver.Q_jax)
    NI = Q.shape[0] - 2 * NGHOST
    NJ = Q.shape[1] - 2 * NGHOST
    
    # Interior solution at this i
    p = Q[NGHOST + i_idx, NGHOST:NGHOST+NJ, 0]
    u = Q[NGHOST + i_idx, NGHOST:NGHOST+NJ, 1]
    v = Q[NGHOST + i_idx, NGHOST:NGHOST+NJ, 2]
    nuHat = Q[NGHOST + i_idx, NGHOST:NGHOST+NJ, 3]
    
    # Compute wall normal direction from face normal at wall
    Sx = Sj_x[i_idx, 0]
    Sy = Sj_y[i_idx, 0]
    S_mag = np.sqrt(Sx**2 + Sy**2)
    nx = Sx / (S_mag + 1e-14)
    ny = Sy / (S_mag + 1e-14)
    
    # Wall point (average of vertices)
    # Wall is at j=0 face, so use the j-face coordinates
    # For cell (i,j), the j=0 face is the wall
    x_wall = xc[i_idx, 0]
    y_wall = yc[i_idx, 0]
    
    # Distance from wall for each cell (using wall_distance array)
    y_dist = wall_dist[i_idx, :]
    
    # Alternative: compute distance along wall normal from cell centers
    y_dist_manual = np.zeros(NJ)
    for j in range(NJ):
        # Distance from wall point to cell center
        dx = xc[i_idx, j] - x_wall
        dy = yc[i_idx, j] - y_wall
        y_dist_manual[j] = np.sqrt(dx**2 + dy**2)
    
    # Velocity magnitude and tangential velocity
    vel_mag = np.sqrt(u**2 + v**2)
    
    # Tangential direction (perpendicular to wall normal, in flow direction)
    tx = -ny  # Rotate normal by 90 degrees
    ty = nx
    
    # Tangential velocity (velocity component parallel to wall)
    u_tan = u * tx + v * ty
    
    # At wall, check the sign - u_tan should be positive if flow is going in +x
    # The wall tangent should point in the direction of flow
    if np.mean(u_tan) < 0:
        tx, ty = -tx, -ty
        u_tan = -u_tan
    
    # Physical quantities
    nu_laminar = 1.0 / solver.config.reynolds
    
    # Effective viscosity
    # chi = nuHat / nu_laminar
    chi = nuHat / nu_laminar
    
    # For SA, nu_tilde contributes to turbulent viscosity
    # mu_t = rho * nu_t, where nu_t = nu_tilde * f_v1(chi)
    # For simplicity, assume chi >> 1 in the log layer, so nu_t ≈ nuHat
    nu_eff = nu_laminar + np.maximum(0.0, nuHat)
    
    # Wall shear stress
    # dy at first cell from wall (vol/area gives full cell height, divide by 2 for wall-to-center distance)
    vol_wall = vol[i_idx, 0]
    area_wall = np.sqrt(Sj_x[i_idx, 0]**2 + Sj_y[i_idx, 0]**2)
    dy_full = vol_wall / (area_wall + 1e-14)  # Full cell height
    dy_first = dy_full / 2.0  # Distance from wall to cell center
    
    # Wall shear using first cell (u=0 at wall, u=u_tan[0] at cell center distance dy_first)
    du_dy_wall = u_tan[0] / dy_first  # Linear profile: du/dy = u / y  # Factor 2 because u=0 at wall
    mu_wall = nu_eff[0]
    tau_wall = mu_wall * du_dy_wall
    
    # Friction velocity
    u_tau = np.sqrt(np.abs(tau_wall))
    
    # y+ and u+
    y_plus = y_dist * u_tau / nu_laminar
    u_plus = u_tan / (u_tau + 1e-14)
    
    # Spalding reference
    u_plus_spalding = spalding_profile(y_plus)
    
    # Cf from wall shear
    V_inf = np.sqrt(solver.freestream.u_inf**2 + solver.freestream.v_inf**2)
    q_inf = 0.5 * V_inf**2
    Cf_local = tau_wall / q_inf
    
    return {
        'i_idx': int(i_idx),
        'x_wall': float(x_wall),
        'y_wall': float(y_wall),
        'chord_frac': float(xc[i_idx, 0]),  # Approximate chord fraction
        # Grid info
        'dy_first': float(dy_first),
        'y_dist': y_dist.tolist(),
        'y_dist_manual': y_dist_manual.tolist(),
        # Solution
        'u': u.tolist(),
        'v': v.tolist(),
        'u_tan': u_tan.tolist(),
        'vel_mag': vel_mag.tolist(),
        'p': p.tolist(),
        'nuHat': nuHat.tolist(),
        'chi': chi.tolist(),
        'nu_eff': nu_eff.tolist(),
        # Dimensionless
        'y_plus': y_plus.tolist(),
        'u_plus': u_plus.tolist(),
        'u_plus_spalding': u_plus_spalding.tolist(),
        # Wall quantities
        'tau_wall': float(tau_wall),
        'u_tau': float(u_tau),
        'Cf_local': float(Cf_local),
        # Parameters
        'nu_laminar': float(nu_laminar),
        'V_inf': float(V_inf),
    }


def find_chord_indices(metrics, chord_fracs: list, n_wake: int):
    """
    Find i-indices corresponding to chord fractions on upper and lower surfaces.
    """
    xc = metrics.xc
    NI = xc.shape[0]
    
    # Upper surface: from LE (around NI/2 - n_wake) to TE (n_wake)
    # Lower surface: from LE (around NI/2 - n_wake) to TE (NI - n_wake)
    
    # Find leading edge (minimum x)
    x_surface = xc[:, 0]
    le_idx = np.argmin(x_surface[n_wake:NI-n_wake]) + n_wake
    
    print(f"Leading edge at i={le_idx}, x={x_surface[le_idx]:.4f}")
    print(f"Wake cells: 0-{n_wake} and {NI-n_wake}-{NI}")
    
    result = {}
    
    # Upper surface: from le_idx going down to n_wake
    for cf in chord_fracs:
        target_x = cf
        
        # Upper surface
        upper_range = range(le_idx, n_wake, -1)
        for i in upper_range:
            if x_surface[i] >= target_x:
                result[f'upper_{cf}'] = i
                break
        
        # Lower surface
        lower_range = range(le_idx, NI - n_wake)
        for i in lower_range:
            if x_surface[i] >= target_x:
                result[f'lower_{cf}'] = i
                break
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Debug boundary layer profiles')
    parser.add_argument('--config', default='config/examples/naca0012_re1m.yaml',
                       help='Config file')
    parser.add_argument('--max-iter', type=int, default=1000,
                       help='Max iterations before extracting profiles')
    parser.add_argument('--output', default='bl_debug.json',
                       help='Output JSON file')
    args = parser.parse_args()
    
    # Load config
    sim_config = load_config(args.config)
    
    # Override max_iter
    sim_config.solver.max_iter = args.max_iter
    
    # Convert to legacy SolverConfig
    config = sim_config.to_solver_config()
    
    # Load grid
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
    
    # Create solver using same pattern as run_airfoil.py
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
    
    # Initialize components
    solver._compute_metrics()
    solver._initialize_state()
    solver._initialize_output()
    
    # Run some iterations
    print(f"\nRunning {args.max_iter} iterations...")
    solver.run_steady_state()
    
    # Extract profiles at key chord locations
    chord_fracs = [0.05, 0.10, 0.25, 0.50, 0.75]
    indices = find_chord_indices(solver.metrics, chord_fracs, n_wake)
    
    print(f"\nChord location indices: {indices}")
    
    profiles = {}
    for name, i_idx in indices.items():
        print(f"\nExtracting profile at {name} (i={i_idx})...")
        profiles[name] = extract_bl_profile(solver, i_idx)
        
        # Print summary
        prof = profiles[name]
        print(f"  x = {prof['chord_frac']:.4f}")
        print(f"  dy_first = {prof['dy_first']:.6e}")
        print(f"  y+ first cell = {prof['y_plus'][0]:.2f}")
        print(f"  u_tau = {prof['u_tau']:.6f}")
        print(f"  Cf = {prof['Cf_local']:.6f}")
        print(f"  chi(wall) = {prof['chi'][0]:.2f}")
        print(f"  nuHat(wall) = {prof['nuHat'][0]:.6e}")
        
        # Compare to Spalding at y+ = 30
        j_30 = None
        for j, yp in enumerate(prof['y_plus']):
            if yp > 30:
                j_30 = j
                break
        if j_30 is not None:
            print(f"  At y+≈30: u+={prof['u_plus'][j_30]:.2f}, Spalding u+={prof['u_plus_spalding'][j_30]:.2f}")
    
    # Save to JSON
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump({
            'config': {
                'reynolds': config.reynolds,
                'alpha': config.alpha,
                'beta': config.beta,
            },
            'profiles': profiles,
        }, f, indent=2)
    
    print(f"\nSaved debug data to: {output_path}")
    
    # Quick diagnostic summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    # Expected Cf for turbulent flat plate at Re=1M
    # Cf ≈ 0.074 / Re^0.2 ≈ 0.0047
    Re = config.reynolds
    Cf_expected = 0.074 / (Re ** 0.2)
    print(f"\nExpected Cf (turbulent flat plate): {Cf_expected:.5f}")
    
    for name, prof in profiles.items():
        print(f"\n{name}:")
        print(f"  Computed Cf: {prof['Cf_local']:.5f} (ratio to expected: {prof['Cf_local']/Cf_expected:.1f}x)")
        print(f"  nu_eff/nu_lam at wall: {prof['nu_eff'][0]/prof['nu_laminar']:.1f}")
        
        # Check if u+ profile matches law of wall
        yp = np.array(prof['y_plus'])
        up = np.array(prof['u_plus'])
        up_ref = np.array(prof['u_plus_spalding'])
        
        # Compare in log layer (30 < y+ < 200)
        mask = (yp > 30) & (yp < 200)
        if np.any(mask):
            error = np.abs(up[mask] - up_ref[mask]) / (up_ref[mask] + 1e-10)
            print(f"  u+ error in log layer (30<y+<200): mean={np.mean(error)*100:.1f}%, max={np.max(error)*100:.1f}%")


if __name__ == '__main__':
    main()
