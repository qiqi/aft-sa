#!/usr/bin/env python3
"""
Generate diagnostic data for momentum and SA physics analysis.

This script runs the solver and saves comprehensive diagnostic data
to output/bl_full_diagnostic.npz for analysis by diagnose_momentum.py
and diagnose_sa_physics.py.
"""

import sys
import numpy as np
from pathlib import Path
from collections import deque

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import device selection BEFORE JAX
from src.physics.jax_config import select_device
select_device()

from src.config.loader import load_yaml as load_config
from src.grid.loader import load_or_generate_grid
from src.solvers.rans_solver import RANSSolver
from src.constants import NGHOST


def main():
    """Generate diagnostic data."""
    # Load configuration
    config_path = project_root / "config/examples/naca0012_re1m.yaml"
    sim_config = load_config(str(config_path))
    
    print(f"Generating diagnostic data from: {config_path}")
    
    # Override max_iter for quick run
    sim_config.solver.max_iter = 10000
    
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
        farfield_radius=sim_config.grid.farfield_radius,
        project_root=project_root,
        verbose=True,
    )
    
    NI = X.shape[0] - 1
    NJ = X.shape[1] - 1
    print(f"Grid loaded: {X.shape[0]} x {X.shape[1]} nodes")
    print(f"            {NI} x {NJ} cells")
    
    # Create solver using same pattern as run_airfoil.py
    solver = RANSSolver.__new__(RANSSolver)
    solver.config = config
    solver.X = X
    solver.Y = Y
    solver.NI = NI
    solver.NJ = NJ
    solver.iteration = 0
    solver.residual_history = []
    solver.iteration_history = []
    solver.converged = False
    solver._divergence_buffer = None
    
    # Initialize components
    solver._compute_metrics()
    solver._initialize_state()
    solver._initialize_output()
    
    # Run solver
    print("\nRunning solver for 10000 iterations...")
    solver.run_steady_state()
    
    # Extract data
    Q = np.asarray(solver.Q_jax)
    nghost = NGHOST
    Q_int = Q[nghost:-nghost, nghost:-nghost, :]  # Interior cells
    
    # Get metrics
    metrics = solver.metrics
    vol = np.asarray(metrics.volume)
    wall_dist = np.asarray(solver.wall_dist_jax)
    xc = np.asarray(metrics.xc)
    yc = np.asarray(metrics.yc)
    Sj_x = np.asarray(metrics.Sj_x)
    Sj_y = np.asarray(metrics.Sj_y)
    Si_x = np.asarray(metrics.Si_x)
    Si_y = np.asarray(metrics.Si_y)
    
    # Flow parameters
    nu = 1.0 / sim_config.flow.reynolds
    
    # Find leading edge (minimum x on surface)
    x_wall = xc[:, 0]
    i_le = np.argmin(np.abs(x_wall - x_wall.min()))
    
    # Wake cell count
    n_wake = sim_config.grid.n_wake
    
    # Station indices at various chord locations
    def find_station(x_target, upper=True):
        """Find I index at given x/c on upper or lower surface."""
        if upper:
            search_range = range(i_le, n_wake, -1)
        else:
            search_range = range(i_le, NI - n_wake)
        
        for i in search_range:
            if abs(x_wall[i] - x_target) < 0.02:
                return i
        return i_le - 10 if upper else i_le + 10
    
    i_05 = find_station(0.05, upper=True)
    i_10 = find_station(0.1, upper=True)
    i_25 = find_station(0.25, upper=True)
    i_50 = find_station(0.5, upper=True)
    
    print(f"\nStation indices: i_05={i_05}, i_10={i_10}, i_25={i_25}, i_50={i_50}")
    print(f"x at stations: {xc[i_05,0]:.3f}, {xc[i_10,0]:.3f}, {xc[i_25,0]:.3f}, {xc[i_50,0]:.3f}")
    
    # Compute SA source terms for diagnostics
    from src.physics.spalart_allmaras import (
        compute_fv1, compute_fv2, compute_S_tilde,
        compute_sa_production, compute_sa_destruction, CB2, SIGMA, KAPPA
    )
    from src.numerics.gradients import compute_gradients_jax
    from jax import numpy as jnp
    
    # Get nuHat and compute derived quantities
    nuHat = Q_int[:, :, 3]
    u = Q_int[:, :, 1]
    v = Q_int[:, :, 2]
    
    chi = nuHat / nu
    fv1 = np.asarray(compute_fv1(chi))
    fv2 = np.asarray(compute_fv2(chi, fv1))
    
    # Compute vorticity (need gradients)
    Q_jax = jnp.array(Q)
    
    grad = compute_gradients_jax(Q_jax, Si_x, Si_y, Sj_x, Sj_y, vol, nghost)
    grad = np.asarray(grad)  # (NI, NJ, 4, 2) - last dim is (dx, dy)
    
    # Extract velocity gradients
    # grad[:, :, var, dir] where var: 0=p, 1=u, 2=v, 3=nuHat; dir: 0=x, 1=y
    du_dx = grad[:, :, 1, 0]
    du_dy = grad[:, :, 1, 1]
    dv_dx = grad[:, :, 2, 0]
    dv_dy = grad[:, :, 2, 1]
    
    omega = np.abs(dv_dx - du_dy)  # |dv/dx - du/dy|
    
    # Production and destruction
    S_tilde = np.asarray(compute_S_tilde(omega, nuHat, wall_dist, chi, fv2))
    P = np.asarray(compute_sa_production(omega, nuHat, wall_dist, nu))
    D = np.asarray(compute_sa_destruction(omega, nuHat, wall_dist, nu))
    
    # Gradient of nuHat for cb2 term
    # nuHat gradient is already in grad[:, :, 3, :]
    grad_nuHat_x = grad[:, :, 3, 0]
    grad_nuHat_y = grad[:, :, 3, 1]
    grad_nuHat_mag2 = grad_nuHat_x**2 + grad_nuHat_y**2
    cb2_term = CB2 / SIGMA * grad_nuHat_mag2
    
    # Save comprehensive data
    output_path = project_root / "output/bl_full_diagnostic.npz"
    np.savez(
        output_path,
        # Grid info
        NI=NI, NJ=NJ, nu=nu, n_wake=n_wake,
        xc=xc, yc=yc, vol=vol, wall_dist=wall_dist,
        Sj_x=Sj_x, Sj_y=Sj_y,
        # Flow state (interior cells)
        Q=Q_int,
        # SA terms
        P=P, D=D, cb2_term=cb2_term,
        omega=omega, S_tilde=S_tilde,
        chi=chi, fv1=fv1, fv2=fv2,
        # Station indices
        i_05=i_05, i_10=i_10, i_25=i_25, i_50=i_50,
    )
    
    print(f"\nSaved diagnostic data to: {output_path}")
    
    # Quick momentum check at x/c = 0.25
    print("\n" + "="*60)
    print("QUICK MOMENTUM CHECK at x/c = 0.25 (upper surface)")
    print("="*60)
    
    i_idx = i_25
    Q_prof = Q_int[i_idx, :]
    wd = wall_dist[i_idx, :]
    
    # Tangent direction
    S0 = np.sqrt(Sj_x[i_idx, 0]**2 + Sj_y[i_idx, 0]**2)
    tx = -Sj_y[i_idx, 0] / S0
    ty = Sj_x[i_idx, 0] / S0
    
    u_tan = Q_prof[:, 1] * tx + Q_prof[:, 2] * ty
    if np.mean(u_tan) < 0:
        u_tan = -u_tan
    
    # Wall shear
    dy_wall = vol[i_idx, 0] / S0 / 2
    tau_wall = nu * u_tan[0] / dy_wall
    u_tau = np.sqrt(np.abs(tau_wall))
    
    y_plus = wd * u_tau / nu
    u_plus = u_tan / u_tau
    chi_prof = Q_prof[:, 3] / nu
    
    print(f"  u_tau = {u_tau:.5f}")
    print(f"  Cf = {2*u_tau**2:.6f}")
    
    # Log layer check
    B = 5.0
    
    print(f"\n  Log layer (y+ = 30-100):")
    for yp_target in [30, 50, 70, 100]:
        idx = np.argmin(np.abs(y_plus - yp_target))
        yp = y_plus[idx]
        up = u_plus[idx]
        chi_val = chi_prof[idx]
        
        up_log = (1/KAPPA) * np.log(yp) + B
        chi_expected = KAPPA * yp
        
        print(f"    y+ = {yp:.1f}: u+ = {up:.2f} (log law: {up_log:.2f}, error: {(up-up_log)/up_log*100:+.1f}%)")
        print(f"           chi = {chi_val:.2f} (expected: {chi_expected:.2f}, error: {(chi_val-chi_expected)/chi_expected*100:+.1f}%)")
    
    # Quick SA check
    print(f"\n  SA term balance (P + Diff - D):")
    for j in range(min(20, NJ)):
        P_val = P[i_idx, j]
        D_val = D[i_idx, j]
        cb2_val = cb2_term[i_idx, j]
        # Diff includes both viscous diffusion and cb2
        balance = P_val + cb2_val - D_val
        if j < 10 or (j % 5 == 0):
            print(f"    j={j:2d}, y+={y_plus[j]:6.1f}: P={P_val:.2e}, D={D_val:.2e}, P/D={P_val/(D_val+1e-20):.2f}")
    
    print("\nDONE. Run diagnose_momentum.py for detailed velocity profile analysis.")


if __name__ == "__main__":
    main()
