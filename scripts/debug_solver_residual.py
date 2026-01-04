#!/usr/bin/env python
"""
Extract the ACTUAL residual from the solver and compare with diagnostic.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.loader import load_yaml as load_config
from src.grid.loader import load_or_generate_grid
from src.solvers.rans_solver import RANSSolver
from src.constants import NGHOST
from src.physics.jax_config import jnp
import jax


def main():
    config_path = 'config/examples/naca0012_re1m.yaml'
    max_iter = 30000
    
    sim_config = load_config(config_path)
    sim_config.solver.max_iter = max_iter
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
    
    print(f"\nRunning {max_iter} iterations...")
    solver.run_steady_state()
    
    # Now compute residual manually using solver's method
    print("\n" + "=" * 80)
    print("Extracting actual solver residual...")
    print("=" * 80)
    
    # Get the residual by calling _compute_residual
    Q = np.asarray(solver.Q_jax)
    nghost = NGHOST
    NI = solver.NI
    NJ = solver.NJ
    
    # Call the residual function
    R_actual = solver._compute_residual(Q)
    R_np = np.asarray(R_actual)
    
    # Get nuHat residual (index 3)
    R_nuHat = R_np[:, :, 3]
    
    # Find i-index for x/c = 0.25
    xc = solver.metrics.xc
    x_surface = xc[:, 0]
    le_idx = np.argmin(x_surface[n_wake:NI-n_wake]) + n_wake
    
    i_025 = None
    for i in range(le_idx, n_wake, -1):
        if x_surface[i] >= 0.25:
            i_025 = i
            break
    
    print(f"\nAt i={i_025} (x/c ≈ 0.25):")
    
    # Wall distance for this i-line
    wall_dist = np.asarray(solver.wall_dist_jax)
    d = wall_dist[i_025, :]
    
    # Compute y+ using u_tau from wall shear
    nu_lam = 1.0 / config.reynolds
    Q_np = np.asarray(Q)
    
    # Get solution at this i
    u = Q_np[nghost + i_025, nghost:nghost+NJ, 1]
    v = Q_np[nghost + i_025, nghost:nghost+NJ, 2]
    nuHat = Q_np[nghost + i_025, nghost:nghost+NJ, 3]
    
    # Wall normal from face
    Sj_x = solver.metrics.Sj_x[i_025, 0]
    Sj_y = solver.metrics.Sj_y[i_025, 0]
    S_mag = np.sqrt(Sj_x**2 + Sj_y**2)
    nx, ny = Sj_x/S_mag, Sj_y/S_mag
    tx, ty = -ny, nx
    u_tan = u * tx + v * ty
    if np.mean(u_tan) < 0:
        u_tan = -u_tan
    
    vol = solver.metrics.volume[i_025, 0]
    area = S_mag
    dy_wall = vol / area / 2
    du_dy_wall = u_tan[0] / dy_wall
    tau_wall = nu_lam * du_dy_wall  # Assuming laminar at wall
    u_tau = np.sqrt(np.abs(tau_wall))
    
    y_plus = d * u_tau / nu_lam
    chi = nuHat / nu_lam
    
    # The residual R_nuHat is already scaled by volume in the solver
    # So it represents: dQ/dt * V = R
    # The source terms are: P - D + cb2 (already multiplied by V)
    # Plus flux contributions
    
    # Get volume for this i-line
    vol_line = solver.metrics.volume[i_025, :]
    
    # R_nuHat / volume gives the actual d(nuHat)/dt at steady state
    R_per_vol = R_nuHat[i_025, :] / vol_line
    
    print(f"\nActual solver residual for nuHat (R/V):")
    print(f"  j |     y+ |    chi |  R_nuHat/V |   nuHat")
    print("-" * 60)
    for j in range(min(25, NJ)):
        if y_plus[j] > 300:
            break
        print(f"{j:3d} | {y_plus[j]:6.1f} | {chi[j]:6.2f} | {R_per_vol[j]:10.2e} | {nuHat[j]:9.2e}")
    
    print()
    print("If R_nuHat/V > 0, nuHat should be growing.")
    print("If R_nuHat/V ≈ 0, we're at steady state.")
    print("If R_nuHat/V < 0, nuHat should be decreasing.")
    
    # Summary
    mask_log = (y_plus > 30) & (y_plus < 100)
    if np.any(mask_log):
        print(f"\nLog layer (30 < y+ < 100):")
        print(f"  Mean R_nuHat/V: {R_per_vol[mask_log].mean():.2e}")
        print(f"  Mean chi: {chi[mask_log].mean():.2f}")


if __name__ == '__main__':
    main()
