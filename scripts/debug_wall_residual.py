#!/usr/bin/env python3
"""
Debug Wall Residual - Check what's happening at the wall boundary.

This script runs a single iteration and examines the residual components
at the wall to understand why the velocity is too high.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import jax.numpy as jnp
import jax

from src.config.loader import load_yaml
from src.solvers.rans_solver import RANSSolver
from src.grid.loader import load_or_generate_grid
from src.solvers.boundary_conditions import FreestreamConditions
from src.constants import NGHOST


def main():
    # Load config
    config = load_yaml('config/examples/naca0012_re1m.yaml')
    
    # Load grid
    grid_params = config.grid
    grid_data = load_or_generate_grid(
        grid_params.airfoil,
        grid_params.ni_airfoil,
        grid_params.nj,
        grid_params.far_field_distance,
        grid_params.le_clustering,
        grid_params.te_clustering,
        grid_params.wall_spacing,
        verbose=False
    )
    
    # Create solver
    solver = RANSSolver.__new__(RANSSolver)
    solver.config = config
    solver.grid_data = grid_data
    solver.NI = grid_data['NI']
    solver.NJ = grid_data['NJ']
    solver.n_wake = grid_params.n_wake
    
    # Create freestream
    freestream = FreestreamConditions.from_mach_alpha(
        mach=config.flow.mach,
        alpha=config.flow.alpha,
        Re=config.flow.reynolds,
        chi_inf=config.solver.chi_inf
    )
    solver.freestream = freestream
    solver.nu = freestream.nu_laminar
    
    # Initialize metrics
    solver._compute_metrics()
    
    # Initialize state
    solver._initialize_state()
    
    print(f"Initial state shape: {solver.Q_jax.shape}")
    print(f"NI={solver.NI}, NJ={solver.NJ}, NGHOST={NGHOST}")
    
    # Check velocity at first interior cell
    Q = np.asarray(solver.Q_jax)
    i_mid = solver.NI // 2 + NGHOST  # Mid-airfoil
    
    print(f"\nVelocity at mid-airfoil (i={i_mid}) near wall:")
    print("   j       u          v         |u|")
    for j in range(NGHOST, NGHOST + 5):
        u = Q[i_mid, j, 1]
        v = Q[i_mid, j, 2]
        mag = np.sqrt(u**2 + v**2)
        print(f"{j:4d} {u:10.6f} {v:10.6f} {mag:10.6f}")
    
    # Check ghost cells
    print("\nGhost cells at mid-airfoil:")
    for j in range(NGHOST):
        u = Q[i_mid, j, 1]
        v = Q[i_mid, j, 2]
        mag = np.sqrt(u**2 + v**2)
        print(f"   j={j}: u={u:10.6f}, v={v:10.6f}, |u|={mag:10.6f}")
    
    # Run one iteration
    print("\nRunning one iteration...")
    solver._initialize_output()
    
    # Get the residual
    R = solver._compute_residual(solver.Q_jax)
    R_np = np.asarray(R)
    
    print("\nResidual at mid-airfoil near wall:")
    print("   j     R_p        R_u        R_v       R_nuHat")
    for j in range(5):
        print(f"{j:4d} {R_np[i_mid-NGHOST, j, 0]:10.2e} {R_np[i_mid-NGHOST, j, 1]:10.2e} "
              f"{R_np[i_mid-NGHOST, j, 2]:10.2e} {R_np[i_mid-NGHOST, j, 3]:10.2e}")
    
    # The key is R_u and R_v at j=0 (first interior cell)
    # If R_u is large and positive, the velocity will increase
    # If R_u is large and negative, the velocity will decrease
    
    print(f"\nResidual at first interior cell (j=0):")
    print(f"  R_u = {R_np[i_mid-NGHOST, 0, 1]:.2e}")
    print(f"  R_v = {R_np[i_mid-NGHOST, 0, 2]:.2e}")
    
    # Run 100 iterations and check again
    print("\nRunning 100 iterations...")
    for _ in range(100):
        solver._rk_step()
    
    Q_after = np.asarray(solver.Q_jax)
    print("\nVelocity after 100 iterations:")
    for j in range(NGHOST, NGHOST + 5):
        u = Q_after[i_mid, j, 1]
        v = Q_after[i_mid, j, 2]
        mag = np.sqrt(u**2 + v**2)
        print(f"{j:4d} {u:10.6f} {v:10.6f} {mag:10.6f}")
    
    # Check if velocity is growing or shrinking
    u_before = np.sqrt(Q[i_mid, NGHOST, 1]**2 + Q[i_mid, NGHOST, 2]**2)
    u_after = np.sqrt(Q_after[i_mid, NGHOST, 1]**2 + Q_after[i_mid, NGHOST, 2]**2)
    print(f"\nVelocity at first interior cell:")
    print(f"  Before: {u_before:.6f}")
    print(f"  After:  {u_after:.6f}")
    print(f"  Change: {u_after - u_before:+.6f} ({(u_after/u_before - 1)*100:+.1f}%)")


if __name__ == '__main__':
    main()
