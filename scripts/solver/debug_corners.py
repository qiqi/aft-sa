#!/usr/bin/env python3
"""
Debug script for farfield corner divergence analysis.

Instruments the solver to track state and residual values at the
farfield corner cells (I=0, J=Jmax) and (I=Imax, J=Jmax).

Usage:
    python scripts/solver/debug_corners.py data/naca0012.dat --super-coarse --max-iter 300 --cfl 2
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import device selection BEFORE JAX
from src.physics.jax_config import select_device


def main():
    parser = argparse.ArgumentParser(description="Debug farfield corner divergence")
    parser.add_argument("grid_file", help="Airfoil data file")
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--reynolds", type=float, default=10000)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--cfl", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--super-coarse", action="store_true")
    parser.add_argument("--coarse", action="store_true")
    parser.add_argument("--print-freq", type=int, default=1, help="Print every N iterations")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    # Device selection
    select_device(args.device)
    
    # Now import JAX-dependent modules
    from src.physics.jax_config import jax, jnp
    from src.solvers.rans_solver import RANSSolver, SolverConfig
    from src.grid.loader import load_or_generate_grid
    from src.numerics.debug_corners import CornerDebugger
    from src.constants import NGHOST
    
    # Grid settings
    if args.super_coarse:
        n_surface, n_normal, n_wake, y_plus = 65, 33, 32, 5.0
        print("Using SUPER-COARSE grid (128x32 cells)")
    elif args.coarse:
        n_surface, n_normal, n_wake, y_plus = 193, 33, 32, 1.0
        print("Using COARSE grid (256x32 cells)")
    else:
        n_surface, n_normal, n_wake, y_plus = 257, 65, 64, 1.0
        print("Using STANDARD grid")
    
    print(f"\n{'='*70}")
    print("   CORNER DEBUG MODE")
    print(f"{'='*70}")
    
    # Load grid
    X, Y = load_or_generate_grid(
        args.grid_file,
        n_surface=n_surface,
        n_normal=n_normal,
        n_wake=n_wake,
        y_plus=y_plus,
        reynolds=args.reynolds,
        project_root=project_root,
        verbose=True
    )
    
    # Create config
    config = SolverConfig(
        mach=0.0,
        alpha=args.alpha,
        reynolds=args.reynolds,
        beta=args.beta,
        cfl_start=0.1,
        cfl_target=args.cfl,
        cfl_ramp_iters=300,
        max_iter=args.max_iter,
        tol=1e-10,
        print_freq=args.print_freq,
        diagnostic_freq=100,
        output_dir="output/debug",
        case_name="corner_debug",
        html_animation=False,  # Disable for speed
        divergence_history=0,
    )
    
    # Create solver manually (bypass __init__ verbose output)
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
    
    print(f"\nGrid size: {solver.NI} x {solver.NJ} cells")
    print(f"Alpha: {args.alpha}°, Re: {args.reynolds:.0f}, Beta: {args.beta}")
    print(f"CFL: {config.cfl_start} -> {config.cfl_target}")
    
    # Create debugger
    debugger = CornerDebugger(solver.NI, solver.NJ, output_dir="output/debug")
    
    # Custom iteration loop with full instrumentation
    print(f"\n{'='*70}")
    print("Starting Debug Iteration")
    print(f"{'='*70}")
    print(f"{'Iter':>6} {'Res':>12} {'CFL':>6} | "
          f"{'C1_p':>12} {'C1_R_p':>12} | "
          f"{'C2_p':>12} {'C2_R_p':>12}")
    print("-" * 90)
    
    initial_residual = None
    
    for n in range(config.max_iter):
        # Step (on GPU)
        solver.step()
        
        # Get current CFL
        cfl = solver._get_cfl(solver.iteration)
        
        # Get residual
        res_rms = solver.get_residual_rms()
        solver.residual_history.append(res_rms)
        solver.iteration_history.append(solver.iteration)
        
        if initial_residual is None:
            initial_residual = res_rms
        
        # Sync to CPU for debugging
        solver.sync_to_cpu()
        R = np.array(solver.R_jax)
        
        # Record corner data
        data = debugger.record(
            Q=solver.Q,
            R=R,
            iteration=solver.iteration,
            cfl=cfl,
            residual=res_rms
        )
        
        # Print every iteration
        if solver.iteration % args.print_freq == 0 or solver.iteration == 1:
            print(f"{solver.iteration:>6d} {res_rms:>12.4e} {cfl:>6.2f} | "
                  f"{data.c1_p:>+12.4e} {data.c1_R_p:>+12.4e} | "
                  f"{data.c2_p:>+12.4e} {data.c2_R_p:>+12.4e}")
        
        # Detailed print at key iterations
        if solver.iteration in [1, 10, 50, 100, 150, 200] or res_rms > 0.01:
            debugger.print_current(data)
        
        # Check for divergence
        if res_rms > 1000 * initial_residual:
            print(f"\n{'='*70}")
            print(f"DIVERGED at iteration {solver.iteration}")
            print(f"Residual: {res_rms:.6e} (initial: {initial_residual:.6e})")
            print(f"{'='*70}")
            debugger.print_current(data)
            break
        
        # Check for convergence
        if res_rms < config.tol:
            print(f"\n{'='*70}")
            print(f"CONVERGED at iteration {solver.iteration}")
            print(f"{'='*70}")
            break
    
    # Save CSV and analyze
    csv_path = debugger.save_csv()
    analysis = debugger.analyze()
    
    print(f"\n{'='*70}")
    print("Debug session complete")
    print(f"CSV saved to: {csv_path}")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
