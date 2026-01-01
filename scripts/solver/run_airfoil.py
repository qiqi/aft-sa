#!/usr/bin/env python3
"""
NACA 0012 Airfoil Simulation Script.

This script sets up and runs a 2D incompressible RANS simulation around
an airfoil using the artificial compressibility method.

Features:
    - On-the-fly grid generation with configurable density
    - Flow field visualization (VTK + PDF) at specified intervals
    - Residual monitoring and divergence detection
    - Surface data extraction

Usage:
    python run_airfoil.py <grid_file.p3d>
    python run_airfoil.py <airfoil.dat> --n-surface 100 --n-normal 50
    python run_airfoil.py <airfoil.dat> --dump-freq 10 --coarse
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.solvers.rans_solver import RANSSolver, SolverConfig
from src.grid.loader import load_or_generate_grid


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run RANS simulation with diagnostic visualization"
    )
    parser.add_argument(
        "grid_file",
        help="Path to grid file (.p3d) or airfoil file (.dat)"
    )
    
    # Flow conditions
    parser.add_argument("--mach", "-M", type=float, default=0.15,
                        help="Mach number (default: 0.15)")
    parser.add_argument("--alpha", "-a", type=float, default=0.0,
                        help="Angle of attack in degrees (default: 0.0)")
    parser.add_argument("--reynolds", "-Re", type=float, default=6e6,
                        help="Reynolds number (default: 6e6)")
    
    # Solver settings
    parser.add_argument("--max-iter", "-n", type=int, default=10000,
                        help="Maximum iterations (default: 10000)")
    parser.add_argument("--cfl", type=float, default=5.0,
                        help="Target CFL number (default: 5.0)")
    parser.add_argument("--cfl-start", type=float, default=0.1,
                        help="Initial CFL for ramping (default: 0.1)")
    parser.add_argument("--cfl-ramp", type=int, default=500,
                        help="CFL ramp iterations (default: 500)")
    parser.add_argument("--tol", type=float, default=1e-10,
                        help="Convergence tolerance (default: 1e-10)")
    parser.add_argument("--beta", type=float, default=10.0,
                        help="Artificial compressibility parameter (default: 10.0)")
    parser.add_argument("--irs", type=float, default=1.0,
                        help="Implicit Residual Smoothing epsilon (0=disabled, default: 1.0)")
    
    # Grid generation options
    parser.add_argument("--n-surface", type=int, default=250,
                        help="Surface points for grid generation (default: 250)")
    parser.add_argument("--n-normal", type=int, default=100,
                        help="Normal points for grid generation (default: 100)")
    parser.add_argument("--n-wake", type=int, default=50,
                        help="Wake points for grid generation (default: 50)")
    parser.add_argument("--coarse", action="store_true",
                        help="Use coarse grid (80x30) for debugging")
    parser.add_argument("--super-coarse", action="store_true",
                        help="Use super-coarse grid (60x20) for fast tests")
    
    # Output settings
    parser.add_argument("--output-dir", "-o", type=str, default="output/solver",
                        help="Output directory (default: output/solver)")
    parser.add_argument("--case-name", type=str, default="naca0012",
                        help="Case name for output files (default: naca0012)")
    parser.add_argument("--dump-freq", type=int, default=100,
                        help="Flow field dump frequency (default: 100)")
    parser.add_argument("--print-freq", type=int, default=10,
                        help="Console print frequency (default: 10)")
    parser.add_argument("--output-freq", type=int, default=100,
                        help="VTK output frequency (default: 100)")
    parser.add_argument("--div-history", type=int, default=0,
                        help="Solutions to keep for divergence visualization (0=disabled)")
    
    # Run mode
    parser.add_argument("--diagnostic", "-d", action="store_true",
                        help="Enable diagnostic mode with enhanced output")
    
    args = parser.parse_args()
    
    # Override grid settings for coarse modes
    if args.super_coarse:
        args.n_surface = 60
        args.n_normal = 20
        args.n_wake = 15
        args.y_plus = 5.0
        print("Using SUPER-COARSE grid mode (60x20)")
    elif args.coarse:
        args.n_surface = 80
        args.n_normal = 30
        args.n_wake = 20
        args.y_plus = 1.0
        print("Using COARSE grid mode (80x30)")
    
    # Print banner
    print("\n" + "="*70)
    print("   2D Incompressible RANS Solver")
    print("="*70)
    
    # Grid generation settings
    y_plus = getattr(args, 'y_plus', 1.0)
    max_first_cell = 0.005 if args.super_coarse else 0.001
    
    # Load or generate grid
    try:
        X, Y = load_or_generate_grid(
            args.grid_file,
            n_surface=args.n_surface,
            n_normal=args.n_normal,
            n_wake=args.n_wake,
            y_plus=y_plus,
            reynolds=args.reynolds,
            farfield_radius=15.0,
            max_first_cell=max_first_cell,
            project_root=project_root,
            verbose=True
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Configure solver with IRS for stability
    config = SolverConfig(
        mach=args.mach,
        alpha=args.alpha,
        reynolds=args.reynolds,
        beta=args.beta,
        cfl_start=args.cfl_start,
        cfl_target=args.cfl,
        cfl_ramp_iters=args.cfl_ramp,
        max_iter=args.max_iter,
        tol=args.tol,
        output_freq=args.output_freq,
        print_freq=args.print_freq,
        output_dir=args.output_dir,
        case_name=args.case_name,
        wall_damping_length=0.1,
        jst_k4=0.04,
        irs_epsilon=args.irs,
        n_wake=args.n_wake,
        diagnostic_mode=args.diagnostic,
        diagnostic_freq=args.dump_freq,
        divergence_history=args.div_history,
    )
    
    # Create solver with pre-loaded grid
    solver = RANSSolver.__new__(RANSSolver)
    solver.config = config
    solver.X = X
    solver.Y = Y
    solver.NI = X.shape[0] - 1
    solver.NJ = X.shape[1] - 1
    solver.iteration = 0
    solver.residual_history = []
    solver.converged = False
    
    # Initialize components
    solver._compute_metrics()
    solver._initialize_state()
    solver._initialize_output()
    
    print(f"\nGrid size: {solver.NI} x {solver.NJ} cells")
    print(f"Reynolds: {args.reynolds:.2e}")
    print(f"Target CFL: {args.cfl} with IRS ε={args.irs}")
    
    # Run
    try:
        if args.diagnostic or args.dump_freq < args.max_iter:
            # Use diagnostic mode with flow field dumps
            converged = solver.run_with_diagnostics(dump_freq=args.dump_freq)
        else:
            # Standard run
            converged = solver.run_steady_state()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        converged = False
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        converged = False
    
    # Final status
    print("\n" + "="*70)
    if converged:
        print("Simulation completed successfully - CONVERGED")
    else:
        print("Simulation completed - NOT CONVERGED")
    print("="*70 + "\n")
    
    return 0 if converged else 1


if __name__ == "__main__":
    sys.exit(main())
