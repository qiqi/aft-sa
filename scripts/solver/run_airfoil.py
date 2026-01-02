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
    
    # Multigrid options
    parser.add_argument("--multigrid", "-mg", action="store_true",
                        help="Enable FAS multigrid acceleration")
    parser.add_argument("--mg-levels", type=int, default=4,
                        help="Maximum multigrid levels (default: 4)")
    parser.add_argument("--mg-nu1", type=int, default=2,
                        help="Pre-smoothing iterations per level (default: 2)")
    parser.add_argument("--mg-nu2", type=int, default=2,
                        help="Post-smoothing iterations per level (default: 2)")
    parser.add_argument("--mg-omega", type=float, default=0.5,
                        help="Multigrid correction relaxation factor (default: 0.5)")
    
    # Grid generation options (nodes = cells + 1, power-of-2 cells for multigrid)
    parser.add_argument("--n-surface", type=int, default=257,
                        help="Surface nodes for grid generation (default: 257 = 256 cells)")
    parser.add_argument("--n-normal", type=int, default=65,
                        help="Normal nodes for grid generation (default: 65 = 64 cells)")
    parser.add_argument("--n-wake", type=int, default=64,
                        help="Wake nodes for grid generation (default: 64)")
    parser.add_argument("--coarse", action="store_true",
                        help="Use coarse grid (128x32 cells) for debugging")
    parser.add_argument("--super-coarse", action="store_true",
                        help="Use super-coarse grid (64x16 cells) for fast tests")
    
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
    # For C-grid: total_I_nodes = n_surface + 2*n_wake, cells = nodes - 1
    # For power-of-2 cells: n_surface + 2*n_wake = power_of_2 + 1
    # Note: construct2d needs at least ~130 surface points for C-grid stability
    if args.super_coarse:
        # Use 32 J-cells (minimum for stability) with coarser I-grid
        # 128 I-cells: n_surface + 2*n_wake = 129, e.g., 65 + 2*32 = 129
        args.n_surface = 65
        args.n_normal = 33    # 32 J-cells (minimum for stability)
        args.n_wake = 32      # 65 + 64 = 129 nodes → 128 cells
        args.y_plus = 5.0
        # Reduce CFL for coarse grid stability
        if args.cfl > 3.0:
            args.cfl = 3.0
        print("Using SUPER-COARSE grid mode (128x32 cells)")
    elif args.coarse:
        # 256 I-cells: n_surface + 2*n_wake = 257, e.g., 193 + 2*32 = 257
        args.n_surface = 193
        args.n_normal = 33    # 32 J-cells  
        args.n_wake = 32
        args.y_plus = 1.0
        print("Using COARSE grid mode (256x32 cells)")
    
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
        # Multigrid options
        use_multigrid=args.multigrid,
        mg_levels=args.mg_levels,
        mg_nu1=args.mg_nu1,
        mg_nu2=args.mg_nu2,
        mg_omega=args.mg_omega,
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
    
    # Initialize multigrid if enabled
    solver.mg_hierarchy = None
    if config.use_multigrid:
        solver._initialize_multigrid()
    
    solver._initialize_output()
    
    print(f"\nGrid size: {solver.NI} x {solver.NJ} cells")
    print(f"Reynolds: {args.reynolds:.2e}")
    mg_info = f" + Multigrid ({args.mg_levels} levels)" if args.multigrid else ""
    print(f"Target CFL: {args.cfl} with IRS ε={args.irs}{mg_info}")
    
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
