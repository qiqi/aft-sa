#!/usr/bin/env python3
"""
NACA 0012 Airfoil Simulation Script.

This script sets up and runs a 2D incompressible RANS simulation around
an airfoil using the artificial compressibility method.

Features:
    - YAML configuration files for easy parameter management
    - On-the-fly grid generation with configurable density
    - Interactive HTML visualization with animation
    - Residual monitoring and divergence detection
    - Surface data extraction

Usage:
    # Using YAML config (recommended)
    python run_airfoil.py --config config/examples/single_case.yaml
    
    # With CLI overrides
    python run_airfoil.py --config config/examples/quick_test.yaml --alpha 4.0
    
    # Legacy mode (positional grid file)
    python run_airfoil.py data/naca0012.dat --super-coarse --max-iter 100
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import device selection BEFORE JAX
from src.physics.jax_config import select_device


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run RANS simulation with diagnostic visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use YAML config
  %(prog)s --config config/examples/quick_test.yaml
  
  # YAML with overrides  
  %(prog)s --config config/examples/single_case.yaml --alpha 8.0
  
  # Legacy mode
  %(prog)s data/naca0012.dat --super-coarse --max-iter 500
"""
    )
    
    # Configuration file (new primary way)
    parser.add_argument(
        "--config", "-c",
        help="Path to YAML configuration file"
    )
    
    # Legacy: positional grid file
    parser.add_argument(
        "grid_file", nargs="?",
        help="Path to grid file (.p3d) or airfoil file (.dat)"
    )
    
    # Flow conditions (can override YAML)
    parser.add_argument("--mach", "-M", type=float,
                        help="Mach number")
    parser.add_argument("--alpha", "-a", type=float,
                        help="Angle of attack in degrees")
    parser.add_argument("--reynolds", "-Re", type=float,
                        help="Reynolds number")
    
    # Solver settings
    parser.add_argument("--max-iter", "-n", type=int,
                        help="Maximum iterations")
    parser.add_argument("--cfl", type=float,
                        help="Target CFL number")
    parser.add_argument("--cfl-start", type=float,
                        help="Initial CFL for ramping")
    parser.add_argument("--cfl-ramp", type=int,
                        help="CFL ramp iterations")
    parser.add_argument("--tol", type=float,
                        help="Convergence tolerance")
    parser.add_argument("--beta", type=float,
                        help="Artificial compressibility parameter")
    parser.add_argument("--irs", type=float,
                        help="Implicit Residual Smoothing epsilon (legacy)")
    
    # Grid generation options
    parser.add_argument("--n-surface", type=int,
                        help="Surface nodes for grid generation")
    parser.add_argument("--n-normal", type=int,
                        help="Normal nodes for grid generation")
    parser.add_argument("--n-wake", type=int,
                        help="Wake nodes for grid generation")
    parser.add_argument("--coarse", action="store_true",
                        help="Use coarse grid (256x32 cells)")
    parser.add_argument("--super-coarse", action="store_true",
                        help="Use super-coarse grid (128x32 cells)")
    
    # Output settings
    parser.add_argument("--output-dir", "-o", type=str,
                        help="Output directory")
    parser.add_argument("--case-name", type=str,
                        help="Case name for output files")
    parser.add_argument("--diagnostic-freq", "--dump-freq", type=int,
                        help="Diagnostic snapshot frequency")
    parser.add_argument("--print-freq", type=int,
                        help="Console print frequency")
    parser.add_argument("--div-history", type=int,
                        help="Solutions to keep for divergence visualization")
    
    # Device selection
    parser.add_argument("--device", "-d", type=str, default=None,
                        help="GPU device: 'auto' (default), 'cpu', or GPU index ('0', '1', 'cuda:0')")
    
    args = parser.parse_args()
    
    # Select device BEFORE importing JAX-dependent modules
    device_spec = args.device
    if args.config:
        # Peek at config for device setting if not specified on CLI
        import yaml
        with open(args.config) as f:
            raw_config = yaml.safe_load(f)
        if device_spec is None and 'device' in raw_config:
            device_spec = raw_config['device'].get('device', 'auto')
    
    print("Device selection:")
    select_device(device_spec, verbose=True)
    
    # Now import JAX-dependent modules (after device selection)
    from src.solvers.rans_solver import RANSSolver, SolverConfig
    from src.grid.loader import load_or_generate_grid
    from src.physics.jax_config import jax
    from src.config import load_yaml, from_dict, apply_cli_overrides, SimulationConfig
    
    # Load configuration
    if args.config:
        # YAML-based configuration
        try:
            sim_config = load_yaml(args.config)
            print(f"Loaded configuration from: {args.config}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR loading config: {e}")
            sys.exit(1)
        
        # Apply CLI overrides
        sim_config = apply_cli_overrides(sim_config, args)
        
    elif args.grid_file:
        # Legacy mode: build config from CLI args
        sim_config = SimulationConfig()
        sim_config.grid.airfoil = args.grid_file
        
        # Apply defaults for --coarse and --super-coarse
        if args.super_coarse:
            sim_config.grid.n_surface = 65
            sim_config.grid.n_normal = 33
            sim_config.grid.n_wake = 32
            sim_config.grid.y_plus = 5.0
            print("Using SUPER-COARSE grid mode (128x32 cells)")
        elif args.coarse:
            sim_config.grid.n_surface = 193
            sim_config.grid.n_normal = 33
            sim_config.grid.n_wake = 32
            sim_config.grid.y_plus = 1.0
            print("Using COARSE grid mode (256x32 cells)")
        
        # Apply CLI overrides
        sim_config = apply_cli_overrides(sim_config, args)
        
    else:
        parser.print_help()
        print("\nERROR: Must provide either --config or a grid file")
        sys.exit(1)
    
    # Handle legacy --coarse/--super-coarse flags with --config
    if args.config:
        if args.super_coarse:
            sim_config.grid.n_surface = 65
            sim_config.grid.n_normal = 33
            sim_config.grid.n_wake = 32
            sim_config.grid.y_plus = 5.0
            print("Override: Using SUPER-COARSE grid (128x32 cells)")
        elif args.coarse:
            sim_config.grid.n_surface = 193
            sim_config.grid.n_normal = 33
            sim_config.grid.n_wake = 32
            sim_config.grid.y_plus = 1.0
            print("Override: Using COARSE grid (256x32 cells)")
    
    # Convert to legacy SolverConfig
    config = sim_config.to_solver_config()
    
    # Print banner
    print("\n" + "="*70)
    print("   2D Incompressible RANS Solver")
    print("="*70)
    
    # Grid generation settings
    y_plus = sim_config.grid.y_plus
    max_first_cell = 0.005 if sim_config.grid.n_normal <= 33 else 0.001
    
    # Load or generate grid
    try:
        X, Y = load_or_generate_grid(
            sim_config.grid.airfoil,
            n_surface=sim_config.grid.n_surface,
            n_normal=sim_config.grid.n_normal,
            n_wake=sim_config.grid.n_wake,
            y_plus=y_plus,
            reynolds=sim_config.flow.reynolds,
            farfield_radius=sim_config.grid.farfield_radius,
            max_first_cell=max_first_cell,
            project_root=project_root,
            verbose=True
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Create solver with pre-loaded grid
    solver = RANSSolver.__new__(RANSSolver)
    solver.config = config
    solver.X = X
    solver.Y = Y
    solver.NI = X.shape[0] - 1
    solver.NJ = X.shape[1] - 1
    solver.iteration = 0
    solver.residual_history = []
    solver.iteration_history = []  # Track which iteration each residual corresponds to
    solver.converged = False
    
    # Initialize components
    solver._compute_metrics()
    solver._initialize_state()
    solver._initialize_output()
    
    print(f"\nGrid size: {solver.NI} x {solver.NJ} cells")
    print(f"Reynolds: {sim_config.flow.reynolds:.2e}")
    print(f"Alpha: {sim_config.flow.alpha}°")
    print(f"Backend: JAX ({jax.devices()[0].device_kind})")
    print(f"Target CFL: {config.cfl_target} (ramp from {config.cfl_start} over {config.cfl_ramp_iters} iters)")
    print(f"Output: HTML animation every {config.diagnostic_freq} iterations")
    
    # Run
    try:
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
