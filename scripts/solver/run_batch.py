#!/usr/bin/env python3
"""
Batch RANS Solver - Run AoA sweeps with GPU parallelization.

Usage:
    python scripts/solver/run_batch.py --config config/examples/aoa_sweep.yaml
    python scripts/solver/run_batch.py data/naca0012.p3d --alpha-sweep -5 15 21

Examples:
    # Using YAML config (recommended)
    python scripts/solver/run_batch.py --config config/examples/aoa_sweep.yaml
    
    # Quick command-line sweep
    python scripts/solver/run_batch.py data/naca0012.p3d --alpha-sweep 0 10 11 --max-iter 2000
    
    # Explicit alpha list
    python scripts/solver/run_batch.py data/naca0012.p3d --alpha-values 0 2 4 6 8
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import device selection BEFORE JAX
from src.physics.jax_config import select_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run batch RANS simulation (AoA sweep)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Grid specification
    parser.add_argument('grid', nargs='?', help="Grid file (.p3d) or airfoil (.dat)")
    parser.add_argument('--config', '-c', help="YAML config file (overrides other args)")
    
    # Flow conditions
    parser.add_argument('--reynolds', '-Re', type=float, default=6e6,
                        help="Reynolds number (default: 6e6)")
    parser.add_argument('--mach', '-M', type=float, default=0.0,
                        help="Mach number (default: 0.0, incompressible)")
    parser.add_argument('--chi-inf', type=float, default=0.0001,
                        help="Initial/farfield turbulent viscosity ratio χ = ν̃/ν (default: 0.0001)")
    
    # Alpha specification (mutually exclusive)
    alpha_group = parser.add_mutually_exclusive_group()
    alpha_group.add_argument('--alpha-sweep', '-as', nargs=3, type=float,
                            metavar=('START', 'END', 'COUNT'),
                            help="Linear AoA sweep: start end count")
    alpha_group.add_argument('--alpha-values', '-av', nargs='+', type=float,
                            help="Explicit list of AoA values")
    alpha_group.add_argument('--alpha', '-a', type=float, default=0.0,
                            help="Single AoA (no batch)")
    
    # Solver settings
    parser.add_argument('--max-iter', type=int, default=5000,
                        help="Maximum iterations (default: 5000)")
    parser.add_argument('--tol', type=float, default=1e-10,
                        help="Convergence tolerance (default: 1e-10)")
    parser.add_argument('--cfl', type=float, default=5.0,
                        help="Target CFL (default: 5.0)")
    parser.add_argument('--beta', type=float, default=10.0,
                        help="Artificial compressibility (default: 10.0)")
    parser.add_argument('--k4', type=float, default=0.04,
                        help="JST 4th-order dissipation (default: 0.04)")
    
    # Output
    parser.add_argument('--output-dir', '-o', default='output/batch',
                        help="Output directory (default: output/batch)")
    parser.add_argument('--print-freq', type=int, default=50,
                        help="Print frequency (default: 50)")
    
    # Device selection
    parser.add_argument('--device', '-d', type=str, default=None,
                        help="GPU device: 'auto' (default), 'cpu', or GPU index ('0', '1', 'cuda:0')")
    
    return parser.parse_args()


def create_flow_conditions(args, BatchFlowConditions):
    """Create BatchFlowConditions from arguments."""
    if args.alpha_sweep:
        start, end, count = args.alpha_sweep
        alpha_spec = {'sweep': [start, end, int(count)]}
    elif args.alpha_values:
        alpha_spec = {'values': args.alpha_values}
    else:
        alpha_spec = args.alpha  # Single value
    
    return BatchFlowConditions.from_sweep(
        alpha_spec=alpha_spec,
        reynolds=args.reynolds,
        mach=args.mach,
        chi_inf=args.chi_inf
    )


def main():
    args = parse_args()
    
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
    
    # Now import JAX-dependent modules
    from src.solvers.batch import BatchRANSSolver, BatchFlowConditions
    
    # Handle config file
    if args.config:
        from src.config.loader import load_config
        config = load_config(args.config)
        
        # Extract grid and flow settings from config
        grid_file = config.grid.get('airfoil') or config.grid.get('file')
        if not grid_file:
            print("Error: Config must specify grid.airfoil or grid.file")
            sys.exit(1)
        
        # Create flow conditions from config
        alpha_spec = config.flow.get('alpha', 0.0)
        reynolds = config.flow.get('reynolds', 6e6)
        mach = config.flow.get('mach', 0.0)
        chi_inf = config.flow.get('chi_inf', 0.0001)
        
        flow_conditions = BatchFlowConditions.from_sweep(
            alpha_spec=alpha_spec,
            reynolds=reynolds,
            mach=mach,
            chi_inf=chi_inf
        )
        
        # Solver settings from config
        max_iter = config.solver.get('max_iter', 5000)
        tol = config.solver.get('tol', 1e-10)
        cfl = config.solver.get('cfl', {}).get('final', 5.0) if isinstance(config.solver.get('cfl'), dict) else config.solver.get('cfl', 5.0)
        beta = config.numerics.get('beta', 10.0)
        k4 = config.numerics.get('jst_k4', 0.04)
        output_dir = config.output.get('directory', 'output/batch')
        print_freq = config.solver.get('print_freq', 50)
    else:
        if not args.grid:
            print("Error: Must specify either --config or grid file")
            sys.exit(1)
        
        grid_file = args.grid
        flow_conditions = create_flow_conditions(args, BatchFlowConditions)
        max_iter = args.max_iter
        tol = args.tol
        cfl = args.cfl
        beta = args.beta
        k4 = args.k4
        output_dir = args.output_dir
        print_freq = args.print_freq
    
    # Verify grid exists
    grid_path = Path(grid_file)
    if not grid_path.exists():
        print(f"Error: Grid file not found: {grid_path}")
        sys.exit(1)
    
    # Create and run solver
    print(f"\n{'='*60}")
    print("Batch RANS Solver")
    print(f"{'='*60}")
    print(f"Grid: {grid_file}")
    print(f"Cases: {flow_conditions.n_batch}")
    print(f"Alpha range: {flow_conditions.alpha_deg.min():.1f}° to {flow_conditions.alpha_deg.max():.1f}°")
    print(f"Reynolds: {flow_conditions.reynolds[0]:.2e}")
    print(f"Turbulence model: SA (fully turbulent)")
    print(f"  Note: AFT transition model available in run_airfoil.py")
    print(f"{'='*60}\n")
    
    solver = BatchRANSSolver(
        grid_file=str(grid_path),
        flow_conditions=flow_conditions,
        beta=beta,
        k4=k4,
        cfl_target=cfl,
        max_iter=max_iter,
        tol=tol,
        print_freq=print_freq,
        output_dir=output_dir,
    )
    
    # Run
    forces = solver.run()
    
    # Print results
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    data = forces.to_dataframe()
    
    # Print table
    ld = forces.CL / (forces.CD + 1e-12)
    print(f"{'alpha':>8} {'CL':>10} {'CD':>10} {'CD_p':>10} {'CD_f':>10} {'L/D':>10}")
    print("-" * 60)
    for i in range(len(forces.alpha_deg)):
        print(f"{forces.alpha_deg[i]:>8.1f} {forces.CL[i]:>10.4f} {forces.CD[i]:>10.5f} "
              f"{forces.CD_p[i]:>10.5f} {forces.CD_f[i]:>10.5f} {ld[i]:>10.2f}")
    
    # Find max L/D
    max_ld_idx = int(np.argmax(ld))
    print(f"\nMax L/D = {ld[max_ld_idx]:.2f} at α = {forces.alpha_deg[max_ld_idx]:.1f}°")
    
    # Save residual history
    solver.save_residual_history()
    
    print(f"\nDone! Results in: {output_dir}")


if __name__ == "__main__":
    main()
