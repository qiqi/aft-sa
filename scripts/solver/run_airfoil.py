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
    python run_airfoil.py config/examples/single_case.yaml
    
    # With CLI overrides
    python run_airfoil.py config/examples/quick_test.yaml --alpha 4.0
"""

import sys
import argparse
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

# Import device selection BEFORE JAX
from src.physics.jax_config import select_device


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run RANS simulation with diagnostic visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config/examples/quick_test.yaml
  %(prog)s config/examples/single_case.yaml --alpha 8.0
  %(prog)s config/examples/naca0012_newton.yaml --max-iter 500
"""
    )
    
    # Configuration file (required positional argument)
    parser.add_argument(
        "config",
        help="Path to YAML configuration file"
    )
    
    # Flow conditions (can override YAML)
    parser.add_argument("--mach", "-M", type=float,
                        help="Mach number")
    parser.add_argument("--alpha", "-a", type=float,
                        help="Angle of attack in degrees")
    parser.add_argument("--reynolds", "-Re", type=float,
                        help="Reynolds number")
    parser.add_argument("--chi-inf", type=float,
                        help="Initial/farfield turbulent viscosity ratio χ = ν̃/ν (default: 0.0001 for AFT)")
    
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
    
    # Grid generation options
    parser.add_argument("--n-surface", type=int,
                        help="Surface nodes for grid generation")
    parser.add_argument("--n-normal", type=int,
                        help="Normal nodes for grid generation")
    parser.add_argument("--n-wake", type=int,
                        help="Wake nodes for grid generation")
    parser.add_argument("--wake-fan-factor", type=float,
                        help="Wake expansion factor (default: 0.005)")
    parser.add_argument("--wake-fan-k", type=float,
                        help="Wake expansion exponent k (default: 10.0)")
    
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
    
    # Always keep JIT enabled for performance (override any env setting).
    os.environ["JAX_DISABLE_JIT"] = "0"

    # Select device BEFORE importing JAX-dependent modules
    # Peek at config for device setting if not specified on CLI
    import yaml
    with open(args.config) as f:
        raw_config = yaml.safe_load(f)
    device_spec = args.device
    if device_spec is None and 'device' in raw_config:
        device_spec = raw_config['device'].get('device', 'auto')
    
    logger.info("Device selection:")
    select_device(device_spec, verbose=True)
    
    # Now import JAX-dependent modules (after device selection)
    from src.solvers.rans_solver import RANSSolver
    from src.grid.loader import load_or_generate_grid
    from src.physics.jax_config import jax
    from src.config import load_yaml, apply_cli_overrides
    
    # Load configuration
    try:
        sim_config = load_yaml(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    except FileNotFoundError as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"ERROR loading config: {e}")
        sys.exit(1)
    
    # Apply CLI overrides
    sim_config = apply_cli_overrides(sim_config, args)
    
    # Convert to legacy SolverConfig
    config = sim_config.to_solver_config()
    
    # Print banner
    logger.info("\n" + "="*70)
    logger.info("   2D Incompressible RANS Solver")
    logger.info("="*70)
    
    # Grid generation settings
    y_plus = sim_config.grid.y_plus
    gradation = sim_config.grid.gradation
    # Coarser grids (high gradation) need larger max_first_cell for safety
    max_first_cell = 0.005
    
    # Load or generate grid
    try:
        X, Y = load_or_generate_grid(
            sim_config.grid.airfoil,
            n_surface=sim_config.grid.n_surface,
            n_wake=sim_config.grid.n_wake,
            y_plus=y_plus,
            gradation=gradation,
            reynolds=sim_config.flow.reynolds,
            farfield_radius=sim_config.grid.farfield_radius,
            wake_fan_factor=sim_config.grid.wake_fan_factor,
            wake_fan_k=sim_config.grid.wake_fan_k,
            max_first_cell=max_first_cell,
            project_root=project_root,
            verbose=True
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)
    
    # Save grid to output folder for debugging/comparison
    output_dir = Path(sim_config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_output_path = output_dir / "grid.npz"
    np.savez(str(grid_output_path), X=X, Y=Y)
    logger.info(f"Saved grid to: {grid_output_path}")
    
    # Create solver with pre-loaded grid
    solver = RANSSolver(grid_data=(X, Y), config=config)

    # Optional: overlay mfoil reference Cp/Cf as dotted lines in HTML plots
    mfoil_future = None
    mfoil_executor = None

    def _apply_mfoil_overlay(mfoil_result):
        if not mfoil_result.get("converged", False):
            logger.warning("mfoil did not converge; skipping Cp/Cf overlay.")
            return

        x_upper = mfoil_result["x_upper"]
        x_lower = mfoil_result["x_lower"]
        cp_upper = mfoil_result["cp_upper"]
        cp_lower = mfoil_result["cp_lower"]
        cf_upper = mfoil_result["cf_upper"]
        cf_lower = mfoil_result["cf_lower"]

        x_surf = 0.5 * (X[:-1, 0] + X[1:, 0])
        y_surf = 0.5 * (Y[:-1, 0] + Y[1:, 0])
        i_start = sim_config.grid.n_wake
        i_end = len(x_surf) - sim_config.grid.n_wake
        x_surf_airfoil = x_surf[i_start:i_end]
        y_surf_airfoil = y_surf[i_start:i_end]

        order_upper = np.argsort(x_upper)
        order_lower = np.argsort(x_lower)
        cp_upper_interp = np.interp(x_surf_airfoil, x_upper[order_upper], cp_upper[order_upper])
        cp_lower_interp = np.interp(x_surf_airfoil, x_lower[order_lower], cp_lower[order_lower])
        cf_upper_interp = np.interp(x_surf_airfoil, x_upper[order_upper], cf_upper[order_upper])
        cf_lower_interp = np.interp(x_surf_airfoil, x_lower[order_lower], cf_lower[order_lower])

        cp_mfoil = np.where(y_surf_airfoil >= 0.0, cp_upper_interp, cp_lower_interp)
        cf_mfoil = np.where(y_surf_airfoil >= 0.0, cf_upper_interp, cf_lower_interp)

        solver.plotter.set_surface_reference(x_surf_airfoil, cp_mfoil, cf_mfoil)
        logger.info("Added mfoil Cp/Cf overlay to HTML surface plots.")

    try:
        from src.validation.mfoil_runner import run_reference

        airfoil_path = Path(sim_config.grid.airfoil)
        if not airfoil_path.is_absolute():
            airfoil_path = project_root / airfoil_path

        mfoil_executor = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
        mfoil_future = mfoil_executor.submit(
            run_reference,
            reynolds=sim_config.flow.reynolds,
            alpha=sim_config.flow.alpha,
            airfoil_file=str(airfoil_path),
            quiet=True,
        )
    except Exception as exc:
        logger.warning(f"mfoil overlay skipped: {exc}")
    
    logger.info(f"Grid size: {solver.NI} x {solver.NJ} cells")
    logger.info(f"Reynolds: {sim_config.flow.reynolds:.2e}")
    logger.info(f"Alpha: {sim_config.flow.alpha}°")
    logger.info(f"Backend: JAX ({jax.devices()[0].device_kind})")
    logger.info(f"Target CFL: {config.cfl_target} (ramp from {config.cfl_start} over {config.cfl_ramp_iters} iters)")
    logger.info(f"Turbulence model: SA-AF (transition), χ_inf={config.chi_inf:.4g}")
    logger.info(f"Output: HTML animation every {config.diagnostic_freq} iterations")
    
    # Run
    try:
        converged = solver.run_steady_state()
    except KeyboardInterrupt:
        logger.warning("\n\nSimulation interrupted by user.")
        converged = False
    except Exception as e:
        logger.exception(f"\nError during simulation: {e}")
        converged = False

    if mfoil_future is not None:
        try:
            mfoil_result = mfoil_future.result()
            _apply_mfoil_overlay(mfoil_result)
            solver.save_diagnostics()
        except Exception as exc:
            logger.warning(f"mfoil overlay skipped: {exc}")
        finally:
            if mfoil_executor is not None:
                mfoil_executor.shutdown(wait=False, cancel_futures=False)
    
    # Final status
    logger.info("\n" + "="*70)
    if converged:
        logger.info("Simulation completed successfully - CONVERGED")
    else:
        logger.info("Simulation completed - NOT CONVERGED")
    logger.info("="*70 + "\n")
    
    return 0 if converged else 1


if __name__ == "__main__":
    sys.exit(main())
