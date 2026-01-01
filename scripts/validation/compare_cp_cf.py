#!/usr/bin/env python3
"""
Compare RANS solver Cp and Cf distributions against mfoil baseline.

This script runs the RANS solver and visualizes the surface pressure and
skin friction distributions compared to mfoil (panel code) results.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.validation.mfoil_runner import run_laminar as run_mfoil_laminar
from src.grid.loader import load_or_generate_grid
from src.solvers.rans_solver import RANSSolver, SolverConfig


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare RANS Cp/Cf against mfoil")
    parser.add_argument("airfoil", nargs="?", default=None,
                        help="Path to airfoil .dat file (default: data/naca0012.dat)")
    parser.add_argument("--naca", type=str, default="0012",
                        help="NACA 4-digit code if no airfoil file (default: 0012)")
    parser.add_argument("--alpha", "-a", type=float, default=0.0,
                        help="Angle of attack in degrees (default: 0.0)")
    parser.add_argument("--reynolds", "-Re", type=float, default=10000,
                        help="Reynolds number (default: 10000)")
    parser.add_argument("--max-iter", "-n", type=int, default=4000,
                        help="Maximum iterations (default: 4000)")
    parser.add_argument("--cfl", type=float, default=3.0,
                        help="CFL number (default: 3.0)")
    parser.add_argument("--n-surface", type=int, default=100,
                        help="Surface points (default: 100)")
    parser.add_argument("--n-normal", type=int, default=40,
                        help="Normal points (default: 40)")
    parser.add_argument("--k4", type=float, default=0.016,
                        help="JST 4th-order dissipation coefficient (default: 0.016)")
    parser.add_argument("--irs", type=float, default=0.5,
                        help="Implicit Residual Smoothing epsilon (default: 0.5)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output PDF path (default: auto-generated)")
    args = parser.parse_args()
    
    # Determine airfoil file
    if args.airfoil is not None:
        airfoil_file = Path(args.airfoil)
        airfoil_name = airfoil_file.stem
    else:
        airfoil_file = project_root / "data" / f"naca{args.naca}.dat"
        airfoil_name = f"naca{args.naca}"
    
    # Generate output path if not specified
    if args.output is None:
        args.output = f"output/{airfoil_name}_a{args.alpha:.1f}_Re{args.reynolds:.0f}.pdf"
    
    print("="*70)
    print("  RANS vs mfoil Cp/Cf Comparison")
    print("="*70)
    print(f"Airfoil:  {airfoil_name}")
    print(f"Alpha:    {args.alpha}°")
    print(f"Reynolds: {args.reynolds}")
    print(f"Grid:     {args.n_surface} x {args.n_normal}")
    print(f"CFL:      {args.cfl} with IRS ε={args.irs}")
    print()
    
    # Run mfoil
    print("Running mfoil...")
    if args.airfoil is not None:
        mfoil_result = run_mfoil_laminar(
            args.reynolds, args.alpha, airfoil_file=str(airfoil_file)
        )
    else:
        mfoil_result = run_mfoil_laminar(
            args.reynolds, args.alpha, naca=args.naca
        )
    
    # Generate grid
    print("\nGenerating grid...")
    N_WAKE = 30
    X, Y = load_or_generate_grid(
        str(airfoil_file),
        n_surface=args.n_surface,
        n_normal=args.n_normal,
        n_wake=N_WAKE,
        y_plus=1.0,
        reynolds=args.reynolds,
        project_root=project_root,
        verbose=True
    )
    
    # Configure solver with IRS for stability
    config = SolverConfig(
        mach=0.1,
        alpha=args.alpha,
        reynolds=args.reynolds,
        beta=10.0,
        cfl_start=args.cfl,
        cfl_target=args.cfl,
        cfl_ramp_iters=1,
        max_iter=args.max_iter,
        tol=1e-10,
        output_freq=args.max_iter + 1,
        print_freq=200,
        output_dir="output/validation",
        case_name="cp_cf_comparison",
        jst_k4=args.k4,
        n_wake=N_WAKE,
        irs_epsilon=args.irs,
    )
    
    # Create and run solver
    print("\nRunning RANS solver...")
    solver = RANSSolver.__new__(RANSSolver)
    solver.config = config
    solver.X = X
    solver.Y = Y
    solver.NI = X.shape[0] - 1
    solver.NJ = X.shape[1] - 1
    solver.iteration = 0
    solver.residual_history = []
    solver.converged = False
    
    solver._compute_metrics()
    solver._initialize_state()
    
    # Dummy VTK writer
    class DummyVTKWriter:
        def write(self, *args, **kwargs): pass
        def finalize(self): return ""
    solver.vtk_writer = DummyVTKWriter()
    
    solver.run_steady_state()
    
    # Compute forces
    rans_forces = solver.compute_forces()
    
    # Print force comparison
    print("\n" + "="*70)
    print("  FORCE COEFFICIENT COMPARISON")
    print("="*70)
    print(f"{'Coefficient':<15} {'mfoil':>12} {'RANS':>12} {'Difference':>12} {'Error %':>10}")
    print("-"*70)
    
    cl_mfoil, cl_rans = mfoil_result['cl'], rans_forces.CL
    cd_mfoil, cd_rans = mfoil_result['cd'], rans_forces.CD
    cdp_mfoil, cdp_rans = mfoil_result['cdp'], rans_forces.CD_p
    cdf_mfoil, cdf_rans = mfoil_result['cdf'], rans_forces.CD_f
    
    for name, mf, ra in [('CL', cl_mfoil, cl_rans), ('CD', cd_mfoil, cd_rans),
                          ('CD_pressure', cdp_mfoil, cdp_rans), ('CD_friction', cdf_mfoil, cdf_rans)]:
        diff = ra - mf
        err = abs(diff / (mf + 1e-10)) * 100 if abs(mf) > 1e-6 else 0
        print(f"{name:<15} {mf:>12.6f} {ra:>12.6f} {diff:>+12.6f} {err:>10.2f}")
    print("-"*70)
    
    # Get surface data
    surface = solver.get_surface_distributions()
    x_rans, cp_rans, cf_rans, y_rans = surface.x, surface.Cp, surface.Cf, surface.y
    
    # Filter to airfoil only (exclude wake: x in [0, 1])
    airfoil_mask = (x_rans >= 0) & (x_rans <= 1.0)
    x_airfoil = x_rans[airfoil_mask]
    y_airfoil = y_rans[airfoil_mask]
    cp_airfoil = cp_rans[airfoil_mask]
    cf_airfoil = cf_rans[airfoil_mask]
    
    # Split upper/lower surface
    upper = y_airfoil >= 0
    lower = y_airfoil < 0
    
    idx_upper = np.argsort(x_airfoil[upper])
    idx_lower = np.argsort(x_airfoil[lower])
    
    x_upper, x_lower = x_airfoil[upper][idx_upper], x_airfoil[lower][idx_lower]
    cp_upper, cp_lower = cp_airfoil[upper][idx_upper], cp_airfoil[lower][idx_lower]
    cf_upper, cf_lower = cf_airfoil[upper][idx_upper], cf_airfoil[lower][idx_lower]
    
    # Create comparison plots
    print("Creating comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    fig.suptitle(f'{airfoil_name.upper()}, α={args.alpha}°, Re={args.reynolds:.0f}', 
                 fontsize=14, fontweight='bold')
    
    # Cp distribution (inverted y-axis: suction up, pressure down)
    ax = axes[0, 0]
    ax.plot(x_upper, cp_upper, 'b-', lw=2, label='RANS upper')
    ax.plot(x_lower, cp_lower, 'b--', lw=2, label='RANS lower')
    ax.plot(mfoil_result['x_upper'], mfoil_result['cp_upper'], 'r-', lw=1.5, alpha=0.7, label='mfoil upper')
    ax.plot(mfoil_result['x_lower'], mfoil_result['cp_lower'], 'r--', lw=1.5, alpha=0.7, label='mfoil lower')
    ax.set_xlabel('x/c')
    ax.set_ylabel('Cp')
    ax.set_title('Pressure Coefficient')
    ax.invert_yaxis()  # Suction (negative Cp) at top
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    # Cf distribution
    ax = axes[0, 1]
    ax.plot(x_upper, cf_upper, 'b-', lw=2, label='RANS upper')
    ax.plot(x_lower, cf_lower, 'b--', lw=2, label='RANS lower')
    ax.plot(mfoil_result['x_upper'], mfoil_result['cf_upper'], 'r-', lw=1.5, alpha=0.7, label='mfoil upper')
    ax.plot(mfoil_result['x_lower'], mfoil_result['cf_lower'], 'r--', lw=1.5, alpha=0.7, label='mfoil lower')
    ax.set_xlabel('x/c')
    ax.set_ylabel('Cf')
    ax.set_title('Skin Friction Coefficient')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    # Full Cp (inverted y-axis)
    ax = axes[1, 0]
    ax.plot(x_airfoil, cp_airfoil, 'b.', markersize=3, label='RANS')
    ax.plot(mfoil_result['x_upper'], mfoil_result['cp_upper'], 'r-', lw=1.5, alpha=0.7, label='mfoil')
    ax.plot(mfoil_result['x_lower'], mfoil_result['cp_lower'], 'r-', lw=1.5, alpha=0.7)
    ax.set_xlabel('x/c')
    ax.set_ylabel('Cp')
    ax.set_title('Cp Distribution (all points)')
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    # Full Cf
    ax = axes[1, 1]
    ax.plot(x_airfoil, cf_airfoil, 'b.', markersize=3, label='RANS')
    ax.plot(mfoil_result['x_upper'], mfoil_result['cf_upper'], 'r-', lw=1.5, alpha=0.7, label='mfoil')
    ax.plot(mfoil_result['x_lower'], mfoil_result['cf_lower'], 'r-', lw=1.5, alpha=0.7)
    ax.set_xlabel('x/c')
    ax.set_ylabel('Cf')
    ax.set_title('Cf Distribution (all points)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    # Add statistics text
    stats_text = (
        f"       mfoil     RANS\n"
        f"CL: {cl_mfoil:8.5f} {cl_rans:8.5f}\n"
        f"CD: {cd_mfoil:8.5f} {cd_rans:8.5f}\n"
        f"CDp:{cdp_mfoil:8.5f} {cdp_rans:8.5f}\n"
        f"CDf:{cdf_mfoil:8.5f} {cdf_rans:8.5f}"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved comparison plot to: {output_path}")
    plt.close()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
