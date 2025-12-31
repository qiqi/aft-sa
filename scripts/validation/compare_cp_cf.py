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

from src.validation.mfoil import mfoil
from src.grid.loader import load_or_generate_grid
from src.solvers.rans_solver import RANSSolver, SolverConfig


def run_mfoil_laminar(reynolds: float, alpha: float = 0.0):
    """Run mfoil and extract Cp distribution."""
    M = mfoil(naca='0012', npanel=199)
    M.param.ncrit = 1000.0  # Force laminar
    M.param.doplot = False
    M.param.verb = 0
    M.setoper(alpha=alpha, Re=reynolds)
    M.solve()
    
    # Extract surface data from mfoil
    x_coords = M.foil.x[0, :].copy()
    y_coords = M.foil.x[1, :].copy()
    
    # Get Cp and Cf from post-processing results
    cp_all = M.post.cp.copy() if hasattr(M.post, 'cp') and M.post.cp is not None else None
    cf_all = M.post.cf.copy() if hasattr(M.post, 'cf') and M.post.cf is not None else None
    
    N = M.foil.N
    n_half = N // 2
    
    # Upper surface: first half, reversed to go LE to TE
    x_upper = x_coords[:n_half+1][::-1]
    y_upper = y_coords[:n_half+1][::-1]
    
    # Lower surface: second half
    x_lower = x_coords[n_half:]
    y_lower = y_coords[n_half:]
    
    if cp_all is not None:
        cp_upper = cp_all[:n_half+1][::-1]
        cp_lower = cp_all[n_half:N]
    else:
        cp_upper = np.zeros(n_half+1)
        cp_lower = np.zeros(N - n_half)
    
    if cf_all is not None:
        cf_upper = cf_all[:n_half+1][::-1]
        cf_lower = cf_all[n_half:N]
    else:
        cf_upper = np.zeros(n_half+1)
        cf_lower = np.zeros(N - n_half)
    
    return {
        'x_upper': x_upper,
        'x_lower': x_lower,
        'cp_upper': cp_upper,
        'cp_lower': cp_lower,
        'cf_upper': cf_upper,
        'cf_lower': cf_lower,
        'cl': M.post.cl,
        'cd': M.post.cd,
        'cdf': M.post.cdf,
        'cdp': M.post.cdp,
    }


def extract_boundary_layer_profiles(solver, x_locations=[0.25, 0.5, 0.75], n_wake=30):
    """
    Extract boundary layer velocity profiles at specified x/c locations.
    
    Parameters
    ----------
    solver : RANSSolver
        Solver object with converged solution.
    x_locations : list
        List of x/c locations to extract profiles.
    n_wake : int
        Number of wake cells at each end of i-direction (to skip).
        
    Returns
    -------
    profiles : dict
        Dictionary with 'upper' and 'lower' keys, each containing list of profiles.
    """
    Q = solver.Q
    X = solver.X
    Y = solver.Y
    metrics = solver.metrics
    
    NI = X.shape[0] - 1
    NJ = X.shape[1] - 1
    
    Q_int = Q[1:-1, 1:-1, :]  # Strip ghost cells
    
    # Cell center coordinates
    xc = metrics.xc
    yc = metrics.yc
    
    # Surface x coordinates (at j=0)
    x_surface = xc[:, 0]
    y_surface = yc[:, 0]
    
    profiles = {'upper': [], 'lower': []}
    
    for x_target in x_locations:
        for surface in ['upper', 'lower']:
            best_i = None
            best_dist = float('inf')
            
            for i in range(n_wake, NI - n_wake):
                x_cell = x_surface[i]
                y_cell = y_surface[i]
                
                if x_cell < 0 or x_cell > 1:
                    continue
                    
                if surface == 'upper' and y_cell < -0.001:
                    continue
                if surface == 'lower' and y_cell > 0.001:
                    continue
                
                dist = abs(x_cell - x_target)
                if dist < best_dist:
                    best_dist = dist
                    best_i = i
            
            if best_i is None:
                continue
            
            y_profile = []
            u_profile = []
            v_profile = []
            
            x_wall = x_surface[best_i]
            y_wall = y_surface[best_i]
            
            n_profile = min(NJ, 20)
            for j in range(n_profile):
                y_dist = metrics.wall_distance[best_i, j]
                u = Q_int[best_i, j, 1]
                v = Q_int[best_i, j, 2]
                
                y_profile.append(y_dist)
                u_profile.append(u)
                v_profile.append(v)
            
            u_edge = np.sqrt(u_profile[-1]**2 + v_profile[-1]**2)
            
            profiles[surface].append({
                'x': x_wall,
                'y_wall': y_wall,
                'y': np.array(y_profile),
                'u': np.array(u_profile),
                'v': np.array(v_profile),
                'u_edge': u_edge,
            })
    
    return profiles


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare RANS Cp/Cf against mfoil")
    parser.add_argument("--reynolds", "-Re", type=float, default=10000,
                        help="Reynolds number (default: 10000)")
    parser.add_argument("--max-iter", "-n", type=int, default=4000,
                        help="Maximum iterations (default: 4000)")
    parser.add_argument("--cfl", type=float, default=0.8,
                        help="CFL number (default: 0.8)")
    parser.add_argument("--n-surface", type=int, default=100,
                        help="Surface points (default: 100)")
    parser.add_argument("--n-normal", type=int, default=40,
                        help="Normal points (default: 40)")
    parser.add_argument("--k4", type=float, default=0.016,
                        help="JST 4th-order dissipation coefficient (default: 0.016)")
    parser.add_argument("--output", "-o", type=str, default="output/cp_cf_comparison.pdf",
                        help="Output PDF path")
    parser.add_argument("--viscous", action="store_true",
                        help="Run with viscous fluxes (Navier-Stokes)")
    parser.add_argument("--inviscid", action="store_true",
                        help="Run without viscous fluxes (Euler)")
    args = parser.parse_args()
    
    # Determine viscous mode
    use_viscous = args.viscous or (not args.inviscid and args.reynolds < 1e6)
    
    print("="*60)
    print("  RANS vs mfoil Cp/Cf Comparison")
    print("="*60)
    print(f"Reynolds: {args.reynolds}")
    print(f"Mode: {'VISCOUS' if use_viscous else 'INVISCID'}")
    print(f"Iterations: {args.max_iter}")
    print(f"JST k4 (4th-order dissipation): {args.k4}")
    print(f"Grid: {args.n_surface} x {args.n_normal}")
    print()
    
    # Run mfoil first
    print("Running mfoil...")
    mfoil_result = run_mfoil_laminar(args.reynolds)
    print(f"  mfoil: CL={mfoil_result['cl']:.6f}, CD={mfoil_result['cd']:.6f}")
    print(f"         Cdf={mfoil_result['cdf']:.6f}, Cdp={mfoil_result['cdp']:.6f}")
    
    # Generate grid
    print("\nGenerating grid...")
    airfoil_file = project_root / "data" / "naca0012.dat"
    X, Y = load_or_generate_grid(
        str(airfoil_file),
        n_surface=args.n_surface,
        n_normal=args.n_normal,
        n_wake=30,
        y_plus=1.0,
        reynolds=args.reynolds,
        project_root=project_root,
        verbose=True
    )
    
    # Configure solver
    config = SolverConfig(
        mach=0.1,
        alpha=0.0,
        reynolds=args.reynolds,
        beta=10.0,
        cfl_start=args.cfl,
        cfl_target=args.cfl,
        cfl_ramp_iters=1,  # No ramping for simple case
        max_iter=args.max_iter,
        tol=1e-10,
        output_freq=args.max_iter + 1,  # No VTK output
        print_freq=200,
        output_dir="output/validation",
        case_name="cp_cf_comparison",
        jst_k2=0.5,
        jst_k4=args.k4,
    )
    
    # Create solver with pre-loaded grid
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
    
    # Initialize components
    solver._compute_metrics()
    solver._initialize_state()
    
    # Override viscous behavior if inviscid requested
    if not use_viscous:
        # Monkey-patch to skip viscous fluxes
        original_compute_residual = solver._compute_residual
        def inviscid_residual(Q):
            from src.numerics.fluxes import compute_fluxes, FluxConfig
            flux_cfg = FluxConfig(k2=config.jst_k2, k4=config.jst_k4)
            return compute_fluxes(Q, solver.flux_metrics, config.beta, flux_cfg)
        solver._compute_residual = inviscid_residual
    
    # Skip VTK output for this validation script
    class DummyVTKWriter:
        def write(self, *args, **kwargs): pass
        def finalize(self): return ""
    solver.vtk_writer = DummyVTKWriter()
    
    # Run
    solver.run_steady_state()
    
    # Get surface data
    surface = solver.get_surface_distributions()
    x_rans = surface.x
    cp_rans = surface.Cp
    cf_rans = surface.Cf
    y_rans = surface.y
    
    # Filter to airfoil only (exclude wake: x must be in [0, 1])
    airfoil_mask = (x_rans >= 0) & (x_rans <= 1.0)
    x_airfoil = x_rans[airfoil_mask]
    y_airfoil = y_rans[airfoil_mask]
    cp_airfoil = cp_rans[airfoil_mask]
    cf_airfoil = cf_rans[airfoil_mask]
    
    # Split upper/lower surface (by y coordinate)
    upper = y_airfoil >= 0
    lower = y_airfoil < 0
    
    # Sort each surface by x
    idx_upper = np.argsort(x_airfoil[upper])
    idx_lower = np.argsort(x_airfoil[lower])
    
    x_upper = x_airfoil[upper][idx_upper]
    x_lower = x_airfoil[lower][idx_lower]
    cp_upper = cp_airfoil[upper][idx_upper]
    cp_lower = cp_airfoil[lower][idx_lower]
    cf_upper = cf_airfoil[upper][idx_upper]
    cf_lower = cf_airfoil[lower][idx_lower]
    
    # mfoil data (already split into upper/lower)
    x_mfoil_upper = mfoil_result['x_upper']
    x_mfoil_lower = mfoil_result['x_lower']
    cp_mfoil_upper = mfoil_result['cp_upper']
    cp_mfoil_lower = mfoil_result['cp_lower']
    cf_mfoil_upper = mfoil_result['cf_upper']
    cf_mfoil_lower = mfoil_result['cf_lower']
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- Plot 1: Cp distribution ---
    ax = axes[0, 0]
    ax.plot(x_upper, -cp_upper, 'b-', lw=2, label='RANS upper')
    ax.plot(x_lower, -cp_lower, 'b--', lw=2, label='RANS lower')
    ax.plot(x_mfoil_upper, -cp_mfoil_upper, 'r-', lw=1.5, alpha=0.7, label='mfoil upper')
    ax.plot(x_mfoil_lower, -cp_mfoil_lower, 'r--', lw=1.5, alpha=0.7, label='mfoil lower')
    ax.set_xlabel('x/c')
    ax.set_ylabel('-Cp')
    ax.set_title(f'Pressure Coefficient (Re={args.reynolds:.0f})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    # --- Plot 2: Cf distribution ---
    ax = axes[0, 1]
    ax.plot(x_upper, cf_upper, 'b-', lw=2, label='RANS upper')
    ax.plot(x_lower, cf_lower, 'b--', lw=2, label='RANS lower')
    ax.plot(x_mfoil_upper, cf_mfoil_upper, 'r-', lw=1.5, alpha=0.7, label='mfoil upper')
    ax.plot(x_mfoil_lower, cf_mfoil_lower, 'r--', lw=1.5, alpha=0.7, label='mfoil lower')
    ax.set_xlabel('x/c')
    ax.set_ylabel('Cf')
    ax.set_title(f'Skin Friction Coefficient (Re={args.reynolds:.0f})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    # --- Plot 3: Full Cp vs x (airfoil only) ---
    ax = axes[1, 0]
    ax.plot(x_airfoil, -cp_airfoil, 'b.', markersize=3, label='RANS')
    ax.plot(x_mfoil_upper, -cp_mfoil_upper, 'r-', lw=1.5, alpha=0.7, label='mfoil')
    ax.plot(x_mfoil_lower, -cp_mfoil_lower, 'r-', lw=1.5, alpha=0.7)
    ax.set_xlabel('x/c')
    ax.set_ylabel('-Cp')
    ax.set_title('Cp Distribution (airfoil)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    # --- Plot 4: Cf distribution (airfoil only) ---
    ax = axes[1, 1]
    ax.plot(x_airfoil, cf_airfoil, 'b.', markersize=3, label='RANS')
    ax.plot(x_mfoil_upper, cf_mfoil_upper, 'r-', lw=1.5, alpha=0.7, label='mfoil')
    ax.plot(x_mfoil_lower, cf_mfoil_lower, 'r-', lw=1.5, alpha=0.7)
    ax.set_xlabel('x/c')
    ax.set_ylabel('Cf')
    ax.set_title('Cf Distribution (airfoil)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    # Add text with statistics (airfoil only)
    stats_text = (
        f"mfoil: CD={mfoil_result['cd']:.5f}, Cdf={mfoil_result['cdf']:.5f}\n"
        f"RANS Cp range: [{cp_airfoil.min():.3f}, {cp_airfoil.max():.3f}]\n"
        f"RANS Cf range: [{cf_airfoil.min():.5f}, {cf_airfoil.max():.5f}]"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save Cp/Cf comparison
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved comparison plot to: {output_path}")
    plt.close()
    
    # --- Boundary Layer Profile Plots ---
    print("Extracting boundary layer profiles...")
    bl_profiles = extract_boundary_layer_profiles(solver, x_locations=[0.25, 0.5, 0.75], n_wake=30)
    
    # Create BL profile figure
    fig_bl, axes_bl = plt.subplots(2, 3, figsize=(15, 10))
    
    # Upper surface profiles
    for idx, profile in enumerate(bl_profiles['upper']):
        if idx >= 3:
            break
        ax = axes_bl[0, idx]
        
        u_mag = np.sqrt(profile['u']**2 + profile['v']**2)
        u_edge = profile['u_edge']
        u_norm = u_mag / (u_edge + 1e-12)
        
        ax.plot(u_norm, profile['y'], 'b-', lw=2, label='|u|/u_e')
        ax.plot(profile['u'] / (u_edge + 1e-12), profile['y'], 'g--', lw=1.5, label='u/u_e')
        ax.plot(profile['v'] / (u_edge + 1e-12), profile['y'], 'r:', lw=1.5, label='v/u_e')
        
        ax.set_xlabel('u/u_e')
        ax.set_ylabel('y (wall distance)')
        ax.set_title(f"Upper Surface x/c = {profile['x']:.2f}")
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.2, 1.2)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=1, color='k', linestyle='--', alpha=0.3)
    
    # Lower surface profiles
    for idx, profile in enumerate(bl_profiles['lower']):
        if idx >= 3:
            break
        ax = axes_bl[1, idx]
        
        u_mag = np.sqrt(profile['u']**2 + profile['v']**2)
        u_edge = profile['u_edge']
        u_norm = u_mag / (u_edge + 1e-12)
        
        ax.plot(u_norm, profile['y'], 'b-', lw=2, label='|u|/u_e')
        ax.plot(profile['u'] / (u_edge + 1e-12), profile['y'], 'g--', lw=1.5, label='u/u_e')
        ax.plot(profile['v'] / (u_edge + 1e-12), profile['y'], 'r:', lw=1.5, label='v/u_e')
        
        ax.set_xlabel('u/u_e')
        ax.set_ylabel('y (wall distance)')
        ax.set_title(f"Lower Surface x/c = {profile['x']:.2f}")
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.2, 1.2)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=1, color='k', linestyle='--', alpha=0.3)
    
    fig_bl.suptitle(f'Boundary Layer Profiles (Re={args.reynolds:.0f})', fontsize=14)
    plt.tight_layout()
    
    # Save BL profiles
    bl_output_path = output_path.parent / (output_path.stem + '_bl_profiles.pdf')
    plt.savefig(bl_output_path, dpi=150)
    print(f"Saved boundary layer profiles to: {bl_output_path}")
    plt.close()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
