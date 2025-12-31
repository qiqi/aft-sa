#!/usr/bin/env python3
"""
Compare RANS solver Cp and Cf distributions against mfoil baseline.

This script runs the RANS solver and visualizes the surface pressure and
skin friction distributions compared to mfoil (panel code) results.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.validation.mfoil import mfoil
from src.grid.mesher import Construct2DWrapper, GridOptions
from src.grid.plot3d import read_plot3d
from src.grid.metrics import MetricComputer
from src.solvers.boundary_conditions import (
    FreestreamConditions, BoundaryConditions, 
    initialize_state, apply_initial_wall_damping
)
from src.numerics.fluxes import compute_fluxes, FluxConfig, GridMetrics as FluxGridMetrics
from src.solvers.time_stepping import compute_local_timestep, TimeStepConfig


def run_mfoil_laminar(reynolds: float, alpha: float = 0.0):
    """Run mfoil and extract Cp distribution."""
    M = mfoil(naca='0012', npanel=199)
    M.param.ncrit = 1000.0  # Force laminar
    M.param.doplot = False
    M.param.verb = 0
    M.setoper(alpha=alpha, Re=reynolds)
    M.solve()
    
    # Extract surface data from mfoil
    # M.foil.x has shape (2, N) where x[0,:] = x-coords, x[1,:] = y-coords
    # M.vsol.Is tells us the indexing:
    #   Is[0] = upper surface (TE to LE, reversed)
    #   Is[1] = lower surface (LE to TE)
    #   Is[2] = wake
    x_coords = M.foil.x[0, :].copy()  # x coordinates on foil
    y_coords = M.foil.x[1, :].copy()  # y coordinates on foil
    
    # Get Cp and Cf from post-processing results (length = N + wake nodes)
    cp_all = M.post.cp.copy() if hasattr(M.post, 'cp') and M.post.cp is not None else None
    cf_all = M.post.cf.copy() if hasattr(M.post, 'cf') and M.post.cf is not None else None
    
    # Extract foil surface only (exclude wake)
    N = M.foil.N
    
    # Reorganize to get upper and lower surfaces
    # Upper surface: indices 0 to N/2 (stored reversed in vsol)
    # Lower surface: indices N/2 to N
    n_half = N // 2
    
    # For mfoil, the foil is ordered from TE (upper) -> LE -> TE (lower)
    # So x_coords[0] is at TE upper, x_coords[n_half] is at LE, x_coords[-1] is at TE lower
    
    # Upper surface: first half, need to reverse to go LE to TE
    x_upper = x_coords[:n_half+1][::-1]
    y_upper = y_coords[:n_half+1][::-1]
    
    # Lower surface: second half
    x_lower = x_coords[n_half:]
    y_lower = y_coords[n_half:]
    
    if cp_all is not None:
        # cp_all has solution at each node including wake
        # First N nodes are foil surface
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


def run_rans_solver(X, Y, reynolds, max_iter=2000, cfl=1.0, k4=0.04):
    """Run the RANS solver and return surface distributions."""
    NI = X.shape[0] - 1
    NJ = X.shape[1] - 1
    
    # Compute metrics
    computer = MetricComputer(X, Y, wall_j=0)
    metrics = computer.compute()
    
    # Initialize state
    freestream = FreestreamConditions.from_mach_alpha(mach=0.1, alpha_deg=0.0)
    Q = initialize_state(NI, NJ, freestream)
    Q = apply_initial_wall_damping(Q, metrics, decay_length=0.1)
    
    # Compute far-field outward unit normals for Riemann BC
    beta = 10.0
    Sj_x_ff = metrics.Sj_x[:, -1]
    Sj_y_ff = metrics.Sj_y[:, -1]
    Sj_mag = np.sqrt(Sj_x_ff**2 + Sj_y_ff**2) + 1e-12
    nx_ff = Sj_x_ff / Sj_mag
    ny_ff = Sj_y_ff / Sj_mag
    
    bc = BoundaryConditions(
        freestream=freestream,
        farfield_normals=(nx_ff, ny_ff),
        beta=beta
    )
    Q = bc.apply(Q)
    
    # Flux configuration
    # k2: 2nd-order dissipation (shock capturing)
    # k4: 4th-order dissipation (background smoothing to reduce oscillations)
    # Typical values: k2=0.5, k4=0.016-0.064
    flux_cfg = FluxConfig(k2=0.5, k4=k4)
    flux_metrics = FluxGridMetrics(
        Si_x=metrics.Si_x, Si_y=metrics.Si_y,
        Sj_x=metrics.Sj_x, Sj_y=metrics.Sj_y,
        volume=metrics.volume
    )
    
    ts_config = TimeStepConfig(cfl=cfl)
    
    print(f"Running RANS solver: {NI}x{NJ} grid, {max_iter} iterations, CFL={cfl}")
    
    # RK4 coefficients
    alphas = [0.25, 0.333333333, 0.5, 1.0]
    
    for n in range(max_iter):
        # Compute time step
        dt = compute_local_timestep(
            Q, metrics.Si_x, metrics.Si_y, metrics.Sj_x, metrics.Sj_y,
            metrics.volume, beta, ts_config
        )
        
        # RK4 integration
        Q0 = Q.copy()
        Qk = Q.copy()
        
        for alpha in alphas:
            Qk = bc.apply(Qk)
            R = compute_fluxes(Qk, flux_metrics, beta, flux_cfg)
            Qk = Q0.copy()
            Qk[1:-1, 1:-1, :] += alpha * (dt / metrics.volume)[:, :, np.newaxis] * R
        
        Q = bc.apply(Qk)
        
        # Print progress every 200 iterations
        if (n + 1) % 200 == 0:
            R = compute_fluxes(Q, flux_metrics, beta, flux_cfg)
            res_rms = np.sqrt(np.sum(R[:, :, 0]**2) / R[:, :, 0].size)
            print(f"  Iter {n+1:5d}: residual = {res_rms:.6e}")
    
    # Extract surface data
    Q_int = Q[1:-1, 1:-1, :]
    
    # Surface is at j=0 in interior indexing
    p_surface = Q_int[:, 0, 0]
    u_surface = Q_int[:, 1, 1]  # First interior cell
    v_surface = Q_int[:, 1, 2]
    
    # Cell centers at the surface
    x_surface = 0.5 * (X[:-1, 0] + X[1:, 0])
    y_surface = 0.5 * (Y[:-1, 0] + Y[1:, 0])
    
    # Freestream values
    p_inf = freestream.p_inf
    V_inf_sq = freestream.u_inf**2 + freestream.v_inf**2
    q_inf = 0.5 * V_inf_sq
    
    # Cp = (p - p_inf) / q_inf
    Cp = (p_surface - p_inf) / (q_inf + 1e-12)
    
    # Cf estimation using wall shear
    # tau_w = mu * du/dy at wall
    # Approximate du/dn using ghost cell formulation: du/dn ≈ 2*u_interior / Delta_n
    # Delta_n is the distance to the first interior cell
    
    # Get first interior cell velocity
    u_int = Q_int[:, 1, 1]
    v_int = Q_int[:, 1, 2]
    
    # Normal distance (use wall distance from metrics for first cell)
    dn = metrics.wall_distance[:, 1]  # Distance at j=1 interior cells
    
    # Wall-tangent velocity (project onto surface tangent)
    # For a C-grid, the tangent direction is roughly the i-direction
    dx = np.diff(X[:, 0])
    dy = np.diff(Y[:, 0])
    ds = np.sqrt(dx**2 + dy**2)
    
    # Tangent vector at cell centers
    tx = np.zeros(NI)
    ty = np.zeros(NI)
    tx[:-1] = 0.5 * (dx[:-1] + dx[1:]) / (0.5 * (ds[:-1] + ds[1:]))
    ty[:-1] = 0.5 * (dy[:-1] + dy[1:]) / (0.5 * (ds[:-1] + ds[1:]))
    tx[-1] = dx[-1] / ds[-1]
    ty[-1] = dy[-1] / ds[-1]
    
    # Normalize
    t_mag = np.sqrt(tx**2 + ty**2)
    tx /= (t_mag + 1e-12)
    ty /= (t_mag + 1e-12)
    
    # Project velocity onto tangent
    u_tan = u_int * tx + v_int * ty
    
    # Wall shear stress (using finite difference)
    mu_laminar = 1.0 / reynolds
    tau_w = mu_laminar * u_tan / (dn + 1e-12)
    
    # Cf = tau_w / q_inf
    Cf = tau_w / (q_inf + 1e-12)
    
    return {
        'x': x_surface,
        'y': y_surface,
        'Cp': Cp,
        'Cf': Cf,
        'Q': Q,
        'freestream': freestream,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare RANS Cp/Cf against mfoil")
    parser.add_argument("--reynolds", "-Re", type=float, default=10000,
                        help="Reynolds number (default: 10000)")
    parser.add_argument("--max-iter", "-n", type=int, default=2000,
                        help="Maximum iterations (default: 2000)")
    parser.add_argument("--cfl", type=float, default=1.0,
                        help="CFL number (default: 1.0)")
    parser.add_argument("--n-surface", type=int, default=100,
                        help="Surface points (default: 100)")
    parser.add_argument("--n-normal", type=int, default=40,
                        help="Normal points (default: 40)")
    parser.add_argument("--k4", type=float, default=0.04,
                        help="JST 4th-order dissipation coefficient (default: 0.04)")
    parser.add_argument("--output", "-o", type=str, default="output/cp_cf_comparison.pdf",
                        help="Output PDF path")
    args = parser.parse_args()
    
    print("="*60)
    print("  RANS vs mfoil Cp/Cf Comparison")
    print("="*60)
    print(f"Reynolds: {args.reynolds}")
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
    binary_paths = [
        project_root / "bin" / "construct2d",
        Path("./construct2d"),
    ]
    
    binary_path = None
    for p in binary_paths:
        if p.exists():
            binary_path = p
            break
    
    if binary_path is None:
        print("ERROR: construct2d binary not found")
        sys.exit(1)
    
    wrapper = Construct2DWrapper(str(binary_path))
    grid_opts = GridOptions(
        n_surface=args.n_surface,
        n_normal=args.n_normal,
        n_wake=30,
        y_plus=1.0,
        reynolds=args.reynolds,
        topology='CGRD',
        farfield_radius=15.0,
    )
    
    airfoil_file = project_root / "data" / "naca0012.dat"
    X, Y = wrapper.generate(str(airfoil_file), grid_opts, verbose=False)
    print(f"  Grid: {X.shape[0]} x {X.shape[1]} nodes")
    
    # Run RANS
    print("\nRunning RANS solver...")
    rans_result = run_rans_solver(X, Y, args.reynolds, args.max_iter, args.cfl, args.k4)
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sort RANS data by x for plotting
    x_rans = rans_result['x']
    cp_rans = rans_result['Cp']
    cf_rans = rans_result['Cf']
    
    # Split upper/lower surface (by y coordinate)
    y_rans = rans_result['y']
    upper = y_rans >= 0
    lower = y_rans < 0
    
    # Sort each surface by x
    idx_upper = np.argsort(x_rans[upper])
    idx_lower = np.argsort(x_rans[lower])
    
    x_upper = x_rans[upper][idx_upper]
    x_lower = x_rans[lower][idx_lower]
    cp_upper = cp_rans[upper][idx_upper]
    cp_lower = cp_rans[lower][idx_lower]
    cf_upper = cf_rans[upper][idx_upper]
    cf_lower = cf_rans[lower][idx_lower]
    
    # mfoil data (already split into upper/lower)
    x_mfoil_upper = mfoil_result['x_upper']
    x_mfoil_lower = mfoil_result['x_lower']
    cp_mfoil_upper = mfoil_result['cp_upper']
    cp_mfoil_lower = mfoil_result['cp_lower']
    cf_mfoil_upper = mfoil_result['cf_upper']
    cf_mfoil_lower = mfoil_result['cf_lower']
    
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
    
    # --- Plot 3: Full Cp vs x ---
    ax = axes[1, 0]
    ax.plot(x_rans, -cp_rans, 'b.', markersize=3, label='RANS')
    ax.plot(x_mfoil_upper, -cp_mfoil_upper, 'r-', lw=1.5, alpha=0.7, label='mfoil')
    ax.plot(x_mfoil_lower, -cp_mfoil_lower, 'r-', lw=1.5, alpha=0.7)
    ax.set_xlabel('x/c')
    ax.set_ylabel('-Cp')
    ax.set_title('Cp Distribution (all points)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    # --- Plot 4: Cf statistics ---
    ax = axes[1, 1]
    ax.plot(x_rans, cf_rans, 'b.', markersize=3, label='RANS')
    ax.plot(x_mfoil_upper, cf_mfoil_upper, 'r-', lw=1.5, alpha=0.7, label='mfoil')
    ax.plot(x_mfoil_lower, cf_mfoil_lower, 'r-', lw=1.5, alpha=0.7)
    ax.set_xlabel('x/c')
    ax.set_ylabel('Cf')
    ax.set_title('Cf Distribution (all points)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    # Add text with statistics
    stats_text = (
        f"mfoil: CD={mfoil_result['cd']:.5f}, Cdf={mfoil_result['cdf']:.5f}\n"
        f"RANS Cp range: [{cp_rans.min():.3f}, {cp_rans.max():.3f}]\n"
        f"RANS Cf range: [{cf_rans.min():.5f}, {cf_rans.max():.5f}]"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved comparison plot to: {output_path}")
    
    plt.close()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
