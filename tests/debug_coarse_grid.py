#!/usr/bin/env python3
"""
Debug script to test coarse grid quality in isolation.

This script:
1. Initializes a fine grid RANSSolver with NACA 0012
2. Builds the multigrid hierarchy
3. Extracts the coarsest level
4. Creates a standalone solver using only the coarse grid
5. Runs it for 1000 steps
6. Plots residual history and residual field

Goal: Determine if the coarse grid by itself has farfield boundary issues.
If it does, the problem is in Coarse Grid BC/Metric generation, not MG transfers.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.solvers.rans_solver import RANSSolver, SolverConfig
from src.solvers.multigrid import build_multigrid_hierarchy
from src.solvers.boundary_conditions import FreestreamConditions, initialize_state, apply_initial_wall_damping
from src.solvers.time_stepping import compute_local_timestep, TimeStepConfig
from src.numerics.fluxes import compute_fluxes, FluxConfig, GridMetrics as FluxGridMetrics
from src.numerics.gradients import compute_gradients, GradientMetrics
from src.numerics.viscous_fluxes import add_viscous_fluxes
from src.numerics.smoothing import apply_residual_smoothing
from src.numerics.diagnostics import compute_total_pressure_loss, compute_solution_bounds
from src.grid.loader import load_or_generate_grid


def create_coarse_grid_solver(level, freestream, config):
    """
    Create a minimal solver structure for running on a single coarse level.
    
    Parameters
    ----------
    level : MultigridLevel
        The coarse level from the hierarchy.
    freestream : FreestreamConditions
        Freestream conditions.
    config : SolverConfig
        Solver configuration.
        
    Returns
    -------
    dict
        Solver-like dictionary with all necessary components.
    """
    return {
        'NI': level.NI,
        'NJ': level.NJ,
        'metrics': level.metrics,
        'Q': level.Q.copy(),
        'bc': level.bc,
        'freestream': freestream,
        'config': config,
        'dt': level.dt,
    }


def compute_residual_coarse(Q, solver_data, mu_laminar):
    """Compute residual on coarse grid."""
    metrics = solver_data['metrics']
    config = solver_data['config']
    NI, NJ = solver_data['NI'], solver_data['NJ']
    
    # Create flux metrics
    flux_metrics = FluxGridMetrics(
        Si_x=metrics.Si_x, Si_y=metrics.Si_y,
        Sj_x=metrics.Sj_x, Sj_y=metrics.Sj_y,
        volume=metrics.volume
    )
    
    # Create gradient metrics
    grad_metrics = GradientMetrics(
        Si_x=metrics.Si_x, Si_y=metrics.Si_y,
        Sj_x=metrics.Sj_x, Sj_y=metrics.Sj_y,
        volume=metrics.volume
    )
    
    # Flux config
    flux_cfg = FluxConfig(
        k2=0.0,  # No 2nd-order dissipation for incompressible
        k4=config.jst_k4
    )
    
    # Compute convective fluxes (JST scheme)
    conv_residual = compute_fluxes(Q, flux_metrics, config.beta, flux_cfg)
    
    # Compute gradients for viscous fluxes
    gradients = compute_gradients(Q, grad_metrics)
    
    # Add viscous fluxes
    residual = add_viscous_fluxes(
        conv_residual, Q, gradients, grad_metrics, mu_laminar
    )
    
    return residual


def step_coarse(solver_data, mu_laminar):
    """Perform one RK5 step on coarse grid."""
    Q = solver_data['Q']
    metrics = solver_data['metrics']
    config = solver_data['config']
    bc = solver_data['bc']
    
    # Jameson 5-stage RK coefficients
    alphas = [0.25, 0.166666667, 0.375, 0.5, 1.0]
    
    # Compute timestep
    ts_config = TimeStepConfig(cfl=config.cfl_target)
    dt = compute_local_timestep(
        Q,
        metrics.Si_x, metrics.Si_y,
        metrics.Sj_x, metrics.Sj_y,
        metrics.volume,
        config.beta,
        ts_config,
        nu=mu_laminar
    )
    solver_data['dt'] = dt
    
    Q0 = Q.copy()
    Qk = Q.copy()
    
    for alpha in alphas:
        Qk = bc.apply(Qk)
        R = compute_residual_coarse(Qk, solver_data, mu_laminar)  # Pass Qk!
        
        # Apply IRS if enabled
        if config.irs_epsilon > 0.0:
            apply_residual_smoothing(R, config.irs_epsilon)
        
        Qk = Q0.copy()
        Qk[1:-1, 1:-1, :] += alpha * (dt / metrics.volume)[:, :, np.newaxis] * R
    
    solver_data['Q'] = bc.apply(Qk)
    
    # Return residual RMS
    R = compute_residual_coarse(solver_data['Q'], solver_data, mu_laminar)
    R_p = R[:, :, 0]
    rms = np.sqrt(np.mean(R_p**2))
    
    return rms, R


def main():
    print("=" * 70)
    print("Coarse Grid Debug Test")
    print("=" * 70)
    
    # Configuration for fine grid (128 x 64 cells)
    # For power-of-2 cells: n_surface + 2*n_wake = NI + 1 nodes
    # We want NI = 128 cells, so n_surface + 2*n_wake = 129 nodes
    # With n_wake = 32: n_surface = 129 - 64 = 65
    n_surface = 65
    n_normal = 65  # 64 cells
    n_wake = 32
    
    config = SolverConfig(
        mach=0.15,
        alpha=0.0,
        reynolds=6e6,
        beta=10.0,
        cfl_start=1.0,
        cfl_target=1.0,  # Lower CFL for stability testing
        cfl_ramp_iters=100,
        max_iter=1000,
        tol=1e-10,
        jst_k4=0.04,
        irs_epsilon=1.0,
        n_wake=n_wake,
    )
    
    # Load or generate grid
    grid_file = project_root / "data" / "naca0012.dat"
    print(f"\nLoading grid from: {grid_file}")
    
    X, Y = load_or_generate_grid(
        str(grid_file),
        n_surface=n_surface,
        n_normal=n_normal,
        n_wake=n_wake,
        y_plus=1.0,
        reynolds=config.reynolds,
        farfield_radius=15.0,
        project_root=project_root,
        verbose=True
    )
    
    NI = X.shape[0] - 1
    NJ = X.shape[1] - 1
    print(f"\nFine grid: {NI} x {NJ} cells")
    
    # Create freestream conditions
    freestream = FreestreamConditions.from_mach_alpha(
        mach=config.mach,
        alpha_deg=config.alpha
    )
    
    # Initialize fine state
    Q_fine = initialize_state(NI, NJ, freestream)
    
    # Build multigrid hierarchy
    print("\nBuilding multigrid hierarchy...")
    hierarchy = build_multigrid_hierarchy(
        X, Y, Q_fine, freestream,
        n_wake=n_wake,
        beta=config.beta,
        min_size=8,
        max_levels=5
    )
    
    print(hierarchy.get_level_info())
    
    # Extract coarsest level
    coarsest_idx = hierarchy.num_levels - 1
    coarsest_level = hierarchy.levels[coarsest_idx]
    print(f"\nUsing coarsest level {coarsest_idx}: {coarsest_level.NI} x {coarsest_level.NJ} cells")
    
    # Debug: Check farfield normals
    print("\n=== Farfield Normals Debug ===")
    nx, ny = coarsest_level.bc.farfield_normals
    print(f"Farfield normals shape: nx={nx.shape}, expected=({coarsest_level.NI},)")
    print(f"nx range: [{nx.min():.4f}, {nx.max():.4f}]")
    print(f"ny range: [{ny.min():.4f}, {ny.max():.4f}]")
    print(f"Normal magnitudes: min={np.sqrt(nx**2 + ny**2).min():.4f}, max={np.sqrt(nx**2 + ny**2).max():.4f}")
    print(f"Sample normals at i=0: ({nx[0]:.4f}, {ny[0]:.4f})")
    print(f"Sample normals at i=NI-1: ({nx[-1]:.4f}, {ny[-1]:.4f})")
    
    # Check if metrics are valid
    print("\n=== Metrics Validation ===")
    for lvl_idx, lvl in enumerate(hierarchy.levels):
        vol = lvl.metrics.volume
        print(f"Level {lvl_idx}: Volume min={vol.min():.6e}, max={vol.max():.6e}, ratio={vol.max()/vol.min():.2e}")
    print(f"\nCoarsest level Si_x shape: {coarsest_level.metrics.Si_x.shape}")
    print(f"Coarsest level Sj_x shape: {coarsest_level.metrics.Sj_x.shape}")
    
    # Compare timesteps
    print("\n=== Timestep Analysis ===")
    ts_config = TimeStepConfig(cfl=1.0)
    for lvl_idx, lvl in enumerate(hierarchy.levels):
        Q_test = initialize_state(lvl.NI, lvl.NJ, freestream)
        Q_test = lvl.bc.apply(Q_test)
        dt_test = compute_local_timestep(
            Q_test,
            lvl.metrics.Si_x, lvl.metrics.Si_y,
            lvl.metrics.Sj_x, lvl.metrics.Sj_y,
            lvl.metrics.volume,
            config.beta,
            ts_config,
            nu=1.0/config.reynolds
        )
        print(f"Level {lvl_idx}: dt min={dt_test.min():.6e}, max={dt_test.max():.6e}, ratio={dt_test.max()/dt_test.min():.2e}")
    
    # Check n_wake scaling
    print("\n=== Wake Parameters ===")
    print(f"Fine grid n_wake: {n_wake}")
    for lvl_idx, lvl in enumerate(hierarchy.levels):
        # Get n_wake_points from BC
        n_wake_lvl = lvl.bc.n_wake_points
        print(f"Level {lvl_idx}: NI={lvl.NI}, n_wake_points={n_wake_lvl}, expected={n_wake // (2**lvl_idx)}")
    
    # Check farfield normals vs J-face normals from metrics
    print("\n=== Farfield Normal Consistency Check ===")
    for lvl_idx, lvl in enumerate(hierarchy.levels):
        # Farfield normals from BC (from coordinates)
        nx_bc, ny_bc = lvl.bc.farfield_normals
        
        # J-face normals from metrics at j=NJ (farfield)
        Sj_x_farfield = lvl.metrics.Sj_x[:, -1]  # Shape: (NI,)
        Sj_y_farfield = lvl.metrics.Sj_y[:, -1]
        
        # Compute magnitude and unit normal
        Sj_mag = np.sqrt(Sj_x_farfield**2 + Sj_y_farfield**2) + 1e-30
        nx_metrics = Sj_x_farfield / Sj_mag
        ny_metrics = Sj_y_farfield / Sj_mag
        
        # Compare
        diff_x = np.abs(nx_bc - nx_metrics)
        diff_y = np.abs(ny_bc - ny_metrics)
        print(f"Level {lvl_idx}: max_diff_nx={diff_x.max():.6f}, max_diff_ny={diff_y.max():.6f}")
    
    # First, run the FINE grid for comparison
    print("\n" + "=" * 70)
    print(f"Running FINE grid ({hierarchy.levels[0].NI}x{hierarchy.levels[0].NJ}) for 100 iterations")
    print("=" * 70)
    
    fine_level = hierarchy.levels[0]
    fine_solver = create_coarse_grid_solver(fine_level, freestream, config)
    fine_solver['Q'] = initialize_state(fine_solver['NI'], fine_solver['NJ'], freestream)
    fine_solver['Q'] = apply_initial_wall_damping(
        fine_solver['Q'], fine_solver['metrics'], 
        decay_length=0.1, n_wake=n_wake
    )
    fine_solver['Q'] = fine_solver['bc'].apply(fine_solver['Q'])
    
    mu_laminar = 1.0 / config.reynolds
    for n in range(500):
        if n < config.cfl_ramp_iters:
            cfl = config.cfl_start + (config.cfl_target - config.cfl_start) * n / config.cfl_ramp_iters
        else:
            cfl = config.cfl_target
        config_step = SolverConfig(**{**config.__dict__, 'cfl_target': cfl})
        fine_solver['config'] = config_step
        res_rms, _ = step_coarse(fine_solver, mu_laminar)
        if (n + 1) % 25 == 0 or n == 0:
            bounds = compute_solution_bounds(fine_solver['Q'])
            print(f"  {n+1:>4}: Res={res_rms:.4e}, |V|_max={bounds['vel_max']:.4f}")
    
    print("\nFine grid completed 100 iterations without divergence")
    
    # Now run coarse grid
    # Create coarse solver data
    solver_data = create_coarse_grid_solver(coarsest_level, freestream, config)
    
    # Re-initialize state to freestream (don't use restricted state)
    solver_data['Q'] = initialize_state(solver_data['NI'], solver_data['NJ'], freestream)
    # Apply wall damping with scaled n_wake
    coarse_n_wake = n_wake // (2 ** coarsest_idx)
    solver_data['Q'] = apply_initial_wall_damping(
        solver_data['Q'], solver_data['metrics'],
        decay_length=0.1, n_wake=coarse_n_wake
    )
    solver_data['Q'] = solver_data['bc'].apply(solver_data['Q'])
    
    # Run for 1000 steps
    print("\n" + "=" * 70)
    print(f"Running coarse grid solver for {config.max_iter} iterations")
    print("=" * 70)
    print(f"{'Iter':>6} {'Residual':>12} {'|V|_max':>10} {'p_range':>20}")
    print("-" * 60)
    
    mu_laminar = 1.0 / config.reynolds
    residual_history = []
    
    for n in range(config.max_iter):
        # CFL ramping
        if n < config.cfl_ramp_iters:
            cfl = config.cfl_start + (config.cfl_target - config.cfl_start) * n / config.cfl_ramp_iters
        else:
            cfl = config.cfl_target
        config_step = SolverConfig(**{**config.__dict__, 'cfl_target': cfl})
        solver_data['config'] = config_step
        
        res_rms, R_field = step_coarse(solver_data, mu_laminar)
        residual_history.append(res_rms)
        
        if (n + 1) % 50 == 0 or n == 0:
            bounds = compute_solution_bounds(solver_data['Q'])
            p_range = f"[{bounds['p_min']:.2f}, {bounds['p_max']:.2f}]"
            print(f"{n+1:>6} {res_rms:>12.4e} {bounds['vel_max']:>10.4f} {p_range:>20}")
        
        # Check for divergence
        if np.isnan(res_rms) or res_rms > 1e10:
            print(f"\nDIVERGED at iteration {n+1}")
            break
    
    # Final state analysis
    print("\n" + "=" * 70)
    print("Final State Analysis")
    print("=" * 70)
    
    Q = solver_data['Q']
    Q_int = Q[1:-1, 1:-1, :]
    metrics = solver_data['metrics']
    
    # Compute C_pt
    C_pt = compute_total_pressure_loss(
        Q_int, freestream.p_inf, freestream.u_inf, freestream.v_inf
    )
    
    print(f"C_pt: min={C_pt.min():.4f}, max={C_pt.max():.4f}, mean={C_pt.mean():.4f}")
    
    # Analyze farfield row
    p_farfield = Q_int[:, -1, 0]
    u_farfield = Q_int[:, -1, 1]
    v_farfield = Q_int[:, -1, 2]
    vel_farfield = np.sqrt(u_farfield**2 + v_farfield**2)
    
    print(f"\nFarfield row (j=-1) analysis:")
    print(f"  Pressure: min={p_farfield.min():.4f}, max={p_farfield.max():.4f}")
    print(f"  Velocity: min={vel_farfield.min():.4f}, max={vel_farfield.max():.4f}")
    print(f"  Expected: p={freestream.p_inf:.4f}, |V|={np.sqrt(freestream.u_inf**2 + freestream.v_inf**2):.4f}")
    
    # Analyze high-i region specifically
    NI_int = Q_int.shape[0]
    high_i_start = int(0.75 * NI_int)
    print(f"\nHigh-i region (i >= {high_i_start}):")
    p_high_i = Q_int[high_i_start:, -1, 0]
    vel_high_i = np.sqrt(Q_int[high_i_start:, -1, 1]**2 + Q_int[high_i_start:, -1, 2]**2)
    print(f"  Pressure: min={p_high_i.min():.4f}, max={p_high_i.max():.4f}")
    print(f"  Velocity: min={vel_high_i.min():.4f}, max={vel_high_i.max():.4f}")
    
    # Plot results
    output_dir = project_root / "output" / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "coarse_grid_debug.pdf"
    
    print(f"\nGenerating plots: {output_file}")
    
    with PdfPages(str(output_file)) as pdf:
        # Page 1: Residual history
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(range(1, len(residual_history) + 1), residual_history, 'b-', lw=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Residual RMS')
        ax.set_title(f'Coarse Grid ({solver_data["NI"]}x{solver_data["NJ"]}) Convergence History')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig, dpi=100)
        plt.close(fig)
        
        # Page 2: Flow field
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Coarse Grid ({solver_data["NI"]}x{solver_data["NJ"]}) Final State', fontsize=14)
        
        xc, yc = metrics.xc, metrics.yc
        p = Q_int[:, :, 0]
        vel_mag = np.sqrt(Q_int[:, :, 1]**2 + Q_int[:, :, 2]**2)
        
        # Pressure - global
        ax = axes[0, 0]
        p_clip = np.clip(p, np.percentile(p, 1), np.percentile(p, 99))
        pc = ax.pcolormesh(xc.T, yc.T, p_clip.T, cmap='RdBu_r', shading='auto')
        ax.set_aspect('equal')
        ax.set_title('Pressure (Global)')
        plt.colorbar(pc, ax=ax, shrink=0.8)
        
        # Pressure - zoomed
        ax = axes[0, 1]
        pc = ax.pcolormesh(xc.T, yc.T, p_clip.T, cmap='RdBu_r', shading='auto')
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title('Pressure (Near Airfoil)')
        plt.colorbar(pc, ax=ax, shrink=0.8)
        
        # Velocity - global
        ax = axes[1, 0]
        vel_clip = np.clip(vel_mag, 0, np.percentile(vel_mag, 99))
        pc = ax.pcolormesh(xc.T, yc.T, vel_clip.T, cmap='viridis', shading='auto')
        ax.set_aspect('equal')
        ax.set_title('Velocity Magnitude (Global)')
        plt.colorbar(pc, ax=ax, shrink=0.8)
        
        # Velocity - zoomed
        ax = axes[1, 1]
        pc = ax.pcolormesh(xc.T, yc.T, vel_clip.T, cmap='viridis', shading='auto')
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title('Velocity Magnitude (Near Airfoil)')
        plt.colorbar(pc, ax=ax, shrink=0.8)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=100)
        plt.close(fig)
        
        # Page 3: Residual field
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Coarse Grid ({solver_data["NI"]}x{solver_data["NJ"]}) Residual Field', fontsize=14)
        
        R_p = np.abs(R_field[:, :, 0])
        R_log = np.log10(R_p + 1e-15)
        
        # Residual - global
        ax = axes[0, 0]
        pc = ax.pcolormesh(xc.T, yc.T, R_log.T, cmap='hot_r', shading='auto')
        ax.set_aspect('equal')
        ax.set_title('log₁₀|Residual| (Global)')
        plt.colorbar(pc, ax=ax, shrink=0.8)
        
        # Residual - zoomed
        ax = axes[0, 1]
        pc = ax.pcolormesh(xc.T, yc.T, R_log.T, cmap='hot_r', shading='auto')
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title('log₁₀|Residual| (Near Airfoil)')
        plt.colorbar(pc, ax=ax, shrink=0.8)
        
        # C_pt - global
        ax = axes[1, 0]
        C_pt_clip = np.clip(C_pt, np.percentile(C_pt, 1), np.percentile(C_pt, 99))
        pc = ax.pcolormesh(xc.T, yc.T, C_pt_clip.T, cmap='RdBu_r', shading='auto')
        ax.set_aspect('equal')
        ax.set_title('Total Pressure Loss C_pt (Global)')
        plt.colorbar(pc, ax=ax, shrink=0.8)
        
        # C_pt - zoomed
        ax = axes[1, 1]
        pc = ax.pcolormesh(xc.T, yc.T, C_pt_clip.T, cmap='RdBu_r', shading='auto')
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title('Total Pressure Loss C_pt (Near Airfoil)')
        plt.colorbar(pc, ax=ax, shrink=0.8)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=100)
        plt.close(fig)
        
        # Page 4: Farfield row analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle('Farfield Row Analysis (j = NJ-1)', fontsize=14)
        
        i_idx = np.arange(len(p_farfield))
        
        ax = axes[0, 0]
        ax.plot(i_idx, p_farfield, 'b-', lw=1.5)
        ax.axhline(y=freestream.p_inf, color='r', linestyle='--', label=f'p_inf={freestream.p_inf:.4f}')
        ax.set_xlabel('i index')
        ax.set_ylabel('Pressure')
        ax.set_title('Farfield Pressure')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.plot(i_idx, vel_farfield, 'b-', lw=1.5)
        V_inf = np.sqrt(freestream.u_inf**2 + freestream.v_inf**2)
        ax.axhline(y=V_inf, color='r', linestyle='--', label=f'|V|_inf={V_inf:.4f}')
        ax.set_xlabel('i index')
        ax.set_ylabel('Velocity Magnitude')
        ax.set_title('Farfield Velocity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        ax.plot(i_idx, u_farfield, 'b-', lw=1.5, label='u')
        ax.plot(i_idx, v_farfield, 'g-', lw=1.5, label='v')
        ax.axhline(y=freestream.u_inf, color='b', linestyle='--', alpha=0.5)
        ax.axhline(y=freestream.v_inf, color='g', linestyle='--', alpha=0.5)
        ax.set_xlabel('i index')
        ax.set_ylabel('Velocity')
        ax.set_title('Farfield Velocity Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Residual at farfield
        ax = axes[1, 1]
        R_farfield = np.abs(R_field[:, -1, 0])
        ax.semilogy(i_idx, R_farfield, 'b-', lw=1.5)
        ax.set_xlabel('i index')
        ax.set_ylabel('|Residual|')
        ax.set_title('Farfield Residual (Pressure Equation)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=100)
        plt.close(fig)
    
    print(f"\nOutput saved to: {output_file}")
    print("\n" + "=" * 70)
    print("Debug complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

