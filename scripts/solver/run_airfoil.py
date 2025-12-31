#!/usr/bin/env python3
"""
NACA 0012 Airfoil Simulation Script with Diagnostic Visualization.

This script sets up and runs a 2D incompressible RANS simulation around
a NACA 0012 airfoil using the artificial compressibility method.

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

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.solvers.rans_solver import RANSSolver, SolverConfig
from src.grid.loader import load_or_generate_grid
from src.grid.metrics import MetricComputer
from src.solvers.boundary_conditions import (
    FreestreamConditions, 
    BoundaryConditions,
    initialize_state,
    apply_initial_wall_damping
)
from src.numerics.fluxes import compute_fluxes, FluxConfig, GridMetrics as FluxGridMetrics
from src.numerics.gradients import compute_gradients, GradientMetrics
from src.numerics.viscous_fluxes import add_viscous_fluxes
from src.solvers.time_stepping import compute_local_timestep, TimeStepConfig
from src.io.output import VTKWriter, write_vtk


def compute_total_pressure_loss(Q: np.ndarray, freestream) -> np.ndarray:
    """
    Compute Total Pressure Loss Coefficient (C_pt) for entropy check.
    
    For inviscid flow, C_pt should be zero everywhere (no entropy generation).
    Non-zero values indicate spurious entropy production (numerical error).
    
    Physics (Incompressible, ρ=1):
        P_0 = p + 0.5 * (u² + v²)           (local total pressure)
        P_0_inf = p_inf + 0.5 * V_inf²      (freestream total pressure)
        q_inf = 0.5 * V_inf²                 (dynamic pressure)
        C_pt = (P_0_inf - P_0_local) / q_inf (total pressure loss coefficient)
    
    Parameters
    ----------
    Q : ndarray, shape (NI, NJ, 4) or (NI+2, NJ+2, 4)
        State vector [p, u, v, nu_t]. Can include ghost cells.
    freestream : FreestreamConditions
        Freestream conditions object with p_inf, u_inf, v_inf.
        
    Returns
    -------
    C_pt : ndarray, shape (NI, NJ)
        Total pressure loss coefficient at each cell.
        Positive = entropy production (energy loss)
        Negative = entropy destruction (unphysical)
    """
    # Handle ghost cells
    if Q.ndim == 3 and Q.shape[2] == 4:
        # Check if ghost cells are present
        # Assume ghost cells if shape is not matching expected interior
        p = Q[:, :, 0]
        u = Q[:, :, 1]
        v = Q[:, :, 2]
    else:
        raise ValueError(f"Unexpected Q shape: {Q.shape}")
    
    # Freestream values
    p_inf = freestream.p_inf
    u_inf = freestream.u_inf
    v_inf = freestream.v_inf
    
    # Dynamic pressure (assume rho = 1.0 for AC formulation)
    V_inf_sq = u_inf**2 + v_inf**2
    q_inf = 0.5 * V_inf_sq
    
    # Freestream total pressure
    P0_inf = p_inf + 0.5 * V_inf_sq
    
    # Local total pressure
    P0_local = p + 0.5 * (u**2 + v**2)
    
    # Total pressure loss coefficient
    # C_pt = (P_0_inf - P_0_local) / q_inf
    # Avoid division by zero for zero freestream velocity
    if q_inf > 1e-12:
        C_pt = (P0_inf - P0_local) / q_inf
    else:
        C_pt = np.zeros_like(p)
    
    return C_pt


def plot_flow_field(X: np.ndarray, Y: np.ndarray, Q: np.ndarray,
                    iteration: int, residual: float, cfl: float,
                    output_dir: str, case_name: str = "flow",
                    freestream=None, residual_field=None):
    """
    Plot flow field with global and zoomed views.
    
    Creates PDF with:
    - Pressure (global and zoomed)
    - Velocity magnitude (global and zoomed)
    - Total Pressure Loss C_pt (global and zoomed) - entropy check
    - Residual field (global and zoomed) - if provided
    
    Parameters
    ----------
    X, Y : ndarray
        Grid node coordinates.
    Q : ndarray
        State vector [p, u, v, nu_t].
    iteration : int
        Current iteration number.
    residual : float
        Current residual value.
    residual_field : ndarray, optional
        Residual field (NI, NJ, 4) for visualization.
    cfl : float
        Current CFL number.
    output_dir : str
        Output directory for PDF files.
    case_name : str
        Base name for output files.
    freestream : FreestreamConditions, optional
        Freestream conditions for C_pt calculation.
    """
    # Strip ghost cells from Q
    NI = X.shape[0] - 1
    NJ = X.shape[1] - 1
    
    if Q.shape[0] == NI + 2:
        Q_int = Q[1:-1, 1:-1, :]
    else:
        Q_int = Q
    
    # Extract fields
    p = Q_int[:, :, 0]
    u = Q_int[:, :, 1]
    v = Q_int[:, :, 2]
    nu_t = Q_int[:, :, 3]
    
    vel_mag = np.sqrt(u**2 + v**2)
    
    # Compute total pressure loss if freestream is provided
    if freestream is not None:
        C_pt = compute_total_pressure_loss(Q_int, freestream)
    else:
        C_pt = None
    
    # Determine number of rows based on available data
    n_rows = 2  # Pressure + Velocity
    if C_pt is not None:
        n_rows += 1
    if residual_field is not None:
        n_rows += 1
    
    # Create figure
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    
    # Ensure axes is always 2D
    if n_rows == 2:
        axes = axes.reshape(2, 2)
    
    # --- Row 1: Pressure ---
    ax = axes[0, 0]
    p_clip = np.clip(p, np.percentile(p, 1), np.percentile(p, 99))
    pc = ax.pcolormesh(X.T, Y.T, p_clip.T, cmap='RdBu_r', shading='flat', rasterized=True)
    ax.set_aspect('equal')
    ax.set_title(f'Pressure (Global) - Iter {iteration}')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    plt.colorbar(pc, ax=ax, shrink=0.8, label='p')
    
    ax = axes[0, 1]
    pc = ax.pcolormesh(X.T, Y.T, p_clip.T, cmap='RdBu_r', shading='flat', rasterized=True)
    ax.set_aspect('equal')
    ax.set_xlim(-0.2, 1.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_title(f'Pressure (Near Airfoil)')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    plt.colorbar(pc, ax=ax, shrink=0.8, label='p')
    
    # --- Row 2: Velocity Magnitude ---
    ax = axes[1, 0]
    vel_clip = np.clip(vel_mag, 0, np.percentile(vel_mag, 99))
    pc = ax.pcolormesh(X.T, Y.T, vel_clip.T, cmap='viridis', shading='flat', rasterized=True)
    ax.set_aspect('equal')
    ax.set_title(f'Velocity Magnitude (Global) - CFL={cfl:.2f}')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    plt.colorbar(pc, ax=ax, shrink=0.8, label='|V|')
    
    ax = axes[1, 1]
    pc = ax.pcolormesh(X.T, Y.T, vel_clip.T, cmap='viridis', shading='flat', rasterized=True)
    ax.set_aspect('equal')
    ax.set_xlim(-0.2, 1.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_title(f'Velocity Magnitude (Near Airfoil) - Res={residual:.2e}')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    plt.colorbar(pc, ax=ax, shrink=0.8, label='|V|')
    
    # Track current row index
    row_idx = 2
    
    # --- Row 3: Total Pressure Loss (Entropy Check) ---
    if C_pt is not None:
        # Use symmetric colormap centered at 0
        # Clip extreme values for visualization
        C_pt_abs_max = max(np.abs(np.percentile(C_pt, 1)), np.abs(np.percentile(C_pt, 99)))
        C_pt_abs_max = max(C_pt_abs_max, 0.01)  # Minimum range
        
        ax = axes[row_idx, 0]
        pc = ax.pcolormesh(X.T, Y.T, C_pt.T, cmap='RdBu_r', shading='flat', 
                           rasterized=True, vmin=-C_pt_abs_max, vmax=C_pt_abs_max)
        ax.set_aspect('equal')
        ax.set_title(f'Total Pressure Loss C_pt (Global) - Entropy Check')
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        plt.colorbar(pc, ax=ax, shrink=0.8, label='C_pt')
        
        ax = axes[row_idx, 1]
        pc = ax.pcolormesh(X.T, Y.T, C_pt.T, cmap='RdBu_r', shading='flat', 
                           rasterized=True, vmin=-C_pt_abs_max, vmax=C_pt_abs_max)
        ax.set_aspect('equal')
        ax.set_xlim(-0.2, 1.4)
        ax.set_ylim(-0.4, 0.4)
        ax.set_title(f'Total Pressure Loss C_pt (Near Airfoil)')
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        
        # Add statistics annotation
        C_pt_stats = f"C_pt: min={C_pt.min():.4f}, max={C_pt.max():.4f}, mean={C_pt.mean():.4f}"
        ax.annotate(C_pt_stats, xy=(0.02, 0.02), xycoords='axes fraction', 
                    fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.colorbar(pc, ax=ax, shrink=0.8, label='C_pt')
        
        row_idx += 1
    
    # --- Residual Field (if provided) ---
    if residual_field is not None:
        # Use pressure residual (component 0) as the main diagnostic
        R_p = np.abs(residual_field[:, :, 0])
        
        # Log scale for better visualization
        R_log = np.log10(R_p + 1e-12)
        R_min = np.percentile(R_log, 1)
        R_max = np.percentile(R_log, 99)
        
        ax = axes[row_idx, 0]
        pc = ax.pcolormesh(X.T, Y.T, R_log.T, cmap='hot_r', shading='flat', 
                           rasterized=True, vmin=R_min, vmax=R_max)
        ax.set_aspect('equal')
        ax.set_title(f'log₁₀|Residual| (Global) - Iter {iteration}')
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        plt.colorbar(pc, ax=ax, shrink=0.8, label='log₁₀|R_p|')
        
        ax = axes[row_idx, 1]
        pc = ax.pcolormesh(X.T, Y.T, R_log.T, cmap='hot_r', shading='flat', 
                           rasterized=True, vmin=R_min, vmax=R_max)
        ax.set_aspect('equal')
        ax.set_xlim(-0.2, 1.4)
        ax.set_ylim(-0.4, 0.4)
        ax.set_title(f'log₁₀|Residual| (Near Airfoil)')
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        
        # Find and mark max residual location
        max_idx = np.unravel_index(np.argmax(R_p), R_p.shape)
        x_max = 0.5 * (X[max_idx[0], max_idx[1]] + X[max_idx[0]+1, max_idx[1]])
        y_max = 0.5 * (Y[max_idx[0], max_idx[1]] + Y[max_idx[0]+1, max_idx[1]])
        ax.plot(x_max, y_max, 'g*', markersize=15, markeredgecolor='white', markeredgewidth=1)
        
        # Add statistics annotation
        R_stats = f"Max |R|={R_p.max():.4e} at ({max_idx[0]},{max_idx[1]})"
        ax.annotate(R_stats, xy=(0.02, 0.02), xycoords='axes fraction', 
                    fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.colorbar(pc, ax=ax, shrink=0.8, label='log₁₀|R_p|')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'{case_name}_iter{iteration:06d}.pdf')
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    return output_path


def plot_residual_history(residuals: list, output_dir: str, case_name: str = "residual"):
    """Plot residual convergence history."""
    if len(residuals) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = np.arange(1, len(residuals) + 1)
    ax.semilogy(iterations, residuals, 'b-', lw=1.5)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual RMS')
    ax.set_title('Convergence History')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(residuals))
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{case_name}_convergence.pdf')
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    return output_path


class DiagnosticRANSSolver:
    """
    Enhanced RANS Solver with diagnostic visualization capabilities.
    
    Extends the basic RANSSolver with:
    - Flow field visualization at specified intervals
    - PDF output for debugging
    - More detailed residual information
    """
    
    def __init__(self, 
                 X: np.ndarray, 
                 Y: np.ndarray,
                 config: SolverConfig):
        """
        Initialize solver with pre-loaded grid.
        
        Parameters
        ----------
        X, Y : ndarray
            Grid node coordinates.
        config : SolverConfig
            Solver configuration.
        """
        self.config = config
        self.X = X
        self.Y = Y
        
        # Grid dimensions
        self.NI = X.shape[0] - 1
        self.NJ = X.shape[1] - 1
        
        # Initialize state
        self.iteration = 0
        self.residual_history = []
        self.converged = False
        
        # Divergence history buffer (rolling buffer for last N solutions)
        self.divergence_history_size = getattr(config, 'divergence_history_size', 5)
        self.solution_history = []  # List of (iteration, Q.copy()) tuples
        
        # Compute metrics
        self._compute_metrics()
        
        # Initialize state
        self._initialize_state()
        
        # Initialize output
        self._initialize_output()
        
        print(f"\n{'='*60}")
        print(f"Diagnostic RANS Solver Initialized")
        print(f"{'='*60}")
        print(f"Grid size: {self.NI} x {self.NJ} cells")
        print(f"Mach: {self.config.mach}, Alpha: {self.config.alpha}°")
        print(f"Reynolds: {self.config.reynolds:.2e}")
        print(f"Target CFL: {self.config.cfl_target}")
        print(f"{'='*60}\n")
    
    def _compute_metrics(self):
        """Compute FVM grid metrics."""
        print("Computing grid metrics...")
        
        computer = MetricComputer(self.X, self.Y, wall_j=0)
        self.metrics = computer.compute()
        
        gcl = computer.validate_gcl()
        print(f"  {gcl}")
        
        # Print some grid statistics
        print(f"  Min cell volume: {self.metrics.volume.min():.6e}")
        print(f"  Max cell volume: {self.metrics.volume.max():.6e}")
        print(f"  Min wall distance: {self.metrics.wall_distance.min():.6e}")
    
    def _initialize_state(self):
        """Initialize state vector."""
        print("Initializing flow state...")
        
        self.freestream = FreestreamConditions.from_mach_alpha(
            mach=self.config.mach,
            alpha_deg=self.config.alpha
        )
        
        self.Q = initialize_state(self.NI, self.NJ, self.freestream)
        
        self.Q = apply_initial_wall_damping(
            self.Q, 
            self.metrics,
            decay_length=self.config.wall_damping_length,
            n_wake=getattr(self.config, 'n_wake', 0)
        )
        
        # Compute far-field outward unit normals for Riemann BC
        Sj_x_ff = self.metrics.Sj_x[:, -1]
        Sj_y_ff = self.metrics.Sj_y[:, -1]
        Sj_mag = np.sqrt(Sj_x_ff**2 + Sj_y_ff**2) + 1e-12
        nx_ff = Sj_x_ff / Sj_mag
        ny_ff = Sj_y_ff / Sj_mag
        
        self.bc = BoundaryConditions(
            freestream=self.freestream,
            farfield_normals=(nx_ff, ny_ff),
            beta=self.config.beta,
            n_wake_points=getattr(self.config, 'n_wake', 0)
        )
        self.Q = self.bc.apply(self.Q)
        
        print(f"  Freestream: u={self.freestream.u_inf:.4f}, v={self.freestream.v_inf:.4f}")
        print(f"  Far-field BC: Riemann-invariant based")
    
    def _initialize_output(self):
        """Initialize output directories and writers."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectory for snapshots
        self.snapshot_dir = output_path / "snapshots"
        self.snapshot_dir.mkdir(exist_ok=True)
        
        # VTK writer
        base_path = output_path / self.config.case_name
        self.vtk_writer = VTKWriter(
            str(base_path),
            self.X, self.Y,
            beta=self.config.beta
        )
        
        # Write initial state with C_pt
        C_pt = self._compute_total_pressure_loss()
        self.vtk_writer.write(self.Q, iteration=0, 
                              additional_scalars={'TotalPressureLoss': C_pt})
        print(f"  Output directory: {output_path}")
    
    def _compute_total_pressure_loss(self) -> np.ndarray:
        """Compute total pressure loss coefficient for entropy check."""
        # Get interior state (strip ghost cells)
        Q_int = self.Q[1:-1, 1:-1, :]
        return compute_total_pressure_loss(Q_int, self.freestream)
    
    def _get_cfl(self, iteration: int) -> float:
        """Get CFL with linear ramping."""
        if iteration >= self.config.cfl_ramp_iters:
            return self.config.cfl_target
        
        t = iteration / self.config.cfl_ramp_iters
        return self.config.cfl_start + t * (self.config.cfl_target - self.config.cfl_start)
    
    def _compute_residual(self, Q: np.ndarray) -> np.ndarray:
        """Compute flux residual (convective + optional viscous)."""
        first_order = getattr(self.config, 'first_order', False)
        nu_min = getattr(self.config, 'nu_min', 0.0)
        flux_cfg = FluxConfig(k2=self.config.jst_k2, k4=self.config.jst_k4, 
                              nu_min=nu_min, first_order=first_order)
        
        flux_metrics = FluxGridMetrics(
            Si_x=self.metrics.Si_x,
            Si_y=self.metrics.Si_y,
            Sj_x=self.metrics.Sj_x,
            Sj_y=self.metrics.Sj_y,
            volume=self.metrics.volume
        )
        
        # Compute convective fluxes (JST scheme)
        conv_residual = compute_fluxes(Q, flux_metrics, self.config.beta, flux_cfg)
        
        # Add viscous fluxes if enabled
        if getattr(self.config, 'use_viscous', False):
            # Compute gradients
            grad_metrics = GradientMetrics(
                Si_x=self.metrics.Si_x,
                Si_y=self.metrics.Si_y,
                Sj_x=self.metrics.Sj_x,
                Sj_y=self.metrics.Sj_y,
                volume=self.metrics.volume
            )
            gradients = compute_gradients(Q, grad_metrics)
            
            # Laminar viscosity from Reynolds number
            mu_laminar = 1.0 / self.config.reynolds if self.config.reynolds > 0 else 0.0
            
            # Add viscous fluxes
            residual = add_viscous_fluxes(
                conv_residual, Q, gradients, grad_metrics, mu_laminar
            )
            
            return residual
        
        return conv_residual
    
    def _compute_residual_rms(self, residual: np.ndarray) -> float:
        """Compute RMS of density residual."""
        R_rho = residual[:, :, 0]
        return np.sqrt(np.sum(R_rho**2) / R_rho.size)
    
    def _check_solution_bounds(self, Q: np.ndarray) -> dict:
        """Check solution for physical bounds and NaN."""
        Q_int = Q[1:-1, 1:-1, :]
        
        info = {
            'has_nan': np.any(np.isnan(Q_int)),
            'has_inf': np.any(np.isinf(Q_int)),
            'p_min': Q_int[:, :, 0].min(),
            'p_max': Q_int[:, :, 0].max(),
            'u_min': Q_int[:, :, 1].min(),
            'u_max': Q_int[:, :, 1].max(),
            'v_min': Q_int[:, :, 2].min(),
            'v_max': Q_int[:, :, 2].max(),
            'nu_min': Q_int[:, :, 3].min(),
            'nu_max': Q_int[:, :, 3].max(),
        }
        
        vel_mag = np.sqrt(Q_int[:, :, 1]**2 + Q_int[:, :, 2]**2)
        info['vel_max'] = vel_mag.max()
        info['vel_max_loc'] = np.unravel_index(np.argmax(vel_mag), vel_mag.shape)
        
        return info
    
    def step(self) -> tuple:
        """Perform one iteration."""
        cfl = self._get_cfl(self.iteration)
        
        # Use local time stepping (let each cell march at its own pace)
        ts_config = TimeStepConfig(cfl=cfl, use_global_dt=False)
        
        # Compute kinematic viscosity for viscous stability limit
        nu = 0.0
        if getattr(self.config, 'use_viscous', False) and self.config.reynolds > 0:
            nu = 1.0 / self.config.reynolds
        
        dt = compute_local_timestep(
            self.Q,
            self.metrics.Si_x, self.metrics.Si_y,
            self.metrics.Sj_x, self.metrics.Sj_y,
            self.metrics.volume,
            self.config.beta,
            ts_config,
            nu=nu
        )
        
        # RK4 integration
        Q0 = self.Q.copy()
        Qk = self.Q.copy()
        
        alphas = [0.25, 0.333333333, 0.5, 1.0]
        
        for alpha in alphas:
            Qk = self.bc.apply(Qk)
            R = self._compute_residual(Qk)
            Qk = Q0.copy()
            Qk[1:-1, 1:-1, :] += alpha * (dt / self.metrics.volume)[:, :, np.newaxis] * R
        
        Qk = self.bc.apply(Qk)
        final_residual = self._compute_residual(Qk)
        
        self.Q = Qk
        
        residual_rms = self._compute_residual_rms(final_residual)
        
        # Find max residual location
        R_abs = np.abs(final_residual[:, :, 0])  # Pressure residual
        max_idx = np.unravel_index(np.argmax(R_abs), R_abs.shape)
        max_res = R_abs[max_idx]
        i_max, j_max = max_idx
        x_max = 0.5 * (self.X[i_max, j_max] + self.X[i_max+1, j_max])
        y_max = 0.5 * (self.Y[i_max, j_max] + self.Y[i_max+1, j_max])
        
        max_res_info = {
            'value': max_res,
            'i': i_max,
            'j': j_max,
            'x': x_max,
            'y': y_max,
        }
        
        self.iteration += 1
        
        return residual_rms, final_residual, cfl, max_res_info
    
    def run_with_diagnostics(self, dump_freq: int = 100) -> bool:
        """
        Run simulation with diagnostic output.
        
        Parameters
        ----------
        dump_freq : int
            Frequency of flow field dumps (PDF + VTK).
        """
        print(f"\n{'='*100}")
        print("Starting Steady-State Iteration (Diagnostic Mode)")
        print(f"Dumping flow field every {dump_freq} iterations")
        print(f"{'='*108}")
        print(f"{'Iter':>8} {'RMS':>12} {'Max':>12} {'MaxLoc':>26} {'CFL':>8} {'|V|_max':>10} {'p_range':>18}")
        print(f"{'-'*108}")
        
        initial_residual = None
        
        # Dump initial state
        plot_flow_field(
            self.X, self.Y, self.Q,
            iteration=0, residual=0, cfl=self._get_cfl(0),
            output_dir=str(self.snapshot_dir),
            case_name=self.config.case_name,
            freestream=self.freestream
        )
        
        for n in range(self.config.max_iter):
            res_rms, R_field, cfl, max_info = self.step()
            
            self.residual_history.append(res_rms)
            self._last_residual_field = R_field  # Store for visualization
            
            # Update divergence history buffer (rolling) - only if enabled
            if self.divergence_history_size > 0:
                self.solution_history.append((self.iteration, self.Q.copy(), R_field.copy()))
                if len(self.solution_history) > self.divergence_history_size:
                    self.solution_history.pop(0)
            
            if initial_residual is None:
                initial_residual = res_rms
            
            # Check solution bounds
            bounds = self._check_solution_bounds(self.Q)
            
            # Print progress every 10 iterations
            if self.iteration % 10 == 0 or self.iteration == 1:
                p_range = f"[{bounds['p_min']:.2f}, {bounds['p_max']:.2f}]"
                max_loc = f"({max_info['i']:3d},{max_info['j']:3d}) x={max_info['x']:5.2f} y={max_info['y']:5.2f}"
                print(f"{self.iteration:>8d} {res_rms:>12.4e} {max_info['value']:>12.4e} {max_loc:>26} "
                      f"{cfl:>8.2f} {bounds['vel_max']:>10.4f} {p_range:>18}")
            
            # Dump flow field
            if self.iteration % dump_freq == 0:
                pdf_path = plot_flow_field(
                    self.X, self.Y, self.Q,
                    iteration=self.iteration, 
                    residual=res_rms, 
                    cfl=cfl,
                    output_dir=str(self.snapshot_dir),
                    case_name=self.config.case_name,
                    freestream=self.freestream,
                    residual_field=R_field
                )
                C_pt = self._compute_total_pressure_loss()
                self.vtk_writer.write(self.Q, iteration=self.iteration,
                                      additional_scalars={'TotalPressureLoss': C_pt})
                print(f"         -> Dumped: {pdf_path}")
            
            # Write VTK at output_freq
            elif self.iteration % self.config.output_freq == 0:
                C_pt = self._compute_total_pressure_loss()
                self.vtk_writer.write(self.Q, iteration=self.iteration,
                                      additional_scalars={'TotalPressureLoss': C_pt})
            
            # Check for NaN/Inf
            if bounds['has_nan'] or bounds['has_inf']:
                print(f"\n{'='*60}")
                print(f"DIVERGED at iteration {self.iteration} - NaN/Inf detected!")
                print(f"  Max velocity location: {bounds['vel_max_loc']}")
                self._dump_divergence_info(bounds)
                break
            
            # Check convergence
            if res_rms < self.config.tol:
                self.converged = True
                print(f"\n{'='*60}")
                print(f"CONVERGED at iteration {self.iteration}")
                print(f"Final residual: {res_rms:.6e}")
                break
            
            # Check divergence
            if res_rms > 1000 * initial_residual:
                print(f"\n{'='*60}")
                print(f"DIVERGED at iteration {self.iteration}")
                print(f"Residual: {res_rms:.6e} (initial: {initial_residual:.6e})")
                self._dump_divergence_info(bounds)
                break
        
        else:
            print(f"\n{'='*60}")
            print(f"Maximum iterations ({self.config.max_iter}) reached")
        
        # Final dumps
        plot_flow_field(
            self.X, self.Y, self.Q,
            iteration=self.iteration, 
            residual=self.residual_history[-1] if self.residual_history else 0, 
            cfl=self._get_cfl(self.iteration),
            output_dir=str(self.snapshot_dir),
            case_name=f"{self.config.case_name}_final",
            freestream=self.freestream
        )
        
        C_pt = self._compute_total_pressure_loss()
        self.vtk_writer.write(self.Q, iteration=self.iteration,
                              additional_scalars={'TotalPressureLoss': C_pt})
        self.vtk_writer.finalize()
        
        # Print C_pt statistics
        print(f"\nEntropy Check (Total Pressure Loss C_pt):")
        print(f"  Min:  {C_pt.min():.6f}")
        print(f"  Max:  {C_pt.max():.6f}")
        print(f"  Mean: {C_pt.mean():.6f}")
        print(f"  (Ideal for inviscid: C_pt = 0 everywhere)")
        
        # Plot residual history
        plot_residual_history(
            self.residual_history,
            str(self.snapshot_dir),
            self.config.case_name
        )
        
        print(f"{'='*60}")
        print(f"Output in: {self.config.output_dir}")
        print(f"Snapshots in: {self.snapshot_dir}")
        
        return self.converged
    
    def _dump_divergence_info(self, bounds: dict):
        """Print detailed info when divergence is detected."""
        print(f"\nDivergence Diagnostics:")
        print(f"  Pressure range: [{bounds['p_min']:.4f}, {bounds['p_max']:.4f}]")
        print(f"  U-velocity range: [{bounds['u_min']:.4f}, {bounds['u_max']:.4f}]")
        print(f"  V-velocity range: [{bounds['v_min']:.4f}, {bounds['v_max']:.4f}]")
        print(f"  Max velocity: {bounds['vel_max']:.4f} at cell {bounds['vel_max_loc']}")
        print(f"  Nu_t range: [{bounds['nu_min']:.6e}, {bounds['nu_max']:.6e}]")
        
        # Dump divergence history
        self._visualize_divergence_history()
    
    def _visualize_divergence_history(self):
        """Visualize the last N flow solutions before divergence."""
        if self.divergence_history_size <= 0:
            return  # Divergence visualization disabled
        if not self.solution_history:
            print("  No solution history available.")
            return
        
        # Create divergence subdirectory
        divergence_dir = self.snapshot_dir / "divergence"
        divergence_dir.mkdir(exist_ok=True)
        
        print(f"\n  Dumping last {len(self.solution_history)} solutions before divergence:")
        
        for idx, (iteration, Q, R_field) in enumerate(self.solution_history):
            # Dump PDF
            pdf_path = plot_flow_field(
                self.X, self.Y, Q,
                iteration=iteration,
                residual=np.sqrt(np.mean(R_field**2)),
                cfl=self._get_cfl(iteration),
                output_dir=str(divergence_dir),
                case_name=f"{self.config.case_name}_div",
                freestream=self.freestream,
                residual_field=R_field
            )
            
            # Dump VTK
            vtk_path = divergence_dir / f"{self.config.case_name}_div_{iteration:06d}.vtk"
            from src.io import write_vtk
            write_vtk(str(vtk_path), self.X, self.Y, Q, self.config.beta)
            
            steps_before = len(self.solution_history) - idx - 1
            print(f"    Step {iteration} ({steps_before} before divergence): {pdf_path}")


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
                        help="Use super-coarse grid (40x15) with relaxed y+ for fast convergence tests")
    parser.add_argument("--inviscid", action="store_true",
                        help="Run inviscid (no viscous fluxes)")
    parser.add_argument("--viscous", action="store_true",
                        help="Run with viscous fluxes (default if Re specified)")
    parser.add_argument("--first-order", action="store_true",
                        help="Use 1st-order dissipation (more stable, less accurate)")
    parser.add_argument("--nu-min", type=float, default=0.0,
                        help="Minimum sensor value for baseline dissipation (default: 0.0)")
    
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
                        help="Number of solutions to keep for divergence visualization (0=disabled, default: 0)")
    
    args = parser.parse_args()
    
    # Override grid settings for coarse/super-coarse modes
    # Note: 80x30 works well for viscous; 100x40 has stability issues with viscous
    if args.super_coarse:
        args.n_surface = 60
        args.n_normal = 20
        args.n_wake = 15
        # Store y_plus override for grid generation
        args.y_plus = 5.0  # Relaxed y+ for bigger cells
        print("Using SUPER-COARSE grid mode (60x20, y+=5) for fast convergence tests")
    elif args.coarse:
        args.n_surface = 80
        args.n_normal = 30
        args.n_wake = 20
        args.y_plus = 1.0
        print("Using COARSE grid mode for debugging (80x30)")
    
    # Print banner
    print("\n" + "="*70)
    print("   2D Incompressible RANS Solver - Diagnostic Mode")
    print("="*70)
    
    # Use y_plus from args if set (e.g., super-coarse mode), else default 1.0
    y_plus = getattr(args, 'y_plus', 1.0)
    # For super-coarse, also increase max_first_cell to allow bigger cells
    max_first_cell = 0.005 if args.super_coarse else 0.001
    
    # Load or generate grid using shared utility
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
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Determine if viscous
    use_viscous = args.viscous or (not args.inviscid and args.reynolds < 1e8)
    if use_viscous:
        print(f"Running VISCOUS simulation (Re = {args.reynolds:.2e})")
    else:
        print(f"Running INVISCID simulation")
    
    # Configure solver
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
        jst_k2=0.5,
        jst_k4=0.04,  # Increased from 0.016 to reduce oscillations
    )
    config.use_viscous = use_viscous  # Add this attribute
    config.first_order = args.first_order  # 1st-order dissipation mode
    config.nu_min = args.nu_min  # Baseline dissipation
    config.divergence_history_size = args.div_history  # Divergence visualization buffer
    config.n_wake = args.n_wake  # Wake points for BC region detection
    
    if args.first_order:
        print("Using FIRST-ORDER dissipation (more stable)")
    elif args.nu_min > 0:
        print(f"Using baseline dissipation nu_min = {args.nu_min}")
    
    # Create solver
    solver = DiagnosticRANSSolver(X, Y, config)
    
    # Run with diagnostics
    try:
        converged = solver.run_with_diagnostics(dump_freq=args.dump_freq)
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

