"""
RANS Solver for 2D Incompressible Flow using Artificial Compressibility.

State vector: Q = [p, u, v, ν̃]
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from ..grid.metrics import MetricComputer, FVMMetrics
from ..grid.mesher import Construct2DWrapper, GridOptions
from ..grid.plot3d import read_plot3d
from ..numerics.fluxes import compute_fluxes, FluxConfig, GridMetrics as FluxGridMetrics
from ..numerics.forces import compute_aerodynamic_forces, AeroForces
from ..numerics.gradients import compute_gradients, GradientMetrics
from ..numerics.viscous_fluxes import add_viscous_fluxes
from ..numerics.smoothing import apply_residual_smoothing
from .boundary_conditions import (
    FreestreamConditions, 
    BoundaryConditions,
    initialize_state,
    apply_initial_wall_damping,
)
from .time_stepping import compute_local_timestep, TimeStepConfig
from .multigrid import MultigridHierarchy, build_multigrid_hierarchy
from ..io.output import VTKWriter
from ..io.plotter import PlotlyDashboard
from ..numerics.forces import compute_surface_distributions, create_surface_vtk_fields
from ..constants import NGHOST
from ..numerics.diagnostics import (
    compute_total_pressure_loss, 
    compute_solution_bounds,
    compute_residual_statistics
)


@dataclass
class SolverConfig:
    """Configuration for RANS solver."""
    
    mach: float = 0.15
    alpha: float = 0.0
    reynolds: float = 6e6
    beta: float = 10.0
    cfl_start: float = 0.1
    cfl_target: float = 5.0
    cfl_ramp_iters: int = 500
    max_iter: int = 10000
    tol: float = 1e-10
    diagnostic_freq: int = 100
    vtk_output_freq: int = 0
    print_freq: int = 50
    output_dir: str = "output"
    case_name: str = "solution"
    wall_damping_length: float = 0.1
    jst_k4: float = 0.04
    irs_epsilon: float = 0.0
    n_wake: int = 30
    html_animation: bool = True
    divergence_history: int = 0
    use_multigrid: bool = False
    mg_levels: int = 4
    mg_nu1: int = 1
    mg_nu2: int = 1
    mg_min_size: int = 8
    mg_omega: float = 0.5
    mg_use_injection: bool = True
    mg_dissipation_scaling: float = 2.0
    mg_coarse_cfl: float = 0.5


class RANSSolver:
    """
    Main RANS Solver for 2D incompressible flow around airfoils.
    """
    
    def __init__(self, 
                 grid_file: str, 
                 config: Optional[Union[SolverConfig, Dict]] = None):
        """Initialize the RANS solver."""
        if config is None:
            self.config = SolverConfig()
        elif isinstance(config, dict):
            self.config = SolverConfig(**config)
        else:
            self.config = config
        
        self.iteration = 0
        self.residual_history = []
        self.converged = False
        
        self._load_grid(grid_file)
        self._compute_metrics()
        self._initialize_state()
        
        self.mg_hierarchy = None
        if self.config.use_multigrid:
            self._initialize_multigrid()
        
        self._initialize_output()
        
        print(f"\n{'='*60}")
        print(f"RANS Solver Initialized")
        print(f"{'='*60}")
        print(f"Grid size: {self.NI} x {self.NJ} cells")
        print(f"Mach: {self.config.mach}, Alpha: {self.config.alpha}°")
        print(f"Reynolds: {self.config.reynolds:.2e}")
        print(f"Target CFL: {self.config.cfl_target}")
        print(f"Max iterations: {self.config.max_iter}")
        print(f"Convergence tolerance: {self.config.tol:.2e}")
        print(f"{'='*60}\n")
    
    def _load_grid(self, grid_file: str):
        """Load grid from file or generate using Construct2D."""
        grid_path = Path(grid_file)
        
        if not grid_path.exists():
            raise FileNotFoundError(f"Grid file not found: {grid_path}")
        
        suffix = grid_path.suffix.lower()
        
        if suffix in ['.p3d', '.x', '.xyz']:
            print(f"Loading grid from: {grid_path}")
            self.X, self.Y = read_plot3d(str(grid_path))
            
        elif suffix == '.dat':
            print(f"Generating grid from airfoil: {grid_path}")
            construct2d_paths = [
                Path("bin/construct2d"),
                Path("./construct2d"),
                Path("/usr/local/bin/construct2d"),
            ]
            
            binary_path = None
            for p in construct2d_paths:
                if p.exists():
                    binary_path = p
                    break
            
            if binary_path is None:
                raise FileNotFoundError(
                    "Construct2D binary not found. Please provide a .p3d grid file "
                    "or install Construct2D."
                )
            
            wrapper = Construct2DWrapper(str(binary_path))
            grid_opts = GridOptions(
                n_surface=250,
                n_normal=100,
                y_plus=1.0,
                reynolds=self.config.reynolds
            )
            self.X, self.Y = wrapper.generate(str(grid_path), grid_opts)
        else:
            raise ValueError(f"Unsupported grid file format: {suffix}")
        
        self.NI = self.X.shape[0] - 1
        self.NJ = self.X.shape[1] - 1
        
        print(f"Grid loaded: {self.X.shape[0]} x {self.X.shape[1]} nodes")
        print(f"            {self.NI} x {self.NJ} cells")
    
    def _compute_metrics(self):
        """Compute FVM grid metrics."""
        print("Computing grid metrics...")
        
        computer = MetricComputer(self.X, self.Y, wall_j=0)
        self.metrics = computer.compute()
        
        self.flux_metrics = FluxGridMetrics(
            Si_x=self.metrics.Si_x,
            Si_y=self.metrics.Si_y,
            Sj_x=self.metrics.Sj_x,
            Sj_y=self.metrics.Sj_y,
            volume=self.metrics.volume
        )
        
        self.grad_metrics = GradientMetrics(
            Si_x=self.metrics.Si_x,
            Si_y=self.metrics.Si_y,
            Sj_x=self.metrics.Sj_x,
            Sj_y=self.metrics.Sj_y,
            volume=self.metrics.volume
        )
        
        gcl = computer.validate_gcl()
        print(f"  {gcl}")
        
        if not gcl.passed:
            print("  WARNING: GCL validation failed. Results may be inaccurate.")
    
    def _initialize_state(self):
        """Initialize state vector with freestream and wall damping."""
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
        
        Sj_x_ff = self.metrics.Sj_x[:, -1]
        Sj_y_ff = self.metrics.Sj_y[:, -1]
        Sj_mag = np.sqrt(Sj_x_ff**2 + Sj_y_ff**2) + 1e-12
        nx_ff = Sj_x_ff / Sj_mag
        ny_ff = Sj_y_ff / Sj_mag
        
        self.bc = BoundaryConditions(
            freestream=self.freestream,
            farfield_normals=(nx_ff, ny_ff),
            beta=self.config.beta,
            n_wake_points=getattr(self.config, 'n_wake', 0),
        )
        self.Q = self.bc.apply(self.Q)
        
        print(f"  Freestream: u={self.freestream.u_inf:.4f}, "
              f"v={self.freestream.v_inf:.4f}")
        print(f"  Far-field BC: Characteristic (non-reflecting)")
        print(f"  Wall damping applied (L={self.config.wall_damping_length})")
    
    def _initialize_multigrid(self):
        """Initialize multigrid hierarchy for FAS scheme."""
        print(f"  Initializing multigrid hierarchy...")
        
        self.mg_hierarchy = build_multigrid_hierarchy(
            X=self.X,
            Y=self.Y,
            Q=self.Q,
            freestream=self.freestream,
            n_wake=getattr(self.config, 'n_wake', 0),
            beta=self.config.beta,
            min_size=self.config.mg_min_size,
            max_levels=self.config.mg_levels,
            base_k4=self.config.jst_k4,
            dissipation_scaling=self.config.mg_dissipation_scaling,
            coarse_cfl_factor=self.config.mg_coarse_cfl
        )
        
        print(f"  Built {self.mg_hierarchy.num_levels} multigrid levels:")
        for i, lvl in enumerate(self.mg_hierarchy.levels):
            print(f"    Level {i}: {lvl.NI} x {lvl.NJ} (k4={lvl.k4:.4f}, cfl_scale={lvl.cfl_scale:.2f})")
    
    def _initialize_output(self):
        """Initialize VTK writer and HTML animation for output."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        base_path = output_path / self.config.case_name
        self.vtk_writer = VTKWriter(
            str(base_path),
            self.X, self.Y,
            beta=self.config.beta
        )
        
        self.plotter = PlotlyDashboard()
        
        if self.config.vtk_output_freq > 0:
            surface_fields = self._compute_surface_fields()
            self.vtk_writer.write(self.Q, iteration=0, additional_scalars=surface_fields)
        
        if self.config.html_animation:
            Q_int = self.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
            C_pt = compute_total_pressure_loss(
                Q_int, self.freestream.p_inf,
                self.freestream.u_inf, self.freestream.v_inf
            )
            initial_R = self._compute_residual(self.Q)
            mg_levels_data = None
            if self.config.use_multigrid and self.mg_hierarchy is not None:
                mg_levels_data = self._get_mg_level_snapshots()
            self.plotter.store_snapshot(
                self.Q, 0, self.residual_history,
                cfl=self._get_cfl(0), C_pt=C_pt, residual_field=initial_R,
                mg_levels=mg_levels_data, freestream=self.freestream
            )
        
        print(f"  Output directory: {output_path}")
    
    def _compute_surface_fields(self) -> dict:
        """Compute surface Cp and Cf fields for VTK output."""
        V_inf = np.sqrt(self.freestream.u_inf**2 + self.freestream.v_inf**2)
        mu_laminar = 1.0 / self.config.reynolds if self.config.reynolds > 0 else 0.0
        
        return create_surface_vtk_fields(
            Q=self.Q,
            X=self.X,
            Y=self.Y,
            metrics=self.flux_metrics,
            mu_laminar=mu_laminar,
            mu_turb=None,  # For laminar flow
            p_inf=self.freestream.p_inf,
            rho_inf=1.0,
            V_inf=V_inf,
        )
    
    def _get_cfl(self, iteration: int) -> float:
        """Get CFL number with linear ramping."""
        if iteration >= self.config.cfl_ramp_iters:
            return self.config.cfl_target
        
        # Linear ramp from cfl_start to cfl_target
        t = iteration / self.config.cfl_ramp_iters
        return self.config.cfl_start + t * (self.config.cfl_target - self.config.cfl_start)
    
    def _compute_residual(self, Q: np.ndarray, 
                           forcing: Optional[np.ndarray] = None,
                           flux_metrics: Optional[FluxGridMetrics] = None,
                           grad_metrics: Optional[GradientMetrics] = None,
                           k4: Optional[float] = None) -> np.ndarray:
        """Compute flux residual using JST scheme + viscous fluxes."""
        if flux_metrics is None:
            flux_metrics = self.flux_metrics
        if grad_metrics is None:
            grad_metrics = self.grad_metrics
        if k4 is None:
            k4 = self.config.jst_k4
        
        flux_cfg = FluxConfig(k4=k4)
        conv_residual = compute_fluxes(Q, flux_metrics, self.config.beta, flux_cfg)
        gradients = compute_gradients(Q, grad_metrics)
        mu_laminar = 1.0 / self.config.reynolds if self.config.reynolds > 0 else 0.0
        
        # Compute turbulent viscosity from SA working variable (ν̃)
        # μ_t = ν̃ * f_v1, where f_v1 = χ³/(χ³ + c_v1³), χ = ν̃/ν
        Q_int = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
        nu_tilde = np.maximum(Q_int[:, :, 3], 0.0)  # SA working variable, prevent negative
        if mu_laminar > 0:
            chi = nu_tilde / mu_laminar  # χ = ν̃/ν
            cv1 = 7.1
            chi3 = chi ** 3
            f_v1 = chi3 / (chi3 + cv1 ** 3)
            mu_turbulent = nu_tilde * f_v1
        else:
            mu_turbulent = np.zeros_like(nu_tilde)
        
        residual = add_viscous_fluxes(
            conv_residual, Q, gradients, grad_metrics, mu_laminar, mu_turbulent
        )
        
        if forcing is not None:
            residual = residual + forcing
        
        return residual
    
    def _apply_bc(self, Q: np.ndarray) -> np.ndarray:
        """Apply boundary conditions."""
        return self.bc.apply(Q)
    
    def _compute_residual_rms(self, residual: np.ndarray) -> float:
        """Compute RMS of density (continuity) residual."""
        # Density residual is the first component (pressure for incompressible)
        R_rho = residual[:, :, 0]
        
        n_cells = R_rho.size
        rms = np.sqrt(np.sum(R_rho**2) / n_cells)
        
        return rms
    
    def step(self) -> Tuple[float, np.ndarray]:
        """Perform one iteration of the solver."""
        cfl = self._get_cfl(self.iteration)
        ts_config = TimeStepConfig(cfl=cfl)
        nu = 1.0 / self.config.reynolds if self.config.reynolds > 0 else 0.0
        
        dt = compute_local_timestep(
            self.Q,
            self.metrics.Si_x, self.metrics.Si_y,
            self.metrics.Sj_x, self.metrics.Sj_y,
            self.metrics.volume,
            self.config.beta,
            ts_config,
            nu=nu
        )
        
        Q0 = self.Q.copy()
        Qk = self.Q.copy()
        
        alphas = [0.25, 0.166666667, 0.375, 0.5, 1.0]
        
        for alpha in alphas:
            Qk = self._apply_bc(Qk)
            R = self._compute_residual(Qk)
            
            if self.config.irs_epsilon > 0.0:
                apply_residual_smoothing(R, self.config.irs_epsilon)
            
            Qk = Q0.copy()
            Qk[NGHOST:-NGHOST, NGHOST:-NGHOST, :] += alpha * (dt / self.metrics.volume)[:, :, np.newaxis] * R
        
        Qk = self._apply_bc(Qk)
        final_residual = self._compute_residual(Qk)
        self.Q = Qk
        residual_rms = self._compute_residual_rms(final_residual)
        self.iteration += 1
        
        return residual_rms, final_residual
    
    def step_multigrid(self) -> Tuple[float, np.ndarray]:
        """Perform one V-cycle iteration with multigrid acceleration."""
        if self.mg_hierarchy is None:
            raise RuntimeError("Multigrid not initialized. Set use_multigrid=True.")
        
        self.mg_hierarchy.levels[0].Q[:] = self.Q
        cfl = self._get_cfl(self.iteration)
        ts_config = TimeStepConfig(cfl=cfl)
        nu = 1.0 / self.config.reynolds if self.config.reynolds > 0 else 0.0
        
        dt = compute_local_timestep(
            self.Q,
            self.metrics.Si_x, self.metrics.Si_y,
            self.metrics.Sj_x, self.metrics.Sj_y,
            self.metrics.volume,
            self.config.beta,
            ts_config,
            nu=nu
        )
        self.mg_hierarchy.levels[0].dt[:] = dt
        
        self._run_v_cycle(0)
        self.Q = self.mg_hierarchy.levels[0].Q.copy()
        final_residual = self._compute_residual(self.Q)
        residual_rms = self._compute_residual_rms(final_residual)
        self.iteration += 1
        
        return residual_rms, final_residual
    
    def _run_v_cycle(self, level: int):
        """Run recursive V-cycle from given level using FAS algorithm."""
        lvl = self.mg_hierarchy.levels[level]
        
        for _ in range(self.config.mg_nu1):
            self._smooth_level(level)
        
        if level >= self.mg_hierarchy.num_levels - 1:
            for _ in range(self.config.mg_nu1 + self.config.mg_nu2):
                self._smooth_level(level)
            return
        
        lvl.Q = lvl.bc.apply(lvl.Q)
        lvl.R = self._compute_residual_level(level)
        
        self.mg_hierarchy.restrict_to_coarse(level)
        
        coarse = self.mg_hierarchy.levels[level + 1]
        coarse.Q = coarse.bc.apply(coarse.Q)
        R_coarse = self._compute_residual_level(level + 1)
        coarse.forcing = coarse.R - R_coarse
        coarse.Q_old[:] = coarse.Q
        
        self._run_v_cycle(level + 1)
        
        lvl.Q_before_correction = lvl.Q.copy()
        self.mg_hierarchy.prolongate_correction(level + 1, 
                                                 use_injection=self.config.mg_use_injection)
        
        omega = self.config.mg_omega
        if omega < 1.0:
            lvl.Q = lvl.Q_before_correction + omega * (lvl.Q - lvl.Q_before_correction)
        
        # Average corrections at wake edges to maintain periodicity
        Q_int = lvl.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
        Q_before = lvl.Q_before_correction[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
        dQ_left = Q_int[0, :, :] - Q_before[0, :, :]
        dQ_right = Q_int[-1, :, :] - Q_before[-1, :, :]
        dQ_avg = 0.5 * (dQ_left + dQ_right)
        Q_int[0, :, :] = Q_before[0, :, :] + dQ_avg
        Q_int[-1, :, :] = Q_before[-1, :, :] + dQ_avg
        
        lvl.Q = lvl.bc.apply(lvl.Q)
        
        for _ in range(self.config.mg_nu2):
            self._smooth_level(level)
    
    def _smooth_level(self, level: int):
        """Perform one smoothing step on a multigrid level using RK."""
        lvl = self.mg_hierarchy.levels[level]
        
        if level > 0:
            from src.numerics.multigrid import restrict_timestep
            fine = self.mg_hierarchy.levels[level - 1]
            restrict_timestep(fine.dt, lvl.dt)
        
        forcing = lvl.forcing if level > 0 else None
        
        Q0 = lvl.Q.copy()
        Qk = lvl.Q.copy()
        
        if level == 0:
            alphas = [0.25, 0.166666667, 0.375, 0.5, 1.0]
        else:
            alphas = [0.333333, 0.5, 1.0]
        
        for alpha in alphas:
            Qk = lvl.bc.apply(Qk)
            R = self._compute_residual_level(level, Q=Qk, forcing=forcing)
            
            if level == 0 and self.config.irs_epsilon > 0.0:
                apply_residual_smoothing(R, self.config.irs_epsilon)
            
            Qk = Q0.copy()
            dt_scaled = lvl.dt * lvl.cfl_scale
            Qk[NGHOST:-NGHOST, NGHOST:-NGHOST, :] += alpha * (dt_scaled / lvl.metrics.volume)[:, :, np.newaxis] * R
        
        lvl.Q = lvl.bc.apply(Qk)
    
    def _compute_residual_level(self, level: int, 
                                  Q: Optional[np.ndarray] = None,
                                  forcing: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute residual on a specific multigrid level."""
        lvl = self.mg_hierarchy.levels[level]
        
        if Q is None:
            Q = lvl.Q
        
        flux_metrics = FluxGridMetrics(
            Si_x=lvl.metrics.Si_x,
            Si_y=lvl.metrics.Si_y,
            Sj_x=lvl.metrics.Sj_x,
            Sj_y=lvl.metrics.Sj_y,
            volume=lvl.metrics.volume
        )
        
        grad_metrics = GradientMetrics(
            Si_x=lvl.metrics.Si_x,
            Si_y=lvl.metrics.Si_y,
            Sj_x=lvl.metrics.Sj_x,
            Sj_y=lvl.metrics.Sj_y,
            volume=lvl.metrics.volume
        )
        
        return self._compute_residual(Q, forcing, flux_metrics, grad_metrics, k4=lvl.k4)
    
    def _get_mg_level_snapshots(self) -> List[Dict[str, np.ndarray]]:
        """Extract multigrid level data for visualization."""
        if self.mg_hierarchy is None:
            return None
        
        mg_levels = []
        for level_idx, lvl in enumerate(self.mg_hierarchy.levels[1:], start=1):
            Q_int = lvl.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
            R = self._compute_residual_level(level_idx)
            residual_rms = np.sqrt(np.mean(R**2, axis=2))
            
            mg_levels.append({
                'p': Q_int[:, :, 0].copy(),
                'u': Q_int[:, :, 1].copy(),
                'v': Q_int[:, :, 2].copy(),
                'xc': lvl.metrics.xc.copy(),
                'yc': lvl.metrics.yc.copy(),
                'residual': residual_rms,
            })
        return mg_levels if mg_levels else None
    
    def run_steady_state(self) -> bool:
        """Run steady-state simulation to convergence."""
        print(f"\n{'='*60}")
        print("Starting Steady-State Iteration")
        print(f"{'='*60}")
        print(f"{'Iter':>8} {'Residual':>14} {'CFL':>8}  "
              f"{'CL':>8} {'CD':>8} {'(p:':>7} {'f:)':>7}")
        print(f"{'-'*72}")
        
        # Initial residual for normalization
        initial_residual = None
        
        for n in range(self.config.max_iter):
            if self.config.use_multigrid:
                res_rms, R_field = self.step_multigrid()
            else:
                res_rms, R_field = self.step()
            
            self.residual_history.append(res_rms)
            
            if initial_residual is None:
                initial_residual = res_rms
            
            res_norm = res_rms / (initial_residual + 1e-30)
            cfl = self._get_cfl(self.iteration)
            
            if self.iteration % self.config.print_freq == 0 or self.iteration == 1:
                forces = self.compute_forces()
                print(f"{self.iteration:>8d} {res_rms:>14.6e} {cfl:>8.2f}  "
                      f"CL={forces.CL:>8.4f} CD={forces.CD:>8.5f} "
                      f"(p:{forces.CD_p:>7.5f} f:{forces.CD_f:>7.5f})")
            
            if self.config.vtk_output_freq > 0 and self.iteration % self.config.vtk_output_freq == 0:
                surface_fields = self._compute_surface_fields()
                self.vtk_writer.write(
                    self.Q, iteration=self.iteration,
                    additional_scalars=surface_fields
                )
            
            if self.config.html_animation and self.iteration % self.config.diagnostic_freq == 0:
                Q_int = self.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
                C_pt = compute_total_pressure_loss(
                    Q_int, self.freestream.p_inf,
                    self.freestream.u_inf, self.freestream.v_inf
                )
                mg_levels_data = None
                if self.config.use_multigrid and self.mg_hierarchy is not None:
                    mg_levels_data = self._get_mg_level_snapshots()
                
                self.plotter.store_snapshot(
                    self.Q, self.iteration, self.residual_history,
                    cfl=cfl, C_pt=C_pt, residual_field=R_field,
                    mg_levels=mg_levels_data, freestream=self.freestream
                )
            
            if res_rms < self.config.tol:
                self.converged = True
                print(f"\n{'='*60}")
                print(f"CONVERGED at iteration {self.iteration}")
                print(f"Final residual: {res_rms:.6e}")
                print(f"{'='*60}")
                break
            
            if res_rms > 1000 * initial_residual:
                print(f"\n{'='*60}")
                print(f"DIVERGED at iteration {self.iteration}")
                print(f"Residual: {res_rms:.6e} (initial: {initial_residual:.6e})")
                print(f"{'='*60}")
                break
        
        else:
            print(f"\n{'='*60}")
            print(f"Maximum iterations ({self.config.max_iter}) reached")
            print(f"Final residual: {self.residual_history[-1]:.6e}")
            print(f"{'='*60}")
        
        if self.config.vtk_output_freq > 0:
            surface_fields = self._compute_surface_fields()
            self.vtk_writer.write(
                self.Q, iteration=self.iteration,
                additional_scalars=surface_fields
            )
            series_file = self.vtk_writer.finalize()
            print(f"VTK series written to: {series_file}")
        
        if self.config.html_animation and self.plotter.num_snapshots > 0:
            html_path = Path(self.config.output_dir) / f"{self.config.case_name}_animation.html"
            self.plotter.save_html(str(html_path), self.metrics)
        
        return self.converged
    
    def get_surface_data(self) -> Dict[str, np.ndarray]:
        """Extract surface quantities for post-processing."""
        p_surface = self.Q[NGHOST:-NGHOST, NGHOST-1, 0]
        u_surface = self.Q[NGHOST:-NGHOST, NGHOST-1, 1]
        v_surface = self.Q[NGHOST:-NGHOST, NGHOST-1, 2]
        
        x_surface = 0.5 * (self.X[:-1, 0] + self.X[1:, 0])
        y_surface = 0.5 * (self.Y[:-1, 0] + self.Y[1:, 0])
        
        V_inf_sq = self.freestream.u_inf**2 + self.freestream.v_inf**2
        cp = 2.0 * (p_surface - self.freestream.p_inf) / (V_inf_sq + 1e-12)
        
        wall_dist = self.metrics.wall_distance[:, 0]
        vel_mag = np.sqrt(u_surface**2 + v_surface**2)
        cf = 2.0 * vel_mag / (wall_dist * self.config.reynolds * V_inf_sq + 1e-12)
        
        return {
            'x': x_surface,
            'y': y_surface,
            'cp': cp,
            'cf': cf
        }
    
    def compute_forces(self) -> AeroForces:
        """Compute aerodynamic force coefficients."""
        V_inf = np.sqrt(self.freestream.u_inf**2 + self.freestream.v_inf**2)
        mu_laminar = 1.0 / self.config.reynolds if self.config.reynolds > 0 else 0.0
        mu_turb = None
        
        n_wake = getattr(self.config, 'n_wake', 0)
        if n_wake == 0 and hasattr(self, 'bc'):
            n_wake = getattr(self.bc, 'n_wake_points', 0)
        
        forces = compute_aerodynamic_forces(
            Q=self.Q,
            metrics=self.flux_metrics,
            mu_laminar=mu_laminar,
            mu_turb=mu_turb,
            alpha_deg=self.config.alpha,
            chord=1.0,
            rho_inf=1.0,
            V_inf=V_inf,
            n_wake=n_wake,
        )
        
        return forces
    
    def get_surface_distributions(self):
        """Get surface Cp and Cf distributions."""
        V_inf = np.sqrt(self.freestream.u_inf**2 + self.freestream.v_inf**2)
        mu_laminar = 1.0 / self.config.reynolds if self.config.reynolds > 0 else 0.0
        
        return compute_surface_distributions(
            Q=self.Q,
            X=self.X,
            Y=self.Y,
            metrics=self.flux_metrics,
            mu_laminar=mu_laminar,
            mu_turb=None,
            p_inf=self.freestream.p_inf,
            rho_inf=1.0,
            V_inf=V_inf,
        )
    
    def save_residual_history(self, filename: str = None):
        """Save residual history to file."""
        if filename is None:
            filename = Path(self.config.output_dir) / "residual_history.dat"
        
        with open(filename, 'w') as f:
            f.write("# Iteration  Residual\n")
            for i, res in enumerate(self.residual_history):
                f.write(f"{i+1:8d}  {res:.10e}\n")
        
        print(f"Residual history saved to: {filename}")
    
    def run_with_diagnostics(self, dump_freq: int = None) -> bool:
        """Run steady-state simulation with enhanced diagnostic output."""
        from ..io.plotting import plot_flow_field, plot_residual_history, plot_multigrid_levels
        
        if dump_freq is None:
            dump_freq = getattr(self.config, 'diagnostic_freq', 100)
        
        snapshot_dir = Path(self.config.output_dir) / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        div_history_size = getattr(self.config, 'divergence_history', 0)
        solution_history = []
        
        print(f"\n{'='*108}")
        print("Starting Steady-State Iteration (Diagnostic Mode)")
        print(f"Dumping flow field every {dump_freq} iterations")
        print(f"{'='*108}")
        print(f"{'Iter':>8} {'RMS':>12} {'Max':>12} {'MaxLoc':>18} "
              f"{'CFL':>8} {'|V|_max':>10} {'p_range':>18}")
        print(f"{'-'*108}")
        
        initial_residual = None
        
        Q_int = self.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
        C_pt = compute_total_pressure_loss(
            Q_int, self.freestream.p_inf, 
            self.freestream.u_inf, self.freestream.v_inf
        )
        plot_flow_field(
            self.X, self.Y, self.Q,
            iteration=0, residual=0, cfl=self._get_cfl(0),
            output_dir=str(snapshot_dir),
            case_name=self.config.case_name,
            C_pt=C_pt
        )
        
        for n in range(self.config.max_iter):
            if self.config.use_multigrid:
                res_rms, R_field = self.step_multigrid()
            else:
                res_rms, R_field = self.step()
            self.residual_history.append(res_rms)
            
            if initial_residual is None:
                initial_residual = res_rms
            
            bounds = compute_solution_bounds(self.Q)
            res_stats = compute_residual_statistics(R_field)
            cfl = self._get_cfl(self.iteration)
            
            if div_history_size > 0:
                solution_history.append((self.iteration, self.Q.copy(), R_field.copy()))
                if len(solution_history) > div_history_size:
                    solution_history.pop(0)
            
            if self.iteration % 10 == 0 or self.iteration == 1:
                p_range = f"[{bounds['p_min']:.2f}, {bounds['p_max']:.2f}]"
                max_loc = f"({res_stats['max_loc'][0]:3d},{res_stats['max_loc'][1]:3d})"
                print(f"{self.iteration:>8d} {res_rms:>12.4e} {res_stats['max_p']:>12.4e} "
                      f"{max_loc:>18} {cfl:>8.2f} {bounds['vel_max']:>10.4f} {p_range:>18}")
            
            if self.iteration % dump_freq == 0:
                Q_int = self.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
                C_pt = compute_total_pressure_loss(
                    Q_int, self.freestream.p_inf,
                    self.freestream.u_inf, self.freestream.v_inf
                )
                
                # Use multigrid plot (all levels in one PDF) or single-level plot
                if self.config.use_multigrid and self.mg_hierarchy is not None:
                    pdf_path = plot_multigrid_levels(
                        self.mg_hierarchy,
                        X_fine=self.X, Y_fine=self.Y,
                        iteration=self.iteration, residual=res_rms, cfl=cfl,
                        output_dir=str(snapshot_dir),
                        case_name=self.config.case_name,
                        C_pt_fine=C_pt, residual_field_fine=R_field,
                        freestream=self.freestream
                    )
                else:
                    pdf_path = plot_flow_field(
                        self.X, self.Y, self.Q,
                        iteration=self.iteration, residual=res_rms, cfl=cfl,
                        output_dir=str(snapshot_dir),
                        case_name=self.config.case_name,
                        C_pt=C_pt, residual_field=R_field
                    )
                print(f"         -> Dumped: {pdf_path}")
                
                # VTK output with C_pt
                surface_fields = self._compute_surface_fields()
                surface_fields['TotalPressureLoss'] = C_pt
                self.vtk_writer.write(
                    self.Q, iteration=self.iteration,
                    additional_scalars=surface_fields
                )
            
            # Check for NaN/Inf
            if bounds['has_nan'] or bounds['has_inf']:
                print(f"\n{'='*60}")
                print(f"DIVERGED at iteration {self.iteration} - NaN/Inf detected!")
                self._print_divergence_info(bounds, solution_history, snapshot_dir)
                break
            
            # Check convergence
            if res_rms < self.config.tol:
                self.converged = True
                print(f"\n{'='*60}")
                print(f"CONVERGED at iteration {self.iteration}")
                print(f"Final residual: {res_rms:.6e}")
                break
            
            # Check for divergence
            if res_rms > 1000 * initial_residual:
                print(f"\n{'='*60}")
                print(f"DIVERGED at iteration {self.iteration}")
                print(f"Residual: {res_rms:.6e} (initial: {initial_residual:.6e})")
                self._print_divergence_info(bounds, solution_history, snapshot_dir)
                break
        
        else:
            print(f"\n{'='*60}")
            print(f"Maximum iterations ({self.config.max_iter}) reached")
        
        # Final dumps
        Q_int = self.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
        C_pt = compute_total_pressure_loss(
            Q_int, self.freestream.p_inf,
            self.freestream.u_inf, self.freestream.v_inf
        )
        plot_flow_field(
            self.X, self.Y, self.Q,
            iteration=self.iteration, 
            residual=self.residual_history[-1] if self.residual_history else 0,
            cfl=self._get_cfl(self.iteration),
            output_dir=str(snapshot_dir),
            case_name=f"{self.config.case_name}_final",
            C_pt=C_pt
        )
        
        surface_fields = self._compute_surface_fields()
        surface_fields['TotalPressureLoss'] = C_pt
        self.vtk_writer.write(self.Q, iteration=self.iteration, additional_scalars=surface_fields)
        self.vtk_writer.finalize()
        
        # Plot residual history
        plot_residual_history(self.residual_history, str(snapshot_dir), self.config.case_name)
        
        # Print C_pt statistics
        print(f"\nEntropy Check (Total Pressure Loss C_pt):")
        print(f"  Min:  {C_pt.min():.6f}")
        print(f"  Max:  {C_pt.max():.6f}")
        print(f"  Mean: {C_pt.mean():.6f}")
        
        print(f"{'='*60}")
        print(f"Output in: {self.config.output_dir}")
        print(f"Snapshots in: {snapshot_dir}")
        
        return self.converged
    
    def _print_divergence_info(self, bounds: dict, solution_history: list, 
                                snapshot_dir: Path):
        """Print detailed info and save snapshots when divergence is detected."""
        print(f"\nDivergence Diagnostics:")
        print(f"  Pressure range: [{bounds['p_min']:.4f}, {bounds['p_max']:.4f}]")
        print(f"  U-velocity range: [{bounds['u_min']:.4f}, {bounds['u_max']:.4f}]")
        print(f"  V-velocity range: [{bounds['v_min']:.4f}, {bounds['v_max']:.4f}]")
        print(f"  Max velocity: {bounds['vel_max']:.4f} at cell {bounds['vel_max_loc']}")
        print(f"  Nu_t range: [{bounds['nu_min']:.6e}, {bounds['nu_max']:.6e}]")
        
        if solution_history:
            from ..io.plotting import plot_flow_field
            
            divergence_dir = snapshot_dir / "divergence"
            divergence_dir.mkdir(exist_ok=True)
            
            print(f"\n  Dumping last {len(solution_history)} solutions before divergence:")
            for iteration, Q, R_field in solution_history:
                Q_int = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
                C_pt = compute_total_pressure_loss(
                    Q_int, self.freestream.p_inf,
                    self.freestream.u_inf, self.freestream.v_inf
                )
                pdf_path = plot_flow_field(
                    self.X, self.Y, Q,
                    iteration=iteration,
                    residual=np.sqrt(np.mean(R_field**2)),
                    cfl=self._get_cfl(iteration),
                    output_dir=str(divergence_dir),
                    case_name=f"{self.config.case_name}_div",
                    C_pt=C_pt, residual_field=R_field
                )
                print(f"    {pdf_path}")

