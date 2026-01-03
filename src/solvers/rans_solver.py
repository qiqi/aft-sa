"""
RANS Solver for 2D Incompressible Flow using Artificial Compressibility.

State vector: Q = [p, u, v, ν̃]

JAX-based implementation with GPU acceleration.
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
from ..numerics.explicit_smoothing import apply_explicit_smoothing
from .boundary_conditions import (
    FreestreamConditions, 
    BoundaryConditions,
    initialize_state,
    apply_initial_wall_damping,
)
from .time_stepping import compute_local_timestep, TimeStepConfig
from ..io.plotter import PlotlyDashboard
from ..numerics.forces import compute_surface_distributions
from ..constants import NGHOST
from ..numerics.diagnostics import (
    compute_total_pressure_loss, 
    compute_solution_bounds,
    compute_residual_statistics
)

# JAX imports (required)
from ..physics.jax_config import jax, jnp
from ..numerics.fluxes import compute_fluxes_jax
from ..numerics.gradients import compute_gradients_jax
from ..numerics.viscous_fluxes import compute_viscous_fluxes_jax
from ..numerics.explicit_smoothing import smooth_explicit_jax
from .time_stepping import compute_local_timestep_jax
from .boundary_conditions import apply_bc_jax, make_apply_bc_jit


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
    print_freq: int = 50
    output_dir: str = "output"
    case_name: str = "solution"
    wall_damping_length: float = 0.1
    jst_k4: float = 0.04
    sponge_thickness: int = 15  # Sponge layer thickness for farfield stabilization
    irs_epsilon: float = 0.0
    # Explicit smoothing (preferred over IRS for GPU)
    smoothing_type: str = "explicit"  # "explicit", "implicit", or "none"
    smoothing_epsilon: float = 0.2
    smoothing_passes: int = 2
    n_wake: int = 30
    html_animation: bool = True
    divergence_history: int = 0


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
        self.iteration_history = []  # Track which iteration each residual corresponds to
        self.converged = False
        
        # Rolling buffer for divergence history (recent snapshots before divergence)
        from collections import deque
        div_history_size = self.config.divergence_history if self.config.divergence_history > 0 else 0
        self._divergence_buffer = deque(maxlen=div_history_size) if div_history_size > 0 else None
        
        self._load_grid(grid_file)
        self._compute_metrics()
        self._initialize_state()
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
        n_wake = getattr(self.config, 'n_wake', 0)
        computer = MetricComputer(self.X, self.Y, wall_j=0, n_wake=n_wake)
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
            alpha_deg=self.config.alpha,
            reynolds=self.config.reynolds
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
    
    def _initialize_jax(self):
        """Initialize JAX arrays for computation."""
        print(f"  JAX backend initialized on: {jax.devices()[0]}")
        
        # Transfer arrays to GPU
        self.Q_jax = jax.device_put(jnp.array(self.Q))
        self.Si_x_jax = jax.device_put(jnp.array(self.metrics.Si_x))
        self.Si_y_jax = jax.device_put(jnp.array(self.metrics.Si_y))
        self.Sj_x_jax = jax.device_put(jnp.array(self.metrics.Sj_x))
        self.Sj_y_jax = jax.device_put(jnp.array(self.metrics.Sj_y))
        self.volume_jax = jax.device_put(jnp.array(self.metrics.volume))
        
        # Farfield normals for BC
        Sj_x_ff = self.metrics.Sj_x[:, -1]
        Sj_y_ff = self.metrics.Sj_y[:, -1]
        Sj_mag = np.sqrt(Sj_x_ff**2 + Sj_y_ff**2) + 1e-12
        self.nx_ff_jax = jax.device_put(jnp.array(Sj_x_ff / Sj_mag))
        self.ny_ff_jax = jax.device_put(jnp.array(Sj_y_ff / Sj_mag))
        
        # Create JIT-compiled BC function with indices baked in
        self.apply_bc_jit = make_apply_bc_jit(
            NI=self.NI, NJ=self.NJ,
            n_wake_points=self.config.n_wake,
            nx=self.nx_ff_jax, ny=self.ny_ff_jax,
            freestream=self.freestream,
            nghost=NGHOST
        )
        
        # Pre-compute inverse volume for RK update on GPU
        self.volume_inv_jax = jax.device_put(jnp.array(1.0 / self.metrics.volume))
        
        # Pre-compute laminar viscosity
        self.mu_laminar = 1.0 / self.config.reynolds if self.config.reynolds > 0 else 0.0
        self.mu_eff_jax = jax.device_put(jnp.full((self.NI, self.NJ), self.mu_laminar))
        
        # Create JIT-compiled step function (key for performance!)
        self._create_jit_step_function()
        
        # Warmup JIT compilation
        print("  Warming up JIT compilation...")
        self._warmup_jax()
        print("  JIT compilation complete.")
    
    def _create_jit_step_function(self):
        """Create JIT-compiled step function with all grid data baked in.
        
        This is the KEY performance optimization: instead of calling many 
        separate JIT functions from Python, we compile a single function
        that does all the work for one RK stage.
        """
        # Capture these in closure (they're constants for this solver instance)
        Si_x = self.Si_x_jax
        Si_y = self.Si_y_jax
        Sj_x = self.Sj_x_jax
        Sj_y = self.Sj_y_jax
        volume = self.volume_jax
        volume_inv = self.volume_inv_jax
        beta = self.config.beta
        k4 = self.config.jst_k4
        nu = self.mu_laminar
        nghost = NGHOST
        smoothing_epsilon = getattr(self.config, 'smoothing_epsilon', 0.2)
        smoothing_passes = getattr(self.config, 'smoothing_passes', 2)
        apply_bc = self.apply_bc_jit
        
        @jax.jit
        def jit_rk_stage(Q, Q0, dt, alpha_rk):
            """Single RK stage - fully JIT compiled."""
            # Apply boundary conditions
            Q = apply_bc(Q)
            
            # Convective fluxes
            R = compute_fluxes_jax(Q, Si_x, Si_y, Sj_x, Sj_y, beta, k4, nghost)
            
            # Gradients
            grad = compute_gradients_jax(Q, Si_x, Si_y, Sj_x, Sj_y, volume, nghost)
            
            # Compute turbulent viscosity from SA variable
            Q_int = Q[nghost:-nghost, nghost:-nghost, :]
            nu_tilde = jnp.maximum(Q_int[:, :, 3], 0.0)
            
            # Turbulent viscosity (SA model)
            chi = nu_tilde / (nu + 1e-30)
            cv1 = 7.1
            chi3 = chi ** 3
            f_v1 = chi3 / (chi3 + cv1 ** 3)
            mu_t = nu_tilde * f_v1
            mu_eff = nu + mu_t
            
            # Viscous fluxes
            R_visc = compute_viscous_fluxes_jax(grad, Si_x, Si_y, Sj_x, Sj_y, mu_eff)
            R = R + R_visc
            
            # Explicit smoothing
            if smoothing_epsilon > 0 and smoothing_passes > 0:
                R = smooth_explicit_jax(R, smoothing_epsilon, smoothing_passes)
            
            # RK update from Q0
            Q_int_new = Q0[nghost:-nghost, nghost:-nghost, :] + \
                alpha_rk * (dt * volume_inv)[:, :, jnp.newaxis] * R
            Q_new = Q0.at[nghost:-nghost, nghost:-nghost, :].set(Q_int_new)
            
            return Q_new, R
        
        @jax.jit
        def jit_compute_dt(Q, cfl):
            """Compute local timestep - JIT compiled."""
            return compute_local_timestep_jax(
                Q, Si_x, Si_y, Sj_x, Sj_y, volume,
                beta, cfl, nghost, nu=nu
            )
        
        self._jit_rk_stage = jit_rk_stage
        self._jit_compute_dt = jit_compute_dt
    
    def _warmup_jax(self):
        """Warm up JAX JIT compilation with dummy iterations."""
        cfl = self.config.cfl_start
        
        # Warm up the new JIT-compiled functions
        Q_test = self.Q_jax
        dt = self._jit_compute_dt(Q_test, cfl)
        
        for alpha in [0.25, 0.5]:
            Q_test, _ = self._jit_rk_stage(Q_test, Q_test, dt, alpha)
        
        jax.block_until_ready(Q_test)
    
    def _initialize_output(self):
        """Initialize HTML animation for output."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.plotter = PlotlyDashboard(reynolds=self.config.reynolds)
        
        # Initialize JAX arrays
        self._initialize_jax()
        
        if self.config.html_animation:
            Q_int = self.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
            C_pt = compute_total_pressure_loss(
                Q_int, self.freestream.p_inf,
                self.freestream.u_inf, self.freestream.v_inf
            )
            initial_R = self._compute_residual(self.Q)
            self.plotter.store_snapshot(
                self.Q, 0, self.residual_history,
                cfl=self._get_cfl(0), C_pt=C_pt, residual_field=initial_R,
                freestream=self.freestream,
                iteration_history=self.iteration_history
            )
        
        print(f"  Output directory: {output_path}")
    
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
    
    def _apply_smoothing(self, R: np.ndarray) -> np.ndarray:
        """Apply residual smoothing based on configuration."""
        smoothing_type = getattr(self.config, 'smoothing_type', 'none')
        
        if smoothing_type == "explicit":
            epsilon = getattr(self.config, 'smoothing_epsilon', 0.2)
            n_passes = getattr(self.config, 'smoothing_passes', 2)
            if epsilon > 0.0 and n_passes > 0:
                return apply_explicit_smoothing(R, epsilon, n_passes)
        elif smoothing_type == "implicit":
            epsilon = self.config.irs_epsilon
            if epsilon > 0.0:
                apply_residual_smoothing(R, epsilon)
        # else: no smoothing
        
        return R
    
    def step(self) -> None:
        """Perform one iteration of the solver.
        
        Fully GPU-accelerated using JIT-compiled RK stages.
        No CPU transfers - state stays on GPU.
        """
        cfl = self._get_cfl(self.iteration)
        
        # Compute timestep using JIT function
        dt = self._jit_compute_dt(self.Q_jax, cfl)
        
        # RK5 coefficients
        alphas = [0.25, 0.166666667, 0.375, 0.5, 1.0]
        
        # Store initial state (on GPU)
        Q0 = self.Q_jax
        Q = self.Q_jax
        
        # RK stages - all using single JIT-compiled function per stage
        for alpha in alphas:
            Q, R = self._jit_rk_stage(Q, Q0, dt, alpha)
        
        # Final BC on GPU
        Q = self.apply_bc_jit(Q)
        
        # Store results on GPU (no CPU transfer!)
        self.Q_jax = Q
        self.R_jax = R  # Last RK residual (for visualization)
        
        self.iteration += 1
    
    def get_residual_rms(self) -> float:
        """Compute RMS of residual on final state (GPU computation, transfers scalar only)."""
        # Use the last computed residual from step() for efficiency
        # This is the residual from the final RK stage
        R_rho = self.R_jax[:, :, 0]
        rms = jnp.sqrt(jnp.mean(R_rho**2))
        return float(rms)
    
    def sync_to_cpu(self) -> None:
        """Transfer Q from GPU to CPU (for visualization/output)."""
        self.Q = np.array(self.Q_jax)
    
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
            # Step stays entirely on GPU
            self.step()
            
            # Check if we need CPU data this iteration
            need_print = (self.iteration % self.config.print_freq == 0) or (self.iteration == 1)
            need_snapshot = self.config.html_animation and (self.iteration % self.config.diagnostic_freq == 0)
            need_residual = need_print or need_snapshot or (initial_residual is None)
            
            # Only transfer residual when needed
            if need_residual:
                res_rms = self.get_residual_rms()
                self.residual_history.append(res_rms)
                self.iteration_history.append(self.iteration)
                
                if initial_residual is None:
                    initial_residual = res_rms
            else:
                # Use previous value for convergence check (approximate)
                res_rms = self.residual_history[-1] if self.residual_history else 1.0
            
            cfl = self._get_cfl(self.iteration)
            
            if need_print:
                # Transfer Q to CPU for force computation
                self.sync_to_cpu()
                forces = self.compute_forces()
                print(f"{self.iteration:>8d} {res_rms:>14.6e} {cfl:>8.2f}  "
                      f"CL={forces.CL:>8.4f} CD={forces.CD:>8.5f} "
                      f"(p:{forces.CD_p:>7.5f} f:{forces.CD_f:>7.5f})")
                
                # Store state in divergence buffer (rolling history)
                if self._divergence_buffer is not None:
                    R_field_copy = np.array(self.R_jax)
                    Q_int = self.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
                    C_pt = compute_total_pressure_loss(
                        Q_int, self.freestream.p_inf,
                        self.freestream.u_inf, self.freestream.v_inf
                    )
                    self._divergence_buffer.append({
                        'Q': self.Q.copy(),
                        'iteration': self.iteration,
                        'residual': res_rms,
                        'cfl': cfl,
                        'C_pt': C_pt,
                        'residual_field': R_field_copy,
                    })
            
            if need_snapshot:
                if not need_print:  # Only sync if not already done
                    self.sync_to_cpu()
                R_field = np.array(self.R_jax)
                Q_int = self.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
                C_pt = compute_total_pressure_loss(
                    Q_int, self.freestream.p_inf,
                    self.freestream.u_inf, self.freestream.v_inf
                )
                self.plotter.store_snapshot(
                    self.Q, self.iteration, self.residual_history,
                    cfl=cfl, C_pt=C_pt, residual_field=R_field,
                    freestream=self.freestream,
                    iteration_history=self.iteration_history
                )
            
            if res_rms < self.config.tol:
                self.converged = True
                print(f"\n{'='*60}")
                print(f"CONVERGED at iteration {self.iteration}")
                print(f"Final residual: {res_rms:.6e}")
                print(f"{'='*60}")
                break
            
            if self.residual_history and res_rms > 1000 * initial_residual:
                print(f"\n{'='*60}")
                print(f"DIVERGED at iteration {self.iteration}")
                print(f"Residual: {res_rms:.6e} (initial: {initial_residual:.6e})")
                print(f"{'='*60}")
                
                # Capture divergence dump(s) for HTML visualization
                if self.config.html_animation:
                    self.sync_to_cpu()
                    
                    # First, dump all snapshots from the divergence buffer (history before divergence)
                    n_history_dumps = 0
                    if self._divergence_buffer:
                        for buf_snap in self._divergence_buffer:
                            self.plotter.store_snapshot(
                                buf_snap['Q'], buf_snap['iteration'], self.residual_history,
                                cfl=buf_snap['cfl'], C_pt=buf_snap['C_pt'], 
                                residual_field=buf_snap['residual_field'],
                                freestream=self.freestream,
                                iteration_history=self.iteration_history,
                                is_divergence_dump=True
                            )
                            n_history_dumps += 1
                    
                    # Also capture current state at divergence detection
                    R_field = np.array(self.R_jax)
                    Q_int = self.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
                    C_pt = compute_total_pressure_loss(
                        Q_int, self.freestream.p_inf,
                        self.freestream.u_inf, self.freestream.v_inf
                    )
                    self.plotter.store_snapshot(
                        self.Q, self.iteration, self.residual_history,
                        cfl=cfl, C_pt=C_pt, residual_field=R_field,
                        freestream=self.freestream,
                        iteration_history=self.iteration_history,
                        is_divergence_dump=True
                    )
                    
                    total_dumps = n_history_dumps + 1
                    print(f"  Divergence snapshots captured: {total_dumps} ({n_history_dumps} from history + 1 current)")
                break
        
        else:
            print(f"\n{'='*60}")
            print(f"Maximum iterations ({self.config.max_iter}) reached")
            print(f"Final residual: {self.residual_history[-1] if self.residual_history else 'N/A':.6e}")
            print(f"{'='*60}")
        
        # Final sync to CPU
        self.sync_to_cpu()
        
        if self.config.html_animation and self.plotter.num_snapshots > 0:
            html_path = Path(self.config.output_dir) / f"{self.config.case_name}_animation.html"
            n_wake = getattr(self.config, 'n_wake', 0)
            self.plotter.save_html(
                str(html_path), self.metrics,
                wall_distance=self.metrics.wall_distance,
                X=self.X, Y=self.Y,
                n_wake=n_wake,
                mu_laminar=self.freestream.nu
            )
        
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

