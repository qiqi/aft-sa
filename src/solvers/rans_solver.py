"""
RANS Solver for 2D Incompressible Flow using Artificial Compressibility.

State vector: Q = [p, u, v, ν̃]

JAX-based implementation with GPU acceleration.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

from ..grid.metrics import MetricComputer
from ..grid.mesher import Construct2DWrapper, GridOptions
from ..grid.plot3d import read_plot3d
from ..numerics.fluxes import compute_fluxes, FluxConfig, GridMetrics as FluxGridMetrics
from ..numerics.forces import compute_aerodynamic_forces, AeroForces, compute_aerodynamic_forces_jax_pure
from ..numerics.gradients import compute_gradients, GradientMetrics
from ..numerics.viscous_fluxes import add_viscous_fluxes, compute_viscous_fluxes_tight_with_ghosts_jax
from ..numerics.smoothing import apply_residual_smoothing
from ..numerics.explicit_smoothing import apply_explicit_smoothing
from .boundary_conditions import (
    FreestreamConditions, 
    BoundaryConditions,
    initialize_state,
    apply_initial_wall_damping,
)
from ..io.plotter import PlotlyDashboard
from ..numerics.forces import compute_surface_distributions
from ..constants import NGHOST
from ..numerics.diagnostics import compute_total_pressure_loss

# JAX imports (required)
from ..physics.jax_config import jax, jnp
from ..numerics.fluxes import compute_fluxes_jax
from ..numerics.gradients import compute_gradients_jax
# viscous_fluxes: tight_with_ghosts version used
from ..numerics.explicit_smoothing import smooth_explicit_jax
from ..numerics.sa_sources import compute_sa_source_jax, compute_aft_sa_source_jax
from ..numerics.aft_sources import compute_Re_Omega, compute_gamma
from ..numerics.gradients import compute_vorticity_jax
from .time_stepping import compute_local_timestep_jax
from .boundary_conditions import make_apply_bc_jit


@dataclass
class SolverConfig:
    """Configuration for RANS solver."""
    
    mach: float = 0.15
    alpha: float = 0.0
    reynolds: float = 6e6
    chi_inf: float = 0.0001  # Initial/farfield turbulent viscosity ratio (ν̃/ν) - low for transition
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
    n_wake: int = 30
    html_animation: bool = True
    html_use_cdn: bool = True  # Use CDN for plotly.js (saves ~3MB, requires internet)
    html_compress: bool = True  # Gzip compress HTML output (typically 5-10x smaller)
    divergence_history: int = 0
    target_yplus: float = 1.0  # Target y+ for grid quality warning
    
    # AFT Transition Model Configuration
    aft_enabled: bool = True          # Enable AFT transition model (False = pure SA)
    aft_gamma_coeff: float = 2.0      # Gamma formula coefficient
    aft_re_omega_scale: float = 1000.0  # Re_Ω normalization
    aft_log_divisor: float = 50.0     # Log term divisor
    aft_sigmoid_center: float = 1.04  # Sigmoid activation center
    aft_sigmoid_slope: float = 35.0   # Sigmoid steepness
    aft_rate_scale: float = 0.2       # Maximum growth rate
    aft_blend_threshold: float = 1.0  # nuHat threshold for transition
    aft_blend_width: float = 4.0      # Blending smoothness


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
        print("RANS Solver Initialized")
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
        
        # Compute face geometry and LS weights for tight-stencil viscous fluxes
        print("  Computing face geometry for tight-stencil viscous fluxes...")
        self.face_geometry = computer.compute_face_geometry()
        self.ls_weights = computer.compute_ls_weights(self.face_geometry)
        
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
            reynolds=self.config.reynolds,
            chi_inf=getattr(self.config, 'chi_inf', 3.0)
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
        print("  Far-field BC: Characteristic (non-reflecting)")
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
        self.wall_dist_jax = jax.device_put(jnp.array(self.metrics.wall_distance))

        # Face geometry for tight-stencil viscous fluxes
        fg = self.face_geometry
        self.d_coord_i_jax = jax.device_put(jnp.array(fg.d_coord_i))
        self.e_coord_i_x_jax = jax.device_put(jnp.array(fg.e_coord_i_x))
        self.e_coord_i_y_jax = jax.device_put(jnp.array(fg.e_coord_i_y))
        self.e_ortho_i_x_jax = jax.device_put(jnp.array(fg.e_ortho_i_x))
        self.e_ortho_i_y_jax = jax.device_put(jnp.array(fg.e_ortho_i_y))
        self.d_coord_j_jax = jax.device_put(jnp.array(fg.d_coord_j))
        self.e_coord_j_x_jax = jax.device_put(jnp.array(fg.e_coord_j_x))
        self.e_coord_j_y_jax = jax.device_put(jnp.array(fg.e_coord_j_y))
        self.e_ortho_j_x_jax = jax.device_put(jnp.array(fg.e_ortho_j_x))
        self.e_ortho_j_y_jax = jax.device_put(jnp.array(fg.e_ortho_j_y))
        
        # LS weights for tight-stencil viscous fluxes
        self.ls_weights_i_jax = jax.device_put(jnp.array(self.ls_weights.weights_i))
        self.ls_weights_j_jax = jax.device_put(jnp.array(self.ls_weights.weights_j))

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
        
        Uses tight-stencil viscous flux computation for improved diagonal dominance
        and stability on non-orthogonal grids.
        """
        # Capture these in closure (they're constants for this solver instance)
        Si_x = self.Si_x_jax
        Si_y = self.Si_y_jax
        Sj_x = self.Sj_x_jax
        Sj_y = self.Sj_y_jax
        volume = self.volume_jax
        volume_inv = self.volume_inv_jax
        wall_dist = self.wall_dist_jax
        beta = self.config.beta
        k4 = self.config.jst_k4
        nu = self.mu_laminar
        nghost = NGHOST
        smoothing_epsilon = getattr(self.config, 'smoothing_epsilon', 0.2)
        smoothing_passes = getattr(self.config, 'smoothing_passes', 2)
        apply_bc = self.apply_bc_jit
        
        # Face geometry for tight-stencil viscous fluxes
        d_coord_i = self.d_coord_i_jax
        e_coord_i_x = self.e_coord_i_x_jax
        e_coord_i_y = self.e_coord_i_y_jax
        e_ortho_i_x = self.e_ortho_i_x_jax
        e_ortho_i_y = self.e_ortho_i_y_jax
        d_coord_j = self.d_coord_j_jax
        e_coord_j_x = self.e_coord_j_x_jax
        e_coord_j_y = self.e_coord_j_y_jax
        e_ortho_j_x = self.e_ortho_j_x_jax
        e_ortho_j_y = self.e_ortho_j_y_jax
        ls_weights_i = self.ls_weights_i_jax
        ls_weights_j = self.ls_weights_j_jax
        
        # AFT transition model parameters
        aft_enabled = self.config.aft_enabled
        aft_gamma_coeff = self.config.aft_gamma_coeff
        aft_re_omega_scale = self.config.aft_re_omega_scale
        aft_log_divisor = self.config.aft_log_divisor
        aft_sigmoid_center = self.config.aft_sigmoid_center
        aft_sigmoid_slope = self.config.aft_sigmoid_slope
        aft_rate_scale = self.config.aft_rate_scale
        aft_blend_threshold = self.config.aft_blend_threshold
        aft_blend_width = self.config.aft_blend_width
        
        if aft_enabled:
            @jax.jit
            def jit_rk_stage(Q, Q0, dt, alpha_rk):
                """Single RK stage with AFT-SA transition model."""
                # Apply boundary conditions
                Q = apply_bc(Q)
                
                # Convective fluxes
                R = compute_fluxes_jax(Q, Si_x, Si_y, Sj_x, Sj_y, beta, k4, nghost)
                
                # Gradients (needed for SA sources and cb2 advection)
                grad = compute_gradients_jax(Q, Si_x, Si_y, Sj_x, Sj_y, volume, nghost)
                
                # Compute turbulent viscosity from SA variable
                Q_int = Q[nghost:-nghost, nghost:-nghost, :]
                nu_tilde = jnp.maximum(Q_int[:, :, 3], 0.0)
                
                # Velocity magnitude for AFT
                vel_mag = jnp.sqrt(Q_int[:, :, 1]**2 + Q_int[:, :, 2]**2)
                
                # Turbulent viscosity (SA model)
                chi = nu_tilde / (nu + 1e-30)
                cv1 = 7.1
                chi3 = chi ** 3
                f_v1 = chi3 / (chi3 + cv1 ** 3)
                mu_t = nu_tilde * f_v1
                mu_eff = nu + mu_t
                
                # Tight-stencil viscous fluxes (momentum + SA diffusion)
                Q_with_ghosts = Q[nghost-1:-(nghost-1), nghost-1:-(nghost-1), :]
                R_visc = compute_viscous_fluxes_tight_with_ghosts_jax(
                    Q_with_ghosts, Si_x, Si_y, Sj_x, Sj_y,
                    d_coord_i, e_coord_i_x, e_coord_i_y, e_ortho_i_x, e_ortho_i_y,
                    d_coord_j, e_coord_j_x, e_coord_j_y, e_ortho_j_x, e_ortho_j_y,
                    ls_weights_i, ls_weights_j,
                    mu_eff, nu, nu_tilde
                )
                R = R.at[:, :, 1:4].add(R_visc[:, :, 1:4])
                
                # AFT-SA source terms with blending
                P, D, cb2_term = compute_aft_sa_source_jax(
                    nu_tilde, grad, wall_dist, vel_mag, nu,
                    aft_gamma_coeff, aft_re_omega_scale, aft_log_divisor,
                    aft_sigmoid_center, aft_sigmoid_slope, aft_rate_scale,
                    aft_blend_threshold, aft_blend_width
                )
                R = R.at[:, :, 3].add((P - D + cb2_term) * volume)
                
                # Explicit smoothing
                if smoothing_epsilon > 0 and smoothing_passes > 0:
                    R = smooth_explicit_jax(R, smoothing_epsilon, smoothing_passes)
                
                # Compute the update increment dQ = alpha * dt/V * R
                Q0_int = Q0[nghost:-nghost, nghost:-nghost, :]
                dQ = alpha_rk * (dt * volume_inv)[:, :, jnp.newaxis] * R
                
                # Start with normal explicit update for all variables
                Q_int_new = Q0_int + dQ
                
                # Patankar-Euler scheme for nuHat (index 3) only
                nuHat_old = Q0_int[:, :, 3]
                dNuHat = dQ[:, :, 3]
                nuHat_old_safe = jnp.maximum(nuHat_old, 1e-20)
                nuHat_patankar = nuHat_old_safe / (1.0 - dNuHat / nuHat_old_safe)
                nuHat_new = jnp.where(dNuHat < 0, nuHat_patankar, Q_int_new[:, :, 3])
                nuHat_new = jnp.maximum(nuHat_new, 1e-20)
                Q_int_new = Q_int_new.at[:, :, 3].set(nuHat_new)
                
                Q_new = Q0.at[nghost:-nghost, nghost:-nghost, :].set(Q_int_new)
                
                return Q_new, R
        else:
            @jax.jit
            def jit_rk_stage(Q, Q0, dt, alpha_rk):
                """Single RK stage - fully JIT compiled with SA turbulence (no AFT).
                
                Uses:
                - Tight-stencil viscous flux for improved diagonal dominance
                - Point-implicit treatment for SA destruction term
                - cb2 term implemented as advection with JST dissipation
                """
                # Apply boundary conditions
                Q = apply_bc(Q)
                
                # Convective fluxes
                R = compute_fluxes_jax(Q, Si_x, Si_y, Sj_x, Sj_y, beta, k4, nghost)
                
                # Gradients (needed for SA sources and cb2 advection)
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
                
                # Tight-stencil viscous fluxes (momentum + SA diffusion)
                # Uses 2-point difference along coordinate line + LS correction for non-orthogonality
                # CRITICAL: Use Q with ghost cells (after BC applied) for correct wall gradient!
                # Extract Q with one layer of ghost cells on each side
                Q_with_ghosts = Q[nghost-1:-(nghost-1), nghost-1:-(nghost-1), :]  # (NI+2, NJ+2, 4)
                R_visc = compute_viscous_fluxes_tight_with_ghosts_jax(
                    Q_with_ghosts, Si_x, Si_y, Sj_x, Sj_y,
                    d_coord_i, e_coord_i_x, e_coord_i_y, e_ortho_i_x, e_ortho_i_y,
                    d_coord_j, e_coord_j_x, e_coord_j_y, e_ortho_j_x, e_ortho_j_y,
                    ls_weights_i, ls_weights_j,
                    mu_eff, nu, nu_tilde
                )
                R = R.at[:, :, 1:4].add(R_visc[:, :, 1:4])
                
                # SA source terms: Production (P), Destruction (D), and cb2 term |∇ν̃|²
                P, D, cb2_term = compute_sa_source_jax(nu_tilde, grad, wall_dist, nu)
                # Add production, subtract destruction, and cb2 source term to residual
                # cb2_term = (cb2/sigma) * |∇ν̃|² is a "anti-diffusion" source that
                # accounts for the conservative form of the diffusion term
                R = R.at[:, :, 3].add((P - D + cb2_term) * volume)
                
                # Explicit smoothing
                if smoothing_epsilon > 0 and smoothing_passes > 0:
                    R = smooth_explicit_jax(R, smoothing_epsilon, smoothing_passes)
                
                # Compute the update increment dQ = alpha * dt/V * R
                Q0_int = Q0[nghost:-nghost, nghost:-nghost, :]
                dQ = alpha_rk * (dt * volume_inv)[:, :, jnp.newaxis] * R
                
                # Start with normal explicit update for all variables
                Q_int_new = Q0_int + dQ
                
                # Patankar-Euler scheme for nuHat (index 3) only
                # Note: Pressure can be negative in artificial compressibility, so don't apply Patankar
                # For positive dNuHat: nuHat_new = nuHat_old + dNuHat (normal explicit update)
                # For negative dNuHat: update reciprocal: nuHat_new = nuHat_old / (1 - dNuHat/nuHat_old)
                #   This ensures nuHat_new stays positive if nuHat_old > 0
                nuHat_old = Q0_int[:, :, 3]
                dNuHat = dQ[:, :, 3]
                nuHat_old_safe = jnp.maximum(nuHat_old, 1e-20)
                nuHat_patankar = nuHat_old_safe / (1.0 - dNuHat / nuHat_old_safe)
                nuHat_new = jnp.where(dNuHat < 0, nuHat_patankar, Q_int_new[:, :, 3])
                # Ensure nuHat stays positive
                nuHat_new = jnp.maximum(nuHat_new, 1e-20)
                Q_int_new = Q_int_new.at[:, :, 3].set(nuHat_new)
                
                Q_new = Q0.at[nghost:-nghost, nghost:-nghost, :].set(Q_int_new)
                
                return Q_new, R
        
        @jax.jit
        def jit_compute_dt(Q, cfl):
            """Compute local timestep - JIT compiled with turbulent viscosity correction."""
            # Compute effective viscosity at each cell for proper viscous stability
            Q_int = Q[nghost:-nghost, nghost:-nghost, :]
            nu_tilde = jnp.maximum(Q_int[:, :, 3], 0.0)
            
            # SA turbulent viscosity: mu_t = nu_tilde * fv1
            chi = nu_tilde / (nu + 1e-30)
            cv1 = 7.1
            chi3 = chi ** 3
            f_v1 = chi3 / (chi3 + cv1 ** 3)
            mu_t = nu_tilde * f_v1
            
            # Use MAX effective viscosity for conservative time step estimate
            # This is conservative but ensures stability in high-mu_t regions
            nu_eff_max = nu + jnp.max(mu_t)
            
            return compute_local_timestep_jax(
                Q, Si_x, Si_y, Sj_x, Sj_y, volume,
                beta, cfl, nghost, nu=nu_eff_max
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
        
        # Use CDN and compression by default to reduce HTML file size
        use_cdn = getattr(self.config, 'html_use_cdn', True)
        compress = getattr(self.config, 'html_compress', True)
        self.plotter = PlotlyDashboard(reynolds=self.config.reynolds, use_cdn=use_cdn, compress=compress)
        
        # Initialize JAX arrays
        self._initialize_jax()
        
        if self.config.html_animation:
            Q_int = self.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
            C_pt = compute_total_pressure_loss(
                Q_int, self.freestream.p_inf,
                self.freestream.u_inf, self.freestream.v_inf
            )
            initial_R = self._compute_residual(self.Q)
            # Compute AFT diagnostic fields if AFT is enabled
            Re_Omega, Gamma = None, None
            if self.config.aft_enabled:
                Re_Omega, Gamma = self._compute_aft_fields()
            self.plotter.store_snapshot(
                self.Q, 0, self.residual_history,
                cfl=self._get_cfl(0), C_pt=C_pt, residual_field=initial_R,
                freestream=self.freestream,
                iteration_history=self.iteration_history,
                Re_Omega=Re_Omega, Gamma=Gamma
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
    
    def _compute_aft_fields(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Re_Omega and Gamma fields for visualization.
        
        Returns
        -------
        Re_Omega : np.ndarray (NI, NJ)
            Vorticity Reynolds number = d² |ω| / ν
        Gamma : np.ndarray (NI, NJ)
            AFT shape factor
        """
        from ..physics.jax_config import jnp
        
        # Get interior values
        Q_int = self.Q_jax[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
        u = Q_int[:, :, 1]
        v = Q_int[:, :, 2]
        
        # Compute velocity magnitude (absolute)
        vel_mag = jnp.sqrt((u + self.freestream.u_inf)**2 + (v + self.freestream.v_inf)**2)
        
        # Compute gradients
        grad = compute_gradients_jax(
            self.Q_jax, 
            self.flux_grid_metrics.Si_x, 
            self.flux_grid_metrics.Si_y,
            self.flux_grid_metrics.Sj_x, 
            self.flux_grid_metrics.Sj_y,
            self.volume, NGHOST
        )
        
        # Compute vorticity magnitude
        omega_mag = compute_vorticity_jax(grad)
        
        # Get wall distance
        wall_dist = self.wall_distance
        
        # Compute AFT fields using existing functions
        nu = 1.0 / self.config.reynolds
        gamma_coeff = self.config.aft_gamma_coeff
        
        Re_Omega = compute_Re_Omega(omega_mag, wall_dist, nu)
        Gamma = compute_gamma(omega_mag, vel_mag, wall_dist, gamma_coeff)
        
        return np.array(Re_Omega), np.array(Gamma)
    
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
    
    def get_residual_rms(self) -> Tuple[float, float, float, float]:
        """Compute RMS of residual for all 4 equations.
        
        Returns
        -------
        Tuple[float, float, float, float]
            RMS residuals for (pressure, u, v, nuHat) equations.
            
        Note: This is NOT mesh-refinement invariant. Use get_residual_l1_scaled()
        for a metric that is invariant to mesh refinement and domain size.
        """
        # Compute RMS for each equation separately
        rms_p = float(jnp.sqrt(jnp.mean(self.R_jax[:, :, 0]**2)))
        rms_u = float(jnp.sqrt(jnp.mean(self.R_jax[:, :, 1]**2)))
        rms_v = float(jnp.sqrt(jnp.mean(self.R_jax[:, :, 2]**2)))
        rms_nu = float(jnp.sqrt(jnp.mean(self.R_jax[:, :, 3]**2)))
        return (rms_p, rms_u, rms_v, rms_nu)
    
    def get_residual_l1_scaled(self) -> Tuple[float, float, float, float]:
        """Compute mesh-refinement-invariant scaled L1 norm of residuals.
        
        This metric is invariant to:
        - Mesh refinement (splitting cells preserves the sum)
        - Domain expansion (adding zero-residual cells doesn't change the sum)
        - Physical unit scaling (each equation is normalized by its characteristic scale)
        
        Formula: R_reported = (1/F_scale) * Σ |r_i / scale_i|
        
        Per-cell scaling (scale_i):
        - Pressure: β (artificial compressibility parameter)
        - u, v: V_inf (freestream velocity magnitude)
        - ν̃: (ν_laminar + ν̃_i) at each cell (local scaling to prevent far-wake dominance)
        
        Global normalization (F_scale):
        - V_inf * chord (freestream volume flux through unit chord, which equals V_inf for chord=1)
        
        Returns
        -------
        Tuple[float, float, float, float]
            Scaled L1 residuals for (pressure, u, v, nuHat) equations.
        """
        # Characteristic scales
        beta = self.config.beta
        V_inf = np.sqrt(self.freestream.u_inf**2 + self.freestream.v_inf**2)
        V_inf = max(V_inf, 1e-30)  # Avoid division by zero
        nu_laminar = 1.0 / self.config.reynolds if self.config.reynolds > 0 else 1e-6
        
        # Get interior Q for local nuHat scaling
        nghost = NGHOST
        Q_int = self.Q_jax[nghost:-nghost, nghost:-nghost, :]
        nuHat = jnp.maximum(Q_int[:, :, 3], 0.0)
        
        # Local effective viscosity for nuHat scaling (prevents far-wake dominance)
        nu_eff = nu_laminar + nuHat
        nu_eff = jnp.maximum(nu_eff, 1e-30)  # Safety
        
        # Residual (this is the integral flux imbalance, NOT divided by volume)
        R = self.R_jax
        
        # Scaled absolute residuals (per-cell normalization)
        R_p_scaled = jnp.abs(R[:, :, 0]) / beta
        R_u_scaled = jnp.abs(R[:, :, 1]) / V_inf
        R_v_scaled = jnp.abs(R[:, :, 2]) / V_inf
        R_nu_scaled = jnp.abs(R[:, :, 3]) / nu_eff
        
        # L1 sums (mesh-refinement invariant: splitting cells preserves the sum)
        L1_p = float(jnp.sum(R_p_scaled))
        L1_u = float(jnp.sum(R_u_scaled))
        L1_v = float(jnp.sum(R_v_scaled))
        L1_nu = float(jnp.sum(R_nu_scaled))
        
        # Global flux scale normalization
        # For 2D airfoil with unit chord: F_scale = V_inf * chord = V_inf
        # This makes the result O(1) for a converged solution
        F_scale = V_inf
        
        return (L1_p / F_scale, L1_u / F_scale, L1_v / F_scale, L1_nu / F_scale)
    
    def sync_to_cpu(self) -> None:
        """Transfer Q from GPU to CPU (for visualization/output)."""
        self.Q = np.array(self.Q_jax)
    
    def run_steady_state(self) -> bool:
        """Run steady-state simulation to convergence."""
        print(f"\n{'='*60}")
        print("Starting Steady-State Iteration")
        print(f"{'='*60}")
        print(f"{'Iter':>8} {'Residual(L1)':>14} {'CFL':>8}  "
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
                res_all = self.get_residual_l1_scaled()  # (p, u, v, nu) tuple - mesh-invariant L1
                self.residual_history.append(res_all)
                self.iteration_history.append(self.iteration)
                
                # Use max of all residuals for convergence check
                res_max = max(res_all)
                if initial_residual is None:
                    initial_residual = res_max
            else:
                # Use previous max residual for convergence check
                res_max = max(self.residual_history[-1]) if self.residual_history else 1.0
            
            cfl = self._get_cfl(self.iteration)
            
            if need_print:
                # Transfer Q to CPU for force computation
                self.sync_to_cpu()
                forces = self.compute_forces()
                # Print max residual and individual components
                print(f"{self.iteration:>8d} {res_max:>14.6e} {cfl:>8.2f}  "
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
                        'residual': res_max,
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
                # Compute AFT diagnostic fields if AFT is enabled
                Re_Omega, Gamma = None, None
                if self.config.aft_enabled:
                    Re_Omega, Gamma = self._compute_aft_fields()
                self.plotter.store_snapshot(
                    self.Q, self.iteration, self.residual_history,
                    cfl=cfl, C_pt=C_pt, residual_field=R_field,
                    freestream=self.freestream,
                    iteration_history=self.iteration_history,
                    Re_Omega=Re_Omega, Gamma=Gamma
                )
            
            if res_max < self.config.tol:
                self.converged = True
                print(f"\n{'='*60}")
                print(f"CONVERGED at iteration {self.iteration}")
                res_final = self.residual_history[-1]
                print(f"Final residual (L1): p={res_final[0]:.2e} u={res_final[1]:.2e} v={res_final[2]:.2e} ν̃={res_final[3]:.2e}")
                print(f"{'='*60}")
                break
            
            # Check for divergence: either residual > 1000x initial, or NaN/Inf
            is_diverged = (
                self.residual_history and 
                (res_max > 1000 * initial_residual or not np.isfinite(res_max))
            )
            if is_diverged:
                print(f"\n{'='*60}")
                print(f"DIVERGED at iteration {self.iteration}")
                if np.isfinite(res_max):
                    print(f"Max residual: {res_max:.6e} (initial: {initial_residual:.6e})")
                else:
                    print(f"Max residual: {res_max} (NaN/Inf detected!)")
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
                                # Note: AFT fields not computed for buffered divergence snapshots
                            )
                            n_history_dumps += 1
                    
                    # Also capture current state at divergence detection
                    R_field = np.array(self.R_jax)
                    Q_int = self.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
                    C_pt = compute_total_pressure_loss(
                        Q_int, self.freestream.p_inf,
                        self.freestream.u_inf, self.freestream.v_inf
                    )
                    # Compute AFT diagnostic fields if AFT is enabled
                    Re_Omega, Gamma = None, None
                    if self.config.aft_enabled:
                        Re_Omega, Gamma = self._compute_aft_fields()
                    self.plotter.store_snapshot(
                        self.Q, self.iteration, self.residual_history,
                        cfl=cfl, C_pt=C_pt, residual_field=R_field,
                        freestream=self.freestream,
                        iteration_history=self.iteration_history,
                        is_divergence_dump=True,
                        Re_Omega=Re_Omega, Gamma=Gamma
                    )
                    
                    total_dumps = n_history_dumps + 1
                    print(f"  Divergence snapshots captured: {total_dumps} ({n_history_dumps} from history + 1 current)")
                break
        
        else:
            print(f"\n{'='*60}")
            print(f"Maximum iterations ({self.config.max_iter}) reached")
            if self.residual_history:
                res_final = self.residual_history[-1]
                print(f"Final residual: p={res_final[0]:.2e} u={res_final[1]:.2e} v={res_final[2]:.2e} ν̃={res_final[3]:.2e}")
            print(f"{'='*60}")
        
        # Final sync to CPU
        self.sync_to_cpu()
        
        if self.config.html_animation and self.plotter.num_snapshots > 0:
            html_path = Path(self.config.output_dir) / f"{self.config.case_name}_animation.html"
            n_wake = getattr(self.config, 'n_wake', 0)
            mu_laminar = 1.0 / self.config.reynolds if self.config.reynolds > 0 else 0.0
            target_yplus = getattr(self.config, 'target_yplus', 1.0)
            self.plotter.save_html(
                str(html_path), self.metrics,
                wall_distance=self.metrics.wall_distance,
                X=self.X, Y=self.Y,
                n_wake=n_wake,
                mu_laminar=mu_laminar,
                target_yplus=target_yplus
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
    
    def compute_forces(self, use_jax: bool = True) -> AeroForces:
        """Compute aerodynamic force coefficients.
        
        Parameters
        ----------
        use_jax : bool
            If True (default), compute directly on GPU using JAX arrays.
            If False, use the legacy NumPy-based computation.
        """
        V_inf = np.sqrt(self.freestream.u_inf**2 + self.freestream.v_inf**2)
        mu_laminar = 1.0 / self.config.reynolds if self.config.reynolds > 0 else 0.0
        
        n_wake = getattr(self.config, 'n_wake', 0)
        if n_wake == 0 and hasattr(self, 'bc'):
            n_wake = getattr(self.bc, 'n_wake_points', 0)
        
        if use_jax and hasattr(self, 'Q_jax'):
            # Fast path: compute directly on GPU with JAX arrays
            return self._compute_forces_jax(mu_laminar, V_inf, n_wake)
        else:
            # Legacy path: uses NumPy arrays
            return self._compute_forces_numpy(mu_laminar, V_inf, n_wake)
    
    def _compute_forces_jax(self, mu_laminar: float, V_inf: float, n_wake: int) -> AeroForces:
        """Compute forces using JAX arrays directly (no GPU↔CPU transfers)."""
        nghost = NGHOST
        
        # Compute turbulent viscosity on GPU
        nuHat = jnp.maximum(self.Q_jax[nghost:-nghost, nghost:-nghost, 3], 0.0)
        chi = nuHat / (mu_laminar + 1e-30)
        cv1 = 7.1
        chi3 = chi ** 3
        fv1 = chi3 / (chi3 + cv1 ** 3)
        mu_turb = nuHat * fv1
        mu_eff = mu_laminar + mu_turb
        
        # Pre-compute constants
        alpha_rad = np.deg2rad(self.config.alpha)
        sin_alpha = np.sin(alpha_rad)
        cos_alpha = np.cos(alpha_rad)
        q_inf = 0.5 * V_inf**2  # rho=1
        ref_area = 1.0  # chord * span (2D: span=1)
        q_inf_ref_area = q_inf * ref_area
        
        # Call pure JAX computation
        CL, CD, CD_p, CD_f = compute_aerodynamic_forces_jax_pure(
            self.Q_jax, self.Sj_x_jax, self.Sj_y_jax, self.volume_jax, mu_eff,
            sin_alpha, cos_alpha, q_inf_ref_area,
            n_wake, nghost
        )
        
        # Transfer only scalars to CPU
        return AeroForces(
            CL=float(CL), CD=float(CD), 
            CD_p=float(CD_p), CD_f=float(CD_f),
            CL_p=float(CL) - 0.0,  # Placeholder (full breakdown needs more work)
            CL_f=0.0,
            Fx=0.0, Fy=0.0  # Not computed in fast path
        )
    
    def _compute_forces_numpy(self, mu_laminar: float, V_inf: float, n_wake: int) -> AeroForces:
        """Compute forces using NumPy arrays (legacy, full breakdown)."""
        # Compute turbulent viscosity from SA variable
        Q = np.array(self.Q_jax) if hasattr(self, 'Q_jax') else self.Q
        nghost = NGHOST
        nuHat = np.maximum(Q[nghost:-nghost, nghost:-nghost, 3], 0.0)
        chi = nuHat / (mu_laminar + 1e-30)
        cv1 = 7.1
        chi3 = chi ** 3
        fv1 = chi3 / (chi3 + cv1 ** 3)
        mu_turb = nuHat * fv1
        
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
            f.write("# Scaled L1 residuals (mesh-refinement invariant)\n")
            f.write("# Scaling: p/β, u/V_inf, v/V_inf, nuHat/(nu_lam+nuHat_local)\n")
            f.write("# Normalization: 1/(V_inf * chord)\n")
            f.write("# Iteration  Res_p  Res_u  Res_v  Res_nuHat\n")
            for i, res in enumerate(self.residual_history):
                if isinstance(res, (tuple, list, np.ndarray)):
                    f.write(f"{i+1:8d}  {res[0]:.10e}  {res[1]:.10e}  {res[2]:.10e}  {res[3]:.10e}\n")
                else:
                    # Legacy: single residual
                    f.write(f"{i+1:8d}  {res:.10e}\n")
        
        print(f"Residual history saved to: {filename}")

