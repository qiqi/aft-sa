"""
RANS Solver for 2D Incompressible Flow using Artificial Compressibility.

State vector: Q = [p, u, v, ν̃]

JAX-based implementation with GPU acceleration.
"""

import numpy as np
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger
from collections import deque

from ..grid.metrics import MetricComputer
from ..grid.mesher import Construct2DWrapper, GridOptions
from ..grid.plot3d import read_plot3d
from ..numerics.fluxes import compute_fluxes, FluxConfig, GridMetrics as FluxGridMetrics
from ..numerics.forces import compute_aerodynamic_forces, AeroForces, compute_aerodynamic_forces_jax_pure
from ..numerics.gradients import compute_gradients, GradientMetrics
from ..numerics.viscous_fluxes import add_viscous_fluxes, compute_viscous_fluxes_tight_with_ghosts_jax
from ..numerics.smoothing import apply_residual_smoothing
from .boundary_conditions import (
    FreestreamConditions, 
    BoundaryConditions,
    initialize_state,
    apply_initial_wall_damping,
)
from ..io.dashboard import PlotlyDashboard
from ..numerics.forces import compute_surface_distributions
from ..constants import NGHOST
from ..numerics.diagnostics import compute_total_pressure_loss
# JAX imports
from ..physics.jax_config import jax, jnp
from ..numerics.fluxes import compute_fluxes_jax
from ..numerics.gradients import compute_gradients_jax
from ..numerics.explicit_smoothing import smooth_explicit_jax
from ..numerics.sa_sources import compute_aft_sa_source_jax, compute_turbulent_fraction
from ..numerics.aft_sources import compute_Re_Omega, compute_gamma
from ..numerics.gradients import compute_vorticity_jax
from .time_stepping import compute_local_timestep_jax
from .boundary_conditions import make_apply_bc_jit
from ..numerics.preconditioner import BlockJacobiPreconditioner
from ..numerics.dissipation import compute_sponge_sigma_jax
from ..numerics.updates import apply_patankar_update
from ..numerics.gmres import gmres_solve
from .params import PhysicsParams



@dataclass
class SolverConfig:
    """Configuration for RANS solver."""
    
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
    html_compress: bool = False  # Gzip compress HTML output (typically 5-10x smaller)
    divergence_history: int = 1
    target_yplus: float = 1.0  # Target y+ for grid quality warning
    
    # Solver mode: "rk5" (explicit RK5), "rk5_precond" (RK5 with block-Jacobi),
    #              "newton" (Newton-GMRES with block-Jacobi)
    solver_mode: str = "rk5"
    
    # GMRES settings (only used for newton mode)
    gmres_restart: int = 20     # GMRES(m) restart parameter
    gmres_maxiter: int = 100    # Maximum GMRES iterations (across restarts)
    gmres_tol: float = 1e-3     # Relative tolerance for GMRES
    newton_relaxation: float = 1.0  # Under-relaxation factor (0 < omega <= 1)
    
    # AFT Transition Model Configuration
    aft_gamma_coeff: float = 2.0      # Gamma formula coefficient
    aft_re_omega_scale: float = 1000.0  # Re_Ω normalization
    aft_log_divisor: float = 50.0     # Log term divisor
    aft_sigmoid_center: float = 1.04  # Sigmoid activation center
    aft_sigmoid_slope: float = 35.0   # Sigmoid steepness
    aft_rate_scale: float = 0.2       # Maximum growth rate
    aft_blend_threshold: float = 1.0  # nuHat threshold for transition
    aft_blend_width: float = 4.0      # Blending smoothness
    
    # Jacobian Control


class RANSSolver:
    """
    Main RANS Solver for 2D incompressible flow around airfoils.
    """
    
    def __init__(self, 
                 grid_file: Optional[str] = None, 
                 config: Optional[Union[SolverConfig, Dict]] = None,
                 grid_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
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
        self._last_gmres_info = None  # GMRES stats for Newton mode printing
        
        
        # Rolling buffer for divergence history (recent snapshots before divergence)
        div_history_size = self.config.divergence_history if self.config.divergence_history > 0 else 0
        self._divergence_buffer = deque(maxlen=div_history_size) if div_history_size > 0 else None
        
        if grid_data is not None:
            self.X, self.Y = grid_data
            self.NI = self.X.shape[0] - 1
            self.NJ = self.X.shape[1] - 1
            logger.info(f"Using provided grid: {self.NI} x {self.NJ} cells")
        elif grid_file is not None:
            self._load_grid(grid_file)
        else:
            raise ValueError("Must provide either grid_file or grid_data to RANSSolver")


        self._compute_metrics()
        self._initialize_state()
        self._initialize_output()
        
        logger.info(f"\n{'='*60}")
        logger.info("RANS Solver Initialized")
        logger.info(f"{'='*60}")
        logger.info(f"Grid size: {self.NI} x {self.NJ} cells")
        logger.info(f"Alpha: {self.config.alpha}°")
        logger.info(f"Reynolds: {self.config.reynolds:.2e}")
        logger.info(f"Target CFL: {self.config.cfl_target}")
        logger.info(f"Max iterations: {self.config.max_iter}")
        logger.info(f"Convergence tolerance: {self.config.tol:.2e}")
        logger.info(f"{'='*60}\n")
    
    def _load_grid(self, grid_file: str):
        """Load grid from file or generate using Construct2D."""
        grid_path = Path(grid_file)
        
        if not grid_path.exists():
            raise FileNotFoundError(f"Grid file not found: {grid_path}")
        
        suffix = grid_path.suffix.lower()
        
        if suffix in ['.p3d', '.x', '.xyz']:
            logger.info(f"Loading grid from: {grid_path}")
            self.X, self.Y = read_plot3d(str(grid_path))
            
        elif suffix == '.dat':
            logger.info(f"Generating grid from airfoil: {grid_path}")
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
        
        logger.info(f"Grid loaded: {self.X.shape[0]} x {self.X.shape[1]} nodes")
        logger.info(f"            {self.NI} x {self.NJ} cells")
    
    def _compute_metrics(self):
        """Compute FVM grid metrics."""
        logger.info("Computing grid metrics...")
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
        logger.info("  Computing face geometry for tight-stencil viscous fluxes...")
        self.face_geometry = computer.compute_face_geometry()
        self.ls_weights = computer.compute_ls_weights(self.face_geometry)
        
        gcl = computer.validate_gcl()
        logger.info(f"  {gcl}")
        
        if not gcl.passed:
            logger.warning("  WARNING: GCL validation failed. Results may be inaccurate.")
    
    def _build_params(self) -> PhysicsParams:
        """Construct PhysicsParams from current configuration."""
        alpha_rad = float(np.radians(self.config.alpha))
        reynolds = float(self.config.reynolds)
        mu_laminar = 1.0 / reynolds if reynolds > 0 else 0.0
        
        # Freestream
        u_inf = float(np.cos(alpha_rad))
        v_inf = float(np.sin(alpha_rad))
        nu_t_inf = float(getattr(self.config, 'chi_inf', 3.0) * mu_laminar)
        
        return PhysicsParams(
            p_inf=0.0,
            u_inf=u_inf,
            v_inf=v_inf,
            nu_t_inf=nu_t_inf,
            beta=float(self.config.beta),
            k4=float(self.config.jst_k4),
            mu_laminar=mu_laminar,
            aft_gamma_coeff=float(self.config.aft_gamma_coeff),
            aft_re_omega_scale=float(self.config.aft_re_omega_scale),
            aft_log_divisor=float(self.config.aft_log_divisor),
            aft_sigmoid_center=float(self.config.aft_sigmoid_center),
            aft_sigmoid_slope=float(self.config.aft_sigmoid_slope),
            aft_rate_scale=float(self.config.aft_rate_scale),
            aft_blend_threshold=float(self.config.aft_blend_threshold),
            aft_blend_width=float(self.config.aft_blend_width)
        )

    def _initialize_state(self):
        """Initialize state vector with freestream and wall damping."""
        logger.info("Initializing flow state...")
        
        self.freestream = FreestreamConditions.from_alpha(
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
        
        logger.info(f"  Freestream: u={self.freestream.u_inf:.4f}, v={self.freestream.v_inf:.4f}")
        logger.info("  Far-field BC: Dirichlet (with sponge layer stabilization)")
        logger.info(f"  Wall damping applied (L={self.config.wall_damping_length})")
    
    def _initialize_jax(self):
        """Initialize JAX arrays for computation."""
        logger.info(f"  JAX backend initialized on: {jax.devices()[0]}")

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
        
        # Pre-compute sponge layer coefficient field
        # Default thickness 15 if not specified
        sponge_thickness = getattr(self.config, 'sponge_thickness', 15)
        sigma = compute_sponge_sigma_jax(self.NI, self.NJ, sponge_thickness)
        self.sigma_jax = jax.device_put(sigma)
        
        # Create JIT-compiled functions (key for performance!)
        self._create_jit_functions()
        
        # Warmup JIT compilation
        logger.info("  Warming up JIT compilation...")
        self._warmup_jax()
        logger.info("  JIT compilation complete.")
    
    def _create_jit_functions(self):
        """Create JIT-compiled functions with all grid data baked in.
        
        This captures ALL constants (metrics, physics parameters, etc.) to 
        create optimized JAX executables for both RK stages and Newton steps.
        """
        # --- Capture Constants (Grid & Metrics) ---
        nghost = NGHOST
        NI, NJ = self.NI, self.NJ
        Si_x, Si_y = self.Si_x_jax, self.Si_y_jax
        Sj_x, Sj_y = self.Sj_x_jax, self.Sj_y_jax
        volume, volume_inv = self.volume_jax, self.volume_inv_jax
        wall_dist = self.wall_dist_jax
        apply_bc = self.apply_bc_jit
        sigma = self.sigma_jax
        
        # Tight-stencil viscous metrics
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
        
        # Newton-GMRES specific constants
        gmres_restart = getattr(self.config, 'gmres_restart', 20)
        gmres_maxiter = getattr(self.config, 'gmres_maxiter', 100)
        smoothing_epsilon = getattr(self.config, 'smoothing_epsilon', 0.2)
        smoothing_passes = getattr(self.config, 'smoothing_passes', 2)

        # Preconditioner Graph Coloring (Static)
        I_idx, J_idx = jnp.meshgrid(jnp.arange(NI), jnp.arange(NJ), indexing='ij')
        color_masks = jnp.stack([((I_idx % 2 == ci) & (J_idx % 2 == cj)).astype(jnp.float64) 
                                 for ci in range(2) for cj in range(2)]) # (4, NI, NJ)

        # --- Common Physical Residual Logic ---
        def compute_R_physical(Q, params: PhysicsParams):
            """Core physics calculation (Convection + Diffusion + Sources)."""
            # Extract dynamic parameters
            beta = params.beta
            k4 = params.k4
            nu = params.mu_laminar
            
            # 1. Convective fluxes
            R = compute_fluxes_jax(Q, Si_x, Si_y, Sj_x, Sj_y, beta, k4, nghost, sigma=sigma)
            
            # 2. Gradients and Effective Viscosity
            grad = compute_gradients_jax(Q, Si_x, Si_y, Sj_x, Sj_y, volume, nghost)
            Q_int = Q[nghost:-nghost, nghost:-nghost, :]
            nu_tilde = jnp.maximum(Q_int[:, :, 3], 0.0)
            
            chi = nu_tilde / (nu + 1e-30)
            f_v1 = chi**3 / (chi**3 + 7.1**3)
            mu_t = nu_tilde * f_v1
            mu_eff = nu + mu_t
            
            # 3. Viscous Fluxes
            Q_with_ghosts = Q[nghost-1:-(nghost-1), nghost-1:-(nghost-1), :]
            R_visc = compute_viscous_fluxes_tight_with_ghosts_jax(
                Q_with_ghosts, Si_x, Si_y, Sj_x, Sj_y,
                d_coord_i, e_coord_i_x, e_coord_i_y, e_ortho_i_x, e_ortho_i_y,
                d_coord_j, e_coord_j_x, e_coord_j_y, e_ortho_j_x, e_ortho_j_y,
                ls_weights_i, ls_weights_j,
                mu_eff, nu, nu_tilde
            )
            R = R.at[:, :, 1:4].add(R_visc[:, :, 1:4])
            
            # 4. AFT-SA Sources
            vel_mag = jnp.sqrt(Q_int[:, :, 1]**2 + Q_int[:, :, 2]**2)
            P, D, cb2_term = compute_aft_sa_source_jax(
                nu_tilde, grad, wall_dist, vel_mag, nu,
                params.aft_gamma_coeff, params.aft_re_omega_scale, params.aft_log_divisor,
                params.aft_sigmoid_center, params.aft_sigmoid_slope, params.aft_rate_scale,
                params.aft_blend_threshold, params.aft_blend_width
            )
            R = R.at[:, :, 3].add((P - D + cb2_term) * volume)
            return R

        # --- JIT-compiles residual function (for preconditioner and diagnostics) ---
        @jax.jit
        def jit_residual(Q, params: PhysicsParams):
            Q = apply_bc(Q, params)
            return compute_R_physical(Q, params)
        
        self._jit_residual = jit_residual

        # --- JIT-compiled RK Stage ---
        @jax.jit
        def jit_rk_stage(Q, Q0, dt, alpha_rk, params: PhysicsParams):
            """Single RK stage with optional residual smoothing."""
            Q = apply_bc(Q, params)
            R = compute_R_physical(Q, params)
            
            # Explicit smoothing (Optional)
            if smoothing_epsilon > 0 and smoothing_passes > 0:
                R = smooth_explicit_jax(R, smoothing_epsilon, smoothing_passes)
            
            # Update increment
            Q0_int = Q0[nghost:-nghost, nghost:-nghost, :]
            dt_factors = jnp.array([1.0, 1.0, 1.0, 0.5])  # Half CFL for nuHat
            dQ = alpha_rk * (dt * volume_inv)[:, :, jnp.newaxis] * dt_factors * R
            
            # Update with Patankar for nuHat
            Q_int_new = apply_patankar_update(Q0_int, dQ)
            
            return Q0.at[nghost:-nghost, nghost:-nghost, :].set(Q_int_new), R

        self._jit_rk_stage = jit_rk_stage
        self._physical_residual_jax = compute_R_physical

        # --- JIT-compiled Newton-GMRES Step ---
        
        @jax.jit
        def jit_newton_step(Q_n, dt, gmres_tol, params: PhysicsParams):
            """Full Newton-GMRES step on GPU."""
            # 1. Block-Jacobi Preconditioner: P = V/dt - J_diag
            def compute_J_diag(Q):
                def make_tangent(k):
                    color, var = k // 4, k % 4
                    mask = color_masks[color]
                    t = jnp.zeros_like(Q)
                    return t.at[nghost:-nghost, nghost:-nghost, var].set(mask)
                
                tangents = jax.vmap(make_tangent)(jnp.arange(16))
                def single_jvp(t):
                    _, res = jax.jvp(lambda q: jit_residual(q, params), (Q,), (t,))
                    return res
                jvps = jax.vmap(single_jvp)(tangents)
                return (jvps.reshape(4, 4, NI, NJ, 4) * color_masks[:, None, :, :, None]).sum(axis=0).transpose(1, 2, 3, 0)
            
            J_diag = compute_J_diag(Q_n)
            P_inv = jax.vmap(jax.vmap(jnp.linalg.inv))( (volume / dt)[:, :, None, None] * jnp.eye(4) - J_diag )
            
            def precond_apply(v):
                v_reshaped = v.reshape(NI, NJ, 4)
                return jnp.einsum('ijkl,ijl->ijk', P_inv, v_reshaped).flatten()

            # 2. Unified Residual F(Q) = (V/dt)(Q-Q_n) - R(Q)
            def F_fn(q_flat):
                q_int = q_flat.reshape(NI, NJ, 4)
                q_full = Q_n.at[nghost:-nghost, nghost:-nghost, :].set(q_int)
                R = jit_residual(q_full, params)
                if smoothing_epsilon > 0 and smoothing_passes > 0:
                    R = smooth_explicit_jax(R, smoothing_epsilon, smoothing_passes)
                ptc = (volume / dt)[:, :, jnp.newaxis] * (q_int - Q_n[nghost:-nghost, nghost:-nghost, :])
                return (ptc - R).flatten()

            # 3. Solve J @ dQ = -F(Q_n)
            q_n_flat = Q_n[nghost:-nghost, nghost:-nghost, :].flatten()
            f0 = F_fn(q_n_flat)
            rhs = -f0 # Note: f0 is -R(Q_n), so rhs is R(Q_n)
            
            def matvec(v):
                _, jv = jax.jvp(F_fn, (q_n_flat,), (v,))
                return jv
            
            x_sol, r_norm, iters, converged = gmres_solve(
                matvec, rhs, jnp.zeros_like(rhs), gmres_tol * jnp.linalg.norm(precond_apply(rhs)),
                gmres_restart, gmres_maxiter, precond_apply
            )
            
            # 4. Apply Update (Patankar)
            dQ = x_sol.reshape(NI, NJ, 4)
            Q0_int = Q_n[nghost:-nghost, nghost:-nghost, :]
            Q_int_new = apply_patankar_update(Q0_int, dQ)
            
            Q_final = apply_bc(Q_n.at[nghost:-nghost, nghost:-nghost, :].set(Q_int_new), params)
            
            return Q_final, iters, r_norm, converged

        self._jit_newton_step = jit_newton_step
        
        @jax.jit
        def jit_compute_dt(Q, cfl, params: PhysicsParams):
            """Compute local timestep - JIT compiled with turbulent viscosity correction."""
            # Compute effective viscosity at each cell for proper viscous stability
            Q_int = Q[nghost:-nghost, nghost:-nghost, :]
            nu_tilde = jnp.maximum(Q_int[:, :, 3], 0.0)
            
            nu = params.mu_laminar
            beta = params.beta

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
        
        self._jit_compute_dt = jit_compute_dt
        
        
        # Warmup (optional, already called in _initialize_jax)
        # self._warmup_jax()
    

    def _get_unified_residual_fn(self, Q0: jnp.ndarray, dt: jnp.ndarray, params: PhysicsParams) -> Callable:
        """Create a unified residual function F(Q) for Newton-Krylov.
        
        F(Q) = (V/dt) * (Q - Q0) - R(Q)
        
        where:
        - Q is the flattened interior state
        - R(Q) includes fluxes, source terms, BCs, and residual smoothing
        - V/dt * (Q - Q0) is the pseudo-transient continuation term
        """
        nghost = NGHOST
        NI, NJ = self.NI, self.NJ
        volume = self.volume_jax
        apply_bc = self.apply_bc_jit
        smoothing_epsilon = getattr(self.config, 'smoothing_epsilon', 0.2)
        smoothing_passes = getattr(self.config, 'smoothing_passes', 2)
        
        def unified_residual(Q_flat):
            # 1. Reshape to interior state
            Q_int = Q_flat.reshape(NI, NJ, 4)
            
            # 2. Insert into full state with ghosts
            Q_full = Q0.at[nghost:-nghost, nghost:-nghost, :].set(Q_int)
            
            # 3. Apply BCs (sensitivities will be captured by AD!)
            Q_full = apply_bc(Q_full, params)
            
            # 4. Compute physical residual
            R = self._physical_residual_jax(Q_full, params)
            
            # 5. Apply residual smoothing if configured
            if smoothing_epsilon > 0 and smoothing_passes > 0:
                R = smooth_explicit_jax(R, smoothing_epsilon, smoothing_passes)
            
            # 6. PTC term: (V/dt) * (Q - Q0)
            Q0_int = Q0[nghost:-nghost, nghost:-nghost, :]
            ptc = (volume / dt)[:, :, jnp.newaxis] * (Q_int - Q0_int)
            
            # F(Q) = PTC - R
            return (ptc - R).flatten()
            
        return unified_residual

    
    def compute_preconditioner(self):
        """Compute block-Jacobi preconditioner at current state.
        
        Stores the preconditioner for use in step() and Newton iterations.
        Should be called:
        - Every iteration for rk5_precond mode
        - Every Newton iteration for newton mode
        """
        # Get current timestep for implicit diagonal
        # Use appropriate CFL function based on solver mode
        solver_mode = getattr(self.config, 'solver_mode', 'rk5')
        if solver_mode == 'newton':
            cfl = self._get_cfl_newton(self.iteration)
        else:
            cfl = self._get_cfl(self.iteration)
            
        params = self._build_params()
        dt = self._jit_compute_dt(self.Q_jax, cfl, params)
        
        # Use lambda to bind params for residual function expected by preconditioner
        residual_fn = lambda q: self._jit_residual(q, params)
        
        self._preconditioner = BlockJacobiPreconditioner.compute(
            residual_fn=residual_fn,
            Q=self.Q_jax,
            dt=dt,
            volume=self.volume_jax,
            nghost=NGHOST,
        )
        
        # Store JIT-compiled apply function
        self._precond_apply = self._preconditioner.apply_jit()
    
    def _warmup_jax(self):
        """Warm up JAX JIT compilation with dummy iterations."""
        solver_mode = getattr(self.config, 'solver_mode', 'rk5')
        
        logger.info("Compiling JAX functions...")
        
        cfl = self.config.cfl_start
        Q_test = self.Q_jax
        params = self._build_params()
        dt = self._jit_compute_dt(Q_test, cfl, params)
        
        if solver_mode == 'newton':
            # Newton mode: warm up with EXACT production settings
            # This forces compilation of the main solver graph.
            logger.info(f"  Compiling Newton-GMRES step (restart={self.config.gmres_restart}, maxiter={self.config.gmres_maxiter})...")
            
            # Use actual JIT function
            # Note: We must block until ready to ensure compilation finishes
            _ = self._jit_newton_step(Q_test, dt, self.config.gmres_tol, params)
            jax.block_until_ready(_)
            
        else:
            # RK mode: warm up RK stages
            for alpha in [0.25, 0.5]:
                Q_test, _ = self._jit_rk_stage(Q_test, Q_test, dt, alpha, params)
            jax.block_until_ready(Q_test)
        
        logger.info("JIT Warmup done")
    
    def _initialize_output(self):
        """Initialize HTML animation for output."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Use CDN by default to reduce HTML file size, no compression for easier debugging
        use_cdn = getattr(self.config, 'html_use_cdn', True)
        compress = getattr(self.config, 'html_compress', False)
        self.plotter = PlotlyDashboard(reynolds=self.config.reynolds, use_cdn=use_cdn, compress=compress)
        
        # Initialize JAX arrays
        self._initialize_jax()
        
        if self.config.html_animation:
            Q_int = self.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
            C_pt = compute_total_pressure_loss(
                Q_int, self.freestream.p_inf,
                self.freestream.u_inf, self.freestream.v_inf
            )
            # Compute initial residual using the JIT function to populate R_jax
            # This is needed for get_scaled_residual_field() to work
            initial_R_raw = self._compute_residual(self.Q)
            self.R_jax = jax.device_put(jnp.array(initial_R_raw))
            initial_R = self.get_scaled_residual_field()
            # Compute AFT diagnostic fields
            Re_Omega, Gamma, is_turb = self._compute_aft_fields()
            self.plotter.store_snapshot(
                self.Q, 0, self.residual_history,
                cfl=self._get_cfl(0), C_pt=C_pt, residual_field=initial_R,
                freestream=self.freestream,
                iteration_history=self.iteration_history,
                Re_Omega=Re_Omega, Gamma=Gamma, is_turb=is_turb
            )
        
        logger.info(f"  Output directory: {output_path}")
    
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
    
    # _apply_smoothing removed (unused legacy code)
    
    def _compute_aft_fields(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Re_Omega, Gamma, and is_turb fields for visualization.
        
        Returns
        -------
        Re_Omega : np.ndarray (NI, NJ)
            Vorticity Reynolds number = d² |ω| / ν
        Gamma : np.ndarray (NI, NJ)
            AFT shape factor
        is_turb : np.ndarray (NI, NJ)
            Turbulent fraction indicator (0=laminar/AFT, 1=turbulent/SA)
        """
        # Get interior values
        # Note: Q stores TOTAL velocity (not perturbation from freestream)
        Q_int = self.Q_jax[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
        u = Q_int[:, :, 1]  # Total u velocity
        v = Q_int[:, :, 2]  # Total v velocity
        nuHat = Q_int[:, :, 3]  # SA working variable
        
        # Compute velocity magnitude (already total velocity, no freestream addition needed)
        vel_mag = jnp.sqrt(u**2 + v**2)
        
        # Compute gradients
        grad = compute_gradients_jax(
            self.Q_jax, 
            self.Si_x_jax, 
            self.Si_y_jax,
            self.Sj_x_jax, 
            self.Sj_y_jax,
            self.volume_jax, NGHOST
        )
        
        # Compute vorticity magnitude
        omega_mag = compute_vorticity_jax(grad)
        
        # Get wall distance (already interior cells, no ghost padding)
        wall_dist = self.wall_dist_jax
        
        # Compute AFT fields using existing functions
        nu = 1.0 / self.config.reynolds
        gamma_coeff = self.config.aft_gamma_coeff
        
        Re_Omega = compute_Re_Omega(omega_mag, wall_dist, nu)
        Gamma = compute_gamma(omega_mag, vel_mag, wall_dist, gamma_coeff)
        
        # Compute turbulent fraction (blending indicator)
        # chi = nuHat / nu is the dimensionless turbulent viscosity ratio
        chi = jnp.maximum(nuHat, 0.0) / nu
        is_turb = compute_turbulent_fraction(
            chi,
            threshold=self.config.aft_blend_threshold,
            width=self.config.aft_blend_width
        )
        
        return np.array(Re_Omega), np.array(Gamma), np.array(is_turb)
    
    def step(self) -> None:
        """Perform one iteration of the solver.
        
        Fully GPU-accelerated using JIT-compiled stages.
        No CPU transfers - state stays on GPU.
        
        Supports three modes:
        - rk5: Standard explicit RK5 (default)
        - rk5_precond: RK5 with block-Jacobi preconditioned residual
        - newton: Newton-GMRES with block-Jacobi preconditioning
        """
        solver_mode = getattr(self.config, 'solver_mode', 'rk5')
        
        if solver_mode == 'newton':
            self._step_newton()
        elif solver_mode == 'rk5_precond':
            self._step_rk5_preconditioned()
        else:  # rk5
            self._step_rk5()
        
        self.iteration += 1
    
    def _step_rk5(self) -> None:
        """Standard explicit RK5 step."""
        params = self._build_params()
        cfl = self._get_cfl(self.iteration)
        dt = self._jit_compute_dt(self.Q_jax, cfl, params)
        
        alphas = [0.25, 0.166666667, 0.375, 0.5, 1.0]
        
        Q0 = self.Q_jax
        Q = self.Q_jax
        
        for alpha in alphas:
            Q, R = self._jit_rk_stage(Q, Q0, dt, alpha, params)
        
        Q = self.apply_bc_jit(Q, params)
        
        self.Q_jax = Q
        self.R_jax = R
    
    def _step_rk5_preconditioned(self) -> None:
        """RK5 step with block-Jacobi preconditioned residual.
        
        The preconditioner P^{-1} is applied to the residual before the RK update:
            dQ = alpha * dt/V * P^{-1} @ R
        
        This accelerates convergence by accounting for local stiffness from
        turbulent viscosity and artificial compressibility.
        """
        params = self._build_params()
        cfl = self._get_cfl(self.iteration)
        dt = self._jit_compute_dt(self.Q_jax, cfl, params)
        
        # Compute preconditioner at current state
        # This is done every iteration for maximum accuracy
        self.compute_preconditioner()
        
        alphas = [0.25, 0.166666667, 0.375, 0.5, 1.0]
        
        Q0 = self.Q_jax
        Q = self.Q_jax
        
        for alpha in alphas:
            Q, R = self._rk_stage_preconditioned(Q, Q0, dt, alpha, params)
        
        Q = self.apply_bc_jit(Q, params)
        
        self.Q_jax = Q
        self.R_jax = R
    
    def _rk_stage_preconditioned(
        self, Q: jnp.ndarray, Q0: jnp.ndarray, dt: jnp.ndarray, alpha: float, params: PhysicsParams
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Single RK stage with preconditioned residual.
        
        Computes: Q_new = Q0 + alpha * dt/V * P^{-1} @ R
        
        Note: This is not fully JIT-compiled because the preconditioner
        changes every iteration. The residual and update are JIT-compiled
        separately.
        """
        nghost = NGHOST
        smoothing_epsilon = getattr(self.config, 'smoothing_epsilon', 0.2)
        smoothing_passes = getattr(self.config, 'smoothing_passes', 2)
        
        # Apply BC and compute residual
        Q = self.apply_bc_jit(Q, params)
        R = self._jit_residual(Q, params)  # (NI, NJ, 4)
        
        # Apply preconditioner: R_precond = P^{-1} @ R
        R_precond = self._precond_apply(R)  # (NI, NJ, 4)
        
        # Apply residual smoothing AFTER preconditioning (for RK modes)
        if smoothing_epsilon > 0 and smoothing_passes > 0:
            R_precond = smooth_explicit_jax(R_precond, smoothing_epsilon, smoothing_passes)
        
        # Compute update with Patankar for nuHat
        Q_int_new = self._apply_rk_update(Q0, R_precond, dt, alpha)
        
        Q_new = Q0.at[nghost:-nghost, nghost:-nghost, :].set(Q_int_new)
        
        return Q_new, R
    
    
    def _step_newton(self) -> None:
        """Newton-GMRES step with consolidated residual and JAX AD."""
        Q_n = self.Q_jax
        params = self._build_params()
        cfl = self._get_cfl_newton(self.iteration)
        dt = self._jit_compute_dt(Q_n, cfl, params)
        gmres_tol = float(getattr(self.config, 'gmres_tol', 1e-3))
        
        # Execute unified JIT Newton step (includes preconditioner + linear solve + update)
        self.Q_jax, iters, r_norm, converged = self._jit_newton_step(Q_n, dt, gmres_tol, params)
        
        self.R_jax = self._jit_residual(self.Q_jax, params)
        self._last_gmres_info = {
            'iterations': int(iters),
            'rhs_norm': float(jnp.linalg.norm(self.R_jax)), # Approximation
            'residual_norm': float(r_norm),
            'converged': bool(converged),
        }

    
    def _get_cfl_newton(self, iteration: int) -> float:
        """Get CFL number for Newton mode with exponential ramping.
        
        Newton mode uses exponential CFL ramping for pseudo-transient
        continuation. CFL starts at cfl_start and grows by a factor of 10
        every cfl_ramp_iters iterations, capped at cfl_target.
        
        Formula: CFL(n) = cfl_start * 10^(n / ramp_iters)
        
        As CFL → ∞, the V/dt diagonal → 0, recovering pure Newton iteration.
        """
        cfl_start = self.config.cfl_start
        cfl_final = self.config.cfl_target  # Final/maximum CFL
        ramp_iters_per_decade = self.config.cfl_ramp_iters
        
        # Exponential growth: multiply by 10 every ramp_iters iterations
        decades = iteration / max(ramp_iters_per_decade, 1)
        cfl = cfl_start * (10.0 ** decades)
        
        return min(cfl, cfl_final)

    def _apply_rk_update(
        self, Q0: jnp.ndarray, R: jnp.ndarray, dt: jnp.ndarray, alpha: float
    ) -> jnp.ndarray:
        """Apply RK update - uses static JIT function."""
        nghost = NGHOST
        Q0_int = Q0[nghost:-nghost, nghost:-nghost, :]
        return self._apply_rk_update_static(Q0_int, R, dt, self.volume_inv_jax, alpha)

    @staticmethod
    @jax.jit
    def _apply_rk_update_static(
        Q0_int: jnp.ndarray, 
        R: jnp.ndarray, 
        dt: jnp.ndarray,
        volume_inv: jnp.ndarray,
        alpha: float
    ) -> jnp.ndarray:
        """Apply RK update with Patankar scheme (uses utility)."""
        dt_factors = jnp.array([1.0, 1.0, 1.0, 0.5])  # Half CFL for nuHat
        dQ = alpha * (dt * volume_inv)[:, :, jnp.newaxis] * dt_factors * R
        return apply_patankar_update(Q0_int, dQ)

    @staticmethod
    def _apply_newton_update_static(Q_int: jnp.ndarray, dQ: jnp.ndarray) -> jnp.ndarray:
        """Apply Newton update with Patankar scheme (for tests)."""
        """Apply Newton update with Patankar scheme (for tests)."""
        return apply_patankar_update(Q_int, dQ)
    
    
    
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
        # nu_eff >= nu_laminar by construction since nuHat >= 0, no clipping needed
        
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
    
    def get_scaled_residual_field(self) -> np.ndarray:
        """Compute scaled residual field for visualization.
        
        Uses the same scaling as get_residual_l1_scaled():
        - p scaled by β
        - u, v scaled by V_inf  
        - nuHat scaled by local (ν_laminar + ν̃)
        
        Then divides by cell volume to get residual density,
        and takes RMS across the 4 equations.
        
        Returns
        -------
        np.ndarray (NI, NJ)
            Scaled RMS residual density at each cell.
        """
        # Characteristic scales (same as get_residual_l1_scaled)
        beta = self.config.beta
        V_inf = np.sqrt(self.freestream.u_inf**2 + self.freestream.v_inf**2)
        V_inf = max(V_inf, 1e-30)
        nu_laminar = 1.0 / self.config.reynolds if self.config.reynolds > 0 else 1e-6
        
        # Get interior Q for local nuHat scaling
        nghost = NGHOST
        Q_int = self.Q_jax[nghost:-nghost, nghost:-nghost, :]
        nuHat = jnp.maximum(Q_int[:, :, 3], 0.0)
        
        # Local effective viscosity for nuHat scaling
        # nu_eff >= nu_laminar by construction since nuHat >= 0
        nu_eff = nu_laminar + nuHat
        
        # Residual and volume
        R = self.R_jax
        volume = self.volume_jax
        
        # Scaled residuals (same scaling as L1 metric)
        R_p_scaled = jnp.abs(R[:, :, 0]) / beta
        R_u_scaled = jnp.abs(R[:, :, 1]) / V_inf
        R_v_scaled = jnp.abs(R[:, :, 2]) / V_inf
        R_nu_scaled = jnp.abs(R[:, :, 3]) / nu_eff
        
        # Divide by cell volume to get residual density
        # Volume is always positive for valid grids
        inv_vol = 1.0 / volume
        R_p_density = R_p_scaled * inv_vol
        R_u_density = R_u_scaled * inv_vol
        R_v_density = R_v_scaled * inv_vol
        R_nu_density = R_nu_scaled * inv_vol
        
        # RMS across 4 equations
        rms_residual = jnp.sqrt(
            (R_p_density**2 + R_u_density**2 + R_v_density**2 + R_nu_density**2) / 4.0
        )
        
        return np.array(rms_residual)
    
    def sync_to_cpu(self) -> None:
        """Transfer Q from GPU to CPU (for visualization/output)."""
        self.Q = np.array(self.Q_jax)
    
    def run_steady_state(self) -> bool:
        """Run steady-state simulation to convergence."""
        logger.info(f"\n{'='*60}")
        logger.info("Starting Steady-State Iteration")
        logger.info(f"{'='*60}")
        logger.info(f"{'Iter':>8} {'Residual(L1)':>14} {'CFL':>8}  "
                    f"{'CL':>8} {'CD':>8} {'(p:':>8} {'f:)':>8}")
        logger.info(f"{'-'*72}")
        
        # Initial residual for normalization
        initial_residual = None
        
        try:
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
                
                # Get CFL using the appropriate method for the solver mode
                solver_mode = getattr(self.config, 'solver_mode', 'rk5')
                if solver_mode == 'newton':
                    cfl = self._get_cfl_newton(self.iteration)
                else:
                    cfl = self._get_cfl(self.iteration)
                
                if need_print:
                    # Print GMRES info for Newton mode (if available)
                    if hasattr(self, '_last_gmres_info') and self._last_gmres_info is not None:
                        info = self._last_gmres_info
                        converged_str = "converged" if info['converged'] else "NOT converged"
                        logger.debug(f"  GMRES: {info['iterations']} iters, "
                                     f"||r||: {info['rhs_norm']:.3e} → {info['residual_norm']:.3e} "
                                     f"({converged_str})")
                    
                    # Transfer Q to CPU for force computation
                    self.sync_to_cpu()
                    forces = self.compute_forces()
                    # Print max residual and individual components
                    logger.info(f"{self.iteration:>8d} {res_max:>14.6e} {cfl:>8.2f}  "
                                f"CL={forces.CL:>7.4f} CD={forces.CD:>7.5f} "
                                f"(p:{forces.CD_p:>7.5f} f:{forces.CD_f:>7.5f})")
                    

            
                    # NuHat Diagnostic (disabled after debugging)
                    # nu_diag = analyze_nuhat_residual(np.array(R_full), np.array(self.Q), self.X, self.Y)
                    # logger.info(f"  NuHat Max: Res={nu_diag.max_res:.4e} at ({nu_diag.i_max},{nu_diag.j_max}) "
                    #             f"x={nu_diag.x_loc:.3f}, y={nu_diag.y_loc:.3f} | nuHat={nu_diag.nu_hat:.4e}")
                    
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
                    # Use scaled residual field (same scaling as L1 metric)
                    R_field = self.get_scaled_residual_field()
                    Q_int = self.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
                    C_pt = compute_total_pressure_loss(
                        Q_int, self.freestream.p_inf,
                        self.freestream.u_inf, self.freestream.v_inf
                    )
                    # Compute AFT diagnostic fields
                    Re_Omega, Gamma, is_turb = self._compute_aft_fields()
                    self.plotter.store_snapshot(
                        self.Q, self.iteration, self.residual_history,
                        cfl=cfl, C_pt=C_pt, residual_field=R_field,
                        freestream=self.freestream,
                        iteration_history=self.iteration_history,
                        Re_Omega=Re_Omega, Gamma=Gamma, is_turb=is_turb
                    )
                
                if res_max < self.config.tol:
                    self.converged = True
                    logger.info(f"\n{'='*60}")
                    logger.info(f"CONVERGED at iteration {self.iteration}")
                    res_final = self.residual_history[-1]
                    logger.info(f"Final residual (L1): p={res_final[0]:.2e} u={res_final[1]:.2e} v={res_final[2]:.2e} ν̃={res_final[3]:.2e}")
                    logger.info(f"{'='*60}")
                    break
                
                # Check for divergence: either residual > 1000x initial, or NaN/Inf
                is_diverged = (
                    self.residual_history and 
                    (res_max > 1000 * initial_residual or not jnp.isfinite(res_max))
                )
                if is_diverged:
                    logger.warning(f"\n{'='*60}")
                    logger.warning(f"DIVERGED at iteration {self.iteration}")
                    if initial_residual:
                        logger.warning(f"Max residual: {res_max:.6e} (initial: {initial_residual:.6e})")
                    else:
                        logger.warning(f"Max residual: {res_max} (NaN/Inf detected!)")
                    logger.warning(f"{'='*60}")
                    
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
                        # Use scaled residual field (same scaling as L1 metric)
                        R_field = self.get_scaled_residual_field()
                        Q_int = self.Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
                        C_pt = compute_total_pressure_loss(
                            Q_int, self.freestream.p_inf,
                            self.freestream.u_inf, self.freestream.v_inf
                        )
                        # Compute AFT diagnostic fields
                        Re_Omega, Gamma, is_turb = self._compute_aft_fields()
                        self.plotter.store_snapshot(
                            self.Q, self.iteration, self.residual_history,
                            cfl=cfl, C_pt=C_pt, residual_field=R_field,
                            freestream=self.freestream,
                            iteration_history=self.iteration_history,
                            is_divergence_dump=True,
                            Re_Omega=Re_Omega, Gamma=Gamma, is_turb=is_turb
                        )
                        
                        total_dumps = n_history_dumps + 1
                        total_dumps = n_history_dumps + 1
                        logger.info(f"  Divergence snapshots captured: {total_dumps} ({n_history_dumps} from history + 1 current)")
                    break
            
            else:
                logger.info(f"\n{'='*60}")
                logger.info(f"Maximum iterations ({self.config.max_iter}) reached")
                if self.residual_history:
                    res_final = self.residual_history[-1]
                    logger.info(f"Final residual: p={res_final[0]:.2e} u={res_final[1]:.2e} v={res_final[2]:.2e} ν̃={res_final[3]:.2e}")
                logger.info(f"{'='*60}")
        
        except KeyboardInterrupt:
            logger.warning("\nSimulation interrupted by user.")
        except Exception as e:
            logger.error(f"\nError during simulation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.save_diagnostics()
            
        return self.converged
        
    def save_diagnostics(self) -> None:
        """Save HTML and VTK diagnostics."""
        self.sync_to_cpu()
        
        if self.config.html_animation and self.plotter.num_snapshots > 0:
            logger.info(f"Saving HTML animation with {self.plotter.num_snapshots} snapshots...")
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
            
            # Save NPY for diagnostics
            np.save(f"{self.config.output_dir}/final_q.npy", self.Q)
            np.save(f"{self.config.output_dir}/final_x.npy", self.X)
            np.save(f"{self.config.output_dir}/final_y.npy", self.Y)
            logger.info(f"Saved checkpoint to {self.config.output_dir}/final_q.npy")

    
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
        
        logger.info(f"Residual history saved to: {filename}")

