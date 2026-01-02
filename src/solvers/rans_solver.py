"""
RANS Solver for 2D Incompressible Flow.

This module provides the main RANSSolver class that orchestrates the
CFD simulation by integrating all components:
- Grid loading and metric computation
- State initialization with wall damping
- Time stepping with CFL ramping
- RK4 integration with JST flux scheme
- Residual monitoring and convergence checking
- VTK output for visualization

Physics: Artificial Compressibility formulation for incompressible RANS
    - State vector: Q = [p, u, v, ν̃]
    - Turbulence model: Spalart-Allmaras (one-equation)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass, field

# Grid components
from ..grid.metrics import MetricComputer, FVMMetrics
from ..grid.mesher import Construct2DWrapper, GridOptions
from ..grid.plot3d import read_plot3d, StructuredGrid

# Numerics
from ..numerics.fluxes import compute_fluxes, FluxConfig, GridMetrics as FluxGridMetrics
from ..numerics.forces import compute_aerodynamic_forces, AeroForces
from ..numerics.gradients import compute_gradients, GradientMetrics
from ..numerics.viscous_fluxes import add_viscous_fluxes
from ..numerics.smoothing import apply_residual_smoothing

# Solvers and BCs
from .boundary_conditions import (
    FreestreamConditions, 
    BoundaryConditions,
    initialize_state,
    apply_initial_wall_damping,
    apply_boundary_conditions
)
from .time_stepping import compute_local_timestep, TimeStepConfig
from .multigrid import MultigridHierarchy, build_multigrid_hierarchy

# IO
from ..io.output import VTKWriter, write_vtk

# Surface analysis
from ..numerics.forces import compute_surface_distributions, create_surface_vtk_fields

# Diagnostics
from ..numerics.diagnostics import (
    compute_total_pressure_loss, 
    compute_solution_bounds,
    compute_residual_statistics
)


@dataclass
class SolverConfig:
    """Configuration for RANS solver."""
    
    # Flow conditions
    mach: float = 0.15              # Reference Mach number
    alpha: float = 0.0              # Angle of attack (degrees)
    reynolds: float = 6e6           # Reynolds number
    
    # Artificial compressibility
    beta: float = 10.0              # Artificial compressibility parameter
    
    # Time stepping
    cfl_start: float = 0.1          # Initial CFL number
    cfl_target: float = 5.0         # Target CFL number
    cfl_ramp_iters: int = 500       # Iterations for CFL ramp
    
    # Convergence
    max_iter: int = 10000           # Maximum iterations
    tol: float = 1e-10              # Residual tolerance for convergence
    
    # Output
    output_freq: int = 100          # VTK output frequency
    print_freq: int = 50            # Console print frequency
    output_dir: str = "output"      # Output directory
    case_name: str = "solution"     # Base name for output files
    
    # Initial conditions
    wall_damping_length: float = 0.1  # Wall damping decay length
    
    # JST flux parameters
    # Note: k2 (2nd-order dissipation) is set to 0 for incompressible flow
    # since there are no shocks to capture. Only k4 (4th-order) is needed.
    jst_k4: float = 0.04            # 4th-order dissipation coefficient
    
    # Implicit Residual Smoothing (IRS)
    irs_epsilon: float = 0.0        # IRS smoothing coefficient (0 = disabled)
                                    # Typical values: 0.5-2.0 for higher CFL
    
    # Grid topology (C-grid)
    n_wake: int = 30                # Number of wake cells at each end of i-direction
    
    # Diagnostic options
    diagnostic_mode: bool = False   # Enable diagnostic output (plots, extra stats)
    diagnostic_freq: int = 100      # Frequency of diagnostic dumps (when enabled)
    divergence_history: int = 0     # Number of solutions to save for divergence analysis
    
    # Multigrid options
    use_multigrid: bool = False     # Enable geometric multigrid (FAS scheme)
    mg_levels: int = 4              # Maximum number of multigrid levels
    mg_nu1: int = 1                 # Pre-smoothing iterations (1 is usually sufficient)
    mg_nu2: int = 1                 # Post-smoothing iterations (1 for stability)
    mg_min_size: int = 8            # Minimum cells per direction on coarsest grid
    mg_omega: float = 0.5           # Prolongation relaxation factor (0.5-1.0)
    mg_use_injection: bool = True   # Use injection instead of bilinear prolongation


class RANSSolver:
    """
    Main RANS Solver for 2D incompressible flow around airfoils.
    
    This class orchestrates the complete CFD simulation workflow:
    1. Grid loading (Plot3D file or Construct2D generation)
    2. Metric computation (cell volumes, face normals, wall distance)
    3. State initialization with wall damping
    4. Time integration with RK4 and CFL ramping
    5. Residual monitoring and convergence checking
    6. VTK output for visualization
    
    Example
    -------
    >>> config = SolverConfig(mach=0.15, alpha=0.0, reynolds=6e6)
    >>> solver = RANSSolver("grid/naca0012.p3d", config)
    >>> solver.run_steady_state()
    
    Attributes
    ----------
    config : SolverConfig
        Solver configuration parameters.
    X, Y : ndarray
        Grid node coordinates.
    metrics : FVMMetrics
        Computed grid metrics.
    Q : ndarray
        Current state vector [p, u, v, ν̃].
    iteration : int
        Current iteration number.
    residual_history : list
        History of density residual RMS values.
    """
    
    def __init__(self, 
                 grid_file: str, 
                 config: Optional[Union[SolverConfig, Dict]] = None):
        """
        Initialize the RANS solver.
        
        Parameters
        ----------
        grid_file : str
            Path to the grid file (.p3d format) or airfoil file (.dat) for
            on-the-fly grid generation.
        config : SolverConfig or dict, optional
            Solver configuration. Can be a SolverConfig object or a dictionary
            of configuration parameters.
        """
        # Parse configuration
        if config is None:
            self.config = SolverConfig()
        elif isinstance(config, dict):
            self.config = SolverConfig(**config)
        else:
            self.config = config
        
        # Initialize state
        self.iteration = 0
        self.residual_history = []
        self.converged = False
        
        # Load grid
        self._load_grid(grid_file)
        
        # Compute metrics
        self._compute_metrics()
        
        # Initialize state
        self._initialize_state()
        
        # Initialize multigrid hierarchy if enabled
        self.mg_hierarchy = None
        if self.config.use_multigrid:
            self._initialize_multigrid()
        
        # Initialize VTK writer
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
            # Load Plot3D grid
            print(f"Loading grid from: {grid_path}")
            self.X, self.Y = read_plot3d(str(grid_path))
            
        elif suffix == '.dat':
            # Airfoil file - generate grid with Construct2D
            print(f"Generating grid from airfoil: {grid_path}")
            # Look for construct2d binary
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
        
        # Store grid dimensions (number of cells)
        self.NI = self.X.shape[0] - 1
        self.NJ = self.X.shape[1] - 1
        
        print(f"Grid loaded: {self.X.shape[0]} x {self.X.shape[1]} nodes")
        print(f"            {self.NI} x {self.NJ} cells")
    
    def _compute_metrics(self):
        """Compute FVM grid metrics."""
        print("Computing grid metrics...")
        
        computer = MetricComputer(self.X, self.Y, wall_j=0)
        self.metrics = computer.compute()
        
        # Create flux metrics (needed for residual computation and force computation)
        self.flux_metrics = FluxGridMetrics(
            Si_x=self.metrics.Si_x,
            Si_y=self.metrics.Si_y,
            Sj_x=self.metrics.Sj_x,
            Sj_y=self.metrics.Sj_y,
            volume=self.metrics.volume
        )
        
        # Create gradient metrics (needed for viscous fluxes)
        self.grad_metrics = GradientMetrics(
            Si_x=self.metrics.Si_x,
            Si_y=self.metrics.Si_y,
            Sj_x=self.metrics.Sj_x,
            Sj_y=self.metrics.Sj_y,
            volume=self.metrics.volume
        )
        
        # Validate GCL
        gcl = computer.validate_gcl()
        print(f"  {gcl}")
        
        if not gcl.passed:
            print("  WARNING: GCL validation failed. Results may be inaccurate.")
    
    def _initialize_state(self):
        """Initialize state vector with freestream and wall damping."""
        print("Initializing flow state...")
        
        # Create freestream conditions
        self.freestream = FreestreamConditions.from_mach_alpha(
            mach=self.config.mach,
            alpha_deg=self.config.alpha
        )
        
        # Initialize to freestream
        self.Q = initialize_state(self.NI, self.NJ, self.freestream)
        
        # Apply wall damping for cold start
        self.Q = apply_initial_wall_damping(
            self.Q, 
            self.metrics,
            decay_length=self.config.wall_damping_length,
            n_wake=getattr(self.config, 'n_wake', 0)
        )
        
        # Compute far-field outward unit normals for characteristic BC
        # Sj at j=NJ-1 points in +j direction (toward far-field)
        Sj_x_ff = self.metrics.Sj_x[:, -1]  # Shape: (NI,)
        Sj_y_ff = self.metrics.Sj_y[:, -1]
        Sj_mag = np.sqrt(Sj_x_ff**2 + Sj_y_ff**2) + 1e-12
        nx_ff = Sj_x_ff / Sj_mag
        ny_ff = Sj_y_ff / Sj_mag
        
        # Apply boundary conditions with characteristic farfield BC
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
            max_levels=self.config.mg_levels
        )
        
        print(f"  Built {self.mg_hierarchy.num_levels} multigrid levels:")
        for i, lvl in enumerate(self.mg_hierarchy.levels):
            print(f"    Level {i}: {lvl.NI} x {lvl.NJ}")
    
    def _initialize_output(self):
        """Initialize VTK writer for output."""
        # Create output directory
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create VTK writer
        base_path = output_path / self.config.case_name
        self.vtk_writer = VTKWriter(
            str(base_path),
            self.X, self.Y,
            beta=self.config.beta
        )
        
        # Write initial state with surface data
        surface_fields = self._compute_surface_fields()
        self.vtk_writer.write(self.Q, iteration=0, additional_scalars=surface_fields)
        print(f"  Initial solution written to: {output_path}")
    
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
                           grad_metrics: Optional[GradientMetrics] = None) -> np.ndarray:
        """
        Compute flux residual using JST scheme + viscous fluxes.
        
        Parameters
        ----------
        Q : ndarray, shape (NI+2, NJ+2, 4)
            State vector with ghost cells.
        forcing : ndarray, shape (NI, NJ, 4), optional
            FAS forcing term to add to residual (for multigrid).
        flux_metrics : FluxGridMetrics, optional
            Grid metrics for flux computation (default: self.flux_metrics).
        grad_metrics : GradientMetrics, optional
            Gradient metrics (default: self.grad_metrics).
            
        Returns
        -------
        residual : ndarray, shape (NI, NJ, 4)
            Computed residual (optionally with forcing added).
        """
        # Use default metrics if not provided
        if flux_metrics is None:
            flux_metrics = self.flux_metrics
        if grad_metrics is None:
            grad_metrics = self.grad_metrics
        
        # Create flux config (k2=0 for incompressible flow - no shock capturing needed)
        flux_cfg = FluxConfig(
            k2=0.0,
            k4=self.config.jst_k4
        )
        
        # Compute convective fluxes (JST scheme)
        conv_residual = compute_fluxes(Q, flux_metrics, self.config.beta, flux_cfg)
        
        # Compute gradients for viscous fluxes
        gradients = compute_gradients(Q, grad_metrics)
        
        # Compute laminar viscosity from Reynolds number
        mu_laminar = 1.0 / self.config.reynolds if self.config.reynolds > 0 else 0.0
        
        # Add viscous fluxes (laminar only for now)
        residual = add_viscous_fluxes(
            conv_residual, Q, gradients, grad_metrics, mu_laminar
        )
        
        # Add FAS forcing term if provided (for multigrid coarse levels)
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
        """
        Perform one iteration of the solver.
        
        Returns
        -------
        residual_rms : float
            RMS of the density residual.
        residual : ndarray
            Full residual array.
        """
        # Get current CFL
        cfl = self._get_cfl(self.iteration)
        
        # Time step config
        ts_config = TimeStepConfig(cfl=cfl)
        
        # Kinematic viscosity for time step stability
        nu = 1.0 / self.config.reynolds if self.config.reynolds > 0 else 0.0
        
        # Compute local timestep (with viscous stability constraint)
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
        
        # Jameson 5-stage RK coefficients (extended stability for multigrid)
        # CFL_max ≈ 4.0 for central differences (vs ~2.8 for 4-stage)
        alphas = [0.25, 0.166666667, 0.375, 0.5, 1.0]
        
        for alpha in alphas:
            # Apply boundary conditions
            Qk = self._apply_bc(Qk)
            
            # Compute residual
            R = self._compute_residual(Qk)
            
            # Apply Implicit Residual Smoothing (IRS) if enabled
            if self.config.irs_epsilon > 0.0:
                apply_residual_smoothing(R, self.config.irs_epsilon)
            
            # Update: Q^(k) = Q^(0) + α * dt/Ω * R
            Qk = Q0.copy()
            Qk[1:-1, 1:-1, :] += alpha * (dt / self.metrics.volume)[:, :, np.newaxis] * R
        
        # Store final residual for monitoring
        Qk = self._apply_bc(Qk)
        final_residual = self._compute_residual(Qk)
        
        # Update state
        self.Q = Qk
        
        # Compute residual RMS
        residual_rms = self._compute_residual_rms(final_residual)
        
        # Update iteration counter
        self.iteration += 1
        
        return residual_rms, final_residual
    
    def step_multigrid(self) -> Tuple[float, np.ndarray]:
        """
        Perform one V-cycle iteration with multigrid acceleration.
        
        Returns
        -------
        residual_rms : float
            RMS of the density residual.
        residual : ndarray
            Full residual array.
        """
        if self.mg_hierarchy is None:
            raise RuntimeError("Multigrid not initialized. Set use_multigrid=True.")
        
        # Sync fine level state
        self.mg_hierarchy.levels[0].Q[:] = self.Q
        
        # Get current CFL
        cfl = self._get_cfl(self.iteration)
        
        # Compute timestep for finest level
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
        
        # Run V-cycle starting from finest level
        self._run_v_cycle(0)
        
        # Copy result back from hierarchy
        self.Q = self.mg_hierarchy.levels[0].Q.copy()
        
        # Compute final residual for monitoring
        final_residual = self._compute_residual(self.Q)
        residual_rms = self._compute_residual_rms(final_residual)
        
        # Update iteration counter
        self.iteration += 1
        
        return residual_rms, final_residual
    
    def _run_v_cycle(self, level: int):
        """
        Run recursive V-cycle from given level.
        
        FAS V-cycle algorithm:
        1. Pre-smoothing (nu1 iterations)
        2. Compute residual R_f
        3. Restrict Q_f -> Q_c and R_f -> R_c
        4. Compute coarse residual R(Q_c)
        5. Compute FAS forcing: P_c = R_c - R(Q_c)
        6. Recurse to coarser level (or solve if coarsest)
        7. Prolongate correction: Q_f += interpolate(Q_c_new - Q_c_old)
        8. Post-smoothing (nu2 iterations)
        
        Parameters
        ----------
        level : int
            Current multigrid level (0 = finest).
        """
        lvl = self.mg_hierarchy.levels[level]
        
        # ===== Pre-smoothing =====
        for _ in range(self.config.mg_nu1):
            self._smooth_level(level)
        
        # ===== If coarsest level, do extra smoothing and return =====
        if level >= self.mg_hierarchy.num_levels - 1:
            # Extra smoothing on coarsest level
            for _ in range(self.config.mg_nu1 + self.config.mg_nu2):
                self._smooth_level(level)
            return
        
        # ===== Compute fine residual =====
        lvl.Q = lvl.bc.apply(lvl.Q)
        lvl.R = self._compute_residual_level(level)
        
        # ===== Restrict to coarse level =====
        self.mg_hierarchy.restrict_to_coarse(level)
        
        # ===== Compute coarse residual and FAS forcing =====
        coarse = self.mg_hierarchy.levels[level + 1]
        coarse.Q = coarse.bc.apply(coarse.Q)
        R_coarse = self._compute_residual_level(level + 1)
        
        # FAS forcing: P_c = R_restricted - R(Q_c)
        coarse.forcing = coarse.R - R_coarse
        
        # Store Q_old for correction computation
        coarse.Q_old[:] = coarse.Q
        
        # ===== Recurse to coarser level =====
        self._run_v_cycle(level + 1)
        
        # ===== Prolongate correction with relaxation =====
        # Store fine Q before correction
        lvl.Q_before_correction = lvl.Q.copy()
        self.mg_hierarchy.prolongate_correction(level + 1, 
                                                 use_injection=self.config.mg_use_injection)
        
        # Apply relaxation: Q = Q_old + omega * (Q_new - Q_old)
        omega = self.config.mg_omega
        if omega < 1.0:
            lvl.Q = lvl.Q_before_correction + omega * (lvl.Q - lvl.Q_before_correction)
        
        # === FIX: Make the CORRECTION periodic at wake cut ===
        # The prolongation adds different corrections at i=1 and i=NI because they
        # map to different coarse cells. We average the corrections at the wake edges
        # to prevent artificial discontinuity from accumulating.
        Q_int = lvl.Q[1:-1, 1:-1, :]  # Current state (interior)
        Q_before = lvl.Q_before_correction[1:-1, 1:-1, :]  # State before correction
        
        # Compute corrections at wake edges
        dQ_left = Q_int[0, :, :] - Q_before[0, :, :]
        dQ_right = Q_int[-1, :, :] - Q_before[-1, :, :]
        
        # Average the corrections and apply to both edges
        dQ_avg = 0.5 * (dQ_left + dQ_right)
        Q_int[0, :, :] = Q_before[0, :, :] + dQ_avg
        Q_int[-1, :, :] = Q_before[-1, :, :] + dQ_avg
        
        # Apply BC (will set ghost cells correctly)
        lvl.Q = lvl.bc.apply(lvl.Q)
        
        # ===== Post-smoothing =====
        for _ in range(self.config.mg_nu2):
            self._smooth_level(level)
    
    def _smooth_level(self, level: int):
        """
        Perform one smoothing step on a multigrid level.
        
        Uses RK4 with local timestepping.
        
        Parameters
        ----------
        level : int
            Multigrid level to smooth.
        """
        lvl = self.mg_hierarchy.levels[level]
        
        # Compute timestep if not already computed
        if level > 0:
            # For coarse levels, average timestep from fine (using Numba kernel)
            from src.numerics.multigrid import restrict_timestep
            fine = self.mg_hierarchy.levels[level - 1]
            restrict_timestep(fine.dt, lvl.dt)
        
        # Get forcing term (zero for finest, computed for coarse)
        forcing = lvl.forcing if level > 0 else None
        
        # RK smoothing step
        # - Finest level: 5-stage for accuracy
        # - Coarse levels: 3-stage for efficiency (just need to smooth)
        Q0 = lvl.Q.copy()
        Qk = lvl.Q.copy()
        
        if level == 0:
            alphas = [0.25, 0.166666667, 0.375, 0.5, 1.0]  # 5-stage Jameson
        else:
            alphas = [0.333333, 0.5, 1.0]  # 3-stage for coarse grids
        
        for alpha in alphas:
            Qk = lvl.bc.apply(Qk)
            
            # Compute residual with forcing
            R = self._compute_residual_level(level, Q=Qk, forcing=forcing)
            
            # Apply IRS only on finest level (coarse levels are more stable)
            if level == 0 and self.config.irs_epsilon > 0.0:
                apply_residual_smoothing(R, self.config.irs_epsilon)
            
            # Update
            Qk = Q0.copy()
            Qk[1:-1, 1:-1, :] += alpha * (lvl.dt / lvl.metrics.volume)[:, :, np.newaxis] * R
        
        lvl.Q = lvl.bc.apply(Qk)
    
    def _compute_residual_level(self, level: int, 
                                  Q: Optional[np.ndarray] = None,
                                  forcing: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute residual on a specific multigrid level.
        
        Parameters
        ----------
        level : int
            Multigrid level.
        Q : ndarray, optional
            State to use (default: level's current Q).
        forcing : ndarray, optional
            FAS forcing term.
            
        Returns
        -------
        R : ndarray, shape (NI, NJ, 4)
            Residual.
        """
        lvl = self.mg_hierarchy.levels[level]
        
        if Q is None:
            Q = lvl.Q
        
        # Build flux metrics for this level
        flux_metrics = FluxGridMetrics(
            Si_x=lvl.metrics.Si_x,
            Si_y=lvl.metrics.Si_y,
            Sj_x=lvl.metrics.Sj_x,
            Sj_y=lvl.metrics.Sj_y,
            volume=lvl.metrics.volume
        )
        
        # Build gradient metrics for this level
        grad_metrics = GradientMetrics(
            Si_x=lvl.metrics.Si_x,
            Si_y=lvl.metrics.Si_y,
            Sj_x=lvl.metrics.Sj_x,
            Sj_y=lvl.metrics.Sj_y,
            volume=lvl.metrics.volume
        )
        
        return self._compute_residual(Q, forcing, flux_metrics, grad_metrics)
    
    def run_steady_state(self) -> bool:
        """
        Run steady-state simulation to convergence.
        
        Returns
        -------
        converged : bool
            True if residual dropped below tolerance.
        """
        print(f"\n{'='*60}")
        print("Starting Steady-State Iteration")
        print(f"{'='*60}")
        print(f"{'Iter':>8} {'Residual':>14} {'CFL':>8}  "
              f"{'CL':>8} {'CD':>8} {'(p:':>7} {'f:)':>7}")
        print(f"{'-'*72}")
        
        # Initial residual for normalization
        initial_residual = None
        
        for n in range(self.config.max_iter):
            # Perform one step (use multigrid if enabled)
            if self.config.use_multigrid:
                res_rms, _ = self.step_multigrid()
            else:
                res_rms, _ = self.step()
            
            # Store history
            self.residual_history.append(res_rms)
            
            # Store initial residual
            if initial_residual is None:
                initial_residual = res_rms
            
            # Normalized residual
            res_norm = res_rms / (initial_residual + 1e-30)
            
            # Current CFL
            cfl = self._get_cfl(self.iteration)
            
            # Print progress
            if self.iteration % self.config.print_freq == 0 or self.iteration == 1:
                # Compute forces for diagnostics
                forces = self.compute_forces()
                print(f"{self.iteration:>8d} {res_rms:>14.6e} {cfl:>8.2f}  "
                      f"CL={forces.CL:>8.4f} CD={forces.CD:>8.5f} "
                      f"(p:{forces.CD_p:>7.5f} f:{forces.CD_f:>7.5f})")
            
            # Write VTK output with surface Cp and Cf
            if self.iteration % self.config.output_freq == 0:
                surface_fields = self._compute_surface_fields()
                self.vtk_writer.write(
                    self.Q, iteration=self.iteration,
                    additional_scalars=surface_fields
                )
            
            # Check convergence
            if res_rms < self.config.tol:
                self.converged = True
                print(f"\n{'='*60}")
                print(f"CONVERGED at iteration {self.iteration}")
                print(f"Final residual: {res_rms:.6e}")
                print(f"{'='*60}")
                break
            
            # Check for divergence (residual increased by factor of 1000)
            if res_rms > 1000 * initial_residual:
                print(f"\n{'='*60}")
                print(f"DIVERGED at iteration {self.iteration}")
                print(f"Residual: {res_rms:.6e} (initial: {initial_residual:.6e})")
                print(f"{'='*60}")
                break
        
        else:
            # Max iterations reached
            print(f"\n{'='*60}")
            print(f"Maximum iterations ({self.config.max_iter}) reached")
            print(f"Final residual: {self.residual_history[-1]:.6e}")
            print(f"{'='*60}")
        
        # Write final solution with surface data
        surface_fields = self._compute_surface_fields()
        self.vtk_writer.write(
            self.Q, iteration=self.iteration,
            additional_scalars=surface_fields
        )
        series_file = self.vtk_writer.finalize()
        print(f"VTK series written to: {series_file}")
        
        return self.converged
    
    def get_surface_data(self) -> Dict[str, np.ndarray]:
        """
        Extract surface quantities for post-processing.
        
        Returns
        -------
        data : dict
            Dictionary containing:
            - 'x': x-coordinates along surface
            - 'y': y-coordinates along surface
            - 'cp': pressure coefficient
            - 'cf': skin friction coefficient (approximate)
        """
        # Surface is at j=0, use first interior cell (j=1 in ghost array)
        # Cell-centered values
        p_surface = self.Q[1:-1, 1, 0]  # Pressure at first interior cell layer
        u_surface = self.Q[1:-1, 1, 1]
        v_surface = self.Q[1:-1, 1, 2]
        
        # Surface coordinates (average of nodes)
        x_surface = 0.5 * (self.X[:-1, 0] + self.X[1:, 0])
        y_surface = 0.5 * (self.Y[:-1, 0] + self.Y[1:, 0])
        
        # Pressure coefficient (Cp = (p - p_inf) / (0.5 * rho * V_inf^2))
        # For incompressible with unit velocity: Cp = 2 * (p - p_inf)
        V_inf_sq = self.freestream.u_inf**2 + self.freestream.v_inf**2
        cp = 2.0 * (p_surface - self.freestream.p_inf) / (V_inf_sq + 1e-12)
        
        # Approximate skin friction from wall-adjacent velocity
        # Cf = tau_w / (0.5 * rho * V_inf^2)
        # tau_w ≈ mu * du/dy ≈ mu * u_cell / (0.5 * cell_height)
        # For unit Reynolds number normalization: mu = 1/Re
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
        """
        Compute aerodynamic force coefficients.
        
        Returns
        -------
        forces : AeroForces
            Named tuple with CL, CD, CD_p, CD_f, etc.
        """
        V_inf = np.sqrt(self.freestream.u_inf**2 + self.freestream.v_inf**2)
        mu_laminar = 1.0 / self.config.reynolds if self.config.reynolds > 0 else 0.0
        
        # Turbulent viscosity (from SA variable nu_tilde)
        # For laminar flow, set to None
        mu_turb = None
        
        # Get n_wake from config or BC handler
        n_wake = getattr(self.config, 'n_wake', 0)
        if n_wake == 0 and hasattr(self, 'bc'):
            n_wake = getattr(self.bc, 'n_wake_points', 0)
        
        forces = compute_aerodynamic_forces(
            Q=self.Q,
            metrics=self.flux_metrics,
            mu_laminar=mu_laminar,
            mu_turb=mu_turb,
            alpha_deg=self.config.alpha,
            chord=1.0,  # Assume unit chord
            rho_inf=1.0,  # AC formulation assumes unit density
            V_inf=V_inf,
            n_wake=n_wake,
        )
        
        return forces
    
    def get_surface_distributions(self):
        """
        Get surface Cp and Cf distributions.
        
        Returns
        -------
        SurfaceData
            Named tuple with x, y, Cp, Cf arrays.
        """
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
        """
        Run steady-state simulation with enhanced diagnostic output.
        
        This method provides detailed diagnostic information including:
        - Flow field visualizations at specified intervals
        - Total pressure loss (entropy) monitoring
        - Max residual location tracking
        - Divergence history for debugging
        
        Parameters
        ----------
        dump_freq : int, optional
            Frequency of diagnostic dumps. If None, uses config.diagnostic_freq.
            
        Returns
        -------
        converged : bool
            True if residual dropped below tolerance.
        """
        # Lazy import plotting to avoid matplotlib dependency issues
        from ..io.plotting import plot_flow_field, plot_residual_history, plot_multigrid_levels
        
        if dump_freq is None:
            dump_freq = getattr(self.config, 'diagnostic_freq', 100)
        
        # Create snapshot directory
        snapshot_dir = Path(self.config.output_dir) / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Solution history buffer for divergence analysis
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
        
        # Dump initial state
        Q_int = self.Q[1:-1, 1:-1, :]
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
            # Use multigrid if enabled
            if self.config.use_multigrid:
                res_rms, R_field = self.step_multigrid()
            else:
                res_rms, R_field = self.step()
            self.residual_history.append(res_rms)
            
            if initial_residual is None:
                initial_residual = res_rms
            
            # Compute diagnostics
            bounds = compute_solution_bounds(self.Q)
            res_stats = compute_residual_statistics(R_field)
            cfl = self._get_cfl(self.iteration)
            
            # Update divergence history buffer
            if div_history_size > 0:
                solution_history.append((self.iteration, self.Q.copy(), R_field.copy()))
                if len(solution_history) > div_history_size:
                    solution_history.pop(0)
            
            # Print progress every 10 iterations
            if self.iteration % 10 == 0 or self.iteration == 1:
                p_range = f"[{bounds['p_min']:.2f}, {bounds['p_max']:.2f}]"
                max_loc = f"({res_stats['max_loc'][0]:3d},{res_stats['max_loc'][1]:3d})"
                print(f"{self.iteration:>8d} {res_rms:>12.4e} {res_stats['max_p']:>12.4e} "
                      f"{max_loc:>18} {cfl:>8.2f} {bounds['vel_max']:>10.4f} {p_range:>18}")
            
            # Dump flow field at specified frequency
            if self.iteration % dump_freq == 0:
                Q_int = self.Q[1:-1, 1:-1, :]
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
        Q_int = self.Q[1:-1, 1:-1, :]
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
                Q_int = Q[1:-1, 1:-1, :]
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

