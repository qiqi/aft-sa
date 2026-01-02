"""
Solver Factory Module.

Provides a standardized way to create and configure the RANSSolver,
ensuring consistent initialization across different scripts.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from .rans_solver import RANSSolver, SolverConfig
from ..grid.loader import load_or_generate_grid


def create_solver(
    grid_file: str,
    # Flow conditions
    mach: float = 0.15,
    alpha: float = 0.0,
    reynolds: float = 6e6,
    # Grid generation options
    n_surface: int = 129,
    n_normal: int = 33,
    n_wake: int = 32,
    y_plus: float = 1.0,
    farfield_radius: float = 15.0,
    # Solver settings
    beta: float = 10.0,
    cfl_start: float = 0.1,
    cfl_target: float = 3.0,
    cfl_ramp_iters: int = 300,
    max_iter: int = 10000,
    tol: float = 1e-10,
    # Numerics
    jst_k4: float = 0.04,
    irs_epsilon: float = 1.0,
    wall_damping_length: float = 0.1,
    # Multigrid options
    use_multigrid: bool = False,
    mg_levels: int = 4,
    mg_nu1: int = 2,
    mg_nu2: int = 2,
    mg_omega: float = 0.5,
    mg_dissipation_scaling: float = 2.0,
    mg_coarse_cfl: float = 0.5,
    # Output options
    output_freq: int = 100,
    print_freq: int = 10,
    output_dir: str = "output/solver",
    case_name: str = "airfoil",
    diagnostic_mode: bool = False,
    diagnostic_freq: int = 100,
    divergence_history: int = 0,
    # Project root for grid generation
    project_root: Optional[Path] = None,
    verbose: bool = True,
) -> RANSSolver:
    """
    Create and initialize a RANSSolver with consistent settings.
    
    This factory function ensures all scripts use the same initialization
    logic, avoiding divergence issues from inconsistent configurations.
    
    Parameters
    ----------
    grid_file : str
        Path to grid file (.p3d) or airfoil file (.dat)
    mach : float
        Mach number (default: 0.15)
    alpha : float
        Angle of attack in degrees (default: 0.0)
    reynolds : float
        Reynolds number (default: 6e6)
    n_surface : int
        Surface nodes for grid generation (default: 129)
    n_normal : int
        Normal nodes for grid generation (default: 33)
    n_wake : int
        Wake nodes for grid generation (default: 32)
    y_plus : float
        Target y+ for first cell (default: 1.0)
    farfield_radius : float
        Farfield radius in chord lengths (default: 15.0)
    beta : float
        Artificial compressibility parameter (default: 10.0)
    cfl_start : float
        Initial CFL for ramping (default: 0.1)
    cfl_target : float
        Target CFL number (default: 3.0)
    cfl_ramp_iters : int
        CFL ramp iterations (default: 300)
    max_iter : int
        Maximum iterations (default: 10000)
    tol : float
        Convergence tolerance (default: 1e-10)
    jst_k4 : float
        JST 4th-order dissipation coefficient (default: 0.04)
    irs_epsilon : float
        Implicit Residual Smoothing epsilon (default: 1.0)
    wall_damping_length : float
        Wall damping length for initialization (default: 0.1)
    use_multigrid : bool
        Enable FAS multigrid acceleration (default: False)
    mg_levels : int
        Maximum multigrid levels (default: 4)
    mg_nu1 : int
        Pre-smoothing iterations per level (default: 2)
    mg_nu2 : int
        Post-smoothing iterations per level (default: 2)
    mg_omega : float
        Multigrid correction relaxation factor (default: 0.5)
    mg_dissipation_scaling : float
        Dissipation scaling per coarse level (default: 2.0)
    mg_coarse_cfl : float
        CFL factor for coarse levels (default: 0.5)
    output_freq : int
        VTK output frequency (default: 100)
    print_freq : int
        Console print frequency (default: 10)
    output_dir : str
        Output directory (default: "output/solver")
    case_name : str
        Case name for output files (default: "airfoil")
    diagnostic_mode : bool
        Enable diagnostic mode (default: False)
    diagnostic_freq : int
        Diagnostic dump frequency (default: 100)
    divergence_history : int
        Solutions to keep for divergence visualization (default: 0)
    project_root : Path, optional
        Project root for grid generation
    verbose : bool
        Print progress messages (default: True)
        
    Returns
    -------
    RANSSolver
        Fully initialized solver ready for time-stepping
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent
    
    # Load or generate grid
    X, Y = load_or_generate_grid(
        grid_file,
        n_surface=n_surface,
        n_normal=n_normal,
        n_wake=n_wake,
        y_plus=y_plus,
        reynolds=reynolds,
        farfield_radius=farfield_radius,
        project_root=project_root,
        verbose=verbose
    )
    
    # Create solver configuration
    config = SolverConfig(
        mach=mach,
        alpha=alpha,
        reynolds=reynolds,
        beta=beta,
        cfl_start=cfl_start,
        cfl_target=cfl_target,
        cfl_ramp_iters=cfl_ramp_iters,
        max_iter=max_iter,
        tol=tol,
        output_freq=output_freq,
        print_freq=print_freq,
        output_dir=output_dir,
        case_name=case_name,
        wall_damping_length=wall_damping_length,
        jst_k4=jst_k4,
        irs_epsilon=irs_epsilon,
        n_wake=n_wake,
        diagnostic_mode=diagnostic_mode,
        diagnostic_freq=diagnostic_freq,
        divergence_history=divergence_history,
        # Multigrid options
        use_multigrid=use_multigrid,
        mg_levels=mg_levels,
        mg_nu1=mg_nu1,
        mg_nu2=mg_nu2,
        mg_omega=mg_omega,
        mg_dissipation_scaling=mg_dissipation_scaling,
        mg_coarse_cfl=mg_coarse_cfl,
    )
    
    # Create solver with pre-loaded grid
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
    
    # Initialize multigrid (set to None even if not used)
    solver.mg_hierarchy = None
    if config.use_multigrid:
        solver._initialize_multigrid()
    
    # Initialize output
    solver._initialize_output()
    
    if verbose:
        print(f"\nGrid size: {solver.NI} x {solver.NJ} cells")
        print(f"Reynolds: {reynolds:.2e}")
        mg_info = f" + Multigrid ({mg_levels} levels)" if use_multigrid else ""
        print(f"CFL: {cfl_start} → {cfl_target} (ramp {cfl_ramp_iters} iters), IRS ε={irs_epsilon}{mg_info}")
    
    return solver


def create_solver_quiet(
    X: np.ndarray,
    Y: np.ndarray,
    n_wake: int,
    alpha: float = 0.0,
    reynolds: float = 6e6,
    mach: float = 0.15,
    beta: float = 10.0,
    cfl_start: float = 0.1,
    cfl_target: float = 3.0,
    cfl_ramp_iters: int = 300,
    max_iter: int = 10000,
    tol: float = 1e-10,
    jst_k4: float = 0.04,
    irs_epsilon: float = 1.0,
    wall_damping_length: float = 0.1,
    print_freq: int = 200,
) -> RANSSolver:
    """
    Create a solver with pre-loaded grid and minimal output.
    
    Useful for validation scripts where you want to control output.
    
    Parameters
    ----------
    X, Y : ndarray
        Grid node coordinates
    n_wake : int
        Number of wake cells
    (other parameters same as create_solver)
        
    Returns
    -------
    RANSSolver
        Initialized solver with dummy VTK writer
    """
    config = SolverConfig(
        mach=mach,
        alpha=alpha,
        reynolds=reynolds,
        beta=beta,
        cfl_start=cfl_start,
        cfl_target=cfl_target,
        cfl_ramp_iters=cfl_ramp_iters,
        max_iter=max_iter,
        tol=tol,
        output_freq=max_iter + 1,  # Disable VTK output
        print_freq=print_freq,
        output_dir="output/validation",
        case_name="validation",
        wall_damping_length=wall_damping_length,
        jst_k4=jst_k4,
        irs_epsilon=irs_epsilon,
        n_wake=n_wake,
    )
    
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
    solver.mg_hierarchy = None
    
    # Dummy VTK writer for quiet operation
    class DummyVTKWriter:
        def write(self, *args, **kwargs): pass
        def finalize(self): return ""
    solver.vtk_writer = DummyVTKWriter()
    
    return solver

