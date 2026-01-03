"""
Solver Factory Module for standardized RANSSolver creation.
"""

from pathlib import Path
from typing import Optional, List
import numpy as np
import numpy.typing as npt

from .rans_solver import RANSSolver, SolverConfig
from ..grid.loader import load_or_generate_grid

NDArrayFloat = npt.NDArray[np.floating]


def create_solver(
    grid_file: str,
    mach: float = 0.15,
    alpha: float = 0.0,
    reynolds: float = 6e6,
    n_surface: int = 129,
    n_normal: int = 33,
    n_wake: int = 32,
    y_plus: float = 1.0,
    farfield_radius: float = 15.0,
    beta: float = 10.0,
    cfl_start: float = 0.1,
    cfl_target: float = 3.0,
    cfl_ramp_iters: int = 300,
    max_iter: int = 10000,
    tol: float = 1e-10,
    jst_k4: float = 0.04,
    irs_epsilon: float = 0.0,
    smoothing_type: str = "explicit",
    smoothing_epsilon: float = 0.2,
    smoothing_passes: int = 2,
    wall_damping_length: float = 0.1,
    output_freq: int = 100,
    print_freq: int = 10,
    output_dir: str = "output/solver",
    case_name: str = "airfoil",
    diagnostic_freq: int = 100,
    divergence_history: int = 0,
    project_root: Optional[Path] = None,
    verbose: bool = True,
) -> RANSSolver:
    """Create and initialize a RANSSolver with consistent settings."""
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent
    
    X: NDArrayFloat
    Y: NDArrayFloat
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
    
    config: SolverConfig = SolverConfig(
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
        smoothing_type=smoothing_type,
        smoothing_epsilon=smoothing_epsilon,
        smoothing_passes=smoothing_passes,
        n_wake=n_wake,
        diagnostic_freq=diagnostic_freq,
        divergence_history=divergence_history,
    )
    
    solver: RANSSolver = RANSSolver.__new__(RANSSolver)
    solver.config = config
    solver.X = X
    solver.Y = Y
    solver.NI = X.shape[0] - 1
    solver.NJ = X.shape[1] - 1
    solver.iteration = 0
    solver.residual_history = []
    solver.iteration_history = []  # Track which iteration each residual corresponds to
    solver.converged = False
    
    # Rolling buffer for divergence history
    from collections import deque
    div_history_size = config.divergence_history if config.divergence_history > 0 else 0
    solver._divergence_buffer = deque(maxlen=div_history_size) if div_history_size > 0 else None
    
    solver._compute_metrics()
    solver._initialize_state()
    solver._initialize_output()
    
    if verbose:
        print(f"\nGrid size: {solver.NI} x {solver.NJ} cells")
        print(f"Reynolds: {reynolds:.2e}")
        smooth_info: str = f"{smoothing_type} (ε={smoothing_epsilon}, {smoothing_passes} passes)" if smoothing_type != "none" else "none"
        print(f"CFL: {cfl_start} → {cfl_target} (ramp {cfl_ramp_iters} iters), Smoothing: {smooth_info}")
    
    return solver


def create_solver_quiet(
    X: NDArrayFloat,
    Y: NDArrayFloat,
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
    irs_epsilon: float = 0.0,
    smoothing_type: str = "explicit",
    smoothing_epsilon: float = 0.2,
    smoothing_passes: int = 2,
    wall_damping_length: float = 0.1,
    print_freq: int = 200,
) -> RANSSolver:
    """Create a solver with pre-loaded grid and minimal output."""
    config: SolverConfig = SolverConfig(
        mach=mach,
        alpha=alpha,
        reynolds=reynolds,
        beta=beta,
        cfl_start=cfl_start,
        cfl_target=cfl_target,
        cfl_ramp_iters=cfl_ramp_iters,
        max_iter=max_iter,
        tol=tol,
        output_freq=max_iter + 1,
        print_freq=print_freq,
        output_dir="output/validation",
        case_name="validation",
        wall_damping_length=wall_damping_length,
        jst_k4=jst_k4,
        irs_epsilon=irs_epsilon,
        smoothing_type=smoothing_type,
        smoothing_epsilon=smoothing_epsilon,
        smoothing_passes=smoothing_passes,
        n_wake=n_wake,
    )
    solver: RANSSolver = RANSSolver.__new__(RANSSolver)
    solver.config = config
    solver.X = X
    solver.Y = Y
    solver.NI = X.shape[0] - 1
    solver.NJ = X.shape[1] - 1
    solver.iteration = 0
    solver.residual_history = []
    solver.iteration_history = []  # Track which iteration each residual corresponds to
    solver.converged = False
    
    # Rolling buffer for divergence history (disabled for quiet mode)
    solver._divergence_buffer = None
    
    solver._compute_metrics()
    solver._initialize_state()
    
    class DummyVTKWriter:
        def write(self, *args: object, **kwargs: object) -> None: pass
        def finalize(self) -> str: return ""
    solver.vtk_writer = DummyVTKWriter()
    
    return solver

