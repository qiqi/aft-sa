"""
Configuration schema for RANS solver.

Dataclass-based configuration that can be loaded from YAML or constructed programmatically.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Union, Dict, Any


@dataclass
class GridConfig:
    """Grid generation configuration."""
    
    airfoil: str = "data/naca0012.dat"
    n_surface: int = 257       # Surface nodes (cells = nodes - 1)
    n_wake: int = 64           # Wake region nodes
    y_plus: float = 1.0        # Target y+ for first cell
    gradation: float = 1.2     # Wall-normal cell growth ratio (1.1-1.3 recommended)
    farfield_radius: float = 15.0  # Farfield distance in chords


@dataclass
class CFLConfig:
    """CFL ramping configuration."""
    
    initial: float = 0.1       # Starting CFL
    final: float = 3.0         # Target CFL
    ramp_iter: int = 300       # Iterations to ramp from initial to final


@dataclass 
class FlowConfig:
    """Flow conditions configuration."""
    
    reynolds: float = 6.0e6
    mach: float = 0.0          # 0 = incompressible
    
    # Alpha can be:
    # - Single value: alpha: 4.0
    # - Sweep spec: alpha: {sweep: [-5, 15, 21]}  (start, end, count)
    # - Explicit list: alpha: {values: [0, 2, 4, 6, 8]}
    alpha: Union[float, Dict[str, Any]] = 0.0
    
    # Initial/farfield turbulent viscosity ratio: χ = ν̃/ν
    # Typical values: 3-5 for external aerodynamics
    chi_inf: float = 3.0
    
    def is_batch(self) -> bool:
        """Check if this is a batch configuration."""
        return isinstance(self.alpha, dict)
    
    def get_batch_size(self) -> int:
        """Get number of cases in batch."""
        from src.solvers.batch import expand_parameter
        return len(expand_parameter(self.alpha))
    
    def get_alpha_values(self) -> List[float]:
        """Get all alpha values (single or expanded sweep)."""
        from src.solvers.batch import expand_parameter
        return expand_parameter(self.alpha)


@dataclass
class SolverSettings:
    """Solver iteration settings."""
    
    max_iter: int = 10000
    tol: float = 1e-10
    print_freq: int = 10
    diagnostic_freq: int = 100
    cfl: CFLConfig = field(default_factory=CFLConfig)


@dataclass
class NumericsConfig:
    """Numerical scheme configuration."""
    
    jst_k4: float = 0.04       # 4th-order dissipation coefficient
    beta: float = 10.0         # Artificial compressibility parameter
    wall_damping_length: float = 0.1  # Wall damping length scale
    sponge_thickness: int = 15  # Sponge layer thickness in cells for farfield stabilization


@dataclass
class OutputConfig:
    """Output configuration."""
    
    directory: str = "output/solver"
    case_name: str = "solution"
    html_animation: bool = True
    divergence_history: int = 0  # Solutions to keep for divergence viz


@dataclass
class DeviceConfig:
    """Device/GPU configuration."""
    
    # Device selection: "auto", "cpu", or GPU index ("0", "1", "cuda:0", etc.)
    device: Optional[str] = "auto"


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    
    grid: GridConfig = field(default_factory=GridConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    solver: SolverSettings = field(default_factory=SolverSettings)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    
    def to_solver_config(self):
        """Convert to legacy SolverConfig for backward compatibility."""
        from src.solvers.rans_solver import SolverConfig
        
        return SolverConfig(
            mach=self.flow.mach,
            alpha=self.flow.alpha,
            reynolds=self.flow.reynolds,
            chi_inf=self.flow.chi_inf,
            beta=self.numerics.beta,
            cfl_start=self.solver.cfl.initial,
            cfl_target=self.solver.cfl.final,
            cfl_ramp_iters=self.solver.cfl.ramp_iter,
            max_iter=self.solver.max_iter,
            tol=self.solver.tol,
            diagnostic_freq=self.solver.diagnostic_freq,
            print_freq=self.solver.print_freq,
            output_dir=self.output.directory,
            case_name=self.output.case_name,
            wall_damping_length=self.numerics.wall_damping_length,
            jst_k4=self.numerics.jst_k4,
            sponge_thickness=self.numerics.sponge_thickness,
            n_wake=self.grid.n_wake,
            html_animation=self.output.html_animation,
            divergence_history=self.output.divergence_history,
            target_yplus=self.grid.y_plus,
        )
    
    def to_dict(self) -> dict:
        """Convert to nested dictionary."""
        return asdict(self)


# Preset configurations
def super_coarse_preset() -> GridConfig:
    """Super-coarse grid for fast testing."""
    return GridConfig(
        n_surface=65,
        n_wake=32,
        y_plus=5.0,
        gradation=1.5,  # Coarser stretching for speed
    )


def coarse_preset() -> GridConfig:
    """Coarse grid for debugging."""
    return GridConfig(
        n_surface=193,
        n_wake=32,
        y_plus=1.0,
        gradation=1.3,
    )


def production_preset() -> GridConfig:
    """Production grid for accurate results."""
    return GridConfig(
        n_surface=385,
        n_wake=64,
        y_plus=1.0,
        gradation=1.15,  # Fine stretching for accuracy
    )
