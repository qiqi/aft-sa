"""
Configuration module for RANS solver.

Provides YAML-based configuration with dataclass schema.
"""

from .schema import (
    SimulationConfig,
    GridConfig,
    FlowConfig,
    SolverSettings,
    NumericsConfig,
    OutputConfig,
    CFLConfig,
    super_coarse_preset,
    coarse_preset,
    production_preset,
)

from .loader import (
    load_yaml,
    from_dict,
    apply_cli_overrides,
    save_yaml,
)

__all__ = [
    # Schema classes
    'SimulationConfig',
    'GridConfig',
    'FlowConfig',
    'SolverSettings',
    'NumericsConfig',
    'OutputConfig',
    'CFLConfig',
    # Presets
    'super_coarse_preset',
    'coarse_preset',
    'production_preset',
    # Loader functions
    'load_yaml',
    'from_dict',
    'apply_cli_overrides',
    'save_yaml',
]
