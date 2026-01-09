"""
YAML configuration loader with validation.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Union
from dataclasses import fields, is_dataclass

from .schema import (
    SimulationConfig, GridConfig, FlowConfig, SolverSettings,
    NumericsConfig, OutputConfig, CFLConfig,
    super_coarse_preset, coarse_preset, production_preset,
)


def _merge_dict(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _coerce_type(value, field_type):
    """Coerce value to the expected field type."""
    # Handle string representations of numbers (e.g., "6.0e6")
    if field_type == float and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    if field_type == int and isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


def _dict_to_dataclass(cls, data: dict):
    """Convert a nested dictionary to a dataclass instance."""
    if not is_dataclass(cls):
        return data
    
    field_types = {f.name: f.type for f in fields(cls)}
    kwargs = {}
    
    for key, value in data.items():
        if key not in field_types:
            continue  # Skip unknown fields
        
        field_type = field_types[key]
        
        # Handle nested dataclasses
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(field_type, value)
        else:
            # Coerce types for primitive values
            kwargs[key] = _coerce_type(value, field_type)
    
    return cls(**kwargs)


def load_yaml(path: Union[str, Path]) -> SimulationConfig:
    """
    Load simulation configuration from a YAML file.
    
    Args:
        path: Path to YAML configuration file
        
    Returns:
        SimulationConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path) as f:
        data = yaml.safe_load(f)
    
    if data is None:
        data = {}
    
    return from_dict(data)


def from_dict(data: Dict[str, Any]) -> SimulationConfig:
    """
    Create SimulationConfig from a dictionary.
    
    Handles nested structures and applies defaults for missing values.
    """
    # Check for preset
    preset = data.pop('preset', None)
    if preset:
        grid_preset = {
            'super-coarse': super_coarse_preset(),
            'coarse': coarse_preset(),
            'production': production_preset(),
        }.get(preset)
        if grid_preset:
            # Merge preset with any explicit grid overrides
            grid_data = data.get('grid', {})
            preset_dict = {f.name: getattr(grid_preset, f.name) for f in fields(GridConfig)}
            data['grid'] = _merge_dict(preset_dict, grid_data)
    
    # Build config section by section
    config_dict = {}
    
    # Grid config
    if 'grid' in data:
        config_dict['grid'] = _dict_to_dataclass(GridConfig, data['grid'])
    
    # Flow config (special handling for alpha sweep specs)
    if 'flow' in data:
        flow_data = data['flow'].copy()
        # Don't coerce alpha if it's a dict (sweep spec) - pass through as-is
        alpha_val = flow_data.get('alpha', 0.0)
        if isinstance(alpha_val, dict):
            # Keep sweep/values spec as dict
            pass
        elif isinstance(alpha_val, str):
            # Coerce string to float for single values
            flow_data['alpha'] = float(alpha_val)
        config_dict['flow'] = FlowConfig(**{
            k: (_coerce_type(v, float) if k != 'alpha' and isinstance(v, str) else v)
            for k, v in flow_data.items()
        })
    
    # Solver settings (with nested CFL)
    if 'solver' in data:
        solver_data = data['solver'].copy()
        if 'cfl' in solver_data and isinstance(solver_data['cfl'], dict):
            solver_data['cfl'] = _dict_to_dataclass(CFLConfig, solver_data['cfl'])
        config_dict['solver'] = _dict_to_dataclass(SolverSettings, solver_data)
    
    # Numerics config
    if 'numerics' in data:
        numerics_data = data['numerics'].copy()
        # Remove legacy smoothing config if present
        numerics_data.pop('smoothing', None)
        config_dict['numerics'] = _dict_to_dataclass(NumericsConfig, numerics_data)
    
    # Output config
    if 'output' in data:
        config_dict['output'] = _dict_to_dataclass(OutputConfig, data['output'])
    
    return SimulationConfig(**config_dict)


def apply_cli_overrides(config: SimulationConfig, args) -> SimulationConfig:
    """
    Apply command-line argument overrides to a configuration.
    
    Only overrides values that were explicitly set (not default).
    
    Args:
        config: Base configuration
        args: argparse.Namespace with CLI arguments
        
    Returns:
        Updated SimulationConfig
    """
    # Convert to dict for easier manipulation
    config_dict = config.to_dict()
    
    # Map CLI args to config paths
    cli_mapping = {
        # Flow conditions
        'alpha': ('flow', 'alpha'),
        'reynolds': ('flow', 'reynolds'),
        'chi_inf': ('flow', 'chi_inf'),
        
        # Grid
        'n_surface': ('grid', 'n_surface'),
        'n_normal': ('grid', 'n_normal'),
        'n_wake': ('grid', 'n_wake'),
        'y_plus': ('grid', 'y_plus'),
        'wake_fan_factor': ('grid', 'wake_fan_factor'),
        'wake_fan_k': ('grid', 'wake_fan_k'),
        
        # Solver
        'max_iter': ('solver', 'max_iter'),
        'tol': ('solver', 'tol'),
        'print_freq': ('solver', 'print_freq'),
        'diagnostic_freq': ('solver', 'diagnostic_freq'),
        'cfl': ('solver', 'cfl', 'final'),
        'cfl_start': ('solver', 'cfl', 'initial'),
        'cfl_ramp': ('solver', 'cfl', 'ramp_iter'),
        
        # Numerics
        'beta': ('numerics', 'beta'),
        'jst_k4': ('numerics', 'jst_k4'),
        
        # Output
        'output_dir': ('output', 'directory'),
        'case_name': ('output', 'case_name'),
        'div_history': ('output', 'divergence_history'),
    }
    
    for cli_name, config_path in cli_mapping.items():
        if hasattr(args, cli_name):
            value = getattr(args, cli_name)
            if value is not None:
                # Navigate to the right nested dict
                target = config_dict
                for key in config_path[:-1]:
                    target = target[key]
                target[config_path[-1]] = value
    
    # Handle grid file override
    if hasattr(args, 'grid_file') and args.grid_file:
        config_dict['grid']['airfoil'] = args.grid_file
    
    return from_dict(config_dict)


def save_yaml(config: SimulationConfig, path: Union[str, Path]) -> None:
    """Save configuration to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
