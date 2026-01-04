"""
HTML dashboard component builders for Plotly-based CFD visualization.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from ._array_utils import sanitize_array, safe_minmax, safe_absmax, to_json_safe_list

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from .plotter import Snapshot


@dataclass
class ColorAxisConfig:
    """Configuration for a color axis."""
    colorscale: str
    cmin: float
    cmax: float
    colorbar: Dict[str, Any]


def compute_color_ranges(
    snapshots: List['Snapshot'],
    nu_laminar: float,
    has_cpt: bool,
    has_res_field: bool,
) -> Dict[str, ColorAxisConfig]:
    """Compute global color ranges for consistent scaling across snapshots.
    
    Parameters
    ----------
    snapshots : List[Snapshot]
        All snapshots to consider for range calculation.
    nu_laminar : float
        Laminar viscosity for chi calculation.
    has_cpt : bool
        Whether C_pt field is available.
    has_res_field : bool
        Whether residual field is available.
        
    Returns
    -------
    Dict[str, ColorAxisConfig]
        Color axis configurations for each field type.
    """
    # Pressure: symmetric around 0
    p_abs_max = max(safe_absmax(s.p, default=1.0) for s in snapshots)
    if has_cpt:
        cpt_abs_max = max(
            safe_absmax(s.C_pt, default=1.0)
            for s in snapshots if s.C_pt is not None
        )
        p_abs_max = max(p_abs_max, cpt_abs_max)
    p_abs_max = max(p_abs_max, 1e-6)
    
    # Velocity: symmetric around 0
    vel_abs_max = max(
        max(safe_absmax(s.u, default=1.0), safe_absmax(s.v, default=1.0))
        for s in snapshots
    )
    vel_abs_max = max(vel_abs_max, 1e-6)
    
    # Residual: 3 orders of magnitude from max
    if has_res_field:
        res_vals = []
        for s in snapshots:
            if s.residual_field is not None:
                sanitized = sanitize_array(s.residual_field, fill_value=1e-12)
                res_vals.append(np.log10(sanitized + 1e-12))
        if res_vals:
            res_max = max(safe_minmax(v, default_min=-12, default_max=0)[1] for v in res_vals)
        else:
            res_max = 0
        res_min = res_max - 3
    else:
        res_min, res_max = -3, 0
    
    # Chi range (log scale)
    chi_max_vals = []
    chi_min_vals = []
    for s in snapshots:
        nu_pos = s.nu[s.nu > 0]
        if len(nu_pos) > 0:
            chi_pos = nu_pos / nu_laminar
            chi_log = np.log10(chi_pos)
            min_val, max_val = safe_minmax(chi_log, default_min=-6, default_max=6)
            chi_max_vals.append(max_val)
            chi_min_vals.append(min_val)
    
    chi_max = max(chi_max_vals) if chi_max_vals else 6.0
    chi_min = min(chi_min_vals) if chi_min_vals else -6.0
    
    return {
        'pressure': ColorAxisConfig(
            colorscale='RdBu_r', cmin=-p_abs_max, cmax=p_abs_max,
            colorbar=dict(title='Δp', len=0.18, y=0.90, x=1.02, tickformat='.2f')
        ),
        'velocity': ColorAxisConfig(
            colorscale='RdBu', cmin=-vel_abs_max, cmax=vel_abs_max,
            colorbar=dict(title='Δvel', len=0.18, y=0.68, x=1.02, tickformat='.2f')
        ),
        'residual': ColorAxisConfig(
            colorscale='Hot', cmin=res_min, cmax=res_max,
            colorbar=dict(title='log₁₀(R)', len=0.18, y=0.46, x=1.02, tickformat='.1f')
        ),
        'chi': ColorAxisConfig(
            colorscale='Viridis', cmin=chi_min, cmax=chi_max,
            colorbar=dict(title='log₁₀(χ)', len=0.18, y=0.24, x=1.02, tickformat='.1f')
        ),
        'wall_dist': ColorAxisConfig(
            colorscale='Viridis', cmin=0, cmax=1,
            colorbar=dict(title='d/c', len=0.12, y=0.06, x=1.02, tickformat='.2f')
        ),
    }


def compute_chi_log(nu_data: np.ndarray, nu_laminar: float) -> np.ndarray:
    """Compute log10(chi) where chi = nuHat / nu_laminar.
    
    Returns NaN for negative/zero nuHat values.
    """
    nu_sanitized = sanitize_array(nu_data, fill_value=0.0)
    chi_log = np.full_like(nu_sanitized, np.nan)
    pos_mask = nu_sanitized > 0
    if np.any(pos_mask):
        chi_log[pos_mask] = np.log10(nu_sanitized[pos_mask] / nu_laminar)
    return chi_log


def compute_cf_distribution(
    snapshot: 'Snapshot',
    volume: np.ndarray,
    Sj_x: np.ndarray,
    Sj_y: np.ndarray,
    mu_laminar: float,
    u_inf: float,
    v_inf: float,
    i_start: int,
    i_end: int,
) -> np.ndarray:
    """Compute skin friction coefficient distribution on airfoil surface.
    
    Parameters
    ----------
    snapshot : Snapshot
        Solution snapshot.
    volume : np.ndarray
        Cell volumes at wall (NI,).
    Sj_x, Sj_y : np.ndarray
        J-face normal components at wall (NI,).
    mu_laminar : float
        Laminar viscosity.
    u_inf, v_inf : float
        Freestream velocity.
    i_start, i_end : int
        Airfoil surface indices (excluding wake).
        
    Returns
    -------
    np.ndarray
        Cf distribution on airfoil surface.
    """
    max_safe = 1e10
    V_inf_sq = u_inf**2 + v_inf**2
    q_inf = max(0.5 * V_inf_sq, 1e-14)
    
    Sj_mag = np.sqrt(Sj_x**2 + Sj_y**2)
    dy = volume / (Sj_mag + 1e-14)
    
    u_wall = np.clip(snapshot.u[:, 0] + u_inf, -max_safe, max_safe)
    v_wall = np.clip(snapshot.v[:, 0] + v_inf, -max_safe, max_safe)
    nu_wall = np.clip(snapshot.nu[:, 0], 0.0, max_safe)
    mu_eff = mu_laminar + np.maximum(0.0, nu_wall)
    
    dudn = 2.0 * u_wall / dy
    dvdn = 2.0 * v_wall / dy
    tau_mag = mu_eff * np.sqrt(dudn**2 + dvdn**2)
    Cf_full = tau_mag / q_inf
    
    return sanitize_array(Cf_full[i_start:i_end], fill_value=0.0)


def make_log_range_slider_steps(
    base_val: float,
    trace_indices: List[int],
    is_residual: bool = False,
    n_steps: int = 50,
) -> List[Dict[str, Any]]:
    """Create continuous logarithmic slider steps (3 orders of magnitude).
    
    Parameters
    ----------
    base_val : float
        Base value for range calculation.
    trace_indices : List[int]
        Plotly trace indices to update.
    is_residual : bool
        If True, treat as log-scale residual field.
    n_steps : int
        Number of slider steps.
        
    Returns
    -------
    List[Dict]
        Slider step configurations.
    """
    steps = []
    log_factors = np.linspace(-1.5, 1.5, n_steps)
    
    for log_f in log_factors:
        if is_residual:
            new_max = base_val + log_f
            new_min = new_max - 3.0
            label = f'{new_max:.1f}'
            steps.append(dict(
                method='restyle',
                args=[{'zmin': new_min, 'zmax': new_max, 'ncontours': 50}, trace_indices],
                label=label,
            ))
        else:
            factor = 10 ** log_f
            new_range = base_val * factor
            steps.append(dict(
                method='restyle',
                args=[{'zmin': -new_range, 'zmax': new_range, 'ncontours': 50}, trace_indices],
                label=f'{new_range:.2e}',
            ))
    return steps


def extract_residual_value(residual: Any, index: int = 0) -> Optional[float]:
    """Extract a single residual value from various formats.
    
    Parameters
    ----------
    residual : Any
        Residual data (array, tuple, list, or scalar).
    index : int
        Index to extract (for multi-component residuals).
        
    Returns
    -------
    float or None
        Extracted value, or None if invalid.
    """
    if isinstance(residual, np.ndarray) and len(residual) > index:
        val = residual[index]
    elif isinstance(residual, (tuple, list)) and len(residual) > index:
        val = residual[index]
    else:
        val = residual
    return val if np.isfinite(val) else None
