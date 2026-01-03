"""
Plotly-based HTML animation for CFD results.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING, Dict, Any
from pathlib import Path
import warnings

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from ..constants import NGHOST


def _sanitize_array(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Replace NaN and Inf values with a finite fill value.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array that may contain NaN/Inf values.
    fill_value : float
        Value to replace NaN/Inf with (default: 0.0).
        
    Returns
    -------
    np.ndarray
        Array with all NaN/Inf values replaced.
    """
    result = np.array(arr, dtype=np.float64)
    mask = ~np.isfinite(result)
    if np.any(mask):
        result[mask] = fill_value
    return result


def _to_json_safe_list(arr: np.ndarray) -> list:
    """Convert numpy array to a list that's safe for JSON serialization.
    
    Converts NaN and Inf values to None (which becomes null in JSON).
    This is needed because Plotly's HTML serialization doesn't always
    handle numpy NaN values correctly.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array that may contain NaN/Inf values.
        
    Returns
    -------
    list
        List with NaN/Inf values replaced by None.
    """
    flat = arr.flatten()
    result = []
    for val in flat:
        if np.isfinite(val):
            result.append(float(val))
        else:
            result.append(None)
    return result


def _safe_minmax(arr: np.ndarray, default_min: float = -1.0, default_max: float = 1.0) -> tuple:
    """Get min/max of array, handling NaN/Inf gracefully.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array.
    default_min, default_max : float
        Default values if array is all NaN/Inf.
        
    Returns
    -------
    tuple
        (min_val, max_val) with finite values.
    """
    finite_vals = arr[np.isfinite(arr)]
    if len(finite_vals) == 0:
        return default_min, default_max
    return float(finite_vals.min()), float(finite_vals.max())


def _safe_absmax(arr: np.ndarray, default: float = 1.0) -> float:
    """Get max absolute value from array, handling NaN/Inf gracefully.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array.
    default : float
        Default value if array is all NaN/Inf.
        
    Returns
    -------
    float
        Maximum absolute finite value in array.
    """
    finite_vals = arr[np.isfinite(arr)]
    if len(finite_vals) == 0:
        return default
    return float(np.abs(finite_vals).max())

if TYPE_CHECKING:
    from ..grid.metrics import FVMMetrics


@dataclass
class Snapshot:
    """Stores a single timestep's solution data (interior cells only)."""
    iteration: int
    residual: np.ndarray  # 4-component: (p, u, v, nuHat) RMS residuals
    cfl: float
    p: np.ndarray  # Relative pressure (p - p_inf)
    u: np.ndarray  # Relative u-velocity (u - u_inf)
    v: np.ndarray  # Relative v-velocity (v - v_inf)
    nu: np.ndarray
    C_pt: Optional[np.ndarray] = None
    residual_field: Optional[np.ndarray] = None
    vel_max: float = 0.0
    p_min: float = 0.0
    p_max: float = 0.0


class PlotlyDashboard:
    """Accumulates CFD solution snapshots and exports as interactive HTML."""
    
    def __init__(self, reynolds: float = 6e6):
        self.snapshots: List[Snapshot] = []
        # residual_history: List of 4-tuples (p, u, v, nuHat) or legacy single floats
        self.residual_history: List = []
        self.iteration_history: List[int] = []  # Track iteration numbers for each residual
        self.divergence_snapshots: List[Snapshot] = []  # Snapshots captured during divergence
        self.p_inf: float = 0.0
        self.u_inf: float = 1.0
        self.v_inf: float = 0.0
        self.nu_laminar: float = 1.0 / reynolds if reynolds > 0 else 1e-6
    
    def store_snapshot(
        self, 
        Q: np.ndarray, 
        iteration: int, 
        residual_history: List,
        cfl: float = 0.0,
        C_pt: Optional[np.ndarray] = None,
        residual_field: Optional[np.ndarray] = None,
        freestream: Any = None,
        iteration_history: Optional[List[int]] = None,
        is_divergence_dump: bool = False,
    ) -> None:
        """Store current solution state with diagnostic data.
        
        NaN/Inf values are automatically sanitized to prevent HTML rendering issues.
        
        Parameters
        ----------
        Q : np.ndarray
            State array.
        iteration : int
            Current iteration number.
        residual_history : List
            List of residual values. Each entry can be:
            - A tuple/array of 4 values (p, u, v, nuHat) [preferred]
            - A single float [legacy]
        cfl : float
            Current CFL number.
        C_pt : np.ndarray, optional
            Total pressure loss coefficient.
        residual_field : np.ndarray, optional
            Spatial residual field.
        freestream : Any, optional
            Freestream conditions.
        iteration_history : List[int], optional
            List of iteration numbers corresponding to residual_history.
            If None, assumes consecutive iterations starting from 0.
        is_divergence_dump : bool
            If True, stores in divergence_snapshots instead of regular snapshots.
        """
        Q_int = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
        
        # Get residual - handle both 4-component and legacy single-value formats
        if residual_history:
            last_res = residual_history[-1]
            if isinstance(last_res, (tuple, list, np.ndarray)):
                residual = np.array([r if np.isfinite(r) else 1e10 for r in last_res])
            else:
                # Legacy: single float - expand to 4-component
                residual = np.array([last_res if np.isfinite(last_res) else 1e10] * 4)
        else:
            residual = np.zeros(4)
        
        # Extract freestream values
        if freestream is not None:
            self.p_inf = freestream.p_inf
            self.u_inf = freestream.u_inf
            self.v_inf = freestream.v_inf
        
        # Store RELATIVE values (p - p_inf, u - u_inf, v - v_inf)
        # Sanitize to replace NaN/Inf with 0
        p_rel = _sanitize_array(Q_int[:, :, 0] - self.p_inf, fill_value=0.0)
        u_rel = _sanitize_array(Q_int[:, :, 1] - self.u_inf, fill_value=0.0)
        v_rel = _sanitize_array(Q_int[:, :, 2] - self.v_inf, fill_value=0.0)
        nu = _sanitize_array(Q_int[:, :, 3], fill_value=0.0)
        
        # Clip to reasonable range to prevent overflow in calculations
        max_safe_vel = 1e10  # Prevents overflow when squared
        u_abs = _sanitize_array(Q_int[:, :, 1], fill_value=self.u_inf)
        v_abs = _sanitize_array(Q_int[:, :, 2], fill_value=self.v_inf)
        u_abs = np.clip(u_abs, -max_safe_vel, max_safe_vel)
        v_abs = np.clip(v_abs, -max_safe_vel, max_safe_vel)
        vel_mag = np.sqrt(u_abs**2 + v_abs**2)
        
        if C_pt is None and freestream is not None:
            p = _sanitize_array(Q_int[:, :, 0], fill_value=self.p_inf)
            p = np.clip(p, -max_safe_vel, max_safe_vel)
            V_inf_sq = self.u_inf**2 + self.v_inf**2
            p_total = p + 0.5 * (u_abs**2 + v_abs**2)
            p_total_inf = self.p_inf + 0.5 * V_inf_sq
            C_pt = (p_total_inf - p_total) / (0.5 * V_inf_sq + 1e-12)
            C_pt = _sanitize_array(C_pt, fill_value=0.0)
        elif C_pt is not None:
            C_pt = _sanitize_array(C_pt, fill_value=0.0)
        
        res_field = None
        if residual_field is not None:
            # Clip to prevent overflow in squaring
            res_clipped = np.clip(residual_field, -max_safe_vel, max_safe_vel)
            res_field = _sanitize_array(
                np.sqrt(np.mean(res_clipped**2, axis=2)), 
                fill_value=1e-12
            )
        
        # Use safe min/max to handle any remaining edge cases
        vel_max = _safe_absmax(vel_mag, default=1.0)
        p_min, p_max = _safe_minmax(p_rel, default_min=-1.0, default_max=1.0)
        
        snapshot = Snapshot(
            iteration=iteration,
            residual=residual,
            cfl=cfl,
            p=p_rel.copy(),
            u=u_rel.copy(),
            v=v_rel.copy(),
            nu=nu.copy(),
            C_pt=C_pt.copy() if C_pt is not None else None,
            residual_field=res_field,
            vel_max=float(vel_max),
            p_min=float(p_min),
            p_max=float(p_max),
        )
        
        if is_divergence_dump:
            self.divergence_snapshots.append(snapshot)
        else:
            self.snapshots.append(snapshot)
        
        # Store residual history with iteration numbers
        # Handle both 4-component and legacy single-value formats
        self.residual_history = []
        for r in residual_history:
            if isinstance(r, (tuple, list, np.ndarray)):
                # 4-component format
                self.residual_history.append(
                    np.array([v if np.isfinite(v) else 1e10 for v in r])
                )
            else:
                # Legacy single float - expand to 4-component
                val = r if np.isfinite(r) else 1e10
                self.residual_history.append(np.array([val, val, val, val]))
        
        if iteration_history is not None:
            self.iteration_history = list(iteration_history)
        elif len(self.iteration_history) != len(self.residual_history):
            # Default: assume consecutive iterations starting from 0
            self.iteration_history = list(range(len(self.residual_history)))
    
    def save_html(self, filename: str, grid_metrics: 'FVMMetrics', 
                  wall_distance: Optional[np.ndarray] = None,
                  X: Optional[np.ndarray] = None,
                  Y: Optional[np.ndarray] = None,
                  n_wake: int = 0,
                  mu_laminar: float = 1e-6) -> str:
        """Export all snapshots as interactive HTML animation.

        Parameters
        ----------
        filename : str
            Output filename for HTML file.
        grid_metrics : FVMMetrics
            Grid metrics containing cell centers.
        wall_distance : np.ndarray, optional
            Wall distance field (NI, NJ). If provided, adds wall distance plot.
        X, Y : np.ndarray, optional
            Grid coordinates (NI+1, NJ+1). If provided, adds Cp and Cf plots.
        n_wake : int
            Number of wake points on each end (airfoil surface is n_wake to NI-n_wake).
        mu_laminar : float
            Laminar viscosity for Cf computation.
        """
        if not HAS_PLOTLY:
            print("Warning: plotly not installed. Skipping HTML animation.")
            return ""
        
        if not self.snapshots:
            print("Warning: No snapshots to save.")
            return ""
        
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        xc = grid_metrics.xc
        yc = grid_metrics.yc
        
        ni, nj = xc.shape
        a = np.arange(ni)
        b = np.arange(nj)
        A, B = np.meshgrid(a, b, indexing='ij')
        
        # Use last snapshot for initial display (matches slider default position)
        snap0 = self.snapshots[0]  # For checking feature availability
        # snapN will be set after combining with divergence snapshots
        has_cpt = snap0.C_pt is not None
        has_res_field = snap0.residual_field is not None
        has_wall_dist = wall_distance is not None
        has_surface_data = X is not None and Y is not None
        
        # Combine regular snapshots with divergence snapshots for animation
        all_snapshots = list(self.snapshots)
        if self.divergence_snapshots:
            all_snapshots.extend(self.divergence_snapshots)
            # Sort by iteration to maintain chronological order
            all_snapshots.sort(key=lambda s: s.iteration)
        
        # Use last snapshot (including divergence) for initial display
        snapN = all_snapshots[-1]
        
        # Layout: Row 1-2: field pairs, Row 3: residual + chi, Row 4: convergence
        # Row 5 (optional): wall distance
        # Row 6 (optional): Cp and Cf
        n_rows = 4
        if has_wall_dist:
            n_rows += 1
        if has_surface_data:
            n_rows += 1
        
        subplot_titles = [
            'Pressure (p - p∞)', 
            'Total Pressure Loss (C_pt)' if has_cpt else 'Turbulent Viscosity (ν)',
            'U-velocity (u - u∞)', 'V-velocity (v - v∞)',
            'Residual Field (log₁₀)' if has_res_field else 'Velocity Magnitude',
            'χ = ν̃/ν (Turbulent/Laminar Viscosity Ratio)',
            'Convergence History',  # colspan=2, so None cell in row 4 col 2 is skipped
        ]
        
        if has_wall_dist:
            subplot_titles.append('Wall Distance (d/c)')
        if has_surface_data:
            subplot_titles.extend(['', ''])  # Cp and Cf use axis labels instead
        
        specs = [
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "scatter", "colspan": 2}, None],  # Convergence spans both columns
        ]
        if has_wall_dist:
            specs.append([{"type": "xy"}, None])  # Wall distance in left column only
        if has_surface_data:
            specs.append([{"type": "scatter"}, {"type": "scatter"}])  # Cp and Cf line plots
        
        fig = make_subplots(
            rows=n_rows, cols=2,
            subplot_titles=subplot_titles,
            specs=specs,
            horizontal_spacing=0.08,
            vertical_spacing=0.06,
        )
        
        field2_data = _sanitize_array(snapN.C_pt if has_cpt else snapN.nu, fill_value=0.0)
        
        if has_res_field and snapN.residual_field is not None:
            field5_data = np.log10(_sanitize_array(snapN.residual_field, fill_value=1e-12) + 1e-12)
        else:
            field5_data = np.zeros_like(snapN.p)
        
        # Chi = nu_hat / nu_laminar (turbulent/laminar viscosity ratio)
        # Use NaN for negative/zero nuHat values (will appear white/transparent in plot)
        nu_sanitized = _sanitize_array(snapN.nu, fill_value=0.0)
        # Compute chi only for positive values, use NaN otherwise
        chi_data = np.full_like(nu_sanitized, np.nan)
        pos_mask = nu_sanitized > 0
        chi_data[pos_mask] = nu_sanitized[pos_mask] / self.nu_laminar
        
        # Compute GLOBAL symmetric ranges for consistent scaling
        # Use safe functions to handle any NaN/Inf values
        
        # Pressure: symmetric around 0
        p_abs_max = max(
            _safe_absmax(s.p, default=1.0) for s in self.snapshots
        )
        if has_cpt:
            cpt_abs_max = max(
                _safe_absmax(s.C_pt, default=1.0)
                for s in self.snapshots if s.C_pt is not None
            )
            p_abs_max = max(p_abs_max, cpt_abs_max)
        
        # Ensure minimum range to avoid zero-width colorscale
        p_abs_max = max(p_abs_max, 1e-6)
        
        # Velocity: symmetric around 0
        vel_abs_max = max(
            max(_safe_absmax(s.u, default=1.0), _safe_absmax(s.v, default=1.0))
            for s in self.snapshots
        )
        vel_abs_max = max(vel_abs_max, 1e-6)
        
        # Residual: 3 orders of magnitude from max
        if has_res_field:
            res_vals = []
            for s in self.snapshots:
                if s.residual_field is not None:
                    sanitized = _sanitize_array(s.residual_field, fill_value=1e-12)
                    res_vals.append(np.log10(sanitized + 1e-12))
            if res_vals:
                res_max = max(_safe_minmax(v, default_min=-12, default_max=0)[1] for v in res_vals)
            else:
                res_max = 0
            res_min = res_max - 3  # 3 orders of magnitude
        else:
            res_min, res_max = -3, 0
        
        # Chi range (log scale for display)
        # Only consider positive nuHat values for range calculation
        chi_max_vals = []
        chi_min_vals = []
        for s in self.snapshots:
            nu_pos = s.nu[s.nu > 0]  # Only positive values
            if len(nu_pos) > 0:
                chi_pos = nu_pos / self.nu_laminar
                chi_log = np.log10(chi_pos)
                min_val, max_val = _safe_minmax(chi_log, default_min=-6, default_max=6)
                chi_max_vals.append(max_val)
                chi_min_vals.append(min_val)
        
        chi_max = max(chi_max_vals) if chi_max_vals else 6.0
        chi_min = min(chi_min_vals) if chi_min_vals else -6.0
        
        # Store ranges for sliders (will be used in layout)
        p_range = p_abs_max
        vel_range = vel_abs_max
        res_range_max = res_max
        
        # Sanitize initial display data
        snapN_p = _sanitize_array(snapN.p, fill_value=0.0)
        snapN_u = _sanitize_array(snapN.u, fill_value=0.0)
        snapN_v = _sanitize_array(snapN.v, fill_value=0.0)
        
        # Chi log scale: NaN values (negative nuHat) will appear white/transparent
        # Avoid log10(0) by only computing log for positive finite values
        chi_log_data = np.full_like(chi_data, np.nan)
        valid_mask = np.isfinite(chi_data) & (chi_data > 0)
        chi_log_data[valid_mask] = np.log10(chi_data[valid_mask])
        
        # Convert to JSON-safe list (NaN -> None) for proper serialization
        chi_log_safe = _to_json_safe_list(chi_log_data)

        contour_configs = [
            (snapN_p.flatten(), 1, 1, 'pressure', True),
            (field2_data.flatten(), 1, 2, 'pressure', False),
            (snapN_u.flatten(), 2, 1, 'velocity', True),
            (snapN_v.flatten(), 2, 2, 'velocity', False),
            (field5_data.flatten(), 3, 1, 'residual', True),
            (chi_log_safe, 3, 2, 'chi', True),  # Chi plot (log scale), None=white for ν̃<0
        ]
        
        # Colorbar positions - 4 colorbars, properly spaced
        coloraxis_config = {
            'pressure': {'colorscale': 'RdBu_r', 'cmin': -p_range, 'cmax': p_range, 
                         'colorbar': dict(title='Δp', len=0.18, y=0.90, x=1.02, 
                                          tickformat='.2f')},
            'velocity': {'colorscale': 'RdBu', 'cmin': -vel_range, 'cmax': vel_range,
                         'colorbar': dict(title='Δvel', len=0.18, y=0.68, x=1.02,
                                          tickformat='.2f')},
            'residual': {'colorscale': 'Hot', 'cmin': res_min, 'cmax': res_max,
                         'colorbar': dict(title='log₁₀(R)', len=0.18, y=0.46, x=1.02,
                                          tickformat='.1f')},
            'chi': {'colorscale': 'Viridis', 'cmin': chi_min, 'cmax': chi_max,
                    'colorbar': dict(title='log₁₀(χ)', len=0.18, y=0.24, x=1.02,
                                     tickformat='.1f')},
            'wall_dist': {'colorscale': 'Viridis', 'cmin': 0, 'cmax': 1,
                          'colorbar': dict(title='d/c', len=0.12, y=0.06, x=1.02,
                                           tickformat='.2f')},
        }
        
        for data, row, col, caxis_group, show_colorbar in contour_configs:
            carpet_id = f'carpet_{row}_{col}'
            caxis_cfg = coloraxis_config[caxis_group]
            
            fig.add_trace(
                go.Carpet(
                    a=A.flatten(), b=B.flatten(),
                    x=xc.flatten(), y=yc.flatten(),
                    carpet=carpet_id,
                    aaxis=dict(showgrid=False, showticklabels='none', showline=False),
                    baxis=dict(showgrid=False, showticklabels='none', showline=False),
                ),
                row=row, col=col
            )
            
            colorbar_cfg = caxis_cfg['colorbar'] if show_colorbar else None
            # data is already flattened (or a list for chi with None for NaN)
            fig.add_trace(
                go.Contourcarpet(
                    a=A.flatten(), b=B.flatten(), z=data,
                    carpet=carpet_id,
                    colorscale=caxis_cfg['colorscale'],
                    zmin=caxis_cfg['cmin'], zmax=caxis_cfg['cmax'],
                    contours=dict(coloring='fill', showlines=False),
                    ncontours=50,
                    colorbar=colorbar_cfg,
                    showscale=show_colorbar,
                ),
                row=row, col=col
            )
        
        # Convergence history: 4 lines for each equation
        eq_names = ['Pressure', 'U-velocity', 'V-velocity', 'ν̃ (nuHat)']
        eq_colors = ['blue', 'red', 'green', 'purple']
        
        if self.residual_history:
            # Use actual iteration numbers if available, otherwise use indices
            if self.iteration_history and len(self.iteration_history) == len(self.residual_history):
                all_iters = self.iteration_history
            else:
                all_iters = list(range(len(self.residual_history)))
            
            # Plot line for each equation
            for eq_idx in range(4):
                # Extract residual for this equation from history
                res_eq = []
                for r in self.residual_history:
                    if isinstance(r, np.ndarray) and len(r) > eq_idx:
                        val = r[eq_idx]
                    elif isinstance(r, (tuple, list)) and len(r) > eq_idx:
                        val = r[eq_idx]
                    else:
                        val = r  # Legacy: single value
                    res_eq.append(val if np.isfinite(val) else None)
                
                fig.add_trace(
                    go.Scatter(
                        x=all_iters, y=res_eq,
                        mode='lines', line=dict(color=eq_colors[eq_idx], width=1.5),
                        showlegend=True, name=eq_names[eq_idx],
                        legendgroup='convergence',
                    ),
                    row=4, col=1  # Convergence plot at row 4, spanning both columns
                )
        
        # Snapshot markers (pressure residual only, to avoid clutter)
        snapshot_iters = [s.iteration for s in self.snapshots]
        snapshot_res = []
        for s in self.snapshots:
            if isinstance(s.residual, np.ndarray) and len(s.residual) > 0:
                val = s.residual[0]  # Pressure
            else:
                val = s.residual
            snapshot_res.append(val if np.isfinite(val) else None)
        
        fig.add_trace(
            go.Scatter(
                x=snapshot_iters, y=snapshot_res,
                mode='markers',
                marker=dict(color='black', size=8, symbol='circle'),
                showlegend=True, name='Snapshots',
                legendgroup='convergence',
            ),
            row=4, col=1  # Convergence plot at row 4
        )
        
        # Divergence snapshots (orange triangles) - show if any divergence was captured
        if self.divergence_snapshots:
            div_iters = [s.iteration for s in self.divergence_snapshots]
            div_res = []
            for s in self.divergence_snapshots:
                if isinstance(s.residual, np.ndarray) and len(s.residual) > 0:
                    val = s.residual[0]  # Pressure
                else:
                    val = s.residual
                div_res.append(val if np.isfinite(val) else None)
            
            fig.add_trace(
                go.Scatter(
                    x=div_iters, y=div_res,
                    mode='markers',
                    marker=dict(color='orange', size=12, symbol='triangle-up'),
                    showlegend=True, name='Divergence',
                    legendgroup='convergence',
                ),
                row=4, col=1
            )
        
        # Wall distance plot (static, at end)
        wall_dist_row = 5
        wall_dist_traces_start = None
        if has_wall_dist:
            wall_dist_traces_start = len(fig.data)
            carpet_id = 'carpet_wall_dist'
            wd_cfg = coloraxis_config['wall_dist']
            
            # Sanitize wall distance data
            wall_distance_sanitized = _sanitize_array(wall_distance, fill_value=0.0)
            
            fig.add_trace(
                go.Carpet(
                    a=A.flatten(), b=B.flatten(),
                    x=xc.flatten(), y=yc.flatten(),
                    carpet=carpet_id,
                    aaxis=dict(showgrid=False, showticklabels='none', showline=False),
                    baxis=dict(showgrid=False, showticklabels='none', showline=False),
                ),
                row=wall_dist_row, col=1
            )
            
            fig.add_trace(
                go.Contourcarpet(
                    a=A.flatten(), b=B.flatten(), z=wall_distance_sanitized.flatten(),
                    carpet=carpet_id,
                    colorscale=wd_cfg['colorscale'],
                    zmin=wd_cfg['cmin'], zmax=wd_cfg['cmax'],
                    contours=dict(coloring='fill', showlines=False),
                    ncontours=50,
                    colorbar=wd_cfg['colorbar'],
                    showscale=True,
                ),
                row=wall_dist_row, col=1
            )
        
        # Cp and Cf surface plots (animated)
        surface_row = (wall_dist_row + 1) if has_wall_dist else 5
        cp_trace_idx = None
        cf_trace_idx = None
        x_surf_airfoil = None
        
        if has_surface_data:
            NI_cells = ni  # Number of cells in i-direction
            
            # Airfoil surface indices (excluding wake)
            i_start = n_wake
            i_end = NI_cells - n_wake
            
            # Surface x-coordinate (cell centers at j=0, airfoil only)
            x_surf_full = 0.5 * (X[:-1, 0] + X[1:, 0])  # Cell center x at wall
            x_surf_airfoil = x_surf_full[i_start:i_end]
            
            # Dynamic pressure for normalization
            V_inf_sq = self.u_inf**2 + self.v_inf**2
            q_inf = 0.5 * V_inf_sq
            if q_inf < 1e-14:
                q_inf = 1e-14
            
            # Compute Cp from last snapshot for initial display
            # snap.p[:, 0] = p_wall - p_inf at first interior cell
            Cp_full = snapN.p[:, 0] / q_inf
            Cp_airfoil = _sanitize_array(Cp_full[i_start:i_end], fill_value=0.0)
            
            # Compute Cf from last snapshot
            # Need wall shear: tau_w = mu * du/dy, where dy is distance from wall to cell center
            vol = grid_metrics.volume[:, 0]
            Sj_mag = np.sqrt(grid_metrics.Sj_x[:, 0]**2 + grid_metrics.Sj_y[:, 0]**2)
            dy = vol / (Sj_mag + 1e-14)  # Distance from wall to first cell center
            
            # Velocity at first interior (relative + freestream)
            # Clip to prevent overflow in squaring
            max_safe = 1e10
            u_wall = np.clip(snapN.u[:, 0] + self.u_inf, -max_safe, max_safe)
            v_wall = np.clip(snapN.v[:, 0] + self.v_inf, -max_safe, max_safe)
            
            # Effective viscosity (laminar + turbulent)
            nu_wall = np.clip(snapN.nu[:, 0], 0.0, max_safe)
            mu_eff = mu_laminar + np.maximum(0.0, nu_wall)
            
            # Wall shear stress magnitude (using 2*u/dy since u=0 at wall)
            dudn = 2.0 * u_wall / dy
            dvdn = 2.0 * v_wall / dy
            tau_mag = mu_eff * np.sqrt(dudn**2 + dvdn**2)
            Cf_full = tau_mag / q_inf
            Cf_airfoil = _sanitize_array(Cf_full[i_start:i_end], fill_value=0.0)
            
            # Compute global Cp range for consistent scaling
            cp_min_global, cp_max_global = -2.0, 1.0
            cf_max_global = 0.05
            for snap in all_snapshots:
                Cp_snap = np.clip(snap.p[:, 0], -max_safe, max_safe) / q_inf
                Cp_snap_airfoil = Cp_snap[i_start:i_end]
                cp_min_snap, cp_max_snap = _safe_minmax(Cp_snap_airfoil, -2.0, 1.0)
                cp_min_global = min(cp_min_global, cp_min_snap)
                cp_max_global = max(cp_max_global, cp_max_snap)
                
                u_snap = np.clip(snap.u[:, 0] + self.u_inf, -max_safe, max_safe)
                v_snap = np.clip(snap.v[:, 0] + self.v_inf, -max_safe, max_safe)
                nu_snap = np.clip(snap.nu[:, 0], 0.0, max_safe)
                mu_eff_snap = mu_laminar + np.maximum(0.0, nu_snap)
                dudn_snap = 2.0 * u_snap / dy
                dvdn_snap = 2.0 * v_snap / dy
                tau_snap = mu_eff_snap * np.sqrt(dudn_snap**2 + dvdn_snap**2)
                Cf_snap = tau_snap / q_inf
                Cf_snap_airfoil = Cf_snap[i_start:i_end]
                cf_max_snap = _safe_absmax(Cf_snap_airfoil, 0.01)
                cf_max_global = max(cf_max_global, cf_max_snap)
            
            # Add Cp trace (negative up convention: invert y-axis later)
            cp_trace_idx = len(fig.data)
            fig.add_trace(
                go.Scatter(
                    x=x_surf_airfoil, y=Cp_airfoil,
                    mode='lines', line=dict(color='blue', width=2),
                    showlegend=False, name='Cp',
                ),
                row=surface_row, col=1
            )
            
            # Add Cf trace
            cf_trace_idx = len(fig.data)
            fig.add_trace(
                go.Scatter(
                    x=x_surf_airfoil, y=Cf_airfoil,
                    mode='lines', line=dict(color='red', width=2),
                    showlegend=False, name='Cf',
                ),
                row=surface_row, col=2
            )
        
        # 6 contour plots: pressure, cpt, u, v, residual, chi
        # Each contour plot has a carpet (even index) and contourcarpet (odd index)
        # So contour indices are: 1, 3, 5, 7, 9, 11
        base_contour_indices = [1, 3, 5, 7, 9, 11]
        # After 6 contour plots (12 traces), we have:
        # - 4 lines (one per equation: p, u, v, nuHat)
        # - 1 black dots (snapshots)
        # - 1 orange triangles (divergence) if present
        # Then optionally wall distance traces (NOT animated)
        # Then optionally Cp and Cf traces (animated)
        # Snapshot marker is trace index: 12 + 4 (4 convergence lines) = 16
        residual_marker_idx = 12 + 4  # 12 contour traces + 4 convergence lines
        animated_indices = base_contour_indices + [residual_marker_idx]
        
        # Add Cp and Cf to animated indices if present
        if has_surface_data and cp_trace_idx is not None and cf_trace_idx is not None:
            animated_indices.append(cp_trace_idx)
            animated_indices.append(cf_trace_idx)
        
        chi_cfg = coloraxis_config['chi']
        
        # Create set of divergence iterations for fast lookup
        divergence_iters = {s.iteration for s in self.divergence_snapshots}
        
        frames = []
        for i, snap in enumerate(all_snapshots):
            # Sanitize all data for this frame
            snap_p = _sanitize_array(snap.p, fill_value=0.0)
            snap_u = _sanitize_array(snap.u, fill_value=0.0)
            snap_v = _sanitize_array(snap.v, fill_value=0.0)
            snap_nu = _sanitize_array(snap.nu, fill_value=1e-12)
            
            field2 = _sanitize_array(snap.C_pt if has_cpt and snap.C_pt is not None else snap.nu, fill_value=0.0)
            if has_res_field and snap.residual_field is not None:
                field5 = np.log10(_sanitize_array(snap.residual_field, fill_value=1e-12) + 1e-12)
            else:
                field5 = np.sqrt(snap_u**2 + snap_v**2)
            
            # Chi for this snapshot (log scale), NaN for negative/zero nuHat
            # Avoid log10(0) by only computing log for positive values
            snap_chi = np.full_like(snap_nu, np.nan)
            pos_mask = snap_nu > 0
            if np.any(pos_mask):
                snap_chi[pos_mask] = np.log10(snap_nu[pos_mask] / self.nu_laminar)
            
            p_cfg = coloraxis_config['pressure']
            r_cfg = coloraxis_config['residual']
            
            # Use JSON-safe conversion for chi (which can have NaN for negative nuHat)
            snap_chi_safe = _to_json_safe_list(snap_chi)
            
            frame_data = [
                go.Contourcarpet(a=A.flatten(), b=B.flatten(), z=snap_p.flatten(), 
                                 carpet='carpet_1_1', colorscale=p_cfg['colorscale'],
                                 zmin=p_cfg['cmin'], zmax=p_cfg['cmax'],
                                 contours=dict(coloring='fill', showlines=False), ncontours=50),
                go.Contourcarpet(a=A.flatten(), b=B.flatten(), z=field2.flatten(),
                                 carpet='carpet_1_2', colorscale=p_cfg['colorscale'],
                                 zmin=p_cfg['cmin'], zmax=p_cfg['cmax'],
                                 contours=dict(coloring='fill', showlines=False), ncontours=50),
                go.Contourcarpet(a=A.flatten(), b=B.flatten(), z=snap_u.flatten(),
                                 carpet='carpet_2_1', colorscale=coloraxis_config['velocity']['colorscale'],
                                 zmin=coloraxis_config['velocity']['cmin'], zmax=coloraxis_config['velocity']['cmax'],
                                 contours=dict(coloring='fill', showlines=False), ncontours=50),
                go.Contourcarpet(a=A.flatten(), b=B.flatten(), z=snap_v.flatten(),
                                 carpet='carpet_2_2', colorscale=coloraxis_config['velocity']['colorscale'],
                                 zmin=coloraxis_config['velocity']['cmin'], zmax=coloraxis_config['velocity']['cmax'],
                                 contours=dict(coloring='fill', showlines=False), ncontours=50),
                go.Contourcarpet(a=A.flatten(), b=B.flatten(), z=field5.flatten(),
                                 carpet='carpet_3_1', colorscale=r_cfg['colorscale'],
                                 zmin=r_cfg['cmin'], zmax=r_cfg['cmax'],
                                 contours=dict(coloring='fill', showlines=False), ncontours=50),
                go.Contourcarpet(a=A.flatten(), b=B.flatten(), z=snap_chi_safe,
                                 carpet='carpet_3_2', colorscale=chi_cfg['colorscale'],
                                 zmin=chi_cfg['cmin'], zmax=chi_cfg['cmax'],
                                 contours=dict(coloring='fill', showlines=False), ncontours=50),
            ]
            
            # Show all snapshots up to this point (including divergence)
            snapshots_so_far = [s for s in all_snapshots[:i+1]]
            snapshot_iters = [s.iteration for s in snapshots_so_far]
            # Sanitize residuals for JSON (NaN -> None) - use pressure (index 0)
            snapshot_res = []
            for s in snapshots_so_far:
                if isinstance(s.residual, np.ndarray) and len(s.residual) > 0:
                    val = s.residual[0]  # Pressure residual
                else:
                    val = s.residual
                snapshot_res.append(val if np.isfinite(val) else None)
            
            # Color divergence snapshots differently (use iteration set for comparison)
            colors = []
            for s in snapshots_so_far:
                if s.iteration in divergence_iters:
                    colors.append('orange')
                else:
                    colors.append('black')
            
            frame_data.append(go.Scatter(
                x=snapshot_iters, y=snapshot_res,
                mode='markers',
                marker=dict(color=colors, size=10, symbol='circle'),
            ))
            
            # Add Cp and Cf for this snapshot if surface data is available
            if has_surface_data and x_surf_airfoil is not None:
                NI_cells = ni
                i_start = n_wake
                i_end = NI_cells - n_wake
                
                V_inf_sq = self.u_inf**2 + self.v_inf**2
                q_inf = max(0.5 * V_inf_sq, 1e-14)
                
                # Cp from this snapshot (clip to prevent overflow)
                max_safe = 1e10
                Cp_snap = np.clip(snap.p[:, 0], -max_safe, max_safe) / q_inf
                Cp_snap_airfoil = _sanitize_array(Cp_snap[i_start:i_end], fill_value=0.0)
                
                # Cf from this snapshot
                vol = grid_metrics.volume[:, 0]
                Sj_mag = np.sqrt(grid_metrics.Sj_x[:, 0]**2 + grid_metrics.Sj_y[:, 0]**2)
                dy = vol / (Sj_mag + 1e-14)
                
                # Clip to prevent overflow in squaring
                u_snap_wall = np.clip(snap.u[:, 0] + self.u_inf, -max_safe, max_safe)
                v_snap_wall = np.clip(snap.v[:, 0] + self.v_inf, -max_safe, max_safe)
                nu_snap_wall = np.clip(snap.nu[:, 0], 0.0, max_safe)
                mu_eff_snap = mu_laminar + np.maximum(0.0, nu_snap_wall)
                
                dudn = 2.0 * u_snap_wall / dy
                dvdn = 2.0 * v_snap_wall / dy
                tau_mag = mu_eff_snap * np.sqrt(dudn**2 + dvdn**2)
                Cf_snap = tau_mag / q_inf
                Cf_snap_airfoil = _sanitize_array(Cf_snap[i_start:i_end], fill_value=0.0)
                
                frame_data.append(go.Scatter(
                    x=x_surf_airfoil, y=Cp_snap_airfoil,
                    mode='lines', line=dict(color='blue', width=2),
                ))
                frame_data.append(go.Scatter(
                    x=x_surf_airfoil, y=Cf_snap_airfoil,
                    mode='lines', line=dict(color='red', width=2),
                ))
            
            frames.append(go.Frame(
                data=frame_data,
                name=str(snap.iteration),
                traces=animated_indices,
            ))
        
        fig.frames = frames
        
        # Iteration slider steps
        slider_steps = [
            dict(
                method='animate',
                args=[[str(snap.iteration)], dict(
                    mode='immediate',
                    frame=dict(duration=100, redraw=True),
                    transition=dict(duration=0)
                )],
                label=f"{snap.iteration}{'*' if snap.iteration in divergence_iters else ''}",
            )
            for snap in all_snapshots
        ]
        
        # Create continuous logarithmic color range sliders
        # 50 steps spanning 3 orders of magnitude
        n_steps = 50
        
        def make_log_range_steps(base_val, trace_indices, is_residual=False):
            """Create continuous logarithmic slider steps (3 orders of magnitude)."""
            steps = []
            # Range from base_val / 10^1.5 to base_val * 10^1.5 (3 orders total)
            log_factors = np.linspace(-1.5, 1.5, n_steps)
            
            for log_f in log_factors:
                if is_residual:
                    # For residual: slider controls max level, range is always 3 decades
                    new_max = base_val + log_f  # base_val is log scale already
                    new_min = new_max - 3.0
                    label = f'{new_max:.1f}'
                    steps.append(dict(
                        method='restyle',
                        args=[{'zmin': new_min, 'zmax': new_max, 'ncontours': 50}, trace_indices],
                        label=label,
                    ))
                else:
                    # For symmetric ranges: log_f controls the multiplier
                    factor = 10 ** log_f
                    new_range = base_val * factor
                    steps.append(dict(
                        method='restyle',
                        args=[{'zmin': -new_range, 'zmax': new_range, 'ncontours': 50}, trace_indices],
                        label=f'{new_range:.2e}',
                    ))
            return steps
        
        # Trace indices for each color group
        pressure_traces = [1, 3]  # carpet_1_1 contour, carpet_1_2 contour
        velocity_traces = [5, 7]  # carpet_2_1 contour, carpet_2_2 contour
        residual_traces = [9]     # carpet_3_1 contour
        chi_traces = [11]         # carpet_3_2 contour (chi = nu_tilde/nu)
        
        p_steps = make_log_range_steps(p_range, pressure_traces, is_residual=False)
        v_steps = make_log_range_steps(vel_range, velocity_traces, is_residual=False)
        r_steps = make_log_range_steps(res_max, residual_traces, is_residual=True)
        chi_steps = make_log_range_steps(chi_max, chi_traces, is_residual=True)
        
        fig.update_layout(
            showlegend=False,
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    y=1.05, x=0.08, xanchor='right',
                    buttons=[
                        dict(
                            label='▶',
                            method='animate',
                            args=[None, dict(
                                frame=dict(duration=200, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                                mode='immediate',
                            )]
                        ),
                        dict(
                            label='⏸',
                            method='animate',
                            args=[[None], dict(
                                frame=dict(duration=0, redraw=False),
                                mode='immediate',
                                transition=dict(duration=0)
                            )]
                        ),
                    ]
                ),
            ],
            sliders=[
                # Iteration slider (top-left)
                dict(
                    active=len(all_snapshots) - 1,
                    yanchor='bottom', xanchor='left',
                    currentvalue=dict(
                        font=dict(size=11),
                        prefix='Iter: ',
                        visible=True,
                        xanchor='right',
                    ),
                    transition=dict(duration=0),
                    pad=dict(b=5, t=5),
                    len=0.42, x=0.08, y=1.04,
                    steps=slider_steps,
                    ticklen=0,
                ),
                # Pressure range slider
                dict(
                    active=n_steps // 2,
                    yanchor='bottom', xanchor='left',
                    currentvalue=dict(
                        font=dict(size=9),
                        prefix='Δp: ',
                        visible=True,
                        xanchor='left',
                    ),
                    pad=dict(b=5, t=5),
                    len=0.11, x=0.52, y=1.04,
                    steps=p_steps,
                    ticklen=0,
                ),
                # Velocity range slider
                dict(
                    active=n_steps // 2,
                    yanchor='bottom', xanchor='left',
                    currentvalue=dict(
                        font=dict(size=9),
                        prefix='Δv: ',
                        visible=True,
                        xanchor='left',
                    ),
                    pad=dict(b=5, t=5),
                    len=0.11, x=0.64, y=1.04,
                    steps=v_steps,
                    ticklen=0,
                ),
                # Residual max slider
                dict(
                    active=n_steps // 2,
                    yanchor='bottom', xanchor='left',
                    currentvalue=dict(
                        font=dict(size=9),
                        prefix='Res: ',
                        visible=True,
                        xanchor='left',
                    ),
                    pad=dict(b=5, t=5),
                    len=0.11, x=0.76, y=1.04,
                    steps=r_steps,
                    ticklen=0,
                ),
                # Chi (ν̃/ν) max slider
                dict(
                    active=n_steps // 2,
                    yanchor='bottom', xanchor='left',
                    currentvalue=dict(
                        font=dict(size=9),
                        prefix='χ: ',
                        visible=True,
                        xanchor='left',
                    ),
                    pad=dict(b=5, t=5),
                    len=0.11, x=0.88, y=1.04,
                    steps=chi_steps,
                    ticklen=0,
                ),
            ],
            height=1600 + (350 if has_wall_dist else 0) + (300 if has_surface_data else 0),
            width=1400,
            margin=dict(t=80),  # Top margin for sliders
        )
        
        # Link zoom/pan across all contourcarpet plots
        # First, set up row 1 col 1 as the reference axis
        fig.update_xaxes(title_text='x', range=[-0.5, 1.5], scaleanchor='y', scaleratio=1, row=1, col=1)
        fig.update_yaxes(title_text='y', range=[-0.5625, 0.5625], row=1, col=1)
        
        # List of all contourcarpet subplot positions
        field_positions = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
        if has_wall_dist:
            field_positions.append((wall_dist_row, 1))
        
        # Link all other field plots to the reference (row 1 col 1 = x, y)
        for row, col in field_positions:
            if row == 1 and col == 1:
                continue  # Skip reference
            fig.update_xaxes(title_text='x', range=[-0.5, 1.5], matches='x', row=row, col=col)
            fig.update_yaxes(title_text='y', range=[-0.5625, 0.5625], matches='y', row=row, col=col)
        
        # Convergence history plot (row 4, col 1, spanning both columns) - keep independent, NOT zoomable
        fig.update_xaxes(title_text='Iteration', matches=None, autorange=True, fixedrange=True, row=4, col=1)
        fig.update_yaxes(title_text='RMS Residual', matches=None, type='log', autorange=True, fixedrange=True, row=4, col=1)
        
        # Note: Wall distance row col 2 is None in specs, so no axes to hide
        
        # Cp and Cf plots configuration
        if has_surface_data:
            # Cp plot: x/c vs Cp (negative up convention)
            fig.update_xaxes(title_text='x/c', range=[0, 1], fixedrange=False, row=surface_row, col=1)
            fig.update_yaxes(title_text='Cp', autorange='reversed', fixedrange=False, row=surface_row, col=1)
            
            # Cf plot: x/c vs Cf
            fig.update_xaxes(title_text='x/c', range=[0, 1], fixedrange=False, row=surface_row, col=2)
            fig.update_yaxes(title_text='Cf', range=[0, cf_max_global * 1.1], fixedrange=False, row=surface_row, col=2)
        
        fig.write_html(str(output_path), auto_play=False)
        
        print(f"Saved HTML animation to: {output_path}")
        print(f"  Use sliders at top-right to adjust color ranges")
        return str(output_path)
    
    def clear(self) -> None:
        """Clear all stored snapshots."""
        self.snapshots.clear()
        self.residual_history.clear()
        self.iteration_history.clear()
        self.divergence_snapshots.clear()
    
    @property
    def num_snapshots(self) -> int:
        return len(self.snapshots)
