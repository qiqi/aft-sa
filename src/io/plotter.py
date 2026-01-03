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
    residual: float
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
        self.residual_history: List[float] = []
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
        residual_history: List[float],
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
        residual_history : List[float]
            List of residual values.
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
        residual = residual_history[-1] if residual_history else 0.0
        
        # Sanitize residual (replace NaN with large value for visibility)
        if not np.isfinite(residual):
            residual = 1e10
        
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
        
        u_abs = _sanitize_array(Q_int[:, :, 1], fill_value=self.u_inf)
        v_abs = _sanitize_array(Q_int[:, :, 2], fill_value=self.v_inf)
        vel_mag = np.sqrt(u_abs**2 + v_abs**2)
        
        if C_pt is None and freestream is not None:
            p = _sanitize_array(Q_int[:, :, 0], fill_value=self.p_inf)
            V_inf_sq = self.u_inf**2 + self.v_inf**2
            p_total = p + 0.5 * (u_abs**2 + v_abs**2)
            p_total_inf = self.p_inf + 0.5 * V_inf_sq
            C_pt = (p_total_inf - p_total) / (0.5 * V_inf_sq + 1e-12)
            C_pt = _sanitize_array(C_pt, fill_value=0.0)
        elif C_pt is not None:
            C_pt = _sanitize_array(C_pt, fill_value=0.0)
        
        res_field = None
        if residual_field is not None:
            res_field = _sanitize_array(
                np.sqrt(np.mean(residual_field**2, axis=2)), 
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
        self.residual_history = [r if np.isfinite(r) else 1e10 for r in residual_history]
        if iteration_history is not None:
            self.iteration_history = list(iteration_history)
        elif len(self.iteration_history) != len(self.residual_history):
            # Default: assume consecutive iterations starting from 0
            self.iteration_history = list(range(len(self.residual_history)))
    
    def save_html(self, filename: str, grid_metrics: 'FVMMetrics', 
                  wall_distance: Optional[np.ndarray] = None) -> str:
        """Export all snapshots as interactive HTML animation.
        
        Parameters
        ----------
        filename : str
            Output filename for HTML file.
        grid_metrics : FVMMetrics
            Grid metrics containing cell centers.
        wall_distance : np.ndarray, optional
            Wall distance field (NI, NJ). If provided, adds wall distance plot.
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
        
        # Combine regular snapshots with divergence snapshots for animation
        all_snapshots = list(self.snapshots)
        if self.divergence_snapshots:
            all_snapshots.extend(self.divergence_snapshots)
            # Sort by iteration to maintain chronological order
            all_snapshots.sort(key=lambda s: s.iteration)
        
        # Use last snapshot (including divergence) for initial display
        snapN = all_snapshots[-1]
        
        # Layout: Row 1-2: field pairs, Row 3: residual + chi, Row 4: convergence (left) + empty (right)
        # Row 5 (optional): wall distance
        n_rows = 5 if has_wall_dist else 4
        
        subplot_titles = [
            'Pressure (p - p∞)', 
            'Total Pressure Loss (C_pt)' if has_cpt else 'Turbulent Viscosity (ν)',
            'U-velocity (u - u∞)', 'V-velocity (v - v∞)',
            'Residual Field (log₁₀)' if has_res_field else 'Velocity Magnitude',
            'χ = ν̃/ν (Turbulent/Laminar Viscosity Ratio)',
            'Convergence History', ''  # Second column empty
        ]
        
        if has_wall_dist:
            subplot_titles.extend(['Wall Distance (d/c)', ''])  # Wall dist in left column only
        
        specs = [
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "scatter", "colspan": 2}, None],  # Convergence spans both columns
        ]
        if has_wall_dist:
            specs.append([{"type": "xy"}, {"type": "xy"}])  # Wall distance in left column only
        
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
        chi_data = np.maximum(_sanitize_array(snapN.nu, fill_value=1e-12), 1e-12) / self.nu_laminar
        
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
        chi_max_vals = []
        chi_min_vals = []
        for s in self.snapshots:
            chi = _sanitize_array(s.nu, fill_value=1e-12) / self.nu_laminar
            chi = np.maximum(chi, 1e-12)  # Ensure positive for log
            chi_log = np.log10(chi)
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
        
        contour_configs = [
            (snapN_p, 1, 1, 'pressure', True),
            (field2_data, 1, 2, 'pressure', False),
            (snapN_u, 2, 1, 'velocity', True),
            (snapN_v, 2, 2, 'velocity', False),
            (field5_data, 3, 1, 'residual', True),
            (np.log10(chi_data + 1e-12), 3, 2, 'chi', True),  # Chi plot (log scale)
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
            fig.add_trace(
                go.Contourcarpet(
                    a=A.flatten(), b=B.flatten(), z=data.flatten(),
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
        
        if self.residual_history:
            # Use actual iteration numbers if available, otherwise use indices
            if self.iteration_history and len(self.iteration_history) == len(self.residual_history):
                all_iters = self.iteration_history
            else:
                all_iters = list(range(len(self.residual_history)))
            fig.add_trace(
                go.Scatter(
                    x=all_iters, y=self.residual_history,
                    mode='lines', line=dict(color='blue', width=1.5),
                    showlegend=False, name='Full History',
                ),
                row=4, col=1  # Convergence plot at row 4, spanning both columns
            )
        
        # Regular snapshots (red dots)
        snapshot_iters = [s.iteration for s in self.snapshots]
        snapshot_res = [s.residual for s in self.snapshots]
        fig.add_trace(
            go.Scatter(
                x=snapshot_iters, y=snapshot_res,
                mode='markers',
                marker=dict(color='red', size=10, symbol='circle'),
                showlegend=False, name='Snapshots',
            ),
            row=4, col=1  # Convergence plot at row 4
        )
        
        # Divergence snapshots (orange triangles) - show if any divergence was captured
        if self.divergence_snapshots:
            div_iters = [s.iteration for s in self.divergence_snapshots]
            div_res = [s.residual for s in self.divergence_snapshots]
            fig.add_trace(
                go.Scatter(
                    x=div_iters, y=div_res,
                    mode='markers',
                    marker=dict(color='orange', size=12, symbol='triangle-up'),
                    showlegend=False, name='Divergence Dumps',
                ),
                row=4, col=1
            )
        
        # Wall distance plot (static, at end)
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
                row=5, col=1
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
                row=5, col=1
            )
        
        # 6 contour plots: pressure, cpt, u, v, residual, chi
        # Each contour plot has a carpet (even index) and contourcarpet (odd index)
        # So contour indices are: 1, 3, 5, 7, 9, 11
        base_contour_indices = [1, 3, 5, 7, 9, 11]
        # After 6 contour plots (12 traces), we have:
        # - 1 blue line (full history)
        # - 1 red dots (snapshots)
        # - 1 orange triangles (divergence) if present
        # Then optionally wall distance traces (NOT animated)
        residual_marker_idx = 12 + 1  # 12 contour traces + 1 scatter line
        animated_indices = base_contour_indices + [residual_marker_idx]
        
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
            
            # Chi for this snapshot (log scale)
            snap_chi = np.log10(np.maximum(snap_nu, 1e-12) / self.nu_laminar + 1e-12)
            
            p_cfg = coloraxis_config['pressure']
            r_cfg = coloraxis_config['residual']
            
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
                go.Contourcarpet(a=A.flatten(), b=B.flatten(), z=snap_chi.flatten(),
                                 carpet='carpet_3_2', colorscale=chi_cfg['colorscale'],
                                 zmin=chi_cfg['cmin'], zmax=chi_cfg['cmax'],
                                 contours=dict(coloring='fill', showlines=False), ncontours=50),
            ]
            
            # Show all snapshots up to this point (including divergence)
            snapshots_so_far = [s for s in all_snapshots[:i+1]]
            snapshot_iters = [s.iteration for s in snapshots_so_far]
            snapshot_res = [s.residual for s in snapshots_so_far]
            
            # Color divergence snapshots differently (use iteration set for comparison)
            colors = []
            for s in snapshots_so_far:
                if s.iteration in divergence_iters:
                    colors.append('orange')
                else:
                    colors.append('red')
            
            frame_data.append(go.Scatter(
                x=snapshot_iters, y=snapshot_res,
                mode='markers',
                marker=dict(color=colors, size=10, symbol='circle'),
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
            height=1600 + (350 if has_wall_dist else 0),
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
            field_positions.append((5, 1))
        
        # Link all other field plots to the reference (row 1 col 1 = x, y)
        for row, col in field_positions:
            if row == 1 and col == 1:
                continue  # Skip reference
            fig.update_xaxes(title_text='x', range=[-0.5, 1.5], matches='x', row=row, col=col)
            fig.update_yaxes(title_text='y', range=[-0.5625, 0.5625], matches='y', row=row, col=col)
        
        # Convergence history plot (row 4, col 1, spanning both columns) - keep independent, NOT zoomable
        fig.update_xaxes(title_text='Iteration', matches=None, autorange=True, fixedrange=True, row=4, col=1)
        fig.update_yaxes(title_text='Residual', matches=None, type='log', autorange=True, fixedrange=True, row=4, col=1)
        
        # Hide the empty right column in row 5 (wall distance)
        if has_wall_dist:
            fig.update_xaxes(visible=False, row=5, col=2)
            fig.update_yaxes(visible=False, row=5, col=2)
        
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
