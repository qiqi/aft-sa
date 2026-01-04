"""
Plotly-based HTML animation for CFD results.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING, Any
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from ..constants import NGHOST
from ._array_utils import sanitize_array, safe_minmax, safe_absmax, to_json_safe_list
from ._vtk_writer import write_vtk
from ._html_components import (
    compute_color_ranges,
    compute_chi_log,
    compute_cf_distribution,
    make_log_range_slider_steps,
    extract_residual_value,
    ColorAxisConfig,
)

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
    
    # Number of contour levels (lower = smaller file size)
    N_CONTOURS = 20
    
    def __init__(self, reynolds: float = 6e6, use_cdn: bool = True):
        """
        Initialize the dashboard.
        
        Parameters
        ----------
        reynolds : float
            Reynolds number for viscosity calculation.
        use_cdn : bool
            If True, reference plotly.js from CDN (saves ~3MB per file).
            If False, embed plotly.js in HTML (works offline but larger).
        """
        self.snapshots: List[Snapshot] = []
        self.residual_history: List = []
        self.iteration_history: List[int] = []
        self.divergence_snapshots: List[Snapshot] = []
        self.p_inf: float = 0.0
        self.u_inf: float = 1.0
        self.v_inf: float = 0.0
        self.nu_laminar: float = 1.0 / reynolds if reynolds > 0 else 1e-6
        self.use_cdn: bool = use_cdn
    
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
        """Store current solution state with diagnostic data."""
        Q_int = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
        
        # Get residual
        residual = self._parse_residual(residual_history)
        
        # Extract freestream values
        if freestream is not None:
            self.p_inf = freestream.p_inf
            self.u_inf = freestream.u_inf
            self.v_inf = freestream.v_inf
        
        # Create snapshot
        snapshot = self._create_snapshot(Q_int, iteration, residual, cfl, C_pt, residual_field)
        
        if is_divergence_dump:
            self.divergence_snapshots.append(snapshot)
        else:
            self.snapshots.append(snapshot)
        
        # Store residual history
        self._update_residual_history(residual_history, iteration_history)
    
    def _parse_residual(self, residual_history: List) -> np.ndarray:
        """Parse residual from history list."""
        if not residual_history:
            return np.zeros(4)
        
        last_res = residual_history[-1]
        if isinstance(last_res, (tuple, list, np.ndarray)):
            return np.array([r if np.isfinite(r) else 1e10 for r in last_res])
        else:
            val = last_res if np.isfinite(last_res) else 1e10
            return np.array([val] * 4)
    
    def _create_snapshot(
        self,
        Q_int: np.ndarray,
        iteration: int,
        residual: np.ndarray,
        cfl: float,
        C_pt: Optional[np.ndarray],
        residual_field: Optional[np.ndarray],
    ) -> Snapshot:
        """Create a Snapshot from solution data."""
        max_safe_vel = 1e10
        
        # Relative values
        p_rel = sanitize_array(Q_int[:, :, 0] - self.p_inf, fill_value=0.0)
        u_rel = sanitize_array(Q_int[:, :, 1] - self.u_inf, fill_value=0.0)
        v_rel = sanitize_array(Q_int[:, :, 2] - self.v_inf, fill_value=0.0)
        nu = sanitize_array(Q_int[:, :, 3], fill_value=0.0)
        
        # Velocity magnitude
        u_abs = np.clip(sanitize_array(Q_int[:, :, 1], fill_value=self.u_inf), -max_safe_vel, max_safe_vel)
        v_abs = np.clip(sanitize_array(Q_int[:, :, 2], fill_value=self.v_inf), -max_safe_vel, max_safe_vel)
        vel_mag = np.sqrt(u_abs**2 + v_abs**2)
        
        # C_pt computation
        if C_pt is None:
            C_pt = self._compute_cpt(Q_int, u_abs, v_abs)
        else:
            C_pt = sanitize_array(C_pt, fill_value=0.0)
        
        # Residual field
        res_field = None
        if residual_field is not None:
            res_clipped = np.clip(residual_field, -max_safe_vel, max_safe_vel)
            res_field = sanitize_array(np.sqrt(np.mean(res_clipped**2, axis=2)), fill_value=1e-12)
        
        return Snapshot(
            iteration=iteration,
            residual=residual,
            cfl=cfl,
            p=p_rel.copy(),
            u=u_rel.copy(),
            v=v_rel.copy(),
            nu=nu.copy(),
            C_pt=C_pt.copy() if C_pt is not None else None,
            residual_field=res_field,
            vel_max=float(safe_absmax(vel_mag, default=1.0)),
            p_min=float(safe_minmax(p_rel)[0]),
            p_max=float(safe_minmax(p_rel)[1]),
        )
    
    def _compute_cpt(self, Q_int: np.ndarray, u_abs: np.ndarray, v_abs: np.ndarray) -> Optional[np.ndarray]:
        """Compute total pressure loss coefficient."""
        V_inf_sq = self.u_inf**2 + self.v_inf**2
        if V_inf_sq < 1e-14:
            return None
        
        p = sanitize_array(Q_int[:, :, 0], fill_value=self.p_inf)
        p = np.clip(p, -1e10, 1e10)
        p_total = p + 0.5 * (u_abs**2 + v_abs**2)
        p_total_inf = self.p_inf + 0.5 * V_inf_sq
        C_pt = (p_total_inf - p_total) / (0.5 * V_inf_sq)
        return sanitize_array(C_pt, fill_value=0.0)
    
    def _update_residual_history(self, residual_history: List, iteration_history: Optional[List[int]]) -> None:
        """Update stored residual history."""
        self.residual_history = []
        for r in residual_history:
            if isinstance(r, (tuple, list, np.ndarray)):
                self.residual_history.append(np.array([v if np.isfinite(v) else 1e10 for v in r]))
            else:
                val = r if np.isfinite(r) else 1e10
                self.residual_history.append(np.array([val, val, val, val]))
        
        if iteration_history is not None:
            self.iteration_history = list(iteration_history)
        elif len(self.iteration_history) != len(self.residual_history):
            self.iteration_history = list(range(len(self.residual_history)))
    
    def save_html(
        self,
        filename: str,
        grid_metrics: 'FVMMetrics', 
        wall_distance: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
        n_wake: int = 0,
        mu_laminar: float = 1e-6,
    ) -> str:
        """Export all snapshots as interactive HTML animation."""
        if not HAS_PLOTLY:
            print("Warning: plotly not installed. Skipping HTML animation.")
            return ""
        
        if not self.snapshots:
            print("Warning: No snapshots to save.")
            return ""
        
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Combine regular and divergence snapshots
        all_snapshots = self._get_all_snapshots()
        
        # Feature flags
        has_cpt = self.snapshots[0].C_pt is not None
        has_res_field = self.snapshots[0].residual_field is not None
        has_wall_dist = wall_distance is not None
        has_surface_data = X is not None and Y is not None
        
        # Compute color ranges
        color_config = compute_color_ranges(self.snapshots, self.nu_laminar, has_cpt, has_res_field)
        
        # Create figure
        fig = self._create_figure(has_wall_dist, has_surface_data, has_cpt, has_res_field)
        
        # Add all traces
        xc, yc = grid_metrics.xc, grid_metrics.yc
        self._add_contour_traces(fig, xc, yc, all_snapshots[-1], color_config, has_cpt, has_res_field)
        self._add_convergence_traces(fig)
        
        if has_wall_dist:
            self._add_wall_distance_trace(fig, xc, yc, wall_distance, color_config)
        
        surface_params = None
        if has_surface_data:
            surface_params = self._add_surface_traces(
                fig, grid_metrics, X, all_snapshots, n_wake, mu_laminar, has_wall_dist
            )
        
        # Create animation frames
        self._add_animation_frames(
            fig, xc, yc, all_snapshots, color_config, has_cpt, has_res_field,
            has_surface_data, surface_params, grid_metrics, n_wake, mu_laminar
        )
        
        # Configure layout
        self._configure_layout(fig, all_snapshots, color_config, has_wall_dist, has_surface_data, surface_params)
        
        # Write HTML with optional CDN to reduce file size
        include_plotlyjs: Any = 'cdn' if self.use_cdn else True
        fig.write_html(str(output_path), auto_play=False, include_plotlyjs=include_plotlyjs)
        print(f"Saved HTML animation to: {output_path}")
        if self.use_cdn:
            print(f"  Note: Requires internet connection (using plotly.js CDN)")
        print(f"  Use sliders at top-right to adjust color ranges")
        
        # Also save VTK file
        vtk_path = output_path.with_suffix('.vtk')
        write_vtk(vtk_path, grid_metrics, all_snapshots[-1], wall_distance,
                  self.u_inf, self.v_inf, self.nu_laminar)
        
        return str(output_path)
    
    def _get_all_snapshots(self) -> List[Snapshot]:
        """Get all snapshots sorted by iteration."""
        all_snapshots = list(self.snapshots)
        if self.divergence_snapshots:
            all_snapshots.extend(self.divergence_snapshots)
            all_snapshots.sort(key=lambda s: s.iteration)
        return all_snapshots
    
    def _create_figure(
        self,
        has_wall_dist: bool,
        has_surface_data: bool,
        has_cpt: bool,
        has_res_field: bool,
    ) -> 'go.Figure':
        """Create the subplot figure layout."""
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
            'Convergence History',
        ]
        if has_wall_dist:
            subplot_titles.append('Wall Distance (d/c)')
        if has_surface_data:
            subplot_titles.extend(['', ''])
        
        specs = [
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "scatter", "colspan": 2}, None],
        ]
        if has_wall_dist:
            specs.append([{"type": "xy"}, None])
        if has_surface_data:
            specs.append([{"type": "scatter"}, {"type": "scatter"}])
        
        return make_subplots(
            rows=n_rows, cols=2,
            subplot_titles=subplot_titles,
            specs=specs,
            horizontal_spacing=0.08,
            vertical_spacing=0.06,
        )
    
    def _add_contour_traces(
        self,
        fig: 'go.Figure',
        xc: np.ndarray,
        yc: np.ndarray,
        snapshot: Snapshot,
        color_config: dict,
        has_cpt: bool,
        has_res_field: bool,
    ) -> None:
        """Add the 6 contour plot traces."""
        ni, nj = xc.shape
        A, B = np.meshgrid(np.arange(ni), np.arange(nj), indexing='ij')
        
        # Prepare field data
        field2 = sanitize_array(snapshot.C_pt if has_cpt else snapshot.nu, fill_value=0.0)
        if has_res_field and snapshot.residual_field is not None:
            field5 = np.log10(sanitize_array(snapshot.residual_field, fill_value=1e-12) + 1e-12)
        else:
            field5 = np.zeros_like(snapshot.p)
        
        chi_log = compute_chi_log(snapshot.nu, self.nu_laminar)
        chi_log_safe = to_json_safe_list(chi_log)
        
        contour_configs = [
            (sanitize_array(snapshot.p).flatten(), 1, 1, 'pressure', True),
            (field2.flatten(), 1, 2, 'pressure', False),
            (sanitize_array(snapshot.u).flatten(), 2, 1, 'velocity', True),
            (sanitize_array(snapshot.v).flatten(), 2, 2, 'velocity', False),
            (field5.flatten(), 3, 1, 'residual', True),
            (chi_log_safe, 3, 2, 'chi', True),
        ]
        
        for data, row, col, caxis_key, show_colorbar in contour_configs:
            carpet_id = f'carpet_{row}_{col}'
            cfg = color_config[caxis_key]
            
            fig.add_trace(go.Carpet(
                a=A.flatten(), b=B.flatten(),
                x=xc.flatten(), y=yc.flatten(),
                carpet=carpet_id,
                aaxis=dict(showgrid=False, showticklabels='none', showline=False),
                baxis=dict(showgrid=False, showticklabels='none', showline=False),
            ), row=row, col=col)
            
            fig.add_trace(go.Contourcarpet(
                a=A.flatten(), b=B.flatten(), z=data,
                carpet=carpet_id,
                colorscale=cfg.colorscale,
                zmin=cfg.cmin, zmax=cfg.cmax,
                contours=dict(coloring='fill', showlines=False),
                ncontours=self.N_CONTOURS,
                colorbar=cfg.colorbar if show_colorbar else None,
                showscale=show_colorbar,
            ), row=row, col=col)
    
    def _add_convergence_traces(self, fig: 'go.Figure') -> None:
        """Add convergence history traces."""
        eq_names = ['Pressure', 'U-velocity', 'V-velocity', 'ν̃ (nuHat)']
        eq_colors = ['blue', 'red', 'green', 'purple']
        
        if self.residual_history:
            all_iters = (self.iteration_history 
                        if len(self.iteration_history) == len(self.residual_history)
                        else list(range(len(self.residual_history))))
            
            for eq_idx in range(4):
                res_eq = [extract_residual_value(r, eq_idx) for r in self.residual_history]
                fig.add_trace(go.Scatter(
                    x=all_iters, y=res_eq,
                    mode='lines', line=dict(color=eq_colors[eq_idx], width=1.5),
                    showlegend=True, name=eq_names[eq_idx],
                    legendgroup='convergence',
                ), row=4, col=1)
        
        # Snapshot markers
        snapshot_iters = [s.iteration for s in self.snapshots]
        snapshot_res = [extract_residual_value(s.residual, 0) for s in self.snapshots]
        fig.add_trace(go.Scatter(
            x=snapshot_iters, y=snapshot_res,
            mode='markers',
            marker=dict(color='black', size=8, symbol='circle'),
            showlegend=True, name='Snapshots',
            legendgroup='convergence',
        ), row=4, col=1)
        
        # Divergence markers
        if self.divergence_snapshots:
            div_iters = [s.iteration for s in self.divergence_snapshots]
            div_res = [extract_residual_value(s.residual, 0) for s in self.divergence_snapshots]
            fig.add_trace(go.Scatter(
                x=div_iters, y=div_res,
                mode='markers',
                marker=dict(color='orange', size=12, symbol='triangle-up'),
                showlegend=True, name='Divergence',
                legendgroup='convergence',
            ), row=4, col=1)
    
    def _add_wall_distance_trace(
        self,
        fig: 'go.Figure',
        xc: np.ndarray,
        yc: np.ndarray,
        wall_distance: np.ndarray,
        color_config: dict,
    ) -> None:
        """Add wall distance contour trace."""
        ni, nj = xc.shape
        A, B = np.meshgrid(np.arange(ni), np.arange(nj), indexing='ij')
        cfg = color_config['wall_dist']
        
        fig.add_trace(go.Carpet(
            a=A.flatten(), b=B.flatten(),
            x=xc.flatten(), y=yc.flatten(),
            carpet='carpet_wall_dist',
            aaxis=dict(showgrid=False, showticklabels='none', showline=False),
            baxis=dict(showgrid=False, showticklabels='none', showline=False),
        ), row=5, col=1)
        
        fig.add_trace(go.Contourcarpet(
            a=A.flatten(), b=B.flatten(),
            z=sanitize_array(wall_distance).flatten(),
            carpet='carpet_wall_dist',
            colorscale=cfg.colorscale,
            zmin=cfg.cmin, zmax=cfg.cmax,
            contours=dict(coloring='fill', showlines=False),
            ncontours=self.N_CONTOURS,
            colorbar=cfg.colorbar,
            showscale=True,
        ), row=5, col=1)
    
    def _add_surface_traces(
        self,
        fig: 'go.Figure',
        grid_metrics: 'FVMMetrics',
        X: np.ndarray,
        all_snapshots: List[Snapshot],
        n_wake: int,
        mu_laminar: float,
        has_wall_dist: bool,
    ) -> dict:
        """Add Cp and Cf surface plot traces."""
        ni = grid_metrics.xc.shape[0]
        surface_row = 6 if has_wall_dist else 5
        
        i_start = n_wake
        i_end = ni - n_wake
        x_surf = 0.5 * (X[:-1, 0] + X[1:, 0])
        x_surf_airfoil = x_surf[i_start:i_end]
        
        V_inf_sq = self.u_inf**2 + self.v_inf**2
        q_inf = max(0.5 * V_inf_sq, 1e-14)
        
        # Compute Cp and Cf for last snapshot
        snap = all_snapshots[-1]
        Cp = sanitize_array(snap.p[:, 0] / q_inf, fill_value=0.0)[i_start:i_end]
        Cf = compute_cf_distribution(
            snap, grid_metrics.volume[:, 0],
            grid_metrics.Sj_x[:, 0], grid_metrics.Sj_y[:, 0],
            mu_laminar, self.u_inf, self.v_inf, i_start, i_end
        )
        
        # Compute global ranges
        cp_min, cp_max = -2.0, 1.0
        cf_max = 0.05
        for snap in all_snapshots:
            Cp_snap = np.clip(snap.p[:, 0], -1e10, 1e10) / q_inf
            cp_min = min(cp_min, safe_minmax(Cp_snap[i_start:i_end], -2, 1)[0])
            cp_max = max(cp_max, safe_minmax(Cp_snap[i_start:i_end], -2, 1)[1])
            Cf_snap = compute_cf_distribution(
                snap, grid_metrics.volume[:, 0],
                grid_metrics.Sj_x[:, 0], grid_metrics.Sj_y[:, 0],
                mu_laminar, self.u_inf, self.v_inf, i_start, i_end
            )
            cf_max = max(cf_max, safe_absmax(Cf_snap, 0.01))
        
        cp_trace_idx = len(fig.data)
        fig.add_trace(go.Scatter(
            x=x_surf_airfoil, y=Cp,
            mode='lines', line=dict(color='blue', width=2),
            showlegend=False, name='Cp',
        ), row=surface_row, col=1)
        
        cf_trace_idx = len(fig.data)
        fig.add_trace(go.Scatter(
            x=x_surf_airfoil, y=Cf,
            mode='lines', line=dict(color='red', width=2),
            showlegend=False, name='Cf',
        ), row=surface_row, col=2)
        
        return {
            'surface_row': surface_row,
            'x_surf_airfoil': x_surf_airfoil,
            'i_start': i_start,
            'i_end': i_end,
            'cp_trace_idx': cp_trace_idx,
            'cf_trace_idx': cf_trace_idx,
            'cf_max': cf_max,
        }
    
    def _add_animation_frames(
        self,
        fig: 'go.Figure',
        xc: np.ndarray,
        yc: np.ndarray,
        all_snapshots: List[Snapshot],
        color_config: dict,
        has_cpt: bool,
        has_res_field: bool,
        has_surface_data: bool,
        surface_params: Optional[dict],
        grid_metrics: 'FVMMetrics',
        n_wake: int,
        mu_laminar: float,
    ) -> None:
        """Add animation frames for all snapshots."""
        ni, nj = xc.shape
        A, B = np.meshgrid(np.arange(ni), np.arange(nj), indexing='ij')
        
        divergence_iters = {s.iteration for s in self.divergence_snapshots}
        
        # Trace indices for animation
        base_contour_indices = [1, 3, 5, 7, 9, 11]
        residual_marker_idx = 12 + 4
        animated_indices = base_contour_indices + [residual_marker_idx]
        
        if has_surface_data and surface_params:
            animated_indices.extend([surface_params['cp_trace_idx'], surface_params['cf_trace_idx']])
        
        frames = []
        for i, snap in enumerate(all_snapshots):
            frame_data = self._create_frame_data(
                snap, A, B, color_config, has_cpt, has_res_field,
                all_snapshots[:i+1], divergence_iters
            )
            
            if has_surface_data and surface_params:
                frame_data.extend(self._create_surface_frame_data(
                    snap, surface_params, grid_metrics, mu_laminar
                ))
            
            frames.append(go.Frame(
                data=frame_data,
                name=str(snap.iteration),
                traces=animated_indices,
            ))
        
        fig.frames = frames
    
    def _create_frame_data(
        self,
        snap: Snapshot,
        A: np.ndarray,
        B: np.ndarray,
        color_config: dict,
        has_cpt: bool,
        has_res_field: bool,
        snapshots_so_far: List[Snapshot],
        divergence_iters: set,
    ) -> list:
        """Create frame data for a single snapshot."""
        snap_p = sanitize_array(snap.p)
        snap_u = sanitize_array(snap.u)
        snap_v = sanitize_array(snap.v)
        snap_nu = sanitize_array(snap.nu, fill_value=1e-12)
        
        field2 = sanitize_array(snap.C_pt if has_cpt and snap.C_pt is not None else snap.nu)
        if has_res_field and snap.residual_field is not None:
            field5 = np.log10(sanitize_array(snap.residual_field, fill_value=1e-12) + 1e-12)
        else:
            field5 = np.sqrt(snap_u**2 + snap_v**2)
        
        chi_log = compute_chi_log(snap_nu, self.nu_laminar)
        
        p_cfg = color_config['pressure']
        v_cfg = color_config['velocity']
        r_cfg = color_config['residual']
        chi_cfg = color_config['chi']
        
        frame_data = [
            go.Contourcarpet(a=A.flatten(), b=B.flatten(), z=snap_p.flatten(),
                            carpet='carpet_1_1', colorscale=p_cfg.colorscale,
                            zmin=p_cfg.cmin, zmax=p_cfg.cmax,
                            contours=dict(coloring='fill', showlines=False), ncontours=self.N_CONTOURS),
            go.Contourcarpet(a=A.flatten(), b=B.flatten(), z=field2.flatten(),
                            carpet='carpet_1_2', colorscale=p_cfg.colorscale,
                            zmin=p_cfg.cmin, zmax=p_cfg.cmax,
                            contours=dict(coloring='fill', showlines=False), ncontours=self.N_CONTOURS),
            go.Contourcarpet(a=A.flatten(), b=B.flatten(), z=snap_u.flatten(),
                            carpet='carpet_2_1', colorscale=v_cfg.colorscale,
                            zmin=v_cfg.cmin, zmax=v_cfg.cmax,
                            contours=dict(coloring='fill', showlines=False), ncontours=self.N_CONTOURS),
            go.Contourcarpet(a=A.flatten(), b=B.flatten(), z=snap_v.flatten(),
                            carpet='carpet_2_2', colorscale=v_cfg.colorscale,
                            zmin=v_cfg.cmin, zmax=v_cfg.cmax,
                            contours=dict(coloring='fill', showlines=False), ncontours=self.N_CONTOURS),
            go.Contourcarpet(a=A.flatten(), b=B.flatten(), z=field5.flatten(),
                            carpet='carpet_3_1', colorscale=r_cfg.colorscale,
                            zmin=r_cfg.cmin, zmax=r_cfg.cmax,
                            contours=dict(coloring='fill', showlines=False), ncontours=self.N_CONTOURS),
            go.Contourcarpet(a=A.flatten(), b=B.flatten(), z=to_json_safe_list(chi_log),
                            carpet='carpet_3_2', colorscale=chi_cfg.colorscale,
                            zmin=chi_cfg.cmin, zmax=chi_cfg.cmax,
                            contours=dict(coloring='fill', showlines=False), ncontours=self.N_CONTOURS),
        ]
        
        # Snapshot markers
        snapshot_iters = [s.iteration for s in snapshots_so_far]
        snapshot_res = [extract_residual_value(s.residual, 0) for s in snapshots_so_far]
        colors = ['orange' if s.iteration in divergence_iters else 'black' for s in snapshots_so_far]
        
        frame_data.append(go.Scatter(
            x=snapshot_iters, y=snapshot_res,
            mode='markers',
            marker=dict(color=colors, size=10, symbol='circle'),
        ))
        
        return frame_data
    
    def _create_surface_frame_data(
        self,
        snap: Snapshot,
        surface_params: dict,
        grid_metrics: 'FVMMetrics',
        mu_laminar: float,
    ) -> list:
        """Create Cp and Cf frame data."""
        i_start = surface_params['i_start']
        i_end = surface_params['i_end']
        x_surf = surface_params['x_surf_airfoil']
        
        V_inf_sq = self.u_inf**2 + self.v_inf**2
        q_inf = max(0.5 * V_inf_sq, 1e-14)
        
        Cp = sanitize_array(np.clip(snap.p[:, 0], -1e10, 1e10) / q_inf)[i_start:i_end]
        Cf = compute_cf_distribution(
            snap, grid_metrics.volume[:, 0],
            grid_metrics.Sj_x[:, 0], grid_metrics.Sj_y[:, 0],
            mu_laminar, self.u_inf, self.v_inf, i_start, i_end
        )
        
        return [
            go.Scatter(x=x_surf, y=Cp, mode='lines', line=dict(color='blue', width=2)),
            go.Scatter(x=x_surf, y=Cf, mode='lines', line=dict(color='red', width=2)),
        ]
    
    def _configure_layout(
        self,
        fig: 'go.Figure',
        all_snapshots: List[Snapshot],
        color_config: dict,
        has_wall_dist: bool,
        has_surface_data: bool,
        surface_params: Optional[dict],
    ) -> None:
        """Configure figure layout, sliders, and axes."""
        divergence_iters = {s.iteration for s in self.divergence_snapshots}
        
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
        
        # Color range sliders
        n_steps = 50
        p_cfg = color_config['pressure']
        v_cfg = color_config['velocity']
        r_cfg = color_config['residual']
        chi_cfg = color_config['chi']
        
        p_steps = make_log_range_slider_steps(p_cfg.cmax, [1, 3], False, n_steps)
        v_steps = make_log_range_slider_steps(v_cfg.cmax, [5, 7], False, n_steps)
        r_steps = make_log_range_slider_steps(r_cfg.cmax, [9], True, n_steps)
        chi_steps = make_log_range_slider_steps(chi_cfg.cmax, [11], True, n_steps)
        
        fig.update_layout(
            showlegend=False,
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                y=1.05, x=0.08, xanchor='right',
                buttons=[
                    dict(label='▶', method='animate',
                         args=[None, dict(frame=dict(duration=200, redraw=True),
                                         fromcurrent=True, transition=dict(duration=0), mode='immediate')]),
                    dict(label='⏸', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                           mode='immediate', transition=dict(duration=0))]),
                ]
            )],
            sliders=[
                dict(active=len(all_snapshots)-1, yanchor='bottom', xanchor='left',
                     currentvalue=dict(font=dict(size=11), prefix='Iter: ', visible=True, xanchor='right'),
                     transition=dict(duration=0), pad=dict(b=5, t=5),
                     len=0.42, x=0.08, y=1.04, steps=slider_steps, ticklen=0),
                dict(active=n_steps//2, yanchor='bottom', xanchor='left',
                     currentvalue=dict(font=dict(size=9), prefix='Δp: ', visible=True, xanchor='left'),
                     pad=dict(b=5, t=5), len=0.11, x=0.52, y=1.04, steps=p_steps, ticklen=0),
                dict(active=n_steps//2, yanchor='bottom', xanchor='left',
                     currentvalue=dict(font=dict(size=9), prefix='Δv: ', visible=True, xanchor='left'),
                     pad=dict(b=5, t=5), len=0.11, x=0.64, y=1.04, steps=v_steps, ticklen=0),
                dict(active=n_steps//2, yanchor='bottom', xanchor='left',
                     currentvalue=dict(font=dict(size=9), prefix='Res: ', visible=True, xanchor='left'),
                     pad=dict(b=5, t=5), len=0.11, x=0.76, y=1.04, steps=r_steps, ticklen=0),
                dict(active=n_steps//2, yanchor='bottom', xanchor='left',
                     currentvalue=dict(font=dict(size=9), prefix='χ: ', visible=True, xanchor='left'),
                     pad=dict(b=5, t=5), len=0.11, x=0.88, y=1.04, steps=chi_steps, ticklen=0),
            ],
            height=1600 + (350 if has_wall_dist else 0) + (300 if has_surface_data else 0),
            width=1400,
            margin=dict(t=80),
        )
        
        # Link axes
        fig.update_xaxes(title_text='x', range=[-0.5, 1.5], scaleanchor='y', scaleratio=1, row=1, col=1)
        fig.update_yaxes(title_text='y', range=[-0.5625, 0.5625], row=1, col=1)
        
        field_positions = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
        if has_wall_dist:
            field_positions.append((5, 1))
        
        for row, col in field_positions:
            if row == 1 and col == 1:
                continue
            fig.update_xaxes(title_text='x', range=[-0.5, 1.5], matches='x', row=row, col=col)
            fig.update_yaxes(title_text='y', range=[-0.5625, 0.5625], matches='y', row=row, col=col)
        
        fig.update_xaxes(title_text='Iteration', matches=None, autorange=True, fixedrange=True, row=4, col=1)
        fig.update_yaxes(title_text='RMS Residual', matches=None, type='log', autorange=True, fixedrange=True, row=4, col=1)
        
        if has_surface_data and surface_params:
            surface_row = surface_params['surface_row']
            fig.update_xaxes(title_text='x/c', range=[0, 1], fixedrange=False, row=surface_row, col=1)
            fig.update_yaxes(title_text='Cp', autorange='reversed', fixedrange=False, row=surface_row, col=1)
            fig.update_xaxes(title_text='x/c', range=[0, 1], fixedrange=False, row=surface_row, col=2)
            fig.update_yaxes(title_text='Cf', range=[0, surface_params['cf_max'] * 1.1], fixedrange=False, row=surface_row, col=2)
    
    def clear(self) -> None:
        """Clear all stored snapshots."""
        self.snapshots.clear()
        self.residual_history.clear()
        self.iteration_history.clear()
        self.divergence_snapshots.clear()
    
    @property
    def num_snapshots(self) -> int:
        return len(self.snapshots)


# Backwards compatibility aliases
_sanitize_array = sanitize_array
_to_json_safe_list = to_json_safe_list
_safe_minmax = safe_minmax
_safe_absmax = safe_absmax
