"""
Animation frame generation for CFD Dashboard.
"""

from typing import List, Optional, Set, Dict, Any, TYPE_CHECKING
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    pass

from .._array_utils import sanitize_array, to_json_safe_list
from .._html_components import (
    compute_yplus_distribution,
    compute_cf_distribution,
    extract_residual_value
)
from .data import Snapshot
from .layout import DashboardLayout
from .traces import AnimatedTraceTracker
from .plot_definitions import (
    build_aft_contour_data,
    build_standard_contour_data,
    get_standard_contour_specs,
)

if TYPE_CHECKING:
    from ...grid.metrics import FVMMetrics


class AnimationManager:
    """Manages creation of animation frames matching the trace layout."""

    def __init__(self, layout: DashboardLayout, tracker: AnimatedTraceTracker, nu_laminar: float):
        self.layout = layout
        self.tracker = tracker
        self.nu_laminar = nu_laminar

    def add_animation_frames(
        self,
        fig: 'go.Figure',
        xc: np.ndarray,
        yc: np.ndarray,
        all_snapshots: List[Snapshot],
        divergence_snapshots: List[Snapshot],
        color_config: Dict[str, Any],
        has_cpt: bool,
        has_res_field: bool,
        has_surface_data: bool,
        surface_params: Optional[Dict[str, Any]],
        grid_metrics: 'FVMMetrics',
        n_wake: int,
        u_inf: float,
        v_inf: float,
        has_aft: bool = False,
        wall_distance: Optional[np.ndarray] = None,
        yplus_params: Optional[dict] = None,
    ) -> None:
        """Add animation frames for all snapshots."""
        ni, nj = xc.shape
        A, B = np.meshgrid(np.arange(ni), np.arange(nj), indexing='ij')
        
        divergence_iters = {s.iteration for s in divergence_snapshots}
        animated_indices = self.tracker.animated_indices
        
        frames = []
        for i, snap in enumerate(all_snapshots):
            frame_data = self._create_frame_data(
                snap, A, B, color_config, has_cpt, has_res_field,
                all_snapshots[:i+1], divergence_iters, has_aft,
                has_surface_data, surface_params, grid_metrics,
                wall_distance, yplus_params, u_inf, v_inf
            )
            
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
        color_config: Dict[str, Any],
        has_cpt: bool,
        has_res_field: bool,
        snapshots_so_far: List[Snapshot],
        divergence_iters: Set[int],
        has_aft: bool,
        has_surface_data: bool,
        surface_params: Optional[dict],
        grid_metrics: Optional['FVMMetrics'],
        wall_distance: Optional[np.ndarray],
        yplus_params: Optional[dict],
        u_inf: float,
        v_inf: float,
    ) -> list:
        """Create frame data dicts for a single snapshot."""
        
        frame_data = []

        standard_specs = get_standard_contour_specs(has_cpt, has_res_field)
        standard_data = build_standard_contour_data(snap, self.nu_laminar, has_cpt, has_res_field)

        for spec in standard_specs:
            name = spec.name
            if not self.layout.has_plot(name):
                continue
            
            cfg = color_config[spec.color_key]
            carpet_id = f'carpet_{name}'
            data = standard_data.get(name)
            if data is None:
                data = np.zeros_like(snap.p).flatten()
            zmin_val = cfg.cmin
            zmax_val = cfg.cmax
            if name == 'amplification_rate':
                zmin_val = np.log10(0.0002)
                zmax_val = np.log10(0.2)
            
            frame_data.append(go.Contourcarpet(
                a=A.flatten(), b=B.flatten(), z=data,
                carpet=carpet_id, 
                colorscale=cfg.colorscale,
                zmin=zmin_val, zmax=zmax_val,
                contours=dict(coloring='fill', showlines=False), 
                ncontours=20
            ))
        
        # AFT fields
        if has_aft:
            frame_data.extend(self._create_aft_frame_data(snap, A, B, color_config))
        
        # Snapshot markers
        snapshot_iters = [s.iteration for s in snapshots_so_far]
        snapshot_res = [extract_residual_value(s.residual, 0) for s in snapshots_so_far]
        # Colors: orange for divergence, black for normal
        colors = ['orange' if s.iteration in divergence_iters else 'black' for s in snapshots_so_far]
        
        # Only add if convergence plot exists, but TraceManager always adds it?
        # TraceManager.add_convergence_traces checks has_plot("convergence")
        if self.layout.has_plot("convergence"):
            frame_data.append(go.Scatter(
                x=snapshot_iters, y=snapshot_res,
                mode='markers',
                marker=dict(color=colors, size=10, symbol='circle'),
            ))
        
        # y+ line
        if yplus_params and wall_distance is not None and grid_metrics is not None and self.layout.has_plot("wall_dist"):
            frame_data.extend(self._create_yplus_frame_data(snap, wall_distance, grid_metrics, yplus_params, u_inf, v_inf))
        
        # Surface lines
        if has_surface_data and surface_params and grid_metrics is not None and self.layout.has_plot("cp"):
            frame_data.extend(self._create_surface_frame_data(snap, surface_params, grid_metrics, u_inf, v_inf))
        
        return frame_data

    def _create_aft_frame_data(
        self,
        snap: Snapshot,
        A: np.ndarray,
        B: np.ndarray,
        color_config: Dict[str, Any],
    ) -> list:
        """Create AFT field frame data."""
        frame_data = []
        
        def add(name, data, cscale, zmin, zmax):
            if self.layout.has_plot(name):
                 frame_data.append(go.Contourcarpet(
                    a=A.flatten(), b=B.flatten(),
                    z=to_json_safe_list(data),
                    carpet=f'carpet_{name}',
                    colorscale=cscale,
                    zmin=zmin, zmax=zmax,
                    contours=dict(coloring='fill', showlines=False),
                    ncontours=20,
                ))
        
        aft_data = build_aft_contour_data(snap)
        if not aft_data:
            return frame_data

        add('re_omega', aft_data['re_omega'], 'Viridis', 1.0, 4.0)
        add('gamma', aft_data['gamma'], 'Reds', 0.0, 2.0)
        add('is_turb', aft_data['is_turb'], 'RdYlBu_r', 0.0, 1.0)
        if 'amplification_ratio' in aft_data and self.layout.has_plot('amplification_ratio'):
            cfg = color_config['amplification_ratio']
            add('amplification_ratio', aft_data['amplification_ratio'], cfg.colorscale, cfg.cmin, cfg.cmax)
        
        return frame_data

    def _create_yplus_frame_data(
        self, snap: Snapshot, wall_dist: np.ndarray, grid_metrics: 'FVMMetrics', 
        params: dict, u_inf: float, v_inf: float
    ) -> list:
        y_plus = compute_yplus_distribution(
            snap, wall_dist, grid_metrics.volume[:, 0],
            grid_metrics.Sj_x[:, 0], grid_metrics.Sj_y[:, 0],
            self.nu_laminar, u_inf, v_inf, params['i_start'], params['i_end']
        )
        return [go.Scatter(x=params['x_surf'], y=y_plus, mode='lines', line=dict(color='green', width=2))]

    def _create_surface_frame_data(
        self, snap: Snapshot, params: dict, grid_metrics: 'FVMMetrics',
        u_inf: float, v_inf: float
    ) -> list:
        V_inf_sq = u_inf**2 + v_inf**2
        q_inf = max(0.5 * V_inf_sq, 1e-14)
        
        Cp = sanitize_array(np.clip(snap.p[:, 0], -1e10, 1e10) / q_inf)[params['i_start']:params['i_end']]
        Cf = compute_cf_distribution(
            snap, grid_metrics.volume[:, 0],
            grid_metrics.Sj_x[:, 0], grid_metrics.Sj_y[:, 0],
            self.nu_laminar, u_inf, v_inf, params['i_start'], params['i_end']
        )
        return [
            go.Scatter(x=params['x_surf_airfoil'], y=Cp, mode='lines', line=dict(color='blue', width=2)),
            go.Scatter(x=params['x_surf_airfoil'], y=Cf, mode='lines', line=dict(color='red', width=2))
        ]
