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
    compute_chi_log, 
    compute_yplus_distribution,
    compute_cf_distribution,
    extract_residual_value
)
from .data import Snapshot
from .layout import DashboardLayout
from .traces import AnimatedTraceTracker

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
        
        frame_data = []
        
        # Standard keys order must match TraceManager.add_contour_traces
        standard_keys = [
            ('pressure', snap_p.flatten()),
            ('u_vel', snap_u.flatten()),
            ('v_vel', snap_v.flatten()),
            ('cpt' if has_cpt else 'nu', field2.flatten()),
            ('residual' if has_res_field else 'vel_mag', field5.flatten()),
            ('grid', None),
            ('chi', to_json_safe_list(chi_log)),
        ]
        
        config_map = {
            'pressure': color_config['pressure'],
            'u_vel': color_config['velocity'],
            'v_vel': color_config['velocity'],
            'cpt': color_config['pressure'], 
            'nu': color_config['pressure'],
            'residual': color_config['residual'],
            'vel_mag': color_config['residual'],
            'chi': color_config['chi'],
            'grid': None
        }

        for name, data in standard_keys:
            if not self.layout.has_plot(name):
                continue
            
            # Grid has no animated trace
            if name == 'grid':
                continue
                
            cfg = config_map[name]
            carpet_id = f'carpet_{name}'
            
            frame_data.append(go.Contourcarpet(
                a=A.flatten(), b=B.flatten(), z=data,
                carpet=carpet_id, 
                colorscale=cfg.colorscale,
                zmin=cfg.cmin, zmax=cfg.cmax,
                contours=dict(coloring='fill', showlines=False), 
                ncontours=20
            ))
        
        # AFT fields
        if has_aft:
            frame_data.extend(self._create_aft_frame_data(snap, A, B))
        
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

    def _create_aft_frame_data(self, snap: Snapshot, A: np.ndarray, B: np.ndarray) -> list:
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
        
        Re_Omega_safe = np.maximum(sanitize_array(snap.Re_Omega, fill_value=10.0), 1.0)
        Re_log = np.log10(Re_Omega_safe)
        Gamma_safe = sanitize_array(snap.Gamma, fill_value=0.0)
        is_turb_safe = sanitize_array(snap.is_turb, fill_value=0.0) if snap.is_turb is not None else np.zeros_like(Gamma_safe)
        
        add('re_omega', Re_log, 'Viridis', 1.0, 4.0)
        add('gamma', Gamma_safe, 'Reds', 0.0, 2.0)
        add('is_turb', is_turb_safe, 'RdYlBu_r', 0.0, 1.0)
        
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
