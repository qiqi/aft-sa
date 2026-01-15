"""
Trace generation logic for CFD animations.
"""

from typing import List, Optional, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    pass

from .._array_utils import sanitize_array, safe_absmax, to_json_safe_list
from .._html_components import (
    compute_yplus_distribution,
    compute_cf_distribution,
    extract_residual_value
)
from .data import Snapshot
from .layout import DashboardLayout
from .plot_definitions import (
    build_aft_contour_data,
    build_standard_contour_data,
    get_aft_contour_specs,
    get_standard_contour_specs,
)

if TYPE_CHECKING:
    from ...grid.metrics import FVMMetrics


class AnimatedTraceTracker:
    """Tracks which traces should be animated and their indices."""
    
    def __init__(self):
        self._trace_count = 0
        self._animated_indices: List[int] = []
        self._static_traces: List[str] = []
    
    def add_static(self, name: str = "") -> int:
        """Add a static (non-animated) trace, returns its index."""
        idx = self._trace_count
        self._trace_count += 1
        if name:
            self._static_traces.append(name)
        return idx
    
    def add_animated(self, name: str = "") -> int:
        """Add an animated trace, returns its index."""
        idx = self._trace_count
        self._animated_indices.append(idx)
        self._trace_count += 1
        return idx
    
    @property
    def animated_indices(self) -> List[int]:
        """Get list of all animated trace indices."""
        return self._animated_indices.copy()
    
    @property
    def count(self) -> int:
        """Total number of traces."""
        return self._trace_count


class TraceManager:
    """Manages the creation of static and initial animated traces."""
    
    # Number of contour levels
    N_CONTOURS = 20

    def __init__(self, layout: DashboardLayout, tracker: AnimatedTraceTracker, nu_laminar: float):
        self.layout = layout
        self.tracker = tracker
        self.nu_laminar = nu_laminar

    def add_contour_traces(
        self,
        fig: 'go.Figure',
        xc: np.ndarray,
        yc: np.ndarray,
        snapshot: Snapshot,
        color_config: Dict[str, Any],
        has_cpt: bool,
        has_res_field: bool,
    ) -> None:
        """Add the main field contour plots."""
        ni, nj = xc.shape
        A, B = np.meshgrid(np.arange(ni), np.arange(nj), indexing='ij')
        
        standard_plots = get_standard_contour_specs(has_cpt, has_res_field)
        standard_data = build_standard_contour_data(snapshot, self.nu_laminar, has_cpt, has_res_field)
        
        colorbar_titles = {spec.name: spec.title for spec in standard_plots}
        
        for spec in standard_plots:
            name = spec.name
            if not self.layout.has_plot(name):
                continue
                
            row, col = self.layout.get_position(name)
            carpet_id = f'carpet_{name}'
            
            # Data selection
            data = standard_data.get(name)
            if data is None:
                data = np.zeros_like(xc).flatten()
            
            # Grid Lines: Default to OFF (False), toggled via UI button
            axis_style = dict(
                showgrid=False, 
                gridcolor='black',
                gridwidth=0.5,
                showticklabels='none', 
                showline=False
            )
            
            self.tracker.add_static(carpet_id)
            fig.add_trace(go.Carpet(
                a=A.flatten(), b=B.flatten(),
                x=xc.flatten(), y=yc.flatten(),
                carpet=carpet_id,
                aaxis=axis_style,
                baxis=axis_style,
            ), row=row, col=col)
            
            if data is not None:
                trace_name = f'contour_{name}'
                show_colorbar = self.layout.should_show_colorbar(name)
                
                colorbar_dict = None
                if show_colorbar:
                    cb_config = self.layout.get_colorbar_config(name)
                    title_key = name
                    if name == 'cpt' and not has_cpt: title_key = 'nu'
                    if name == 'residual' and not has_res_field: title_key = 'vel_mag'
                    
                    colorbar_dict = dict(
                        title=colorbar_titles.get(title_key, spec.title),
                        x=cb_config['x'],
                        y=cb_config['y'],
                        len=cb_config['len'],
                        tickformat='.2f' if name in ['pressure', 'u_vel', 'v_vel'] else '.1f',
                    )

                # Prepare color limits (override for amplification rate as requested)
                zmin_val = color_config[spec.color_key].cmin
                zmax_val = color_config[spec.color_key].cmax
                if name == 'amplification_rate':
                    zmin_val = np.log10(0.0002)
                    zmax_val = np.log10(0.2)

                self.tracker.add_animated(trace_name)
                fig.add_trace(go.Contourcarpet(
                    a=A.flatten(), b=B.flatten(), z=data,
                    carpet=carpet_id,
                    colorscale=color_config[spec.color_key].colorscale,
                    zmin=zmin_val, zmax=zmax_val,
                    contours=dict(coloring='fill', showlines=False),
                    ncontours=self.N_CONTOURS,
                    colorbar=colorbar_dict,
                    showscale=show_colorbar,
                ), row=row, col=col)
                
                # Add Grid Overlay (Hidden by default, toggled via button)
                # We reuse the same carpet definition but control its axis visibility
                # Actually, the carpet trace IS the coordinate system. We declared it above at line 170.
                # We simply need to ensure it was created with showgrid=False initially.
                # Lines 146-153 create the carpet trace.
                # I should MODIFY that creation to be False by default.
                # And remove this duplicate adding code which I added in previous step.
                pass 

    def add_aft_traces(
        self,
        fig: 'go.Figure',
        xc: np.ndarray,
        yc: np.ndarray,
        snapshot: Snapshot,
        color_config: Dict[str, Any],
    ) -> None:
        """Add AFT diagnostic traces."""
        if snapshot.Re_Omega is None or snapshot.Gamma is None:
            return
        
        ni, nj = xc.shape
        A, B = np.meshgrid(np.arange(ni), np.arange(nj), indexing='ij')
        
        aft_data = build_aft_contour_data(snapshot)
        aft_specs = get_aft_contour_specs(include_amplification_ratio=True)
        
        # Helper to add contour
        def add_aft_contour(name, data, cscale, zmin, zmax, title):
            if not self.layout.has_plot(name): return
            row, col = self.layout.get_position(name)
            cb_config = self.layout.get_colorbar_config(name)
            
            self.tracker.add_static(f'carpet_{name}')
            fig.add_trace(go.Carpet(
                a=A.flatten(), b=B.flatten(),
                x=xc.flatten(), y=yc.flatten(),
                carpet=f'carpet_{name}',
                aaxis=dict(showgrid=False, showticklabels='none', showline=False),
                baxis=dict(showgrid=False, showticklabels='none', showline=False),
            ), row=row, col=col)
            
            self.tracker.add_animated(f'contour_{name}')
            fig.add_trace(go.Contourcarpet(
                a=A.flatten(), b=B.flatten(), z=to_json_safe_list(data),
                carpet=f'carpet_{name}',
                colorscale=cscale,
                zmin=zmin, zmax=zmax,
                contours=dict(coloring='fill', showlines=False),
                ncontours=self.N_CONTOURS,
                colorbar=dict(title=title, **cb_config),
                showscale=True,
            ), row=row, col=col)

        for spec in aft_specs:
            data = aft_data.get(spec.name)
            if data is None:
                continue
            if spec.name == 're_omega':
                add_aft_contour(spec.name, data, 'Viridis', 1.0, 4.0, spec.title)
            elif spec.name == 'gamma':
                add_aft_contour(spec.name, data, 'Reds', 0.0, 2.0, spec.title)
            elif spec.name == 'is_turb':
                add_aft_contour(spec.name, data, 'RdYlBu_r', 0.0, 1.0, spec.title)
            elif spec.name == 'amplification_ratio':
                cfg = color_config['amplification_ratio']
                add_aft_contour(spec.name, data, cfg.colorscale, cfg.cmin, cfg.cmax, spec.title)

    def add_convergence_traces(
        self, 
        fig: 'go.Figure', 
        residual_history: List, 
        iteration_history: List[int],
        snapshots: List[Snapshot],
        divergence_snapshots: List[Snapshot],
    ) -> None:
        """Add static convergence history traces."""
        if not self.layout.has_plot("convergence"):
            return
            
        eq_names = ['Pressure', 'U-velocity', 'V-velocity', 'ν̃ (nuHat)']
        eq_colors = ['blue', 'red', 'green', 'purple']
        
        conv_row, _ = self.layout.get_position("convergence")
        
        if residual_history:
            iters = iteration_history if len(iteration_history) == len(residual_history) else list(range(len(residual_history)))
            
            for eq_idx in range(4):
                res_eq = [extract_residual_value(r, eq_idx) for r in residual_history]
                self.tracker.add_static(f'convergence_line_{eq_idx}')
                fig.add_trace(go.Scatter(
                    x=iters, y=res_eq,
                    mode='lines', line=dict(color=eq_colors[eq_idx], width=1.5),
                    showlegend=True, name=eq_names[eq_idx],
                    legendgroup='convergence',
                ), row=conv_row, col=1)
        
        # Snapshot markers (animated)
        snapshot_iters = [s.iteration for s in snapshots]
        snapshot_res = [extract_residual_value(s.residual, 0) for s in snapshots]
        self.tracker.add_animated('snapshot_markers')
        fig.add_trace(go.Scatter(
            x=snapshot_iters, y=snapshot_res,
            mode='markers',
            marker=dict(color='black', size=8, symbol='circle'),
            showlegend=True, name='Snapshots',
            legendgroup='convergence',
        ), row=conv_row, col=1)
        
        # Divergence markers
        if divergence_snapshots:
            div_iters = [s.iteration for s in divergence_snapshots]
            div_res = [extract_residual_value(s.residual, 0) for s in divergence_snapshots]
            self.tracker.add_static('divergence_markers')
            fig.add_trace(go.Scatter(
                x=div_iters, y=div_res,
                mode='markers',
                marker=dict(color='orange', size=12, symbol='triangle-up'),
                showlegend=True, name='Divergence',
                legendgroup='convergence',
            ), row=conv_row, col=1)

    def add_wall_distance_trace(
        self,
        fig: 'go.Figure',
        xc: np.ndarray,
        yc: np.ndarray,
        wall_distance: np.ndarray,
        color_config: dict,
        grid_metrics: 'FVMMetrics',
        all_snapshots: List[Snapshot],
        n_wake: int,
        u_inf: float,
        v_inf: float,
    ) -> Optional[dict]:
        """Add wall distance contour and y+ distribution."""
        if not self.layout.has_plot("wall_dist"):
            return None
        
        ni, nj = xc.shape
        A, B = np.meshgrid(np.arange(ni), np.arange(nj), indexing='ij')
        cfg = color_config['wall_dist']
        
        wall_row, _ = self.layout.get_position("wall_dist")
        wall_cb = self.layout.get_colorbar_config("wall_dist")
        
        # Wall distance contour (static)
        self.tracker.add_static('carpet_wall_dist')
        fig.add_trace(go.Carpet(
            a=A.flatten(), b=B.flatten(),
            x=xc.flatten(), y=yc.flatten(),
            carpet='carpet_wall_dist',
            aaxis=dict(showgrid=False, showticklabels='none', showline=False),
            baxis=dict(showgrid=False, showticklabels='none', showline=False),
        ), row=wall_row, col=1)
        
        self.tracker.add_static('contour_wall_dist')
        fig.add_trace(go.Contourcarpet(
            a=A.flatten(), b=B.flatten(),
            z=sanitize_array(wall_distance).flatten(),
            carpet='carpet_wall_dist',
            colorscale=cfg.colorscale,
            zmin=cfg.cmin, zmax=cfg.cmax,
            contours=dict(coloring='fill', showlines=False),
            ncontours=self.N_CONTOURS,
            colorbar=dict(title='d/c', **wall_cb),
            showscale=True,
        ), row=wall_row, col=1)
        
        # y+ distribution
        i_start = n_wake
        i_end = ni - n_wake
        x_surf = xc[i_start:i_end, 0]
        
        snap = all_snapshots[-1]
        y_plus = compute_yplus_distribution(
            snap, wall_distance,
            grid_metrics.volume[:, 0],
            grid_metrics.Sj_x[:, 0], grid_metrics.Sj_y[:, 0],
            self.nu_laminar, u_inf, v_inf, i_start, i_end
        )
        
        self.tracker.add_animated('yplus_line')
        fig.add_trace(go.Scatter(
            x=x_surf, y=y_plus,
            mode='lines', line=dict(color='green', width=2),
            showlegend=False, name='y⁺',
        ), row=wall_row, col=2)
        
        self.tracker.add_static('yplus_ref_line')
        fig.add_trace(go.Scatter(
            x=[x_surf.min(), x_surf.max()], y=[1.0, 1.0],
            mode='lines', line=dict(color='gray', width=1, dash='dash'),
            showlegend=False, name='y⁺=1',
        ), row=wall_row, col=2)
        
        return {
            'wall_row': wall_row,
            'i_start': i_start,
            'i_end': i_end,
            'x_surf': x_surf,
            'yplus_max': float(np.max(y_plus)) if len(y_plus) > 0 else 1.0,
        }

    def add_surface_traces(
        self,
        fig: 'go.Figure',
        grid_metrics: 'FVMMetrics',
        X: np.ndarray,
        all_snapshots: List[Snapshot],
        n_wake: int,
        u_inf: float,
        v_inf: float,
        surface_reference: Optional[dict] = None,
    ) -> dict:
        """Add Cp and Cf surface plots."""
        if not self.layout.has_plot("cp"):
            return {}
            
        ni = grid_metrics.xc.shape[0]
        surface_row, _ = self.layout.get_position("cp")
        
        i_start = n_wake
        i_end = ni - n_wake
        x_surf = 0.5 * (X[:-1, 0] + X[1:, 0])
        x_surf_airfoil = x_surf[i_start:i_end]
        
        V_inf_sq = u_inf**2 + v_inf**2
        q_inf = max(0.5 * V_inf_sq, 1e-14)
        
        snap = all_snapshots[-1]
        Cp = sanitize_array(snap.p[:, 0] / q_inf, fill_value=0.0)[i_start:i_end]
        Cf = compute_cf_distribution(
            snap, grid_metrics.volume[:, 0],
            grid_metrics.Sj_x[:, 0], grid_metrics.Sj_y[:, 0],
            self.nu_laminar, u_inf, v_inf, i_start, i_end
        )
        
        # Calculate max Cf for constant y-axis range
        cf_max = max(0.01, safe_absmax(Cf, 0.01))
        
        self.tracker.add_animated('cp_line')
        fig.add_trace(go.Scatter(
            x=x_surf_airfoil, y=Cp,
            mode='lines', line=dict(color='blue', width=2),
            showlegend=False, name='Cp',
        ), row=surface_row, col=1)
        
        self.tracker.add_animated('cf_line')
        fig.add_trace(go.Scatter(
            x=x_surf_airfoil, y=Cf,
            mode='lines', line=dict(color='red', width=2),
            showlegend=False, name='Cf',
        ), row=surface_row, col=2)

        if surface_reference is not None:
            ref_x = surface_reference.get("x")
            ref_cp = surface_reference.get("cp")
            ref_cf = surface_reference.get("cf")
            if ref_x is not None and ref_cp is not None:
                self.tracker.add_static('cp_mfoil_line')
                fig.add_trace(go.Scatter(
                    x=ref_x, y=ref_cp,
                    mode='lines', line=dict(color='blue', width=2, dash='dot'),
                    showlegend=True, name='Cp (mfoil)',
                ), row=surface_row, col=1)
            if ref_x is not None and ref_cf is not None:
                self.tracker.add_static('cf_mfoil_line')
                fig.add_trace(go.Scatter(
                    x=ref_x, y=ref_cf,
                    mode='lines', line=dict(color='red', width=2, dash='dot'),
                    showlegend=True, name='Cf (mfoil)',
                ), row=surface_row, col=2)
        
        return {
            'surface_row': surface_row,
            'x_surf_airfoil': x_surf_airfoil,
            'i_start': i_start,
            'i_end': i_end,
            'cf_max': cf_max,
        }
