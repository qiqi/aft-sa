"""
Main entry point for the CFD Dashboard.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Any, TYPE_CHECKING

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from loguru import logger

from .._vtk_writer import write_vtk
from .._html_components import compute_color_ranges
from .data import DataManager
from .layout import create_standard_layout, configure_dashboard_controls
from .traces import AnimatedTraceTracker, TraceManager
from .animation import AnimationManager

if TYPE_CHECKING:
    from ...grid.metrics import FVMMetrics


class PlotlyDashboard:
    """Accumulates CFD solution snapshots and exports as interactive HTML."""
    
    def __init__(self, reynolds: float = 6e6, use_cdn: bool = True, compress: bool = True):
        self.data_mgr = DataManager()
        self.nu_laminar = 1.0 / reynolds if reynolds > 0 else 1e-6
        self.use_cdn = use_cdn
        self.compress = compress
        
        # Facade properties for backward compatibility if accessed directly
        self.snapshots = self.data_mgr.snapshots
        self.residual_history = self.data_mgr.residual_history
        self.divergence_snapshots = self.data_mgr.divergence_snapshots

    @property
    def num_snapshots(self) -> int:
        """Total number of snapshots."""
        return len(self.data_mgr.snapshots)

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
        Re_Omega: Optional[np.ndarray] = None,
        Gamma: Optional[np.ndarray] = None,
        is_turb: Optional[np.ndarray] = None,
    ) -> None:
        """Store current solution state with diagnostic data."""
        if freestream is not None:
            self.data_mgr.update_freestream(freestream.p_inf, freestream.u_inf, freestream.v_inf)
            
        self.data_mgr.store_snapshot(
            Q, iteration, residual_history, cfl, C_pt, residual_field,
            iteration_history, is_divergence_dump, Re_Omega, Gamma, is_turb
        )

    def save_html(
        self,
        filename: str,
        grid_metrics: 'FVMMetrics', 
        wall_distance: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
        n_wake: int = 0,
        mu_laminar: float = 1e-6,
        target_yplus: float = 1.0,
    ) -> str:
        """Export all snapshots as interactive HTML animation."""
        if not HAS_PLOTLY:
            logger.warning("Plotly is required for HTML output but not installed. Skipping.")
            return ""
        
        if not self.data_mgr.snapshots:
            logger.warning("No snapshots to save. Skipping HTML generation.")
            return ""
        
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get data
        all_snapshots = self.data_mgr.get_all_snapshots()
        last_snap = all_snapshots[-1]
        
        # Feature flags
        has_cpt = self.data_mgr.snapshots[0].C_pt is not None
        has_res_field = self.data_mgr.snapshots[0].residual_field is not None
        has_wall_dist = wall_distance is not None
        has_surface_data = X is not None and Y is not None
        has_aft = self.data_mgr.snapshots[0].Re_Omega is not None and self.data_mgr.snapshots[0].Gamma is not None
        
        # 1. Create Layout
        layout_mgr = create_standard_layout(
            has_cpt=has_cpt,
            has_res_field=has_res_field,
            has_aft=has_aft,
            has_wall_dist=has_wall_dist,
            has_surface=has_surface_data,
        )
        fig = layout_mgr.build_figure()
        
        # 2. Compute Color Ranges
        color_config = compute_color_ranges(
            self.data_mgr.snapshots, self.nu_laminar, has_cpt, has_res_field,
            has_aft=has_aft, has_wall_dist=has_wall_dist, has_surface=has_surface_data
        )
        
        # 3. Initialize Helpers
        tracker = AnimatedTraceTracker()
        trace_mgr = TraceManager(layout_mgr, tracker, self.nu_laminar)
        anim_mgr = AnimationManager(layout_mgr, tracker, self.nu_laminar)
        
        # 4. Add Traces
        xc, yc = grid_metrics.xc, grid_metrics.yc
        trace_mgr.add_contour_traces(fig, xc, yc, last_snap, color_config, has_cpt, has_res_field)
        
        if has_aft:
            trace_mgr.add_aft_traces(fig, xc, yc, last_snap, color_config)
            
        trace_mgr.add_convergence_traces(fig, self.data_mgr.residual_history, 
                                        self.data_mgr.iteration_history, 
                                        self.data_mgr.snapshots, 
                                        self.data_mgr.divergence_snapshots)
        
        yplus_params = None
        if has_wall_dist:
            yplus_params = trace_mgr.add_wall_distance_trace(
                fig, xc, yc, wall_distance, color_config, grid_metrics, 
                all_snapshots, n_wake, self.data_mgr.u_inf, self.data_mgr.v_inf
            )
            if yplus_params and yplus_params.get('yplus_max', 0) > 2 * target_yplus:
                logger.warning(f"Max y+ = {yplus_params['yplus_max']:.1f} exceeds 2x target ({target_yplus:.1f})")

        surface_params = None
        if has_surface_data:
            surface_params = trace_mgr.add_surface_traces(
                fig, grid_metrics, X, all_snapshots, n_wake, 
                self.data_mgr.u_inf, self.data_mgr.v_inf
            )
            
        # 5. Create Animation Frames
        anim_mgr.add_animation_frames(
            fig, xc, yc, all_snapshots, self.data_mgr.divergence_snapshots,
            color_config, has_cpt, has_res_field, has_surface_data, surface_params,
            grid_metrics, n_wake, self.data_mgr.u_inf, self.data_mgr.v_inf,
            has_aft, wall_distance, yplus_params
        )
        
        # 6. Configure Layout (Controls, Sliders)
        configure_dashboard_controls(
            fig, layout_mgr, all_snapshots, color_config, has_wall_dist, has_surface_data,
            surface_params, yplus_params, has_aft, has_cpt, has_res_field
        )
        
        # 7. Write Output
        self._write_html(fig, output_path)
            
        # Write VTK
        vtk_path = output_path.with_suffix('.vtk')
        write_vtk(vtk_path, grid_metrics, last_snap, wall_distance,
                  self.data_mgr.u_inf, self.data_mgr.v_inf, self.nu_laminar)
        
        return str(output_path)
    
    def _write_html(self, fig: 'go.Figure', output_path: Path) -> None:
        """Write HTML file with optional compression."""
        include_plotlyjs: Any = 'cdn' if self.use_cdn else True
        
        if self.compress:
            import gzip
            html_str = fig.to_html(auto_play=False, include_plotlyjs=include_plotlyjs)
            gz_path = output_path.with_suffix('.html.gz')
            with gzip.open(str(gz_path), 'wt', encoding='utf-8', compresslevel=9) as f:
                f.write(html_str)
            
            uncompressed_size = len(html_str.encode('utf-8'))
            compressed_size = gz_path.stat().st_size
            ratio = uncompressed_size / compressed_size if compressed_size > 0 else 1
            uncompressed_size = len(html_str.encode('utf-8'))
            compressed_size = gz_path.stat().st_size
            ratio = uncompressed_size / compressed_size if compressed_size > 0 else 1
            logger.info(f"Saved compressed HTML to: {gz_path} (Size: {compressed_size / 1e6:.1f} MB, {ratio:.1f}x compression)")
        else:
            fig.write_html(str(output_path), auto_play=False, include_plotlyjs=include_plotlyjs)
            logger.info(f"Saved HTML animation to: {output_path}")
        
        if self.use_cdn:
            logger.info("Note: Animation requires internet connection (using plotly.js CDN)")
        logger.info("Use sliders at top-right to adjust color ranges")
