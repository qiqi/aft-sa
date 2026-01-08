"""
Plotly-based HTML animation for CFD results.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING, Any
from pathlib import Path

try:
    import plotly.graph_objects as go
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
    compute_yplus_distribution,
    make_log_range_slider_steps,
    extract_residual_value,
)
from ._layout import create_standard_layout

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
    # AFT diagnostic fields (optional)
    Re_Omega: Optional[np.ndarray] = None  # Vorticity Reynolds number
    Gamma: Optional[np.ndarray] = None     # Shape factor for AFT
    is_turb: Optional[np.ndarray] = None   # Turbulent fraction (0=laminar, 1=turbulent)


class AnimatedTraceTracker:
    """Tracks which traces should be animated and their indices."""
    
    def __init__(self):
        self._trace_count = 0
        self._animated_indices: List[int] = []
        self._static_traces: List[str] = []  # Names of static traces (e.g., 'carpet_1_1')
    
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


class PlotlyDashboard:
    """Accumulates CFD solution snapshots and exports as interactive HTML."""
    
    # Number of contour levels (lower = smaller file size)
    N_CONTOURS = 20
    
    def __init__(self, reynolds: float = 6e6, use_cdn: bool = True, compress: bool = True):
        """
        Initialize the dashboard.
        
        Parameters
        ----------
        reynolds : float
            Reynolds number for viscosity calculation.
        use_cdn : bool
            If True, reference plotly.js from CDN (saves ~3MB per file).
            If False, embed plotly.js in HTML (works offline but larger).
        compress : bool
            If True, gzip compress the HTML output (typically 5-10x smaller).
            Output file will have .html.gz extension.
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
        self.compress: bool = compress
    
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
        Q_int = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
        
        # Get residual
        residual = self._parse_residual(residual_history)
        
        # Extract freestream values
        if freestream is not None:
            self.p_inf = freestream.p_inf
            self.u_inf = freestream.u_inf
            self.v_inf = freestream.v_inf
        
        # Create snapshot
        snapshot = self._create_snapshot(Q_int, iteration, residual, cfl, C_pt, residual_field,
                                         Re_Omega, Gamma, is_turb)
        
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
        Re_Omega: Optional[np.ndarray] = None,
        Gamma: Optional[np.ndarray] = None,
        is_turb: Optional[np.ndarray] = None,
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
        
        # Residual field (already scaled RMS if from get_scaled_residual_field)
        res_field = None
        if residual_field is not None:
            # Handle both old-style 4-component arrays and new-style 2D scalar arrays
            if residual_field.ndim == 3:
                # Legacy: 4-component array, compute RMS across equations
                res_clipped = np.clip(residual_field, -max_safe_vel, max_safe_vel)
                res_field = sanitize_array(np.sqrt(np.mean(res_clipped**2, axis=2)), fill_value=1e-12)
            else:
                # New: already a 2D scalar field (scaled RMS)
                res_clipped = np.clip(residual_field, 0, max_safe_vel)
                res_field = sanitize_array(res_clipped, fill_value=1e-12)
        
        # AFT diagnostic fields
        Re_Omega_arr = None
        if Re_Omega is not None:
            Re_Omega_arr = sanitize_array(Re_Omega, fill_value=1.0).copy()
        Gamma_arr = None
        if Gamma is not None:
            Gamma_arr = sanitize_array(Gamma, fill_value=0.0).copy()
        is_turb_arr = None
        if is_turb is not None:
            is_turb_arr = sanitize_array(is_turb, fill_value=0.0).copy()
        
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
            Re_Omega=Re_Omega_arr,
            Gamma=Gamma_arr,
            is_turb=is_turb_arr,
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
        target_yplus: float = 1.0,
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
        has_aft = self.snapshots[0].Re_Omega is not None and self.snapshots[0].Gamma is not None
        
        # Create figure (and layout) first - needed for colorbar positioning
        fig = self._create_figure(has_wall_dist, has_surface_data, has_cpt, has_res_field, has_aft)
        
        # Compute color ranges (without colorbar positions - those come from layout)
        color_config = compute_color_ranges(
            self.snapshots, self.nu_laminar, has_cpt, has_res_field,
            has_aft=has_aft, has_wall_dist=has_wall_dist, has_surface=has_surface_data
        )
        
        # Initialize trace tracker for animation
        self.tracker = AnimatedTraceTracker()
        
        # Add all traces (tracker records which are animated)
        xc, yc = grid_metrics.xc, grid_metrics.yc
        self._add_contour_traces(fig, xc, yc, all_snapshots[-1], color_config, has_cpt, has_res_field)
        
        # Add AFT diagnostic traces (animated)
        if has_aft:
            self._add_aft_traces(fig, xc, yc, all_snapshots[-1], color_config,
                                has_wall_dist, has_surface_data)
        
        # Convergence traces are NOT animated (static line plot that shows full history)
        self._add_convergence_traces(fig, has_aft)
        
        yplus_params = None
        if has_wall_dist:
            yplus_params = self._add_wall_distance_trace(
                fig, xc, yc, wall_distance, color_config,
                grid_metrics, all_snapshots, n_wake, mu_laminar, has_aft
            )
            # Warn if max y+ exceeds 2x target
            if yplus_params and yplus_params.get('yplus_max', 0) > 2 * target_yplus:
                print(f"  WARNING: Max y+ = {yplus_params['yplus_max']:.1f} exceeds 2x target ({target_yplus:.1f})")
                print("           Grid may be too coarse near the wall")
        
        surface_params = None
        if has_surface_data:
            surface_params = self._add_surface_traces(
                fig, grid_metrics, X, all_snapshots, n_wake, mu_laminar, has_wall_dist, has_aft
            )
        
        # Create animation frames using tracker's animated indices
        self._add_animation_frames(
            fig, xc, yc, all_snapshots, color_config, has_cpt, has_res_field,
            has_surface_data, surface_params, grid_metrics, n_wake, mu_laminar, 
            has_aft, wall_distance, yplus_params
        )
        
        # Configure layout
        self._configure_layout(fig, all_snapshots, color_config, has_wall_dist, has_surface_data, 
                              surface_params, yplus_params, has_aft, has_cpt, has_res_field)
        
        # Write HTML with optional CDN and gzip compression
        include_plotlyjs: Any = 'cdn' if self.use_cdn else True
        
        if self.compress:
            import gzip
            # Write to string first, then compress
            html_str = fig.to_html(auto_play=False, include_plotlyjs=include_plotlyjs)
            gz_path = output_path.with_suffix('.html.gz')
            with gzip.open(str(gz_path), 'wt', encoding='utf-8', compresslevel=9) as f:
                f.write(html_str)
            # Get file sizes for comparison
            uncompressed_size = len(html_str.encode('utf-8'))
            compressed_size = gz_path.stat().st_size
            ratio = uncompressed_size / compressed_size if compressed_size > 0 else 1
            print(f"Saved compressed HTML to: {gz_path}")
            print(f"  Size: {compressed_size / 1e6:.1f} MB (was {uncompressed_size / 1e6:.1f} MB, {ratio:.1f}x compression)")
            output_path = gz_path
        else:
            fig.write_html(str(output_path), auto_play=False, include_plotlyjs=include_plotlyjs)
            print(f"Saved HTML animation to: {output_path}")
        
        if self.use_cdn:
            print("  Note: Requires internet connection (using plotly.js CDN)")
        print("  Use sliders at top-right to adjust color ranges")
        
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
        has_aft: bool = False,
    ) -> 'go.Figure':
        """Create the subplot figure layout using DashboardLayout."""
        # Create layout using the standard factory
        self.layout = create_standard_layout(
            has_cpt=has_cpt,
            has_res_field=has_res_field,
            has_aft=has_aft,
            has_wall_dist=has_wall_dist,
            has_surface=has_surface_data,
        )
        return self.layout.build_figure()
    
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
        
        # Standard plots that might be in the layout
        standard_plots = [
            ('pressure', 'p', color_config['pressure']),
            ('u_vel', 'u', color_config['velocity']),
            ('v_vel', 'v', color_config['velocity']),
            ('cpt' if has_cpt else 'nu', 'cpt' if has_cpt else 'nu', color_config['pressure']),
            ('residual' if has_res_field else 'vel_mag', 'res' if has_res_field else 'mag', color_config['residual']),
            ('grid', 'grid', None),  # New grid pane
            ('chi', 'chi', color_config['chi']),
        ]
        
        # Colorbar title mapping
        colorbar_titles = {
            'pressure': 'Δp',
            'cpt': 'C_pt',
            'nu': 'ν̃',
            'u_vel': 'Δu',
            'v_vel': 'Δv',
            'residual': 'log₁₀(R)',
            'vel_mag': '|V|',
            'chi': 'log₁₀(χ)',
        }
        
        for name, data_key, cfg in standard_plots:
            if not self.layout.has_plot(name):
                continue
                
            row, col = self.layout.get_position(name)
            carpet_id = f'carpet_{name}'
            
            # Prepare data
            data = None
            if name == 'pressure':
                data = sanitize_array(snapshot.p).flatten()
            elif name == 'cpt':
                data = field2.flatten()
            elif name == 'nu': # if cpt not present
                data = field2.flatten()
            elif name == 'u_vel':
                data = sanitize_array(snapshot.u).flatten()
            elif name == 'v_vel':
                data = sanitize_array(snapshot.v).flatten()
            elif name == 'residual':
                data = field5.flatten()
            elif name == 'vel_mag':
                data = field5.flatten()
            elif name == 'chi':
                data = chi_log_safe
            elif name == 'grid':
                # Grid only needs carpet, no data
                pass

            # Carpet trace
            is_grid_panel = (name == 'grid')
            
            # For grid panel, show grid lines! For others, hide them.
            axis_style = dict(
                showgrid=is_grid_panel, 
                gridcolor='black' if is_grid_panel else None,
                gridwidth=0.5 if is_grid_panel else None,
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
            
            # Add ContourCarpet ONLY if it's not the grid panel
            if not is_grid_panel and data is not None:
                trace_name = f'contour_{name}'
                
                # Get colorbar config
                show_colorbar = self.layout.should_show_colorbar(name)
                colorbar_dict = None
                if show_colorbar:
                    cb_config = self.layout.get_colorbar_config(name)
                    # Determine title key - handle cpt/nu and residual/vel_mag aliasing
                    title_key = name
                    if name == 'cpt' and not has_cpt: title_key = 'nu'
                    if name == 'residual' and not has_res_field: title_key = 'vel_mag'
                    
                    colorbar_dict = dict(
                        title=colorbar_titles.get(title_key, ''),
                        x=cb_config['x'],
                        y=cb_config['y'],
                        len=cb_config['len'],
                        tickformat='.2f' if name in ['pressure', 'u_vel', 'v_vel'] else '.1f',
                    )

                self.tracker.add_animated(trace_name)
                fig.add_trace(go.Contourcarpet(
                    a=A.flatten(), b=B.flatten(), z=data,
                    carpet=carpet_id,
                    colorscale=cfg.colorscale,
                    zmin=cfg.cmin, zmax=cfg.cmax,
                    contours=dict(coloring='fill', showlines=False),
                    ncontours=self.N_CONTOURS,
                    colorbar=colorbar_dict,
                    showscale=show_colorbar,
                ), row=row, col=col)
    
    def _add_aft_traces(
        self,
        fig: 'go.Figure',
        xc: np.ndarray,
        yc: np.ndarray,
        snapshot: Snapshot,
        color_config: dict,
        has_wall_dist: bool = False,
        has_surface: bool = False,
    ) -> None:
        """Add AFT diagnostic field traces (Re_Omega, Gamma, and is_turb)."""
        if snapshot.Re_Omega is None or snapshot.Gamma is None:
            return
        
        ni, nj = xc.shape
        A, B = np.meshgrid(np.arange(ni), np.arange(nj), indexing='ij')
        
        # Re_Omega: log scale from 10 to 10000
        # Take log10 for display, clamp to valid range
        Re_Omega_safe = np.maximum(sanitize_array(snapshot.Re_Omega, fill_value=10.0), 1.0)
        Re_Omega_log = np.log10(Re_Omega_safe)
        
        # Gamma: linear scale from 0 to 2
        Gamma_safe = sanitize_array(snapshot.Gamma, fill_value=0.0)
        
        # is_turb: linear scale from 0 to 1
        is_turb_safe = sanitize_array(snapshot.is_turb, fill_value=0.0) if snapshot.is_turb is not None else np.zeros_like(Gamma_safe)
        
        # Get row positions from layout
        aft_row1, _ = self.layout.get_position("re_omega")
        # is_turb is in different row now (row 5) than re_omega (row 4)
        is_turb_row, _ = self.layout.get_position("is_turb")
        
        # Get colorbar configs from layout
        re_omega_cb = self.layout.get_colorbar_config("re_omega")
        gamma_cb = self.layout.get_colorbar_config("gamma")
        is_turb_cb = self.layout.get_colorbar_config("is_turb")
        
        # Re_Omega contour (col 1)
        self.tracker.add_static('carpet_re_omega')
        fig.add_trace(go.Carpet(
            a=A.flatten(), b=B.flatten(),
            x=xc.flatten(), y=yc.flatten(),
            carpet='carpet_re_omega',
            aaxis=dict(showgrid=False, showticklabels='none', showline=False),
            baxis=dict(showgrid=False, showticklabels='none', showline=False),
        ), row=aft_row1, col=1)
        
        self.tracker.add_animated('contour_re_omega')
        fig.add_trace(go.Contourcarpet(
            a=A.flatten(), b=B.flatten(),
            z=to_json_safe_list(Re_Omega_log),
            carpet='carpet_re_omega',
            colorscale='Viridis',
            zmin=1.0, zmax=4.0,  # log10(10) to log10(10000)
            contours=dict(coloring='fill', showlines=False),
            ncontours=self.N_CONTOURS,
            colorbar=dict(title='log₁₀(Re_Ω)', **re_omega_cb),
            showscale=True,
        ), row=aft_row1, col=1)
        
        # Gamma contour (col 2)
        self.tracker.add_static('carpet_gamma')
        fig.add_trace(go.Carpet(
            a=A.flatten(), b=B.flatten(),
            x=xc.flatten(), y=yc.flatten(),
            carpet='carpet_gamma',
            aaxis=dict(showgrid=False, showticklabels='none', showline=False),
            baxis=dict(showgrid=False, showticklabels='none', showline=False),
        ), row=aft_row1, col=2)
        
        self.tracker.add_animated('contour_gamma')
        fig.add_trace(go.Contourcarpet(
            a=A.flatten(), b=B.flatten(),
            z=to_json_safe_list(Gamma_safe),
            carpet='carpet_gamma',
            colorscale='Reds',
            zmin=0.0, zmax=2.0,
            contours=dict(coloring='fill', showlines=False),
            ncontours=self.N_CONTOURS,
            colorbar=dict(title='Γ', **gamma_cb),
            showscale=True,
        ), row=aft_row1, col=2)
        
        # is_turb contour (col 1, single column) - turbulent fraction indicator
        self.tracker.add_static('carpet_is_turb')
        fig.add_trace(go.Carpet(
            a=A.flatten(), b=B.flatten(),
            x=xc.flatten(), y=yc.flatten(),
            carpet='carpet_is_turb',
            aaxis=dict(showgrid=False, showticklabels='none', showline=False),
            baxis=dict(showgrid=False, showticklabels='none', showline=False),
        ), row=is_turb_row, col=2) # is_turb is now in col 2 (next to chi)
        
        self.tracker.add_animated('contour_is_turb')
        fig.add_trace(go.Contourcarpet(
            a=A.flatten(), b=B.flatten(),
            z=to_json_safe_list(is_turb_safe),
            carpet='carpet_is_turb',
            colorscale='RdYlBu_r',  # Red=turbulent, Blue=laminar
            zmin=0.0, zmax=1.0,
            contours=dict(coloring='fill', showlines=False),
            ncontours=self.N_CONTOURS,
            colorbar=dict(title='is_turb', **is_turb_cb),
            showscale=True,
        ), row=is_turb_row, col=2)
    
    def _add_convergence_traces(self, fig: 'go.Figure', has_aft: bool = False) -> None:
        """Add convergence history traces (static - not animated)."""
        eq_names = ['Pressure', 'U-velocity', 'V-velocity', 'ν̃ (nuHat)']
        eq_colors = ['blue', 'red', 'green', 'purple']
        
        # Get row from layout
        conv_row, _ = self.layout.get_position("convergence")
        
        if self.residual_history:
            all_iters = (self.iteration_history 
                        if len(self.iteration_history) == len(self.residual_history)
                        else list(range(len(self.residual_history))))
            
            for eq_idx in range(4):
                res_eq = [extract_residual_value(r, eq_idx) for r in self.residual_history]
                self.tracker.add_static(f'convergence_line_{eq_idx}')
                fig.add_trace(go.Scatter(
                    x=all_iters, y=res_eq,
                    mode='lines', line=dict(color=eq_colors[eq_idx], width=1.5),
                    showlegend=True, name=eq_names[eq_idx],
                    legendgroup='convergence',
                ), row=conv_row, col=1)
        
        # Snapshot markers - animated to show progress through iterations
        snapshot_iters = [s.iteration for s in self.snapshots]
        snapshot_res = [extract_residual_value(s.residual, 0) for s in self.snapshots]
        self.tracker.add_animated('snapshot_markers')
        fig.add_trace(go.Scatter(
            x=snapshot_iters, y=snapshot_res,
            mode='markers',
            marker=dict(color='black', size=8, symbol='circle'),
            showlegend=True, name='Snapshots',
            legendgroup='convergence',
        ), row=conv_row, col=1)
        
        # Divergence markers (static)
        if self.divergence_snapshots:
            div_iters = [s.iteration for s in self.divergence_snapshots]
            div_res = [extract_residual_value(s.residual, 0) for s in self.divergence_snapshots]
            self.tracker.add_static('divergence_markers')
            fig.add_trace(go.Scatter(
                x=div_iters, y=div_res,
                mode='markers',
                marker=dict(color='orange', size=12, symbol='triangle-up'),
                showlegend=True, name='Divergence',
                legendgroup='convergence',
            ), row=conv_row, col=1)
    
    def _add_wall_distance_trace(
        self,
        fig: 'go.Figure',
        xc: np.ndarray,
        yc: np.ndarray,
        wall_distance: np.ndarray,
        color_config: dict,
        grid_metrics: 'FVMMetrics',
        all_snapshots: List[Snapshot],
        n_wake: int,
        mu_laminar: float,
        has_aft: bool = False,
    ) -> Optional[dict]:
        """Add wall distance contour trace and y+ distribution plot."""
        ni, nj = xc.shape
        A, B = np.meshgrid(np.arange(ni), np.arange(nj), indexing='ij')
        cfg = color_config['wall_dist']
        
        # Get row from layout
        wall_row, _ = self.layout.get_position("wall_dist")
        
        # Get colorbar config from layout
        wall_cb = self.layout.get_colorbar_config("wall_dist")
        
        # Left: Wall distance contour (static - wall distance doesn't change)
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
        
        # Right: y+ distribution along airfoil surface (animated - depends on solution)
        i_start = n_wake
        i_end = ni - n_wake
        x_surf = xc[i_start:i_end, 0]  # x-coordinate at wall
        
        # Compute y+ for final snapshot
        snap = all_snapshots[-1]
        y_plus = compute_yplus_distribution(
            snap, wall_distance,
            grid_metrics.volume[:, 0],
            grid_metrics.Sj_x[:, 0], grid_metrics.Sj_y[:, 0],
            self.nu_laminar, self.u_inf, self.v_inf, i_start, i_end
        )
        
        self.tracker.add_animated('yplus_line')
        fig.add_trace(go.Scatter(
            x=x_surf, y=y_plus,
            mode='lines', line=dict(color='green', width=2),
            showlegend=False, name='y⁺',
        ), row=wall_row, col=2)
        
        # Add reference line at y+ = 1 (static)
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
    
    def _add_surface_traces(
        self,
        fig: 'go.Figure',
        grid_metrics: 'FVMMetrics',
        X: np.ndarray,
        all_snapshots: List[Snapshot],
        n_wake: int,
        mu_laminar: float,
        has_wall_dist: bool,
        has_aft: bool = False,
    ) -> dict:
        """Add Cp and Cf surface plot traces (animated)."""
        ni = grid_metrics.xc.shape[0]
        # Get row from layout
        surface_row, _ = self.layout.get_position("cp")
        
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
        
        # Compute ranges based on LAST snapshot (converged solution)
        # Using last snapshot prevents early unconverged data from dominating the scale
        Cp_last = np.clip(snap.p[:, 0], -1e10, 1e10) / q_inf
        _cp_min, _cp_max = safe_minmax(Cp_last[i_start:i_end], -2, 1)
        cf_max = max(0.01, safe_absmax(Cf, 0.01))
        
        # Cp and Cf are animated (change per snapshot)
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
        
        return {
            'surface_row': surface_row,
            'x_surf_airfoil': x_surf_airfoil,
            'i_start': i_start,
            'i_end': i_end,
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
        has_aft: bool = False,
        wall_distance: Optional[np.ndarray] = None,
        yplus_params: Optional[dict] = None,
    ) -> None:
        """Add animation frames for all snapshots.
        
        Uses self.tracker.animated_indices to determine which traces to update.
        Frame data order must match the order traces were added with add_animated().
        """
        ni, nj = xc.shape
        A, B = np.meshgrid(np.arange(ni), np.arange(nj), indexing='ij')
        
        divergence_iters = {s.iteration for s in self.divergence_snapshots}
        animated_indices = self.tracker.animated_indices
        
        frames = []
        for i, snap in enumerate(all_snapshots):
            frame_data = self._create_frame_data(
                snap, A, B, color_config, has_cpt, has_res_field,
                all_snapshots[:i+1], divergence_iters, has_aft,
                has_surface_data, surface_params, grid_metrics, mu_laminar,
                wall_distance, yplus_params,
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
        color_config: dict,
        has_cpt: bool,
        has_res_field: bool,
        snapshots_so_far: List[Snapshot],
        divergence_iters: set,
        has_aft: bool = False,
        has_surface_data: bool = False,
        surface_params: Optional[dict] = None,
        grid_metrics: Optional['FVMMetrics'] = None,
        mu_laminar: float = 1e-6,
        wall_distance: Optional[np.ndarray] = None,
        yplus_params: Optional[dict] = None,
    ) -> list:
        """Create frame data for a single snapshot.
        
        IMPORTANT: The order of items in frame_data must exactly match
        the order that animated traces were added via tracker.add_animated().
        """
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
        
        
        # Assemble frame data in the EXACT same order as traces were added to tracker
        frame_data = []
        
        standard_keys = [
            ('pressure', snap_p.flatten()),
            ('u_vel', snap_u.flatten()),
            ('v_vel', snap_v.flatten()),
            ('cpt' if has_cpt else 'nu', field2.flatten()),
            ('residual' if has_res_field else 'vel_mag', field5.flatten()),
            ('grid', None), # No z-data for grid
            ('chi', to_json_safe_list(chi_log)),
        ]
        
        # Color config mapping
        config_map = {
            'pressure': color_config['pressure'],
            'u_vel': color_config['velocity'],
            'v_vel': color_config['velocity'],
            'cpt': color_config['pressure'], # share range
            'nu': color_config['pressure'],  # share range
            'residual': color_config['residual'],
            'vel_mag': color_config['residual'], # share range
            'chi': color_config['chi'],
            'grid': None
        }

        for name, data in standard_keys:
            if not self.layout.has_plot(name):
                continue
            
            # Grid has no animated trace, so it contributes nothing to frame data!
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
                ncontours=self.N_CONTOURS
            ))
        
        # AFT fields: contour_re_omega, contour_gamma, contour_is_turb
        if has_aft:
            frame_data.extend(self._create_aft_frame_data(snap, A, B))
        
        # Snapshot markers (convergence plot)
        snapshot_iters = [s.iteration for s in snapshots_so_far]
        snapshot_res = [extract_residual_value(s.residual, 0) for s in snapshots_so_far]
        colors = ['orange' if s.iteration in divergence_iters else 'black' for s in snapshots_so_far]
        
        frame_data.append(go.Scatter(
            x=snapshot_iters, y=snapshot_res,
            mode='markers',
            marker=dict(color=colors, size=10, symbol='circle'),
        ))
        
        # y+ line (if wall distance available)
        if yplus_params is not None and wall_distance is not None and grid_metrics is not None:
            frame_data.extend(self._create_yplus_frame_data(snap, wall_distance, grid_metrics, yplus_params))
        
        # Cp and Cf lines (if surface data available)
        if has_surface_data and surface_params is not None and grid_metrics is not None:
            frame_data.extend(self._create_surface_frame_data(snap, surface_params, grid_metrics, mu_laminar))
        
        return frame_data
    
    def _create_aft_frame_data(
        self,
        snap: Snapshot,
        A: np.ndarray,
        B: np.ndarray,
    ) -> list:
        """Create AFT field frame data (Re_Omega, Gamma, is_turb)."""
        frame_data = []
        
        # Re_Omega
        Re_Omega_safe = np.maximum(sanitize_array(snap.Re_Omega, fill_value=10.0), 1.0)
        Re_Omega_log = np.log10(Re_Omega_safe)
        frame_data.append(go.Contourcarpet(
            a=A.flatten(), b=B.flatten(),
            z=to_json_safe_list(Re_Omega_log),
            carpet='carpet_re_omega',
            colorscale='Viridis',
            zmin=1.0, zmax=4.0,
            contours=dict(coloring='fill', showlines=False),
            ncontours=self.N_CONTOURS,
        ))
        
        # Gamma
        Gamma_safe = sanitize_array(snap.Gamma, fill_value=0.0)
        frame_data.append(go.Contourcarpet(
            a=A.flatten(), b=B.flatten(),
            z=to_json_safe_list(Gamma_safe),
            carpet='carpet_gamma',
            colorscale='Reds',
            zmin=0.0, zmax=2.0,
            contours=dict(coloring='fill', showlines=False),
            ncontours=self.N_CONTOURS,
        ))
        
        # is_turb
        is_turb_safe = sanitize_array(snap.is_turb, fill_value=0.0) if snap.is_turb is not None else np.zeros_like(Gamma_safe)
        frame_data.append(go.Contourcarpet(
            a=A.flatten(), b=B.flatten(),
            z=to_json_safe_list(is_turb_safe),
            carpet='carpet_is_turb',
            colorscale='RdYlBu_r',
            zmin=0.0, zmax=1.0,
            contours=dict(coloring='fill', showlines=False),
            ncontours=self.N_CONTOURS,
        ))
        
        return frame_data
    
    def _create_yplus_frame_data(
        self,
        snap: Snapshot,
        wall_distance: np.ndarray,
        grid_metrics: 'FVMMetrics',
        yplus_params: dict,
    ) -> list:
        """Create y+ distribution frame data."""
        i_start = yplus_params['i_start']
        i_end = yplus_params['i_end']
        x_surf = yplus_params['x_surf']
        
        y_plus = compute_yplus_distribution(
            snap, wall_distance,
            grid_metrics.volume[:, 0],
            grid_metrics.Sj_x[:, 0], grid_metrics.Sj_y[:, 0],
            self.nu_laminar, self.u_inf, self.v_inf, i_start, i_end
        )
        
        return [
            go.Scatter(x=x_surf, y=y_plus, mode='lines', line=dict(color='green', width=2)),
        ]
    
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
        yplus_params: Optional[dict] = None,
        has_aft: bool = False,
        has_cpt: bool = True,
        has_res_field: bool = True,
    ) -> None:
        """Configure figure layout, sliders, and axes."""
        divergence_iters = {s.iteration for s in self.divergence_snapshots}
        
        # Get row numbers from layout
        conv_row, _ = self.layout.get_position("convergence")
        wall_row = self.layout.get_position("wall_dist")[0] if has_wall_dist else None
        
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
        
        # Calculate dynamic trace indices
        # Order MUST match _add_contour_traces: pressure, u, v, (cpt/nu), (res/mag), grid, chi
        idx = 0
        indices = {}
        # Keys used in standard_plots loop in _add_contour_traces
        trace_order = ['pressure', 'u_vel', 'v_vel', 
                      'cpt' if has_cpt else 'nu', 
                      'residual' if has_res_field else 'vel_mag', 
                      'grid', 'chi']
        
        for name in trace_order:
            if self.layout.has_plot(name):
                # Carpet trace
                idx += 1
                if name != 'grid':
                    # Contour trace (this is the one sliders affect)
                    indices[name] = idx
                    idx += 1
        
        # Get indices for each slider
        p_idx = indices.get('pressure')
        u_idx = indices.get('u_vel')
        v_idx = indices.get('v_vel')
        cpt_key = 'cpt' if has_cpt else 'nu'
        cpt_idx = indices.get(cpt_key)
        res_key = 'residual' if has_res_field else 'vel_mag'
        res_idx = indices.get(res_key)
        chi_idx = indices.get('chi')
        
        # Define slider targets
        # Pressure slider -> Pressure plot AND Cpt/Nu plot (since they share P scaling often)
        p_targets = [i for i in [p_idx, cpt_idx] if i is not None]
        
        # Velocity slider -> U and V plots
        v_targets = [i for i in [u_idx, v_idx] if i is not None]
        
        # Residual slider
        r_targets = [res_idx] if res_idx is not None else []
        
        # Chi slider
        chi_targets = [chi_idx] if chi_idx is not None else []
        
        p_steps = make_log_range_slider_steps(p_cfg.cmax, p_targets, False, n_steps)
        v_steps = make_log_range_slider_steps(v_cfg.cmax, v_targets, False, n_steps)
        # Residual: 8 decades slider range, 8 decades display
        r_steps = make_log_range_slider_steps(r_cfg.cmax, r_targets, True, n_steps, 
                                              slider_decades=8.0, display_decades=8.0)
        # Chi: 4 decades slider range, 5 decades display
        chi_steps = make_log_range_slider_steps(chi_cfg.cmax, chi_targets, True, n_steps,
                                                slider_decades=4.0, display_decades=5.0)
        
        # Height: ~350px per row, with some base margin
        height = 350 * self.layout.n_rows + 100
        
        fig.update_layout(
            showlegend=False,
            dragmode='pan',  # Default to pan instead of box zoom
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
            height=height,
            width=1400,
            margin=dict(t=80),
        )
        
        # Get all contour plot positions from layout for isotropic axes
        contour_plot_names = ["pressure", "cpt" if has_cpt else "nu",
                              "u_vel", "v_vel", "residual" if has_res_field else "vel_mag", "grid", "chi"]
        if has_aft:
            contour_plot_names.extend(["re_omega", "gamma", "is_turb"])
        if has_wall_dist:
            contour_plot_names.append("wall_dist")
        
        field_positions = []
        for name in contour_plot_names:
            if self.layout.has_plot(name):
                field_positions.append(self.layout.get_position(name))
        
        # Apply isotropic scaling to ALL 2D contour plots
        # Use scaleanchor for isotropic aspect, matches for linked pan/zoom
        # IMPORTANT: all contour plots share the same x and y ranges for consistent behavior
        for i, (row, col) in enumerate(field_positions):
            # Get the yaxis key for scaleanchor reference
            yaxis_key = self._get_subplot_yaxis_key(fig, row, col)
            yaxis_name = yaxis_key.replace('axis', '')
            
            # Common settings for all contour plots
            x_settings = dict(
                title_text='x', range=[-0.5, 1.5],
                scaleanchor=yaxis_name, scaleratio=1,
            )
            y_settings = dict(
                title_text='y', range=[-0.5625, 0.5625],
            )
            
            # Add matches for all plots except the first
            if i > 0:
                x_settings['matches'] = 'x'
                y_settings['matches'] = 'y'
            
            fig.update_xaxes(**x_settings, row=row, col=col)
            fig.update_yaxes(**y_settings, row=row, col=col)
        
        # Convergence plot
        fig.update_xaxes(title_text='Iteration', matches=None, autorange=True, fixedrange=True, row=conv_row, col=1)
        fig.update_yaxes(title_text='RMS Residual', matches=None, type='log', autorange=True, fixedrange=True, row=conv_row, col=1)
        
        if has_wall_dist and yplus_params and wall_row is not None:
            # y+ plot - log scale with fixed range
            fig.update_xaxes(title_text='x/c', range=[0, 1], matches=None, fixedrange=True, row=wall_row, col=2)
            fig.update_yaxes(title_text='y⁺', type='log', range=[np.log10(0.0005), np.log10(5)], 
                           matches=None, fixedrange=True, row=wall_row, col=2)
        
        if has_surface_data and surface_params:
            surface_row = surface_params['surface_row']
            # Cp and Cf plots have independent axes (don't scale with 2D contours)
            fig.update_xaxes(title_text='x/c', range=[0, 1], matches=None, fixedrange=True, row=surface_row, col=1)
            fig.update_yaxes(title_text='Cp', autorange='reversed', matches=None, fixedrange=True, row=surface_row, col=1)
            fig.update_xaxes(title_text='x/c', range=[0, 1], matches=None, fixedrange=True, row=surface_row, col=2)
            fig.update_yaxes(title_text='Cf', range=[0, surface_params['cf_max'] * 1.1], matches=None, fixedrange=True, row=surface_row, col=2)
    
    def _get_subplot_xaxis_key(self, fig: 'go.Figure', row: int, col: int) -> str:
        """Get the xaxis key for a subplot, handling complex layouts.
        
        Plotly's subplot indices can be non-sequential when rows have
        colspan or single-column configurations.
        """
        # Get grid reference from figure - this handles the complexity
        refs = fig._grid_ref
        if refs is not None and row <= len(refs):
            row_refs = refs[row - 1]  # 0-indexed
            if col <= len(row_refs) and row_refs[col - 1] is not None:
                subplot_ref = row_refs[col - 1]
                # subplot_ref is a tuple with SubplotRef objects
                if subplot_ref and len(subplot_ref) > 0:
                    return subplot_ref[0].layout_keys[0]
        # Fallback
        return 'xaxis'
    
    def _get_subplot_yaxis_key(self, fig: 'go.Figure', row: int, col: int) -> str:
        """Get the yaxis key for a subplot, handling complex layouts."""
        refs = fig._grid_ref
        if refs is not None and row <= len(refs):
            row_refs = refs[row - 1]  # 0-indexed
            if col <= len(row_refs) and row_refs[col - 1] is not None:
                subplot_ref = row_refs[col - 1]
                if subplot_ref and len(subplot_ref) > 0:
                    return subplot_ref[0].layout_keys[1]
        # Fallback
        return 'yaxis'
    
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
