"""
Dashboard layout manager for Plotly-based CFD visualization.

This module provides a declarative way to define subplot layouts:
- DashboardLayout: Manages the grid of subplots
- Automatic row/column positioning
- Automatic colorbar positioning
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from .._plot_registry import PlotSpec, PlotType
from .._html_components import make_log_range_slider_steps  # Needed for control configuration


@dataclass
class LayoutRow:
    """A single row in the dashboard layout."""
    left: PlotSpec
    right: Optional[PlotSpec] = None  # None if left spans full width or single column
    
    @property
    def is_full_width(self) -> bool:
        """True only if explicitly set to span 2 columns."""
        return self.left.col_span == 2
    
    @property
    def is_single_column(self) -> bool:
        """True if row has only left plot but not full-width."""
        return self.right is None and self.left.col_span == 1


class DashboardLayout:
    """Declarative layout manager for dashboard plots."""
    
    # Layout constants
    HORIZONTAL_SPACING = 0.12
    VERTICAL_SPACING = 0.06
    COLORBAR_WIDTH = 0.04
    TOP_MARGIN = 0.05
    BOTTOM_MARGIN = 0.10
    
    # Colorbar x-positions
    COL1_COLORBAR_X = 0.44
    COL2_COLORBAR_X = 1.00
    
    def __init__(self):
        self.rows: List[LayoutRow] = []
        self._position_cache: Dict[str, Tuple[int, int]] = {}
        self._built = False
        self._figure = None
    
    def add_row(self, left: PlotSpec, right: Optional[PlotSpec] = None) -> 'DashboardLayout':
        """Add a row with one or two plots."""
        self.rows.append(LayoutRow(left, right))
        self._built = False
        return self
    
    def add_full_width(self, plot: PlotSpec) -> 'DashboardLayout':
        """Add a full-width plot (spans both columns)."""
        plot.col_span = 2
        self.rows.append(LayoutRow(plot, None))
        self._built = False
        return self
    
    @property
    def n_rows(self) -> int:
        """Total number of rows in the layout."""
        return len(self.rows)
    
    def _build_position_cache(self) -> None:
        """Build the position lookup cache."""
        self._position_cache.clear()
        for row_idx, row in enumerate(self.rows):
            row_num = row_idx + 1  # Plotly uses 1-based indexing
            self._position_cache[row.left.name] = (row_num, 1)
            if row.right is not None:
                self._position_cache[row.right.name] = (row_num, 2)
        self._built = True
    
    def get_position(self, name: str) -> Tuple[int, int]:
        """Get (row, col) position for a named plot."""
        if not self._built:
            self._build_position_cache()
        if name not in self._position_cache:
            # It's possible the plot wasn't added to layout (e.g. optional features like AFT off)
            # Returun None or raise error? The original raised error, but we should check has_plot first.
            raise KeyError(f"Plot '{name}' not found in layout. Available: {list(self._position_cache.keys())}")
        return self._position_cache[name]
    
    def get_row_height(self) -> float:
        """Get the height of each row as a fraction of total height."""
        usable_height = 1.0 - self.TOP_MARGIN - self.BOTTOM_MARGIN
        return usable_height / self.n_rows if self.n_rows > 0 else 0.0
    
    def compute_colorbar_y(self, name: str) -> float:
        """Compute the y-position for a colorbar."""
        row, _ = self.get_position(name)
        row_height = self.get_row_height()
        return 1.0 - self.TOP_MARGIN - (row - 0.5) * row_height
    
    def compute_colorbar_len(self) -> float:
        """Compute the length of colorbars."""
        return self.get_row_height() * 0.9
    
    def get_colorbar_x(self, col: int) -> float:
        """Get colorbar x-position for a column."""
        if col == 1:
            return self.COL1_COLORBAR_X
        else:
            return self.COL2_COLORBAR_X
    
    def _get_subplot_titles(self) -> List[str]:
        """Build the list of subplot titles."""
        titles = []
        for row in self.rows:
            titles.append(row.left.title)
            if row.right is not None:
                titles.append(row.right.title)
        return titles
    
    def _get_specs(self) -> List[List[Optional[Dict[str, Any]]]]:
        """Build the specs list for make_subplots."""
        specs = []
        for row in self.rows:
            if row.is_full_width:
                specs.append([{"type": row.left.get_subplot_type(), "colspan": 2}, None])
            elif row.is_single_column:
                left_type = row.left.get_subplot_type()
                specs.append([{"type": left_type}, None])
            else:
                left_type = row.left.get_subplot_type()
                right_type = row.right.get_subplot_type() if row.right else "xy"
                specs.append([{"type": left_type}, {"type": right_type}])
        return specs
    
    def build_figure(self) -> 'go.Figure':
        """Build the Plotly figure with the configured layout."""
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required for HTML output")
        
        self._build_position_cache()
        
        fig = make_subplots(
            rows=self.n_rows,
            cols=2,
            subplot_titles=self._get_subplot_titles(),
            specs=self._get_specs(),
            horizontal_spacing=self.HORIZONTAL_SPACING,
            vertical_spacing=self.VERTICAL_SPACING,
            column_widths=[0.5, 0.5],
        )
        
        self._figure = fig
        return fig
    
    def get_subplot_y_domain(self, row: int, col: int) -> Tuple[float, float]:
        """Get the y-domain for a subplot from the actual figure."""
        if self._figure is None:
            # Fallback
            row_height = self.get_row_height()
            y_max = 1.0 - self.TOP_MARGIN - (row - 1) * (row_height + self.VERTICAL_SPACING)
            y_min = y_max - row_height
            return (y_min, y_max)
        
        subplot_idx = (row - 1) * 2 + col
        yaxis_key = 'yaxis' if subplot_idx == 1 else f'yaxis{subplot_idx}'
        
        layout_dict = self._figure.layout.to_plotly_json()
        if yaxis_key in layout_dict and 'domain' in layout_dict[yaxis_key]:
            return tuple(layout_dict[yaxis_key]['domain'])
        
        # Fallback
        row_height = self.get_row_height()
        y_max = 1.0 - self.TOP_MARGIN - (row - 1) * (row_height + self.VERTICAL_SPACING)
        y_min = y_max - row_height
        return (y_min, y_max)
    
    def has_plot(self, name: str) -> bool:
        """Check if a plot with the given name exists."""
        if not self._built:
            self._build_position_cache()
        return name in self._position_cache
    
    def get_plot_spec(self, name: str) -> Optional[PlotSpec]:
        """Get the PlotSpec for a named plot."""
        for row in self.rows:
            if row.left.name == name:
                return row.left
            if row.right is not None and row.right.name == name:
                return row.right
        return None
    
    def should_show_colorbar(self, name: str) -> bool:
        """Check if a plot should show its own colorbar."""
        spec = self.get_plot_spec(name)
        if spec is None:
            return True
        return spec.sharecontour is None
    
    def get_colorbar_config(self, name: str) -> Dict[str, Any]:
        """Get colorbar configuration for a named plot."""
        row, col = self.get_position(name)
        y_min, y_max = self.get_subplot_y_domain(row, col)
        
        y_center = (y_min + y_max) / 2
        cb_len = (y_max - y_min) * 0.9
        
        return {
            'x': self.get_colorbar_x(col),
            'y': y_center,
            'len': cb_len,
        }
    
    def _get_subplot_yaxis_key(self, fig: 'go.Figure', row: int, col: int) -> str:
        """Get the yaxis key for a subplot, handling complex layouts."""
        refs = fig._grid_ref
        if refs is not None and row <= len(refs):
            row_refs = refs[row - 1]
            if col <= len(row_refs) and row_refs[col - 1] is not None:
                subplot_ref = row_refs[col - 1]
                if subplot_ref and len(subplot_ref) > 0:
                    return subplot_ref[0].layout_keys[1]
        return 'yaxis'


def create_standard_layout(
    has_cpt: bool = True,
    has_res_field: bool = True,
    has_aft: bool = False,
    has_wall_dist: bool = False,
    has_surface: bool = False,
) -> DashboardLayout:
    """Create the standard CFD dashboard layout."""
    layout = DashboardLayout()
    
    # Row 1: Pressure + C_pt/Nu
    layout.add_row(
        PlotSpec("pressure", PlotType.CONTOUR_2D, "Pressure (p - p∞)", data_key="p"),
        PlotSpec("cpt" if has_cpt else "nu", PlotType.CONTOUR_2D, 
                 "Total Pressure Loss (C_pt)" if has_cpt else "Turbulent Viscosity (ν)",
                 data_key="C_pt" if has_cpt else "nu"),
    )
    
    # Row 2: U-velocity + V-velocity
    layout.add_row(
        PlotSpec("u_vel", PlotType.CONTOUR_2D, "U-velocity (u - u∞)", data_key="u"),
        PlotSpec("v_vel", PlotType.CONTOUR_2D, "V-velocity (v - v∞)", data_key="v"),
    )
    
    # Row 3: Residual + Grid
    layout.add_row(
        PlotSpec("residual" if has_res_field else "vel_mag", PlotType.CONTOUR_2D,
                 "Residual Field (log₁₀)" if has_res_field else "Velocity Magnitude",
                 data_key="residual_field" if has_res_field else "vel_mag"),
        PlotSpec("grid", PlotType.CONTOUR_2D, "Grid", data_key="grid"),
    )
    
    # Optional AFT rows
    if has_aft:
        layout.add_row(
            PlotSpec("re_omega", PlotType.CONTOUR_2D, "Re_Ω (Vorticity Reynolds Number)",
                     data_key="Re_Omega"),
            PlotSpec("gamma", PlotType.CONTOUR_2D, "Γ (AFT Shape Factor)",
                     data_key="Gamma"),
        )
        layout.add_row(
            PlotSpec("chi", PlotType.CONTOUR_2D, "χ = ν̃/ν (Turbulent/Laminar Viscosity Ratio)",
                     data_key="nu"),
            PlotSpec("is_turb", PlotType.CONTOUR_2D, "is_turb (Turbulent Fraction)",
                     data_key="is_turb"),
        )
    else:
        layout.add_row(
            PlotSpec("chi", PlotType.CONTOUR_2D, "χ = ν̃/ν (Turbulent/Laminar Viscosity Ratio)",
                     data_key="nu"),
        )
    
    # Convergence
    layout.add_full_width(
        PlotSpec("convergence", PlotType.CONVERGENCE, "Convergence History")
    )
    
    if has_wall_dist:
        layout.add_row(
            PlotSpec("wall_dist", PlotType.CONTOUR_2D, "Wall Distance (d/c)",
                     data_key="wall_dist"),
            PlotSpec("yplus", PlotType.LINE_1D, "y⁺ Distribution"),
        )
    
    if has_surface:
        layout.add_row(
            PlotSpec("cp", PlotType.LINE_SURFACE, "Pressure Coefficient (Cp)"),
            PlotSpec("cf", PlotType.LINE_SURFACE, "Skin Friction (Cf)"),
        )
    
    return layout


def configure_dashboard_controls(
    fig: 'go.Figure',
    layout_mgr: DashboardLayout,
    all_snapshots: List,
    color_config: Dict,
    has_wall_dist: bool,
    has_surface_data: bool,
    surface_params: Optional[Dict],
    yplus_params: Optional[Dict] = None,
    has_aft: bool = False,
    has_cpt: bool = True,
    has_res_field: bool = True,
) -> None:
    """Configure figure layout, sliders, and axes."""
    
    # Iteration slider steps
    # Note: divergence snapshots need to be identified but we only have the raw list here.
    # We'll assume the caller passes fully processed snapshots or we simply use iterations.
    snapshot_labels = [str(s.iteration) for s in all_snapshots]
    
    slider_steps = [
        dict(
            method='animate',
            args=[[label], dict(
                mode='immediate',
                frame=dict(duration=100, redraw=True),
                transition=dict(duration=0)
            )],
            label=label,
        )
        for label in snapshot_labels
    ]
    
    # Color range sliders
    n_steps = 50
    p_cfg = color_config['pressure']
    v_cfg = color_config['velocity']
    r_cfg = color_config['residual']
    chi_cfg = color_config['chi']
    
    # Map trace names to indices for sliders
    # Order matches trace generation order in traces.py
    # We must replicate the order logic here effectively or pass it in.
    # The original logic dynamically computed indices based on `has_plot`.
    
    indices = {}
    idx = 0
    trace_order = ['pressure', 'u_vel', 'v_vel', 
                  'cpt' if has_cpt else 'nu', 
                  'residual' if has_res_field else 'vel_mag', 
                  'grid', 'chi']
    
    # "grid" is created but usually static carpet + static contour, but here we only care about
    # the SLIDERS affecting the contour traces.
    # In traces.py, we add Carpet (static), them ContourCarpet (animated/colorable).
    
    for name in trace_order:
        if layout_mgr.has_plot(name):
            idx += 1 # Carpet trace (static)
            if name != 'grid':
                indices[name] = idx # Contour trace
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
    p_targets = [i for i in [p_idx, cpt_idx] if i is not None]
    v_targets = [i for i in [u_idx, v_idx] if i is not None]
    r_targets = [res_idx] if res_idx is not None else []
    chi_targets = [chi_idx] if chi_idx is not None else []
    
    p_steps = make_log_range_slider_steps(p_cfg.cmax, p_targets, False, n_steps)
    v_steps = make_log_range_slider_steps(v_cfg.cmax, v_targets, False, n_steps)
    r_steps = make_log_range_slider_steps(r_cfg.cmax, r_targets, True, n_steps, 
                                          slider_decades=8.0, display_decades=8.0)
    chi_steps = make_log_range_slider_steps(chi_cfg.cmax, chi_targets, True, n_steps,
                                            slider_decades=4.0, display_decades=5.0)
    
    height = 350 * layout_mgr.n_rows + 100
    
    fig.update_layout(
        showlegend=False,
        dragmode='pan',
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
    
    # Configure axes for all contour plots
    contour_plot_names = ["pressure", "cpt" if has_cpt else "nu",
                          "u_vel", "v_vel", "residual" if has_res_field else "vel_mag", "grid", "chi"]
    if has_aft:
        contour_plot_names.extend(["re_omega", "gamma", "is_turb"])
    if has_wall_dist:
        contour_plot_names.append("wall_dist")
    
    field_positions = []
    for name in contour_plot_names:
        if layout_mgr.has_plot(name):
            field_positions.append(layout_mgr.get_position(name))
    
    for i, (row, col) in enumerate(field_positions):
        yaxis_key = layout_mgr._get_subplot_yaxis_key(fig, row, col)
        yaxis_name = yaxis_key.replace('axis', '')
        
        x_settings = dict(
            title_text='x', range=[-0.5, 1.5],
            scaleanchor=yaxis_name, scaleratio=1,
            row=row, col=col
        )
        y_settings = dict(
            title_text='y', range=[-0.5625, 0.5625],
            row=row, col=col
        )
        
        if i > 0:
            x_settings['matches'] = 'x'
            y_settings['matches'] = 'y'
        
        fig.update_xaxes(**x_settings)
        fig.update_yaxes(**y_settings)
    
    # Convergence
    conv_row, _ = layout_mgr.get_position("convergence")
    fig.update_xaxes(title_text='Iteration', matches=None, autorange=True, fixedrange=True, row=conv_row, col=1)
    fig.update_yaxes(title_text='RMS Residual', matches=None, type='log', autorange=True, fixedrange=True, row=conv_row, col=1)
    
    # Wall dist
    if has_wall_dist and yplus_params and layout_mgr.has_plot("wall_dist"):
        wall_row, _ = layout_mgr.get_position("wall_dist")
        fig.update_xaxes(title_text='x/c', range=[0, 1], matches=None, fixedrange=True, row=wall_row, col=2)
        fig.update_yaxes(title_text='y⁺', type='log', range=[np.log10(0.0005), np.log10(5)], 
                       matches=None, fixedrange=True, row=wall_row, col=2)
    
    # Surface
    if has_surface_data and surface_params:
        surface_row = surface_params['surface_row']
        fig.update_xaxes(title_text='x/c', range=[0, 1], matches=None, fixedrange=True, row=surface_row, col=1)
        fig.update_yaxes(title_text='Cp', autorange='reversed', matches=None, fixedrange=True, row=surface_row, col=1)
        fig.update_xaxes(title_text='x/c', range=[0, 1], matches=None, fixedrange=True, row=surface_row, col=2)
        fig.update_yaxes(title_text='Cf', range=[0, surface_params['cf_max'] * 1.1], matches=None, fixedrange=True, row=surface_row, col=2)

