"""
Dashboard layout manager for Plotly-based CFD visualization.

This module provides a declarative way to define subplot layouts:
- DashboardLayout: Manages the grid of subplots
- Automatic row/column positioning
- Automatic colorbar positioning
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from ._plot_registry import PlotSpec, PlotType


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
    """Declarative layout manager for dashboard plots.
    
    Usage:
        layout = DashboardLayout()
        layout.add_row(PlotSpec("pressure", ...), PlotSpec("cpt", ...))
        layout.add_full_width(PlotSpec("convergence", ...))
        fig = layout.build_figure()
        row, col = layout.get_position("pressure")
    """
    
    # Layout constants
    # Increased horizontal spacing to leave room for colorbars between columns
    HORIZONTAL_SPACING = 0.12
    VERTICAL_SPACING = 0.06
    COLORBAR_WIDTH = 0.04
    TOP_MARGIN = 0.05
    BOTTOM_MARGIN = 0.10
    
    # Colorbar x-positions for each column
    # Col 1 colorbar: just to the right of left plot (before the gap)
    COL1_COLORBAR_X = 0.44
    # Col 2 colorbar: at right edge of figure
    COL2_COLORBAR_X = 1.00
    
    def __init__(self):
        self.rows: List[LayoutRow] = []
        self._position_cache: Dict[str, Tuple[int, int]] = {}
        self._built = False
    
    def add_row(self, left: PlotSpec, right: Optional[PlotSpec] = None) -> 'DashboardLayout':
        """Add a row with one or two plots.
        
        Parameters
        ----------
        left : PlotSpec
            Left subplot (or full-width if right is None and col_span=2)
        right : Optional[PlotSpec]
            Right subplot (None for single-column row)
            
        Returns
        -------
        self : DashboardLayout
            For method chaining
        """
        self.rows.append(LayoutRow(left, right))
        self._built = False
        return self
    
    def add_full_width(self, plot: PlotSpec) -> 'DashboardLayout':
        """Add a full-width plot (spans both columns).
        
        Parameters
        ----------
        plot : PlotSpec
            The plot specification (col_span will be set to 2)
            
        Returns
        -------
        self : DashboardLayout
            For method chaining
        """
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
        """Get (row, col) position for a named plot.
        
        Parameters
        ----------
        name : str
            The plot name (from PlotSpec.name)
            
        Returns
        -------
        Tuple[int, int]
            (row, col) in 1-based Plotly indexing
        """
        if not self._built:
            self._build_position_cache()
        if name not in self._position_cache:
            raise KeyError(f"Plot '{name}' not found in layout. Available: {list(self._position_cache.keys())}")
        return self._position_cache[name]
    
    def get_row_height(self) -> float:
        """Get the height of each row as a fraction of total height."""
        usable_height = 1.0 - self.TOP_MARGIN - self.BOTTOM_MARGIN
        return usable_height / self.n_rows
    
    def compute_colorbar_y(self, name: str) -> float:
        """Compute the y-position for a colorbar.
        
        Parameters
        ----------
        name : str
            The plot name
            
        Returns
        -------
        float
            Y-position for the colorbar center (0-1 scale)
        """
        row, _ = self.get_position(name)
        row_height = self.get_row_height()
        # Center of the row, counting from top
        return 1.0 - self.TOP_MARGIN - (row - 0.5) * row_height
    
    def compute_colorbar_len(self) -> float:
        """Compute the length of colorbars."""
        return self.get_row_height() * 0.9
    
    def get_colorbar_x(self, col: int) -> float:
        """Get colorbar x-position for a column.
        
        Parameters
        ----------
        col : int
            Column number (1 or 2)
            
        Returns
        -------
        float
            X-position for the colorbar (just to the right of the plot)
        """
        if col == 1:
            return self.COL1_COLORBAR_X
        else:
            return self.COL2_COLORBAR_X
    
    def _get_subplot_titles(self) -> List[str]:
        """Build the list of subplot titles.
        
        Plotly expects one title per subplot cell. For full-width (colspan=2)
        and single-column rows, the right cell is None in specs, so no title.
        """
        titles = []
        for row in self.rows:
            titles.append(row.left.title)
            if row.right is not None:
                # Two-column row: add right title
                titles.append(row.right.title)
            # For full-width and single-column: right cell is None in specs,
            # so no title needed (plotly matches titles to existing subplots)
        return titles
    
    def _get_specs(self) -> List[List[Optional[Dict[str, Any]]]]:
        """Build the specs list for make_subplots."""
        specs = []
        for row in self.rows:
            if row.is_full_width:
                specs.append([{"type": row.left.get_subplot_type(), "colspan": 2}, None])
            elif row.is_single_column:
                # Single left column only - right cell is empty
                left_type = row.left.get_subplot_type()
                specs.append([{"type": left_type}, None])
            else:
                left_type = row.left.get_subplot_type()
                right_type = row.right.get_subplot_type() if row.right else "xy"
                specs.append([{"type": left_type}, {"type": right_type}])
        return specs
    
    def build_figure(self) -> 'go.Figure':
        """Build the Plotly figure with the configured layout.
        
        Returns
        -------
        go.Figure
            Plotly figure with subplots configured
        """
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
            column_widths=[0.5, 0.5],  # Equal column widths
        )
        
        # Store reference to figure for domain-based colorbar positioning
        self._figure = fig
        
        return fig
    
    def get_subplot_y_domain(self, row: int, col: int) -> Tuple[float, float]:
        """Get the y-domain for a subplot from the actual figure.
        
        Parameters
        ----------
        row : int
            Row number (1-indexed)
        col : int  
            Column number (1 or 2)
            
        Returns
        -------
        Tuple[float, float]
            (y_min, y_max) domain values
        """
        if not hasattr(self, '_figure') or self._figure is None:
            # Fallback to manual calculation if figure not built yet
            row_height = self.get_row_height()
            y_max = 1.0 - self.TOP_MARGIN - (row - 1) * (row_height + self.VERTICAL_SPACING)
            y_min = y_max - row_height
            return (y_min, y_max)
        
        # Calculate yaxis index: row 1 col 1 = yaxis, row 1 col 2 = yaxis2, etc.
        subplot_idx = (row - 1) * 2 + col
        if subplot_idx == 1:
            yaxis_key = 'yaxis'
        else:
            yaxis_key = f'yaxis{subplot_idx}'
        
        layout_dict = self._figure.layout.to_plotly_json()
        if yaxis_key in layout_dict and 'domain' in layout_dict[yaxis_key]:
            return tuple(layout_dict[yaxis_key]['domain'])
        
        # Fallback
        row_height = self.get_row_height()
        y_max = 1.0 - self.TOP_MARGIN - (row - 1) * (row_height + self.VERTICAL_SPACING)
        y_min = y_max - row_height
        return (y_min, y_max)
    
    def get_all_plots(self) -> List[Tuple[PlotSpec, int, int]]:
        """Get all plots with their positions.
        
        Returns
        -------
        List[Tuple[PlotSpec, int, int]]
            List of (spec, row, col) tuples
        """
        if not self._built:
            self._build_position_cache()
        
        result = []
        for row_idx, row in enumerate(self.rows):
            row_num = row_idx + 1
            result.append((row.left, row_num, 1))
            if row.right is not None:
                result.append((row.right, row_num, 2))
        return result
    
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
        """Check if a plot should show its own colorbar.
        
        Returns False if the plot has sharecontour set to another plot.
        """
        spec = self.get_plot_spec(name)
        if spec is None:
            return True
        return spec.sharecontour is None
    
    def get_colorbar_config(self, name: str) -> Dict[str, Any]:
        """Get colorbar configuration for a named plot.
        
        Uses actual subplot domains from the figure for accurate positioning.
        
        Parameters
        ----------
        name : str
            The plot name
            
        Returns
        -------
        Dict[str, Any]
            Colorbar configuration dict with x, y, len keys
        """
        row, col = self.get_position(name)
        y_min, y_max = self.get_subplot_y_domain(row, col)
        
        # Colorbar y is the center of the subplot domain
        y_center = (y_min + y_max) / 2
        # Colorbar length is 90% of the subplot height
        cb_len = (y_max - y_min) * 0.9
        
        return {
            'x': self.get_colorbar_x(col),
            'y': y_center,
            'len': cb_len,
        }


def create_standard_layout(
    has_cpt: bool = True,
    has_res_field: bool = True,
    has_aft: bool = False,
    has_wall_dist: bool = False,
    has_surface: bool = False,
) -> DashboardLayout:
    """Create the standard CFD dashboard layout.
    
    Parameters
    ----------
    has_cpt : bool
        Include total pressure loss plot
    has_res_field : bool
        Include residual field plot
    has_aft : bool
        Include AFT diagnostic plots (Re_Omega, Gamma, is_turb)
    has_wall_dist : bool
        Include wall distance and y+ plots
    has_surface : bool
        Include surface distribution plots (Cp, Cf)
        
    Returns
    -------
    DashboardLayout
        Configured layout
    """
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
    
    # Row 3: Residual/VelMag + Chi
    layout.add_row(
        PlotSpec("residual" if has_res_field else "vel_mag", PlotType.CONTOUR_2D,
                 "Residual Field (log₁₀)" if has_res_field else "Velocity Magnitude",
                 data_key="residual_field" if has_res_field else "vel_mag"),
        PlotSpec("chi", PlotType.CONTOUR_2D, "χ = ν̃/ν (Turbulent/Laminar Viscosity Ratio)",
                 data_key="nu"),
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
            PlotSpec("is_turb", PlotType.CONTOUR_2D, "is_turb (Turbulent Fraction)",
                     data_key="is_turb"),
        )
    
    # Convergence (always full width)
    layout.add_full_width(
        PlotSpec("convergence", PlotType.CONVERGENCE, "Convergence History")
    )
    
    # Optional wall distance row
    if has_wall_dist:
        layout.add_row(
            PlotSpec("wall_dist", PlotType.CONTOUR_2D, "Wall Distance (d/c)",
                     data_key="wall_dist"),
            PlotSpec("yplus", PlotType.LINE_1D, "y⁺ Distribution"),
        )
    
    # Optional surface plots row
    if has_surface:
        layout.add_row(
            PlotSpec("cp", PlotType.LINE_SURFACE, "Pressure Coefficient (Cp)"),
            PlotSpec("cf", PlotType.LINE_SURFACE, "Skin Friction (Cf)"),
        )
    
    return layout
