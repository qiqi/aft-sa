"""
Plot type definitions and specifications for the dashboard layout system.

This module defines the building blocks for declarative dashboard layouts:
- PlotType: Enum of supported plot types
- ColorbarConfig: Configuration for colorbars on 2D plots
- PlotSpec: Specification for a single subplot
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Tuple


class PlotType(Enum):
    """Types of plots supported in the dashboard."""
    CONTOUR_2D = "contour"       # 2D field contour (pressure, velocity, etc.)
    LINE_SURFACE = "surface"     # 1D line plot along airfoil surface (Cp, Cf)
    CONVERGENCE = "convergence"  # Convergence history vs iteration
    LINE_1D = "line"             # Generic 1D line plot


@dataclass
class ColorbarConfig:
    """Configuration for a colorbar on a 2D contour plot."""
    title: str = ""
    colorscale: str = "RdBu_r"
    symmetric: bool = False      # If True, range is symmetric around 0
    log_scale: bool = False      # If True, data is log-transformed
    fixed_range: Optional[Tuple[float, float]] = None  # (min, max) or None for auto
    
    # Position adjustment (relative to auto-computed position)
    x_offset: float = 0.0
    
    def to_dict(self, x: float, y: float, length: float) -> Dict[str, Any]:
        """Convert to Plotly colorbar dict."""
        return {
            "title": self.title,
            "x": x + self.x_offset,
            "y": y,
            "len": length,
        }


@dataclass  
class PlotSpec:
    """Specification for a single subplot in the dashboard.
    
    Parameters
    ----------
    name : str
        Unique identifier for this plot (e.g., "pressure", "convergence")
    plot_type : PlotType
        Type of plot (contour, line, convergence)
    title : str
        Display title shown above the plot
    col_span : int
        Number of columns this plot spans (1 = half width, 2 = full width)
    colorbar : Optional[ColorbarConfig]
        Colorbar configuration for contour plots
    data_key : str
        Key to extract data from Snapshot (e.g., "p", "u", "is_turb")
    sharecontour : Optional[str]
        Name of another plot whose colorbar to share (no colorbar shown for this plot)
    """
    name: str
    plot_type: PlotType
    title: str
    col_span: int = 1
    colorbar: Optional[ColorbarConfig] = None
    data_key: str = ""
    sharecontour: Optional[str] = None
    
    def get_subplot_type(self) -> str:
        """Return Plotly subplot type string."""
        if self.plot_type == PlotType.CONVERGENCE:
            return "scatter"
        elif self.plot_type in (PlotType.LINE_SURFACE, PlotType.LINE_1D):
            return "scatter"
        else:
            return "xy"  # Contour plots use xy type


# Pre-defined colorbar configurations for common field types
COLORBARS = {
    "pressure": ColorbarConfig(
        title="p - p∞",
        colorscale="RdBu_r",
        symmetric=True,
    ),
    "cpt": ColorbarConfig(
        title="C_pt",
        colorscale="RdBu_r",
        symmetric=True,
    ),
    "velocity": ColorbarConfig(
        title="",
        colorscale="RdBu_r",
        symmetric=True,
    ),
    "chi": ColorbarConfig(
        title="log₁₀(χ)",
        colorscale="Viridis",
        log_scale=True,
    ),
    "residual": ColorbarConfig(
        title="log₁₀(R)",
        colorscale="Viridis",
        log_scale=True,
    ),
    "re_omega": ColorbarConfig(
        title="log₁₀(Re_Ω)",
        colorscale="Plasma",
        log_scale=True,
    ),
    "gamma": ColorbarConfig(
        title="Γ",
        colorscale="Viridis",
    ),
    "is_turb": ColorbarConfig(
        title="is_turb",
        colorscale="RdYlBu_r",  # Red=turbulent, Blue=laminar
        fixed_range=(0.0, 1.0),
    ),
    "wall_dist": ColorbarConfig(
        title="d/c",
        colorscale="Viridis",
    ),
}
