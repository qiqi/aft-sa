"""
Plotly-based HTML Animation for CFD Results.

This module provides a PlotlyDashboard class that accumulates solution
snapshots during simulation and exports them as an interactive HTML
animation with comprehensive diagnostic information.

Features:
- 3x2 subplot layout: Pressure, Velocity Magnitude, Total Pressure Loss,
  U-velocity, V-velocity, and Residual History
- Interactive slider to scrub through iterations
- Play/Pause button for auto-advance
- Shared axes for synchronized zooming
- Auto-scaling color ranges per frame
- Residual history line plot updated per frame
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING, Dict, Any
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from ..constants import NGHOST

if TYPE_CHECKING:
    from ..grid.metrics import FVMMetrics


@dataclass
class Snapshot:
    """
    Stores a single timestep's solution data.
    
    All fields are interior cells only (no ghost cells).
    """
    iteration: int
    residual: float
    cfl: float
    p: np.ndarray           # Pressure, shape (NI, NJ)
    u: np.ndarray           # U-velocity, shape (NI, NJ)
    v: np.ndarray           # V-velocity, shape (NI, NJ)
    nu: np.ndarray          # Turbulent viscosity, shape (NI, NJ)
    C_pt: Optional[np.ndarray] = None  # Total pressure loss coefficient
    residual_field: Optional[np.ndarray] = None  # RMS of residual per cell
    # Multigrid level data (list of dicts with 'p', 'u', 'v', 'xc', 'yc')
    mg_levels: Optional[List[Dict[str, np.ndarray]]] = None
    # Scalar diagnostics
    vel_max: float = 0.0
    p_min: float = 0.0
    p_max: float = 0.0


class PlotlyDashboard:
    """
    Accumulates CFD solution snapshots and exports as interactive HTML.
    
    Provides comprehensive diagnostic visualization including:
    - Primary flow variables (p, u, v)
    - Derived quantities (velocity magnitude, total pressure loss)
    - Residual history and field
    - Multigrid level solutions (if enabled)
    
    Usage:
        dashboard = PlotlyDashboard()
        
        for iteration in range(max_iter):
            # ... solver step ...
            if iteration % diagnostic_freq == 0:
                dashboard.store_snapshot(Q, iteration, residual_history, cfl,
                                         C_pt=C_pt, residual_field=R_field)
        
        dashboard.save_html("animation.html", grid_metrics)
    """
    
    def __init__(self):
        """Initialize empty snapshot storage."""
        self.snapshots: List[Snapshot] = []
        self.residual_history: List[float] = []
    
    def store_snapshot(
        self, 
        Q: np.ndarray, 
        iteration: int, 
        residual_history: List[float],
        cfl: float = 0.0,
        C_pt: Optional[np.ndarray] = None,
        residual_field: Optional[np.ndarray] = None,
        mg_levels: Optional[List[Dict[str, Any]]] = None,
        freestream: Any = None,
    ) -> None:
        """
        Store the current solution state with diagnostic data.
        
        Parameters
        ----------
        Q : ndarray, shape (NI + 2*NGHOST, NJ + 2*NGHOST, 4)
            Full state vector including ghost cells.
            Components: [p, u, v, nu]
        iteration : int
            Current iteration number.
        residual_history : list of float
            Full history of residuals.
        cfl : float
            Current CFL number.
        C_pt : ndarray, optional
            Total pressure loss coefficient field.
        residual_field : ndarray, optional
            Residual magnitude field.
        mg_levels : list of dict, optional
            Multigrid level data for visualization.
        freestream : FreestreamConditions, optional
            Freestream conditions for C_pt calculation.
        """
        # Extract interior cells only
        Q_int = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
        
        # Get current residual
        residual = residual_history[-1] if residual_history else 0.0
        
        # Compute derived quantities
        p = Q_int[:, :, 0]
        u = Q_int[:, :, 1]
        v = Q_int[:, :, 2]
        vel_mag = np.sqrt(u**2 + v**2)
        
        # Compute C_pt if freestream provided but C_pt not given
        if C_pt is None and freestream is not None:
            p_inf = freestream.p_inf
            u_inf = freestream.u_inf
            v_inf = freestream.v_inf
            V_inf_sq = u_inf**2 + v_inf**2
            p_total = p + 0.5 * (u**2 + v**2)
            p_total_inf = p_inf + 0.5 * V_inf_sq
            C_pt = (p_total_inf - p_total) / (0.5 * V_inf_sq + 1e-12)
        
        # Extract residual field magnitude if provided (take RMS of components)
        res_field = None
        if residual_field is not None:
            # residual_field shape: (NI, NJ, 4) - take RMS
            res_field = np.sqrt(np.mean(residual_field**2, axis=2))
        
        # Create snapshot with copies
        snapshot = Snapshot(
            iteration=iteration,
            residual=residual,
            cfl=cfl,
            p=p.copy(),
            u=u.copy(),
            v=v.copy(),
            nu=Q_int[:, :, 3].copy(),
            C_pt=C_pt.copy() if C_pt is not None else None,
            residual_field=res_field,
            mg_levels=mg_levels,
            vel_max=float(vel_mag.max()),
            p_min=float(p.min()),
            p_max=float(p.max()),
        )
        
        self.snapshots.append(snapshot)
        self.residual_history = list(residual_history)
    
    def save_html(self, filename: str, grid_metrics: 'FVMMetrics') -> str:
        """
        Export all snapshots as an interactive HTML animation.
        
        Parameters
        ----------
        filename : str
            Output HTML file path.
        grid_metrics : FVMMetrics
            Grid metrics object containing cell center coordinates.
        
        Returns
        -------
        str
            Path to the saved HTML file.
        """
        if not HAS_PLOTLY:
            print("Warning: plotly not installed. Skipping HTML animation.")
            return ""
        
        if not self.snapshots:
            print("Warning: No snapshots to save.")
            return ""
        
        # Create output directory if needed
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get grid coordinates (cell centers) - slice to match interior data
        xc = grid_metrics.xc[NGHOST:-NGHOST, NGHOST:-NGHOST]
        yc = grid_metrics.yc[NGHOST:-NGHOST, NGHOST:-NGHOST]
        
        # Determine layout based on available data
        snap0 = self.snapshots[0]
        has_cpt = snap0.C_pt is not None
        has_res_field = snap0.residual_field is not None
        
        # Create 3x2 subplot figure
        # Row 1: Pressure, Velocity Magnitude
        # Row 2: U-velocity, V-velocity  
        # Row 3: Total Pressure Loss / Residual Field, Residual History
        subplot_titles = [
            'Pressure (p)', 'Velocity Magnitude',
            'U-velocity', 'V-velocity',
            'Total Pressure Loss (C_pt)' if has_cpt else 'Turbulent Viscosity (ν)',
            'Residual History'
        ]
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=subplot_titles,
            specs=[
                [{"type": "heatmap"}, {"type": "heatmap"}],
                [{"type": "heatmap"}, {"type": "heatmap"}],
                [{"type": "heatmap"}, {"type": "scatter"}],  # Last is line plot
            ],
            horizontal_spacing=0.08,
            vertical_spacing=0.08,
        )
        
        # Compute velocity magnitude for first snapshot
        vel_mag0 = np.sqrt(snap0.u**2 + snap0.v**2)
        
        # Field 5: C_pt or nu
        field5_data = snap0.C_pt if has_cpt else snap0.nu
        field5_cmap = 'Reds' if has_cpt else 'Plasma'
        
        # Add base heatmaps (5 heatmaps)
        heatmap_configs = [
            (snap0.p, 1, 1, 'RdBu_r', 'p'),
            (vel_mag0, 1, 2, 'Viridis', '|V|'),
            (snap0.u, 2, 1, 'RdBu', 'u'),
            (snap0.v, 2, 2, 'RdBu', 'v'),
            (field5_data, 3, 1, field5_cmap, 'C_pt' if has_cpt else 'ν'),
        ]
        
        for data, row, col, colorscale, name in heatmap_configs:
            fig.add_trace(
                go.Heatmap(
                    z=data.T,
                    x=xc[:, 0],
                    y=yc[0, :],
                    colorscale=colorscale,
                    colorbar=dict(
                        len=0.25,
                        y=1.0 - (row - 0.5) / 3,
                        title=dict(text=name, side='right'),
                    ),
                    name=name,
                ),
                row=row, col=col
            )
        
        # Add residual history line plot (trace index 5)
        iterations = [s.iteration for s in self.snapshots]
        residuals = [s.residual for s in self.snapshots]
        
        # Add full history as background
        if self.residual_history:
            all_iters = list(range(len(self.residual_history)))
            fig.add_trace(
                go.Scatter(
                    x=all_iters,
                    y=self.residual_history,
                    mode='lines',
                    line=dict(color='lightgray', width=1),
                    name='Full History',
                    showlegend=False,
                ),
                row=3, col=2
            )
        
        # Add marker for current frame (will be updated in animation)
        fig.add_trace(
            go.Scatter(
                x=[snap0.iteration],
                y=[snap0.residual],
                mode='markers+lines',
                marker=dict(color='red', size=10),
                line=dict(color='blue', width=2),
                name='Snapshots',
            ),
            row=3, col=2
        )
        
        # Make residual y-axis logarithmic
        fig.update_yaxes(type='log', title_text='Residual', row=3, col=2)
        fig.update_xaxes(title_text='Iteration', row=3, col=2)
        
        # Create frames for animation
        frames = []
        for i, snap in enumerate(self.snapshots):
            vel_mag = np.sqrt(snap.u**2 + snap.v**2)
            field5 = snap.C_pt if has_cpt and snap.C_pt is not None else snap.nu
            
            # Heatmap data for this frame
            frame_data = [
                go.Heatmap(z=snap.p.T, colorscale='RdBu_r'),
                go.Heatmap(z=vel_mag.T, colorscale='Viridis'),
                go.Heatmap(z=snap.u.T, colorscale='RdBu'),
                go.Heatmap(z=snap.v.T, colorscale='RdBu'),
                go.Heatmap(z=field5.T, colorscale=field5_cmap),
            ]
            
            # Update residual marker position
            snapshot_iters = [s.iteration for s in self.snapshots[:i+1]]
            snapshot_res = [s.residual for s in self.snapshots[:i+1]]
            frame_data.append(go.Scatter(
                x=snapshot_iters,
                y=snapshot_res,
                mode='markers+lines',
                marker=dict(color='red', size=10),
                line=dict(color='blue', width=2),
            ))
            
            frames.append(go.Frame(
                data=frame_data,
                name=str(snap.iteration),
                traces=[0, 1, 2, 3, 4, 6],  # Skip trace 5 (background)
            ))
        
        fig.frames = frames
        
        # Create slider steps
        slider_steps = []
        for snap in self.snapshots:
            step = dict(
                method='animate',
                args=[
                    [str(snap.iteration)],
                    dict(
                        mode='immediate',
                        frame=dict(duration=100, redraw=True),
                        transition=dict(duration=0)
                    )
                ],
                label=f"{snap.iteration}",
            )
            slider_steps.append(step)
        
        # Build title with diagnostics
        final_snap = self.snapshots[-1]
        title_text = (
            f'CFD Simulation - {len(self.snapshots)} frames | '
            f'Final: iter={final_snap.iteration}, res={final_snap.residual:.2e}, '
            f'|V|_max={final_snap.vel_max:.3f}'
        )
        
        # Add slider and play button
        fig.update_layout(
            title=dict(text=title_text, x=0.5, font=dict(size=14)),
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    y=1.02,
                    x=0.1,
                    xanchor='right',
                    buttons=[
                        dict(
                            label='▶ Play',
                            method='animate',
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=200, redraw=True),
                                    fromcurrent=True,
                                    transition=dict(duration=0),
                                    mode='immediate',
                                )
                            ]
                        ),
                        dict(
                            label='⏸ Pause',
                            method='animate',
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode='immediate',
                                    transition=dict(duration=0)
                                )
                            ]
                        ),
                    ]
                ),
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor='top',
                    xanchor='left',
                    currentvalue=dict(
                        font=dict(size=12),
                        prefix='Iteration: ',
                        visible=True,
                        xanchor='right',
                    ),
                    transition=dict(duration=0),
                    pad=dict(b=10, t=30),
                    len=0.9,
                    x=0.05,
                    y=0,
                    steps=slider_steps,
                )
            ],
            height=1000,
            width=1200,
        )
        
        # Set axis labels
        for row in [1, 2, 3]:
            fig.update_xaxes(title_text='x', row=row, col=1)
            fig.update_yaxes(title_text='y', row=row, col=1)
        for row in [1, 2]:
            fig.update_xaxes(title_text='x', row=row, col=2)
            fig.update_yaxes(title_text='y', row=row, col=2)
        
        # Write HTML
        fig.write_html(str(output_path), auto_play=False)
        
        print(f"Saved HTML animation to: {output_path}")
        return str(output_path)
    
    def clear(self) -> None:
        """Clear all stored snapshots."""
        self.snapshots.clear()
        self.residual_history.clear()
    
    @property
    def num_snapshots(self) -> int:
        """Return number of stored snapshots."""
        return len(self.snapshots)
