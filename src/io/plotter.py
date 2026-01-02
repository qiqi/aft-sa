"""
Plotly-based HTML Animation for CFD Results.

This module provides a PlotlyDashboard class that accumulates solution
snapshots during simulation and exports them as an interactive HTML
animation with 2x2 heatmap subplots and playback controls.

Features:
- 2x2 subplot layout: Pressure, U-velocity, V-velocity, Turbulent Viscosity
- Interactive slider to scrub through iterations
- Play/Pause button for auto-advance
- Shared axes for synchronized zooming
- Auto-scaling color ranges per frame
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
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
    p: np.ndarray   # Pressure, shape (NI, NJ)
    u: np.ndarray   # U-velocity, shape (NI, NJ)
    v: np.ndarray   # V-velocity, shape (NI, NJ)
    nu: np.ndarray  # Turbulent viscosity, shape (NI, NJ)


class PlotlyDashboard:
    """
    Accumulates CFD solution snapshots and exports as interactive HTML.
    
    Usage:
        dashboard = PlotlyDashboard()
        
        for iteration in range(max_iter):
            # ... solver step ...
            if iteration % output_freq == 0:
                dashboard.store_snapshot(Q, iteration, residual_history)
        
        dashboard.save_html("animation.html", grid_metrics)
    """
    
    def __init__(self):
        """Initialize empty snapshot storage."""
        self.snapshots: List[Snapshot] = []
    
    def store_snapshot(self, Q: np.ndarray, iteration: int, 
                       residual_history: List[float]) -> None:
        """
        Store the current solution state.
        
        Parameters
        ----------
        Q : ndarray, shape (NI + 2*NGHOST, NJ + 2*NGHOST, 4)
            Full state vector including ghost cells.
            Components: [p, u, v, nu]
        iteration : int
            Current iteration number.
        residual_history : list of float
            History of residuals (used to get current residual).
        """
        # Extract interior cells only
        Q_int = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
        
        # Get current residual
        residual = residual_history[-1] if residual_history else 0.0
        
        # Create snapshot with copies (to avoid aliasing issues)
        snapshot = Snapshot(
            iteration=iteration,
            residual=residual,
            p=Q_int[:, :, 0].copy(),
            u=Q_int[:, :, 1].copy(),
            v=Q_int[:, :, 2].copy(),
            nu=Q_int[:, :, 3].copy(),
        )
        
        self.snapshots.append(snapshot)
    
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
        
        # Get grid coordinates (cell centers)
        xc = grid_metrics.xc
        yc = grid_metrics.yc
        
        # Create 2x2 subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Pressure (p)', 'U-velocity', 
                           'V-velocity', 'Viscosity (ν)'),
            shared_xaxes=True,
            shared_yaxes=True,
            horizontal_spacing=0.08,
            vertical_spacing=0.1,
        )
        
        # Get first snapshot for base traces
        snap0 = self.snapshots[0]
        
        # Field configurations: (data, row, col, colorscale)
        fields = [
            ('p', snap0.p, 1, 1, 'RdBu_r'),
            ('u', snap0.u, 1, 2, 'Viridis'),
            ('v', snap0.v, 2, 1, 'RdBu'),
            ('nu', snap0.nu, 2, 2, 'Plasma'),
        ]
        
        # Add base heatmaps
        for name, data, row, col, colorscale in fields:
            fig.add_trace(
                go.Heatmap(
                    z=data.T,  # Transpose for correct orientation
                    x=xc[:, 0],  # Use first column for x coords
                    y=yc[0, :],  # Use first row for y coords
                    colorscale=colorscale,
                    colorbar=dict(
                        len=0.4,
                        y=0.8 if row == 1 else 0.2,
                        title=name,
                    ),
                    name=name,
                ),
                row=row, col=col
            )
        
        # Create frames for animation
        frames = []
        for snap in self.snapshots:
            frame_data = [
                go.Heatmap(z=snap.p.T, colorscale='RdBu_r'),
                go.Heatmap(z=snap.u.T, colorscale='Viridis'),
                go.Heatmap(z=snap.v.T, colorscale='RdBu'),
                go.Heatmap(z=snap.nu.T, colorscale='Plasma'),
            ]
            frames.append(go.Frame(
                data=frame_data,
                name=str(snap.iteration),
                traces=[0, 1, 2, 3],  # Update these trace indices
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
        
        # Add slider and play button
        fig.update_layout(
            title=dict(
                text=f'CFD Simulation ({len(self.snapshots)} frames)',
                x=0.5,
            ),
            updatemenus=[
                # Play button
                dict(
                    type='buttons',
                    showactive=False,
                    y=1.15,
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
                    pad=dict(b=10, t=50),
                    len=0.9,
                    x=0.05,
                    y=0,
                    steps=slider_steps,
                )
            ],
            height=800,
            width=1200,
        )
        
        # Set axis properties
        fig.update_xaxes(title_text='x', row=2, col=1)
        fig.update_xaxes(title_text='x', row=2, col=2)
        fig.update_yaxes(title_text='y', row=1, col=1)
        fig.update_yaxes(title_text='y', row=2, col=1)
        
        # Ensure equal aspect ratio
        fig.update_yaxes(scaleanchor='x', scaleratio=1)
        
        # Write HTML
        fig.write_html(str(output_path), auto_play=False)
        
        print(f"Saved HTML animation to: {output_path}")
        return str(output_path)
    
    def clear(self) -> None:
        """Clear all stored snapshots."""
        self.snapshots.clear()
    
    @property
    def num_snapshots(self) -> int:
        """Return number of stored snapshots."""
        return len(self.snapshots)

