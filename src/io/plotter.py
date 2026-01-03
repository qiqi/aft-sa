"""
Plotly-based HTML animation for CFD results.
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
    """Stores a single timestep's solution data (interior cells only)."""
    iteration: int
    residual: float
    cfl: float
    p: np.ndarray  # Relative pressure (p - p_inf)
    u: np.ndarray  # Relative u-velocity (u - u_inf)
    v: np.ndarray  # Relative v-velocity (v - v_inf)
    nu: np.ndarray
    C_pt: Optional[np.ndarray] = None
    residual_field: Optional[np.ndarray] = None
    mg_levels: Optional[List[Dict[str, np.ndarray]]] = None
    vel_max: float = 0.0
    p_min: float = 0.0
    p_max: float = 0.0


class PlotlyDashboard:
    """Accumulates CFD solution snapshots and exports as interactive HTML."""
    
    def __init__(self, reynolds: float = 6e6):
        self.snapshots: List[Snapshot] = []
        self.residual_history: List[float] = []
        self.p_inf: float = 0.0
        self.u_inf: float = 1.0
        self.v_inf: float = 0.0
        self.nu_laminar: float = 1.0 / reynolds if reynolds > 0 else 1e-6
    
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
        """Store current solution state with diagnostic data."""
        Q_int = Q[NGHOST:-NGHOST, NGHOST:-NGHOST, :]
        residual = residual_history[-1] if residual_history else 0.0
        
        # Extract freestream values
        if freestream is not None:
            self.p_inf = freestream.p_inf
            self.u_inf = freestream.u_inf
            self.v_inf = freestream.v_inf
        
        # Store RELATIVE values (p - p_inf, u - u_inf, v - v_inf)
        p_rel = Q_int[:, :, 0] - self.p_inf
        u_rel = Q_int[:, :, 1] - self.u_inf
        v_rel = Q_int[:, :, 2] - self.v_inf
        
        u_abs = Q_int[:, :, 1]
        v_abs = Q_int[:, :, 2]
        vel_mag = np.sqrt(u_abs**2 + v_abs**2)
        
        if C_pt is None and freestream is not None:
            p = Q_int[:, :, 0]
            V_inf_sq = self.u_inf**2 + self.v_inf**2
            p_total = p + 0.5 * (u_abs**2 + v_abs**2)
            p_total_inf = self.p_inf + 0.5 * V_inf_sq
            C_pt = (p_total_inf - p_total) / (0.5 * V_inf_sq + 1e-12)
        
        res_field = None
        if residual_field is not None:
            res_field = np.sqrt(np.mean(residual_field**2, axis=2))
        
        # Process MG levels to also store relative values
        mg_levels_rel = None
        if mg_levels is not None:
            mg_levels_rel = []
            for mg in mg_levels:
                mg_rel = {
                    'p': mg['p'] - self.p_inf,
                    'u': mg['u'] - self.u_inf,
                    'v': mg['v'] - self.v_inf,
                    'xc': mg['xc'],
                    'yc': mg['yc'],
                }
                if 'residual' in mg:
                    mg_rel['residual'] = mg['residual']
                mg_levels_rel.append(mg_rel)
        
        snapshot = Snapshot(
            iteration=iteration,
            residual=residual,
            cfl=cfl,
            p=p_rel.copy(),
            u=u_rel.copy(),
            v=v_rel.copy(),
            nu=Q_int[:, :, 3].copy(),
            C_pt=C_pt.copy() if C_pt is not None else None,
            residual_field=res_field,
            mg_levels=mg_levels_rel,
            vel_max=float(vel_mag.max()),
            p_min=float(p_rel.min()),
            p_max=float(p_rel.max()),
        )
        
        self.snapshots.append(snapshot)
        self.residual_history = list(residual_history)
    
    def save_html(self, filename: str, grid_metrics: 'FVMMetrics') -> str:
        """Export all snapshots as interactive HTML animation."""
        if not HAS_PLOTLY:
            print("Warning: plotly not installed. Skipping HTML animation.")
            return ""
        
        if not self.snapshots:
            print("Warning: No snapshots to save.")
            return ""
        
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        xc = grid_metrics.xc
        yc = grid_metrics.yc
        
        ni, nj = xc.shape
        a = np.arange(ni)
        b = np.arange(nj)
        A, B = np.meshgrid(a, b, indexing='ij')
        
        # Use last snapshot for initial display (matches slider default position)
        snap0 = self.snapshots[0]  # For checking feature availability
        snapN = self.snapshots[-1]  # For initial display
        has_cpt = snap0.C_pt is not None
        has_res_field = snap0.residual_field is not None
        has_mg = snap0.mg_levels is not None and len(snap0.mg_levels) > 0
        
        n_mg_levels = min(len(snap0.mg_levels), 2) if has_mg else 0
        # Layout: Row 1-2: field pairs, Row 3: residual + chi, Row 4: convergence (full width), Rows 5+: MG
        n_rows = 4 + n_mg_levels
        
        subplot_titles = [
            'Pressure (p - p∞)', 
            'Total Pressure Loss (C_pt)' if has_cpt else 'Turbulent Viscosity (ν)',
            'U-velocity (u - u∞)', 'V-velocity (v - v∞)',
            'Residual Field (log₁₀)' if has_res_field else 'Velocity Magnitude',
            'χ = ν̃/ν (Turbulent/Laminar Viscosity Ratio)',
            'Convergence History', ''  # Second column empty for colspan
        ]
        
        for i in range(n_mg_levels):
            subplot_titles.extend([f'MG Level {i+1}: Pressure', f'MG Level {i+1}: Residual (log₁₀)'])
        
        specs = [
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "scatter", "colspan": 2}, None],  # Convergence spans both columns
        ]
        for _ in range(n_mg_levels):
            specs.append([{"type": "xy"}, {"type": "xy"}])
        
        fig = make_subplots(
            rows=n_rows, cols=2,
            subplot_titles=subplot_titles,
            specs=specs,
            horizontal_spacing=0.08,
            vertical_spacing=0.06,
        )
        
        field2_data = snapN.C_pt if has_cpt else snapN.nu
        
        if has_res_field and snapN.residual_field is not None:
            field5_data = np.log10(snapN.residual_field + 1e-12)
        else:
            field5_data = np.zeros_like(snapN.p)
        
        # Chi = nu_hat / nu_laminar (turbulent/laminar viscosity ratio)
        chi_data = snapN.nu / self.nu_laminar
        
        # Compute GLOBAL symmetric ranges for consistent scaling
        # Pressure: symmetric around 0
        p_abs_max = max(
            max(abs(s.p.min()), abs(s.p.max())) for s in self.snapshots
        )
        if has_cpt:
            cpt_abs_max = max(
                max(abs(s.C_pt.min()), abs(s.C_pt.max())) 
                for s in self.snapshots if s.C_pt is not None
            )
            p_abs_max = max(p_abs_max, cpt_abs_max)
        
        # Velocity: symmetric around 0
        vel_abs_max = max(
            max(abs(s.u.min()), abs(s.u.max()), abs(s.v.min()), abs(s.v.max()))
            for s in self.snapshots
        )
        
        # Residual: 3 orders of magnitude from max
        if has_res_field:
            res_vals = [np.log10(s.residual_field + 1e-12) for s in self.snapshots if s.residual_field is not None]
            res_max = max(v.max() for v in res_vals) if res_vals else 0
            res_min = res_max - 3  # 3 orders of magnitude
        else:
            res_min, res_max = -3, 0
        
        # Chi range (log scale for display)
        chi_vals = [s.nu / self.nu_laminar for s in self.snapshots]
        chi_max = max(np.log10(v.max() + 1e-12) for v in chi_vals)
        chi_min = min(np.log10(np.maximum(v, 1e-12).min()) for v in chi_vals)
        
        # Store ranges for sliders (will be used in layout)
        p_range = p_abs_max
        vel_range = vel_abs_max
        res_range_max = res_max
        
        contour_configs = [
            (snapN.p, 1, 1, 'pressure', True),
            (field2_data, 1, 2, 'pressure', False),
            (snapN.u, 2, 1, 'velocity', True),
            (snapN.v, 2, 2, 'velocity', False),
            (field5_data, 3, 1, 'residual', True),
            (np.log10(chi_data + 1e-12), 3, 2, 'chi', True),  # Chi plot (log scale)
        ]
        
        # Colorbar positions - 4 colorbars, properly spaced
        coloraxis_config = {
            'pressure': {'colorscale': 'RdBu_r', 'cmin': -p_range, 'cmax': p_range, 
                         'colorbar': dict(title='Δp', len=0.18, y=0.90, x=1.02, 
                                          tickformat='.2f')},
            'velocity': {'colorscale': 'RdBu', 'cmin': -vel_range, 'cmax': vel_range,
                         'colorbar': dict(title='Δvel', len=0.18, y=0.68, x=1.02,
                                          tickformat='.2f')},
            'residual': {'colorscale': 'Hot', 'cmin': res_min, 'cmax': res_max,
                         'colorbar': dict(title='log₁₀(R)', len=0.18, y=0.46, x=1.02,
                                          tickformat='.1f')},
            'chi': {'colorscale': 'Viridis', 'cmin': chi_min, 'cmax': chi_max,
                    'colorbar': dict(title='log₁₀(χ)', len=0.18, y=0.24, x=1.02,
                                     tickformat='.1f')},
        }
        
        for data, row, col, caxis_group, show_colorbar in contour_configs:
            carpet_id = f'carpet_{row}_{col}'
            caxis_cfg = coloraxis_config[caxis_group]
            
            fig.add_trace(
                go.Carpet(
                    a=A.flatten(), b=B.flatten(),
                    x=xc.flatten(), y=yc.flatten(),
                    carpet=carpet_id,
                    aaxis=dict(showgrid=False, showticklabels='none', showline=False),
                    baxis=dict(showgrid=False, showticklabels='none', showline=False),
                ),
                row=row, col=col
            )
            
            colorbar_cfg = caxis_cfg['colorbar'] if show_colorbar else None
            fig.add_trace(
                go.Contourcarpet(
                    a=A.flatten(), b=B.flatten(), z=data.flatten(),
                    carpet=carpet_id,
                    colorscale=caxis_cfg['colorscale'],
                    zmin=caxis_cfg['cmin'], zmax=caxis_cfg['cmax'],
                    contours=dict(coloring='fill', showlines=False),
                    ncontours=50,
                    colorbar=colorbar_cfg,
                    showscale=show_colorbar,
                ),
                row=row, col=col
            )
        
        mg_carpet_indices = []
        mg_contour_indices = []
        
        mg_level_info = []
        p_cfg = coloraxis_config['pressure']
        r_cfg = coloraxis_config['residual']
        
        if has_mg and snapN.mg_levels:
            for mg_idx, mg_level in enumerate(snapN.mg_levels[:n_mg_levels]):
                mg_xc = mg_level['xc']
                mg_yc = mg_level['yc']
                mg_p = mg_level['p']
                mg_res = mg_level.get('residual', np.zeros_like(mg_p))
                
                mg_ni, mg_nj = mg_xc.shape
                mg_a = np.arange(mg_ni)
                mg_b = np.arange(mg_nj)
                mg_A, mg_B = np.meshgrid(mg_a, mg_b, indexing='ij')
                
                row = 5 + mg_idx  # MG levels start at row 5 (after convergence plot)
                mg_level_info.append((mg_A, mg_B, row))
                
                carpet_id_p = f'mg_carpet_p_{mg_idx}'
                mg_carpet_indices.append(len(fig.data))
                fig.add_trace(
                    go.Carpet(
                        a=mg_A.flatten(), b=mg_B.flatten(),
                        x=mg_xc.flatten(), y=mg_yc.flatten(),
                        carpet=carpet_id_p,
                        aaxis=dict(showgrid=False, showticklabels='none', showline=False),
                        baxis=dict(showgrid=False, showticklabels='none', showline=False),
                    ),
                    row=row, col=1
                )
                mg_contour_indices.append(len(fig.data))
                fig.add_trace(
                    go.Contourcarpet(
                        a=mg_A.flatten(), b=mg_B.flatten(), z=mg_p.flatten(),
                        carpet=carpet_id_p,
                        colorscale=p_cfg['colorscale'],
                        zmin=p_cfg['cmin'], zmax=p_cfg['cmax'],
                        contours=dict(coloring='fill', showlines=False),
                        ncontours=50,
                        showscale=False,
                    ),
                    row=row, col=1
                )
                
                carpet_id_r = f'mg_carpet_r_{mg_idx}'
                mg_carpet_indices.append(len(fig.data))
                fig.add_trace(
                    go.Carpet(
                        a=mg_A.flatten(), b=mg_B.flatten(),
                        x=mg_xc.flatten(), y=mg_yc.flatten(),
                        carpet=carpet_id_r,
                        aaxis=dict(showgrid=False, showticklabels='none', showline=False),
                        baxis=dict(showgrid=False, showticklabels='none', showline=False),
                    ),
                    row=row, col=2
                )
                mg_contour_indices.append(len(fig.data))
                fig.add_trace(
                    go.Contourcarpet(
                        a=mg_A.flatten(), b=mg_B.flatten(),
                        z=np.log10(mg_res + 1e-12).flatten(),
                        carpet=carpet_id_r,
                        colorscale=r_cfg['colorscale'],
                        zmin=r_cfg['cmin'], zmax=r_cfg['cmax'],
                        contours=dict(coloring='fill', showlines=False),
                        ncontours=50,
                        showscale=False,
                    ),
                    row=row, col=2
                )
        
        if self.residual_history:
            all_iters = list(range(len(self.residual_history)))
            fig.add_trace(
                go.Scatter(
                    x=all_iters, y=self.residual_history,
                    mode='lines', line=dict(color='blue', width=1.5),
                    showlegend=False, name='Full History',
                ),
                row=4, col=1  # Convergence plot now at row 4, spanning both columns
            )
        
        snapshot_iters = [s.iteration for s in self.snapshots]
        snapshot_res = [s.residual for s in self.snapshots]
        fig.add_trace(
            go.Scatter(
                x=snapshot_iters, y=snapshot_res,
                mode='markers',
                marker=dict(color='red', size=10, symbol='circle'),
                showlegend=False, name='Snapshots',
            ),
            row=4, col=1  # Convergence plot now at row 4
        )
        
        # Convergence history y-axis: log scale (x-axis configured at end with matches=None)
        
        # 6 contour plots: pressure, cpt, u, v, residual, chi
        # Each contour plot has a carpet (even index) and contourcarpet (odd index)
        # So contour indices are: 1, 3, 5, 7, 9, 11
        base_contour_indices = [1, 3, 5, 7, 9, 11]
        n_mg_traces = len(mg_contour_indices) * 2
        residual_marker_idx = 12 + n_mg_traces + 1  # Updated for 6 base plots (12 traces) + MG + scatter line
        animated_indices = base_contour_indices + mg_contour_indices + [residual_marker_idx]
        
        chi_cfg = coloraxis_config['chi']
        
        frames = []
        for i, snap in enumerate(self.snapshots):
            field2 = snap.C_pt if has_cpt and snap.C_pt is not None else snap.nu
            if has_res_field and snap.residual_field is not None:
                field5 = np.log10(snap.residual_field + 1e-12)
            else:
                field5 = np.sqrt(snap.u**2 + snap.v**2)
            
            # Chi for this snapshot (log scale)
            snap_chi = np.log10(snap.nu / self.nu_laminar + 1e-12)
            
            frame_data = [
                go.Contourcarpet(a=A.flatten(), b=B.flatten(), z=snap.p.flatten(), 
                                 carpet='carpet_1_1', colorscale=p_cfg['colorscale'],
                                 zmin=p_cfg['cmin'], zmax=p_cfg['cmax'],
                                 contours=dict(coloring='fill', showlines=False), ncontours=50),
                go.Contourcarpet(a=A.flatten(), b=B.flatten(), z=field2.flatten(),
                                 carpet='carpet_1_2', colorscale=p_cfg['colorscale'],
                                 zmin=p_cfg['cmin'], zmax=p_cfg['cmax'],
                                 contours=dict(coloring='fill', showlines=False), ncontours=50),
                go.Contourcarpet(a=A.flatten(), b=B.flatten(), z=snap.u.flatten(),
                                 carpet='carpet_2_1', colorscale=coloraxis_config['velocity']['colorscale'],
                                 zmin=coloraxis_config['velocity']['cmin'], zmax=coloraxis_config['velocity']['cmax'],
                                 contours=dict(coloring='fill', showlines=False), ncontours=50),
                go.Contourcarpet(a=A.flatten(), b=B.flatten(), z=snap.v.flatten(),
                                 carpet='carpet_2_2', colorscale=coloraxis_config['velocity']['colorscale'],
                                 zmin=coloraxis_config['velocity']['cmin'], zmax=coloraxis_config['velocity']['cmax'],
                                 contours=dict(coloring='fill', showlines=False), ncontours=50),
                go.Contourcarpet(a=A.flatten(), b=B.flatten(), z=field5.flatten(),
                                 carpet='carpet_3_1', colorscale=r_cfg['colorscale'],
                                 zmin=r_cfg['cmin'], zmax=r_cfg['cmax'],
                                 contours=dict(coloring='fill', showlines=False), ncontours=50),
                go.Contourcarpet(a=A.flatten(), b=B.flatten(), z=snap_chi.flatten(),
                                 carpet='carpet_3_2', colorscale=chi_cfg['colorscale'],
                                 zmin=chi_cfg['cmin'], zmax=chi_cfg['cmax'],
                                 contours=dict(coloring='fill', showlines=False), ncontours=50),
            ]
            
            if has_mg and snap.mg_levels:
                for mg_idx, mg_level in enumerate(snap.mg_levels[:n_mg_levels]):
                    mg_xc = mg_level['xc']
                    mg_ni, mg_nj = mg_xc.shape
                    mg_a = np.arange(mg_ni)
                    mg_b = np.arange(mg_nj)
                    mg_A, mg_B = np.meshgrid(mg_a, mg_b, indexing='ij')
                    mg_res = mg_level.get('residual', np.zeros_like(mg_level['p']))
                    
                    frame_data.append(
                        go.Contourcarpet(
                            a=mg_A.flatten(), b=mg_B.flatten(),
                            z=mg_level['p'].flatten(),
                            carpet=f'mg_carpet_p_{mg_idx}', colorscale=p_cfg['colorscale'],
                            zmin=p_cfg['cmin'], zmax=p_cfg['cmax'],
                            contours=dict(coloring='fill', showlines=False), ncontours=50
                        )
                    )
                    frame_data.append(
                        go.Contourcarpet(
                            a=mg_A.flatten(), b=mg_B.flatten(),
                            z=np.log10(mg_res + 1e-12).flatten(),
                            carpet=f'mg_carpet_r_{mg_idx}', colorscale=r_cfg['colorscale'],
                            zmin=r_cfg['cmin'], zmax=r_cfg['cmax'],
                            contours=dict(coloring='fill', showlines=False), ncontours=50
                        )
                    )
            
            snapshot_iters = [s.iteration for s in self.snapshots[:i+1]]
            snapshot_res = [s.residual for s in self.snapshots[:i+1]]
            frame_data.append(go.Scatter(
                x=snapshot_iters, y=snapshot_res,
                mode='markers',
                marker=dict(color='red', size=10, symbol='circle'),
            ))
            
            frames.append(go.Frame(
                data=frame_data,
                name=str(snap.iteration),
                traces=animated_indices,
            ))
        
        fig.frames = frames
        
        # Iteration slider steps
        slider_steps = [
            dict(
                method='animate',
                args=[[str(snap.iteration)], dict(
                    mode='immediate',
                    frame=dict(duration=100, redraw=True),
                    transition=dict(duration=0)
                )],
                label=f"{snap.iteration}",
            )
            for snap in self.snapshots
        ]
        
        # Create continuous logarithmic color range sliders
        # 50 steps spanning 3 orders of magnitude
        n_steps = 50
        
        def make_log_range_steps(base_val, trace_indices, is_residual=False):
            """Create continuous logarithmic slider steps (3 orders of magnitude)."""
            steps = []
            # Range from base_val / 10^1.5 to base_val * 10^1.5 (3 orders total)
            log_factors = np.linspace(-1.5, 1.5, n_steps)
            
            for log_f in log_factors:
                if is_residual:
                    # For residual: slider controls max level, range is always 3 decades
                    new_max = base_val + log_f  # base_val is log scale already
                    new_min = new_max - 3.0
                    label = f'{new_max:.1f}'
                    steps.append(dict(
                        method='restyle',
                        args=[{'zmin': new_min, 'zmax': new_max, 'ncontours': 50}, trace_indices],
                        label=label,
                    ))
                else:
                    # For symmetric ranges: log_f controls the multiplier
                    factor = 10 ** log_f
                    new_range = base_val * factor
                    steps.append(dict(
                        method='restyle',
                        args=[{'zmin': -new_range, 'zmax': new_range, 'ncontours': 50}, trace_indices],
                        label=f'{new_range:.2e}',
                    ))
            return steps
        
        # Trace indices for each color group
        pressure_traces = [1, 3]  # carpet_1_1 contour, carpet_1_2 contour
        velocity_traces = [5, 7]  # carpet_2_1 contour, carpet_2_2 contour
        residual_traces = [9]     # carpet_3_1 contour
        chi_traces = [11]         # carpet_3_2 contour (chi = nu_tilde/nu)
        
        # Add MG traces to their groups
        if has_mg:
            for i in range(n_mg_levels):
                pressure_traces.append(mg_contour_indices[i * 2] if i * 2 < len(mg_contour_indices) else None)
                residual_traces.append(mg_contour_indices[i * 2 + 1] if i * 2 + 1 < len(mg_contour_indices) else None)
            pressure_traces = [t for t in pressure_traces if t is not None]
            residual_traces = [t for t in residual_traces if t is not None]
        
        p_steps = make_log_range_steps(p_range, pressure_traces, is_residual=False)
        v_steps = make_log_range_steps(vel_range, velocity_traces, is_residual=False)
        r_steps = make_log_range_steps(res_max, residual_traces, is_residual=True)
        chi_steps = make_log_range_steps(chi_max, chi_traces, is_residual=True)
        
        fig.update_layout(
            showlegend=False,
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    y=1.05, x=0.08, xanchor='right',
                    buttons=[
                        dict(
                            label='▶',
                            method='animate',
                            args=[None, dict(
                                frame=dict(duration=200, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                                mode='immediate',
                            )]
                        ),
                        dict(
                            label='⏸',
                            method='animate',
                            args=[[None], dict(
                                frame=dict(duration=0, redraw=False),
                                mode='immediate',
                                transition=dict(duration=0)
                            )]
                        ),
                    ]
                ),
            ],
            sliders=[
                # Iteration slider (top-left)
                dict(
                    active=len(self.snapshots) - 1,
                    yanchor='bottom', xanchor='left',
                    currentvalue=dict(
                        font=dict(size=11),
                        prefix='Iter: ',
                        visible=True,
                        xanchor='right',
                    ),
                    transition=dict(duration=0),
                    pad=dict(b=5, t=5),
                    len=0.42, x=0.08, y=1.04,
                    steps=slider_steps,
                    ticklen=0,
                ),
                # Pressure range slider
                dict(
                    active=n_steps // 2,
                    yanchor='bottom', xanchor='left',
                    currentvalue=dict(
                        font=dict(size=9),
                        prefix='Δp: ',
                        visible=True,
                        xanchor='left',
                    ),
                    pad=dict(b=5, t=5),
                    len=0.11, x=0.52, y=1.04,
                    steps=p_steps,
                    ticklen=0,
                ),
                # Velocity range slider
                dict(
                    active=n_steps // 2,
                    yanchor='bottom', xanchor='left',
                    currentvalue=dict(
                        font=dict(size=9),
                        prefix='Δv: ',
                        visible=True,
                        xanchor='left',
                    ),
                    pad=dict(b=5, t=5),
                    len=0.11, x=0.64, y=1.04,
                    steps=v_steps,
                    ticklen=0,
                ),
                # Residual max slider
                dict(
                    active=n_steps // 2,
                    yanchor='bottom', xanchor='left',
                    currentvalue=dict(
                        font=dict(size=9),
                        prefix='Res: ',
                        visible=True,
                        xanchor='left',
                    ),
                    pad=dict(b=5, t=5),
                    len=0.11, x=0.76, y=1.04,
                    steps=r_steps,
                    ticklen=0,
                ),
                # Chi (ν̃/ν) max slider
                dict(
                    active=n_steps // 2,
                    yanchor='bottom', xanchor='left',
                    currentvalue=dict(
                        font=dict(size=9),
                        prefix='χ: ',
                        visible=True,
                        xanchor='left',
                    ),
                    pad=dict(b=5, t=5),
                    len=0.11, x=0.88, y=1.04,
                    steps=chi_steps,
                    ticklen=0,
                ),
            ],
            # Height sized for ~16:9 aspect ratio per subplot
            # With 2 columns at ~650px each, 16:9 needs ~365px height per row
            height=1600 + (400 * n_mg_levels),
            width=1400,
            margin=dict(t=80),  # Top margin for sliders
        )
        
        # Link zoom/pan across all contourcarpet plots
        # All field plots share the same x and y ranges (except row 3 col 2 which is scatter)
        # Use 'matches' to link axes, and 'scaleanchor' for aspect ratio
        
        # First, set up row 1 col 1 as the reference axis
        # For 16:9 aspect ratio subplots with equal data scaling:
        # x range: [-0.5, 1.5] = 2.0 extent
        # y range for 16:9: 2.0 * 9/16 = 1.125, so [-0.5625, 0.5625]
        fig.update_xaxes(title_text='x', range=[-0.5, 1.5], scaleanchor='y', scaleratio=1, row=1, col=1)
        fig.update_yaxes(title_text='y', range=[-0.5625, 0.5625], row=1, col=1)
        
        # List of all contourcarpet subplot positions (row, col) - rows 1-3 have 2D field plots
        # Row 4 has convergence (scatter), MG levels start at row 5
        mg_rows = list(range(5, 5 + n_mg_levels)) if n_mg_levels > 0 else []
        field_positions = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]  # Include chi at (3, 2)
        for mg_row in mg_rows:
            field_positions.extend([(mg_row, 1), (mg_row, 2)])
        
        # Link all other field plots to the reference (row 1 col 1 = x, y)
        for row, col in field_positions:
            if row == 1 and col == 1:
                continue  # Skip reference
            fig.update_xaxes(title_text='x', range=[-0.5, 1.5], matches='x', row=row, col=col)
            fig.update_yaxes(title_text='y', range=[-0.5625, 0.5625], matches='y', row=row, col=col)
        
        # Convergence history plot (row 4, col 1, spanning both columns) - keep independent
        # fixedrange=True prevents modebar zoom buttons from affecting this subplot
        fig.update_xaxes(title_text='Iteration', matches=None, autorange=True, fixedrange=True, row=4, col=1)
        fig.update_yaxes(title_text='Residual', matches=None, type='log', autorange=True, fixedrange=True, row=4, col=1)
        
        fig.write_html(str(output_path), auto_play=False)
        
        print(f"Saved HTML animation to: {output_path}")
        print(f"  Use sliders at top-right to adjust color ranges")
        return str(output_path)
    
    def clear(self) -> None:
        """Clear all stored snapshots."""
        self.snapshots.clear()
        self.residual_history.clear()
    
    @property
    def num_snapshots(self) -> int:
        return len(self.snapshots)
