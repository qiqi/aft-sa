"""
Visualization utilities for CFD solutions.

This module provides functions for plotting flow fields, residual history,
and other diagnostic visualizations.
"""

import os
import numpy as np
from typing import Optional, List
from pathlib import Path

# Lazy import matplotlib to avoid issues when not installed
_plt = None
_matplotlib = None


def _ensure_matplotlib():
    """Ensure matplotlib is available and configured."""
    global _plt, _matplotlib
    if _plt is None:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        _matplotlib = matplotlib
        _plt = plt
    return _plt


def plot_flow_field(X: np.ndarray, Y: np.ndarray, Q: np.ndarray,
                    iteration: int, residual: float, cfl: float,
                    output_dir: str, case_name: str = "flow",
                    C_pt: Optional[np.ndarray] = None,
                    residual_field: Optional[np.ndarray] = None) -> str:
    """
    Plot flow field with global and zoomed views.
    
    Creates PDF with:
    - Pressure (global and zoomed)
    - Velocity magnitude (global and zoomed)
    - Total Pressure Loss C_pt (if provided)
    - Residual field (if provided)
    
    Parameters
    ----------
    X, Y : ndarray
        Grid node coordinates.
    Q : ndarray
        State vector [p, u, v, nu_t]. Can include ghost cells.
    iteration : int
        Current iteration number.
    residual : float
        Current residual value.
    cfl : float
        Current CFL number.
    output_dir : str
        Output directory for PDF files.
    case_name : str
        Base name for output files.
    C_pt : ndarray, optional
        Total pressure loss coefficient field.
    residual_field : ndarray, optional
        Residual field (NI, NJ, 4) for visualization.
        
    Returns
    -------
    output_path : str
        Path to saved PDF file.
    """
    plt = _ensure_matplotlib()
    
    # Strip ghost cells from Q
    NI = X.shape[0] - 1
    NJ = X.shape[1] - 1
    
    # Q has 2 J-ghosts at wall/wake, 1 at farfield
    if Q.shape[0] == NI + 2 and Q.shape[1] == NJ + 3:
        Q_int = Q[1:-1, 2:-1, :]
    else:
        Q_int = Q
    
    # Extract fields
    p = Q_int[:, :, 0]
    u = Q_int[:, :, 1]
    v = Q_int[:, :, 2]
    
    vel_mag = np.sqrt(u**2 + v**2)
    
    # Determine number of rows
    n_rows = 2  # Pressure + Velocity
    if C_pt is not None:
        n_rows += 1
    if residual_field is not None:
        n_rows += 1
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    if n_rows == 2:
        axes = axes.reshape(2, 2)
    
    # --- Row 1: Pressure ---
    ax = axes[0, 0]
    p_clip = np.clip(p, np.percentile(p, 1), np.percentile(p, 99))
    pc = ax.pcolormesh(X.T, Y.T, p_clip.T, cmap='RdBu_r', shading='flat', rasterized=True)
    ax.set_aspect('equal')
    ax.set_title(f'Pressure (Global) - Iter {iteration}')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    plt.colorbar(pc, ax=ax, shrink=0.8, label='p')
    
    ax = axes[0, 1]
    pc = ax.pcolormesh(X.T, Y.T, p_clip.T, cmap='RdBu_r', shading='flat', rasterized=True)
    ax.set_aspect('equal')
    ax.set_xlim(-0.2, 1.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_title(f'Pressure (Near Airfoil)')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    plt.colorbar(pc, ax=ax, shrink=0.8, label='p')
    
    # --- Row 2: Velocity Magnitude ---
    ax = axes[1, 0]
    vel_clip = np.clip(vel_mag, 0, np.percentile(vel_mag, 99))
    pc = ax.pcolormesh(X.T, Y.T, vel_clip.T, cmap='viridis', shading='flat', rasterized=True)
    ax.set_aspect('equal')
    ax.set_title(f'Velocity Magnitude (Global) - CFL={cfl:.2f}')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    plt.colorbar(pc, ax=ax, shrink=0.8, label='|V|')
    
    ax = axes[1, 1]
    pc = ax.pcolormesh(X.T, Y.T, vel_clip.T, cmap='viridis', shading='flat', rasterized=True)
    ax.set_aspect('equal')
    ax.set_xlim(-0.2, 1.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_title(f'Velocity Magnitude (Near Airfoil) - Res={residual:.2e}')
    ax.set_xlabel('x/c')
    ax.set_ylabel('y/c')
    plt.colorbar(pc, ax=ax, shrink=0.8, label='|V|')
    
    row_idx = 2
    
    # --- Total Pressure Loss ---
    if C_pt is not None:
        C_pt_abs_max = max(np.abs(np.percentile(C_pt, 1)), np.abs(np.percentile(C_pt, 99)))
        C_pt_abs_max = max(C_pt_abs_max, 0.01)
        
        ax = axes[row_idx, 0]
        pc = ax.pcolormesh(X.T, Y.T, C_pt.T, cmap='RdBu_r', shading='flat',
                           rasterized=True, vmin=-C_pt_abs_max, vmax=C_pt_abs_max)
        ax.set_aspect('equal')
        ax.set_title(f'Total Pressure Loss C_pt (Global)')
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        plt.colorbar(pc, ax=ax, shrink=0.8, label='C_pt')
        
        ax = axes[row_idx, 1]
        pc = ax.pcolormesh(X.T, Y.T, C_pt.T, cmap='RdBu_r', shading='flat',
                           rasterized=True, vmin=-C_pt_abs_max, vmax=C_pt_abs_max)
        ax.set_aspect('equal')
        ax.set_xlim(-0.2, 1.4)
        ax.set_ylim(-0.4, 0.4)
        ax.set_title(f'Total Pressure Loss C_pt (Near Airfoil)')
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        plt.colorbar(pc, ax=ax, shrink=0.8, label='C_pt')
        
        row_idx += 1
    
    # --- Residual Field ---
    if residual_field is not None:
        R_p = np.abs(residual_field[:, :, 0])
        R_log = np.log10(R_p + 1e-12)
        R_min = np.percentile(R_log, 1)
        R_max = np.percentile(R_log, 99)
        
        ax = axes[row_idx, 0]
        pc = ax.pcolormesh(X.T, Y.T, R_log.T, cmap='hot_r', shading='flat',
                           rasterized=True, vmin=R_min, vmax=R_max)
        ax.set_aspect('equal')
        ax.set_title(f'log₁₀|Residual| (Global)')
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        plt.colorbar(pc, ax=ax, shrink=0.8, label='log₁₀|R_p|')
        
        ax = axes[row_idx, 1]
        pc = ax.pcolormesh(X.T, Y.T, R_log.T, cmap='hot_r', shading='flat',
                           rasterized=True, vmin=R_min, vmax=R_max)
        ax.set_aspect('equal')
        ax.set_xlim(-0.2, 1.4)
        ax.set_ylim(-0.4, 0.4)
        ax.set_title(f'log₁₀|Residual| (Near Airfoil)')
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        
        # Mark max residual location
        max_idx = np.unravel_index(np.argmax(R_p), R_p.shape)
        x_max = 0.5 * (X[max_idx[0], max_idx[1]] + X[max_idx[0]+1, max_idx[1]])
        y_max = 0.5 * (Y[max_idx[0], max_idx[1]] + Y[max_idx[0]+1, max_idx[1]])
        ax.plot(x_max, y_max, 'g*', markersize=15, markeredgecolor='white', markeredgewidth=1)
        plt.colorbar(pc, ax=ax, shrink=0.8, label='log₁₀|R_p|')
    
    plt.tight_layout()
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f'{case_name}_iter{iteration:06d}.pdf')
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    return output_path


def plot_residual_history(residuals: List[float], output_dir: str,
                          case_name: str = "residual") -> Optional[str]:
    """
    Plot residual convergence history.
    
    Parameters
    ----------
    residuals : list of float
        Residual values at each iteration.
    output_dir : str
        Output directory.
    case_name : str
        Base name for output file.
        
    Returns
    -------
    output_path : str or None
        Path to saved PDF, or None if not enough data.
    """
    if len(residuals) < 2:
        return None
    
    plt = _ensure_matplotlib()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = np.arange(1, len(residuals) + 1)
    ax.semilogy(iterations, residuals, 'b-', lw=1.5)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual RMS')
    ax.set_title('Convergence History')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(residuals))
    
    plt.tight_layout()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f'{case_name}_convergence.pdf')
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    return output_path


def plot_multigrid_levels(mg_hierarchy, X_fine: np.ndarray, Y_fine: np.ndarray,
                          iteration: int, residual: float, cfl: float,
                          output_dir: str, case_name: str = "flow",
                          C_pt_fine: Optional[np.ndarray] = None,
                          residual_field_fine: Optional[np.ndarray] = None,
                          freestream: Optional[object] = None,
                          residual_fields: Optional[List[np.ndarray]] = None) -> str:
    """
    Plot flow field for all multigrid levels in a single multi-page PDF.
    
    Each page shows the same content as plot_flow_field:
    - Pressure (global and zoomed)
    - Velocity magnitude (global and zoomed)
    - Total Pressure Loss C_pt (computed for all levels)
    - Residual field (for all levels if provided)
    
    Parameters
    ----------
    mg_hierarchy : MultigridHierarchy
        The multigrid hierarchy object containing all levels.
    X_fine, Y_fine : ndarray
        Node coordinates for finest grid (for proper pcolormesh).
    iteration : int
        Current iteration number.
    residual : float
        Current residual value.
    cfl : float
        Current CFL number.
    output_dir : str
        Output directory for PDF files.
    case_name : str
        Base name for output files.
    C_pt_fine : ndarray, optional
        Total pressure loss for finest level (used if freestream not provided).
    residual_field_fine : ndarray, optional
        Residual field for finest level (used if residual_fields not provided).
    freestream : FreestreamConditions, optional
        Freestream conditions to compute C_pt for all levels.
    residual_fields : list of ndarray, optional
        Residual field for each level (same length as mg_hierarchy.levels).
        
    Returns
    -------
    output_path : str
        Path to saved PDF file.
    """
    plt = _ensure_matplotlib()
    from matplotlib.backends.backend_pdf import PdfPages
    import warnings
    
    # Import for C_pt computation
    from ..numerics.diagnostics import compute_total_pressure_loss
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f'{case_name}_iter{iteration:06d}.pdf')
    
    # Suppress pcolormesh warnings for curvilinear grids
    with warnings.catch_warnings(), PdfPages(output_path) as pdf:
        warnings.filterwarnings('ignore', category=UserWarning, 
                                message='.*pcolormesh.*monotonically.*')
        
        for level_idx, level in enumerate(mg_hierarchy.levels):
            # Get level data
            NI, NJ = level.NI, level.NJ
            Q = level.Q
            
            # Strip ghost cells (2 J-ghosts at wall/wake, 1 at farfield)
            if Q.shape[0] == NI + 2 and Q.shape[1] == NJ + 3:
                Q_int = Q[1:-1, 2:-1, :]
            else:
                Q_int = Q
            
            # Get grid coordinates for this level
            step = 2 ** level_idx
            if level_idx == 0:
                X, Y = X_fine, Y_fine
            else:
                X = X_fine[::step, ::step]
                Y = Y_fine[::step, ::step]
            
            # Compute C_pt for this level
            if freestream is not None:
                C_pt = compute_total_pressure_loss(
                    Q_int, freestream.p_inf, freestream.u_inf, freestream.v_inf
                )
            elif level_idx == 0 and C_pt_fine is not None:
                C_pt = C_pt_fine
            else:
                C_pt = None
            
            # Get residual field for this level
            if residual_fields is not None and level_idx < len(residual_fields):
                residual_field = residual_fields[level_idx]
            elif level_idx == 0 and residual_field_fine is not None:
                residual_field = residual_field_fine
            else:
                # Use forcing term as proxy for coarse level residual
                if level_idx > 0 and hasattr(level, 'forcing') and level.forcing is not None:
                    residual_field = level.forcing
                else:
                    residual_field = None
            
            # Extract fields
            p = Q_int[:, :, 0]
            u = Q_int[:, :, 1]
            v = Q_int[:, :, 2]
            vel_mag = np.sqrt(u**2 + v**2)
            
            # Determine number of rows for this level
            n_rows = 2  # Pressure + Velocity always
            if C_pt is not None:
                n_rows += 1
            if residual_field is not None:
                n_rows += 1
            
            fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
            if n_rows == 2:
                axes = axes.reshape(2, 2)
            
            level_title = f"Level {level_idx}: {NI}x{NJ} cells"
            fig.suptitle(f'{level_title} - Iter {iteration}, Res={residual:.2e}, CFL={cfl:.2f}', 
                        fontsize=14, fontweight='bold')
            
            # --- Row 1: Pressure ---
            ax = axes[0, 0]
            p_clip = np.clip(p, np.percentile(p, 1), np.percentile(p, 99))
            pc = ax.pcolormesh(X.T, Y.T, p_clip.T, cmap='RdBu_r', shading='flat', rasterized=True)
            ax.set_aspect('equal')
            ax.set_title('Pressure (Global)')
            ax.set_xlabel('x/c')
            ax.set_ylabel('y/c')
            plt.colorbar(pc, ax=ax, shrink=0.8, label='p')
            
            ax = axes[0, 1]
            pc = ax.pcolormesh(X.T, Y.T, p_clip.T, cmap='RdBu_r', shading='flat', rasterized=True)
            ax.set_aspect('equal')
            ax.set_xlim(-0.2, 1.4)
            ax.set_ylim(-0.4, 0.4)
            ax.set_title('Pressure (Near Airfoil)')
            ax.set_xlabel('x/c')
            ax.set_ylabel('y/c')
            plt.colorbar(pc, ax=ax, shrink=0.8, label='p')
            
            # --- Row 2: Velocity Magnitude ---
            ax = axes[1, 0]
            vel_clip = np.clip(vel_mag, 0, np.percentile(vel_mag, 99))
            pc = ax.pcolormesh(X.T, Y.T, vel_clip.T, cmap='viridis', shading='flat', rasterized=True)
            ax.set_aspect('equal')
            ax.set_title('Velocity Magnitude (Global)')
            ax.set_xlabel('x/c')
            ax.set_ylabel('y/c')
            plt.colorbar(pc, ax=ax, shrink=0.8, label='|V|')
            
            ax = axes[1, 1]
            pc = ax.pcolormesh(X.T, Y.T, vel_clip.T, cmap='viridis', shading='flat', rasterized=True)
            ax.set_aspect('equal')
            ax.set_xlim(-0.2, 1.4)
            ax.set_ylim(-0.4, 0.4)
            ax.set_title('Velocity Magnitude (Near Airfoil)')
            ax.set_xlabel('x/c')
            ax.set_ylabel('y/c')
            plt.colorbar(pc, ax=ax, shrink=0.8, label='|V|')
            
            row_idx = 2
            
            # --- Total Pressure Loss (finest level only) ---
            if C_pt is not None:
                C_pt_abs_max = max(np.abs(np.percentile(C_pt, 1)), np.abs(np.percentile(C_pt, 99)))
                C_pt_abs_max = max(C_pt_abs_max, 0.01)
                
                ax = axes[row_idx, 0]
                pc = ax.pcolormesh(X.T, Y.T, C_pt.T, cmap='RdBu_r', shading='flat',
                                   rasterized=True, vmin=-C_pt_abs_max, vmax=C_pt_abs_max)
                ax.set_aspect('equal')
                ax.set_title('Total Pressure Loss C_pt (Global)')
                ax.set_xlabel('x/c')
                ax.set_ylabel('y/c')
                plt.colorbar(pc, ax=ax, shrink=0.8, label='C_pt')
                
                ax = axes[row_idx, 1]
                pc = ax.pcolormesh(X.T, Y.T, C_pt.T, cmap='RdBu_r', shading='flat',
                                   rasterized=True, vmin=-C_pt_abs_max, vmax=C_pt_abs_max)
                ax.set_aspect('equal')
                ax.set_xlim(-0.2, 1.4)
                ax.set_ylim(-0.4, 0.4)
                ax.set_title('Total Pressure Loss C_pt (Near Airfoil)')
                ax.set_xlabel('x/c')
                ax.set_ylabel('y/c')
                plt.colorbar(pc, ax=ax, shrink=0.8, label='C_pt')
                
                row_idx += 1
            
            # --- Residual Field (finest level only) ---
            if residual_field is not None:
                R_p = np.abs(residual_field[:, :, 0])
                R_log = np.log10(R_p + 1e-12)
                R_min = np.percentile(R_log, 1)
                R_max = np.percentile(R_log, 99)
                
                ax = axes[row_idx, 0]
                pc = ax.pcolormesh(X.T, Y.T, R_log.T, cmap='hot_r', shading='flat',
                                   rasterized=True, vmin=R_min, vmax=R_max)
                ax.set_aspect('equal')
                ax.set_title('log₁₀|Residual| (Global)')
                ax.set_xlabel('x/c')
                ax.set_ylabel('y/c')
                plt.colorbar(pc, ax=ax, shrink=0.8, label='log₁₀|R_p|')
                
                ax = axes[row_idx, 1]
                pc = ax.pcolormesh(X.T, Y.T, R_log.T, cmap='hot_r', shading='flat',
                                   rasterized=True, vmin=R_min, vmax=R_max)
                ax.set_aspect('equal')
                ax.set_xlim(-0.2, 1.4)
                ax.set_ylim(-0.4, 0.4)
                ax.set_title('log₁₀|Residual| (Near Airfoil)')
                ax.set_xlabel('x/c')
                ax.set_ylabel('y/c')
                
                # Mark max residual location
                max_idx = np.unravel_index(np.argmax(R_p), R_p.shape)
                x_max = 0.5 * (X[max_idx[0], max_idx[1]] + X[max_idx[0]+1, max_idx[1]])
                y_max = 0.5 * (Y[max_idx[0], max_idx[1]] + Y[max_idx[0]+1, max_idx[1]])
                ax.plot(x_max, y_max, 'g*', markersize=15, markeredgecolor='white', markeredgewidth=1)
                plt.colorbar(pc, ax=ax, shrink=0.8, label='log₁₀|R_p|')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, dpi=100)
            plt.close(fig)
    
    return output_path


def plot_surface_distributions(x: np.ndarray, Cp: np.ndarray, Cf: np.ndarray,
                                output_dir: str, case_name: str = "surface",
                                reference_data: Optional[dict] = None) -> str:
    """
    Plot surface Cp and Cf distributions.
    
    Parameters
    ----------
    x : ndarray
        Surface x-coordinates.
    Cp : ndarray
        Pressure coefficient.
    Cf : ndarray
        Skin friction coefficient.
    output_dir : str
        Output directory.
    case_name : str
        Base name for output file.
    reference_data : dict, optional
        Reference data for comparison (e.g., from mfoil).
        Should contain 'x_upper', 'x_lower', 'cp_upper', 'cp_lower', etc.
        
    Returns
    -------
    output_path : str
        Path to saved PDF.
    """
    plt = _ensure_matplotlib()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cp distribution (inverted y-axis: suction up)
    ax = axes[0]
    ax.plot(x, Cp, 'b.', markersize=3, label='CFD')
    if reference_data is not None:
        if 'cp_upper' in reference_data:
            ax.plot(reference_data['x_upper'], reference_data['cp_upper'],
                    'r-', lw=1.5, alpha=0.7, label='Reference')
        if 'cp_lower' in reference_data:
            ax.plot(reference_data['x_lower'], reference_data['cp_lower'],
                    'r-', lw=1.5, alpha=0.7)
    ax.set_xlabel('x/c')
    ax.set_ylabel('Cp')
    ax.set_title('Pressure Coefficient')
    ax.invert_yaxis()  # Suction (negative Cp) at top
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    # Cf distribution
    ax = axes[1]
    ax.plot(x, Cf, 'b.', markersize=3, label='CFD')
    if reference_data is not None:
        if 'cf_upper' in reference_data:
            ax.plot(reference_data['x_upper'], reference_data['cf_upper'],
                    'r-', lw=1.5, alpha=0.7, label='Reference')
        if 'cf_lower' in reference_data:
            ax.plot(reference_data['x_lower'], reference_data['cf_lower'],
                    'r-', lw=1.5, alpha=0.7)
    ax.set_xlabel('x/c')
    ax.set_ylabel('Cf')
    ax.set_title('Skin Friction Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f'{case_name}_surface.pdf')
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    return output_path

